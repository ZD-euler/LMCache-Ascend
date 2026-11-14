# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the packed main byte adapter and connector wiring."""

import ast
import hashlib
import json
import logging
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
FORMAT_PATH = ROOT / "lmcache_ascend/v1/kv_format.py"
ADAPTER_PATH = ROOT / "lmcache_ascend/v1/npu_connector/packed_main.py"
CONNECTOR_PATH = ROOT / "lmcache_ascend/v1/npu_connector/npu_connectors.py"


def _load_formats(monkeypatch):
    logger_module = ModuleType("lmcache.logging")
    logger_module.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "lmcache.logging", logger_module)
    return runpy.run_path(str(FORMAT_PATH))


@pytest.fixture
def api(monkeypatch):
    formats = _load_formats(monkeypatch)
    for name in (
        "lmcache_ascend",
        "lmcache_ascend.v1",
        "lmcache_ascend.v1.npu_connector",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    format_module = ModuleType("lmcache_ascend.v1.kv_format")
    for name in ("KVCacheFormat", "PackedMainKVLayout"):
        setattr(format_module, name, formats[name])
    monkeypatch.setitem(sys.modules, "lmcache_ascend.v1.kv_format", format_module)
    adapter = runpy.run_path(str(ADAPTER_PATH))
    return SimpleNamespace(
        KVCacheFormat=formats["KVCacheFormat"],
        PackedMainKVLayout=formats["PackedMainKVLayout"],
        recognize=adapter["recognize_packed_main"],
        Adapter=adapter["PackedMainByteAdapter"],
    )


def _packed(blocks=2):
    return torch.zeros(blocks, 128, 1, 656, dtype=torch.int8)


def test_packed_main_needs_group_zero_and_explicit_schema(api):
    schema = api.PackedMainKVLayout.SCHEMA
    packed = _packed()

    # Shape alone never opts in; regular detection must still run.
    assert api.recognize([(packed,)], kv_group=0, schema=None) is None

    layout = api.recognize([(packed,)], kv_group=0, schema=schema)
    assert isinstance(layout, api.PackedMainKVLayout)
    assert layout.layers == 1 and layout.blocks == 2

    for group, marker in (
        (1, schema),
        (False, schema),
        (0, "DSA_INDEX"),
        (0, ""),
    ):
        with pytest.raises(ValueError, match="group 0 and its explicit schema"):
            api.recognize([(packed,)], kv_group=group, schema=marker)


def test_chunk_geometry_and_page_byte_count(api):
    packed = torch.zeros(3, 128, 1, 656, dtype=torch.int8)
    layout = api.recognize(
        [(packed,), (torch.empty_like(packed),)],
        kv_group=0,
        schema=api.PackedMainKVLayout.SCHEMA,
    )
    adapter = api.Adapter(layout)

    assert adapter.token_bytes == 656
    assert adapter.layer_bytes == 3 * 128 * 656
    assert adapter.chunk_shape(0) == (0,)
    assert adapter.chunk_shape(7) == (7 * 656,)
    assert adapter.chunk_shape(7).numel() == 7 * 656
    assert adapter.chunk_shape(128).numel() == 128 * 656
    assert layout.payload_schema(7)["page_bytes"] == 2 * 7 * 656
    assert layout.payload_schema(7)["token_bytes"] == 656


def test_indexer_and_mla_detection_unchanged(api):
    fmt = api.KVCacheFormat
    assert {item.name: item.value for item in fmt} == {
        "UNDEFINED": 0,
        "MERGED_KV": 1,
        "SEPARATE_KV": 2,
        "MLA_KV": 3,
        "DSA_KV": 4,
        "MLA_LATENT": 5,
        "DSA_INDEX": 6,
    }
    bf16 = torch.empty(2, 128, 1, 128, dtype=torch.bfloat16)
    assert fmt.detect([(bf16,)], use_mla=True, dsa_two_groups=True) is fmt.DSA_INDEX
    assert (
        fmt.detect([(bf16, bf16)], use_mla=True, dsa_two_groups=True)
        is fmt.MLA_LATENT
    )
    assert (
        fmt.detect([(bf16, bf16)], use_mla=False, dsa_two_groups=False)
        is fmt.SEPARATE_KV
    )
    assert fmt.MLA_LATENT.kv_group == 0
    assert fmt.DSA_INDEX.kv_group == 1


def test_gather_scatter_round_trip_is_exact(api):
    generator = torch.Generator().manual_seed(20261114)
    packed = torch.randint(
        -128, 128, (4, 128, 1, 656), dtype=torch.int8, generator=generator
    )
    slots = torch.tensor([0, 1, 127, 128, 255, 511], dtype=torch.int64)
    layout = api.recognize(
        [(packed,)], kv_group=0, schema=api.PackedMainKVLayout.SCHEMA
    )
    adapter = api.Adapter(layout)

    view = adapter.layer_view((packed,))
    assert view.dtype == torch.uint8
    assert view.numel() == adapter.layer_bytes == 4 * 128 * 656
    assert view.data_ptr() == packed.data_ptr()

    saved = adapter.gather((packed,), slots)
    assert saved.dtype == torch.uint8
    assert saved.numel() == len(slots) * 656
    rows = packed.view(torch.uint8).reshape(-1, 656)
    assert torch.equal(saved.view(-1, 656), rows[slots])

    restored = torch.full_like(packed, 7)
    adapter.scatter((restored,), slots, saved)
    restored_rows = restored.view(torch.uint8).reshape(-1, 656)
    assert torch.equal(restored_rows[slots], saved.view(-1, 656))
    untouched = torch.ones(4 * 128, dtype=torch.bool)
    untouched[slots] = False
    assert torch.all(restored_rows[untouched] == 7)

    zeroed = torch.zeros_like(packed)
    adapter.scatter((zeroed,), slots, saved)
    assert torch.equal(
        zeroed.view(torch.uint8).reshape(-1, 656)[slots],
        packed.view(torch.uint8).reshape(-1, 656)[slots],
    )


@pytest.mark.parametrize(
    "payload_builder, message",
    [
        (lambda saved: saved[:-1], "contiguous uint8 bytes"),
        (lambda saved: saved.to(torch.int8), "contiguous uint8 bytes"),
        (
            lambda saved: saved.view(-1, 2, 328)[:, ::2].reshape(-1),
            "contiguous uint8 bytes",
        ),
    ],
)
def test_scatter_rejects_malformed_chunks(api, payload_builder, message):
    packed = _packed()
    slots = torch.tensor([0, 2], dtype=torch.int64)
    layout = api.recognize(
        [(packed,)], kv_group=0, schema=api.PackedMainKVLayout.SCHEMA
    )
    adapter = api.Adapter(layout)
    saved = adapter.gather((packed,), slots)
    with pytest.raises(ValueError, match=message):
        adapter.scatter((packed,), slots, payload_builder(saved))


def test_adapter_rejects_bad_slots(api):
    packed = _packed()
    layout = api.recognize(
        [(packed,)], kv_group=0, schema=api.PackedMainKVLayout.SCHEMA
    )
    adapter = api.Adapter(layout)
    for slots in (
        torch.tensor([[0]], dtype=torch.int64),
        torch.tensor([0.0]),
        torch.tensor([0], dtype=torch.int16),
    ):
        with pytest.raises(ValueError, match="one-dimensional integer"):
            adapter.gather((packed,), slots)
    with pytest.raises(ValueError, match="outside the registered"):
        adapter.gather((packed,), torch.tensor([256], dtype=torch.int64))
    with pytest.raises(ValueError, match="outside the registered"):
        adapter.gather((packed,), torch.tensor([-1], dtype=torch.int64))
    with pytest.raises(ValueError, match="must be unique"):
        adapter.gather((packed,), torch.tensor([0, 0], dtype=torch.int64))


@pytest.fixture
def connector(api):
    tree = ast.parse(CONNECTOR_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    wanted = {
        "_initialize_packed_main_buffer",
        "_lazy_initialize_buffer",
        "get_shape",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_GroupLayout"
    ]
    body += [
        node
        for node in cls.body
        if getattr(node, "name", None) in wanted
    ]
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + body,
        type_ignores=[],
    )
    env = {
        "torch": torch,
        "hashlib": hashlib,
        "json": json,
        "logger": logging.getLogger("test_packed_main"),
        "KVCacheFormat": api.KVCacheFormat,
        "PackedMainKVLayout": api.PackedMainKVLayout,
        "recognize_packed_main": api.recognize,
    }
    exec(compile(module, str(CONNECTOR_PATH), "exec"), env)
    assert wanted <= set(env)
    assert "_GroupLayout" in env
    return env


class FakeConnector:
    def __init__(self, schema, chunk_size=128, use_gpu=True):
        self.packed_main_schema = schema
        self.lmcache_chunk_size = chunk_size
        self.use_gpu = use_gpu
        self._group_layouts = {}
        self._current_kv_group = 0
        self.kv_format = None
        self.kv_device = None
        self.gpu_buffer_allocator = None
        self.allocations = []
        self.resets = 0
        self.mirrored = []

    def _reset_sparse_direct_layer_states(self):
        self.resets += 1

    def _mirror_layout(self, layout):
        self.mirrored.append(layout)
        self.kv_format = layout.kv_format
        self.kv_device = layout.kv_device
        self.gpu_buffer_allocator = layout.gpu_buffer_allocator

    def _layerwise_staging_num_tokens(self, max_tokens):
        return min(max_tokens, 512)

    def _layerwise_staging_pool_slots(self):
        return 2

    def _assign_group_gpu_allocator(self, size, layout):
        layout.gpu_buffer_allocator = size
        self.gpu_buffer_allocator = size
        self.allocations.append(size)


def test_connector_initializes_opaque_group_zero(connector, api):
    schema = api.PackedMainKVLayout.SCHEMA
    caches = [(_packed(3),), (_packed(3),)]
    fake = FakeConnector(schema)
    layout = connector["_initialize_packed_main_buffer"](
        fake, caches, kv_group=0, init_staging=True
    )

    assert layout is fake._group_layouts[0]
    assert layout.kv_format is api.KVCacheFormat.UNDEFINED
    assert layout.storage_dtype is torch.uint8
    assert layout.num_layers == 2 and layout.layer_indices == (0, 1)
    assert layout.layer_token_bytes == (656, 656)
    assert layout.packed_main.layers == 2 and layout.packed_main.blocks == 3
    assert layout.kv_device == caches[0][0].device
    assert len(layout.layout_signature) == 16
    assert layout.staging_bytes_per_slot == 384 * 656
    assert layout.gpu_buffer_allocator == 2 * 384 * 656
    assert fake.allocations == [2 * 384 * 656]
    assert fake.resets == 1
    assert fake.mirrored[-1] is layout

    again = connector["_initialize_packed_main_buffer"](
        fake, caches, kv_group=0, init_staging=True
    )
    assert again is layout
    assert fake.allocations == [2 * 384 * 656]
    assert fake.resets == 1


def test_connector_dispatch_uses_explicit_schema(connector, api):
    calls = []

    class Fake:
        packed_main_schema = api.PackedMainKVLayout.SCHEMA

        def _initialize_packed_main_buffer(
            self, caches, *, kv_group, init_staging
        ):
            calls.append((caches, kv_group, init_staging))
            return "packed-layout"

    caches = [(_packed(),)]
    dispatch = connector["_lazy_initialize_buffer"]
    assert dispatch(Fake(), caches, kv_group=0) == "packed-layout"
    assert dispatch(Fake(), caches, kv_group=0, init_staging=True) == (
        "packed-layout"
    )
    assert calls == [(caches, 0, None), (caches, 0, True)]


def test_connector_rejects_packed_main_for_group_one(connector, api):
    fake = FakeConnector(api.PackedMainKVLayout.SCHEMA)
    with pytest.raises(ValueError, match="group 0 and its explicit schema"):
        connector["_initialize_packed_main_buffer"](
            fake, [(_packed(),)], kv_group=1, init_staging=False
        )
    assert fake._group_layouts == {}


def test_connector_dispatch_keeps_group_one_on_normal_path(connector, api):
    fake = FakeConnector(api.PackedMainKVLayout.SCHEMA)
    indexer = connector["_GroupLayout"]()
    indexer.kv_format = api.KVCacheFormat.DSA_INDEX
    indexer.gpu_buffer_allocator = "existing"
    fake._group_layouts[1] = indexer
    result = connector["_lazy_initialize_buffer"](
        fake, [(_packed(),)], kv_group=1, init_staging=False
    )
    assert result is indexer
    assert indexer.packed_main is None
    assert 0 not in fake._group_layouts


def test_connector_get_shape_uses_byte_geometry(connector, api):
    packed = connector["_GroupLayout"]()
    packed.packed_main = api.recognize(
        [(_packed(3),)], kv_group=0, schema=api.PackedMainKVLayout.SCHEMA
    )
    self = SimpleNamespace(
        _current_kv_group=0,
        _group_layouts={0: packed},
        kv_format=api.KVCacheFormat.UNDEFINED,
        k_hidden_dims=0,
        v_hidden_dims=0,
        dsa_hidden_dims=0,
        hidden_dim_size=128,
    )
    get_shape = connector["get_shape"]
    assert get_shape(self, 5) == (5 * 656,)
    assert get_shape(self, 5, 0) == (5 * 656,)
    assert get_shape(self, 0, 0) == (0,)

    indexer = connector["_GroupLayout"]()
    indexer.kv_format = api.KVCacheFormat.DSA_INDEX
    indexer.dsa_hidden_dims = 128
    self._group_layouts[1] = indexer
    assert get_shape(self, 5, 1) == (5 * 128,)
