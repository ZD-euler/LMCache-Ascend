# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest
import torch


@pytest.fixture
def formats(monkeypatch):
    # Load the actual format definitions without package/NPU bootstrap.
    logger_module = ModuleType("lmcache.logging")
    logger_module.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "lmcache.logging", logger_module)
    path = Path(__file__).resolve().parents[2] / "lmcache_ascend/v1/kv_format.py"
    return runpy.run_path(str(path))


def test_packed_main_explicit_group_has_opaque_byte_storage(formats):
    cls = formats["PackedMainKVLayout"]
    backing = torch.zeros(1 + 2 * 128 * 656, dtype=torch.int8)
    first = backing[1:].view(2, 128, 1, 656)
    second = torch.empty_like(first)
    layout = cls.from_group([(first,), (second,)], kv_group=0, layout=cls.SCHEMA)
    assert layout.layers == 2 and layout.blocks == 2
    rows = layout.storage_rows((first,))
    assert rows.shape == (256, 656) and rows.dtype == torch.uint8
    assert rows.data_ptr() == first.data_ptr()
    assert rows.storage_offset() == first.storage_offset() == 1
    first[1, 0, 0, 511:513] = torch.tensor([-128, -1], dtype=torch.int8)
    assert rows[128, 511:513].tolist() == [128, 255]
    rows[128, 640] = 254
    assert first[1, 0, 0, 640].item() == -2
    assert layout.chunk_shape(0) == (0,)
    assert layout.chunk_shape(32) == (32 * 656,)
    assert layout.payload_schema(32) == {
        "format": "dsa_packed_main_c8_bf16_rope_fp32_scale_v1",
        "layers": 2,
        "chunk_size": 32,
        "storage_dtype": "torch.uint8",
        "token_bytes": 656,
        "page_bytes": 2 * 32 * 656,
    }


def test_packed_main_schema_does_not_depend_on_device_addresses(formats):
    cls = formats["PackedMainKVLayout"]
    a = torch.empty(2, 128, 1, 656, dtype=torch.int8)
    b = torch.empty(3, 128, 1, 656, dtype=torch.int8)
    layout_a = cls.from_group([(a,)], kv_group=0, layout=cls.SCHEMA)
    layout_b = cls.from_group([(b,)], kv_group=0, layout=cls.SCHEMA)
    assert layout_a.payload_schema(128) == layout_b.payload_schema(128)
    with pytest.raises(ValueError, match="capacity and device"):
        layout_a.storage_rows((b,))
    with pytest.raises(ValueError, match="capacity and device"):
        cls.from_group([(a,), (b,)], kv_group=0, layout=cls.SCHEMA)
    with pytest.raises(ValueError, match="nonnegative integer"):
        layout_a.payload_schema(True)
    with pytest.raises(ValueError, match="nonnegative integer"):
        layout_a.chunk_shape(-1)


@pytest.mark.parametrize(
    ("kv_group", "layout_name"),
    [(1, "packed"), (False, "packed"), (0, None), (0, "DSA_INDEX")],
)
def test_packed_main_requires_both_explicit_signals(formats, kv_group, layout_name):
    cls = formats["PackedMainKVLayout"]
    packed = torch.empty(2, 128, 1, 656, dtype=torch.int8)
    marker = cls.SCHEMA if layout_name == "packed" else layout_name
    with pytest.raises(ValueError, match="group 0 and its explicit schema"):
        cls.from_group([(packed,)], kv_group=kv_group, layout=marker)


def test_packed_main_validates_every_layer_without_legacy_fallback(formats):
    cls = formats["PackedMainKVLayout"]
    packed = torch.empty(2, 128, 1, 656, dtype=torch.int8)
    indexer = torch.empty(2, 128, 1, 128, dtype=torch.int8)
    for bad, error in (
        ((indexer,), "contiguous INT8"),
        ((packed, packed), "one tensor per layer"),
        ((torch.empty(2, 128, 1, 656, dtype=torch.uint8),), "contiguous INT8"),
        ((torch.empty(4, 128, 1, 656, dtype=torch.int8)[::2],), "contiguous INT8"),
        ((packed.to("meta"),), "capacity and device"),
    ):
        with pytest.raises(ValueError, match=error):
            cls.from_group([(packed,), bad], kv_group=0, layout=cls.SCHEMA)
    with pytest.raises(ValueError, match="at least one layer"):
        cls.from_group([], kv_group=0, layout=cls.SCHEMA)
    with pytest.raises(ValueError, match="contiguous INT8"):
        cls.from_group([(packed[:0],)], kv_group=0, layout=cls.SCHEMA)


def test_native_format_values_and_two_group_detection_remain_unchanged(formats):
    fmt = formats["KVCacheFormat"]
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
    latent = fmt.detect([(bf16, bf16)], use_mla=True, dsa_two_groups=True)
    assert latent is fmt.MLA_LATENT
    assert fmt.detect([(bf16,)], use_mla=True, dsa_two_groups=True) is fmt.DSA_INDEX
    separate = fmt.detect([(bf16, bf16)], use_mla=False, dsa_two_groups=False)
    assert separate is fmt.SEPARATE_KV
