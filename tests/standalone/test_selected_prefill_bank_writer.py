# SPDX-License-Identifier: Apache-2.0
"""Host checks for the external writer's bank and allocator handoff."""

import ast
from contextlib import contextmanager
from pathlib import Path
from threading import Thread, current_thread
from types import SimpleNamespace

import pytest


@pytest.fixture
def harness():
    log = []
    device = SimpleNamespace(type="npu", index=0)

    class Stream:
        def __init__(self, name):
            self.name = name
            self.device = device
            self.fail_sync = False

        def wait_event(self, event):
            log.append(("wait", self.name, event.name))

        def synchronize(self):
            log.append(("sync", self.name))
            if self.fail_sync:
                raise RuntimeError("stream fence failed")

    class Event:
        def __init__(self, name="selected", event_device=device):
            self.name = name
            self.device = event_device

        def record(self, stream):
            log.append(("record", stream.name, self.name))

        def synchronize(self):
            log.append(("sync-event", self.name))

    class Owner:
        def is_valid(self):
            return True

        def ref_count_up(self):
            log.append(("retain",))

        def ref_count_down(self):
            log.append(("release",))

    @contextmanager
    def on_stream(stream):
        log.append(("enter", stream.name))
        yield

    compute = Stream("compute")
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (root / "lmcache_ascend/v1/npu_connector/npu_connectors.py").read_text()
    )
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VLLMPagedMemLayerwiseNPUConnector"
    )
    methods = {
        "_layerwise_prefill_transfer_state",
        "_layerwise_prefill_transfer_generation",
        "_check_layerwise_prefill_transfer_generation",
        "reset_layerwise_prefill_transfer_state",
        "_layerwise_prefill_bank",
        "_layerwise_prefill_dma_stream",
        "_synchronize_layerwise_prefill_dma_streams",
        "_require_layerwise_prefill_bank_use",
        "_retain_unfenced_layerwise_prefill_load",
        "wait_for_layerwise_prefill_stores",
        "release_layerwise_prefill_dma_cache",
        "record_layerwise_prefill_bank_use",
        "_check_selected_prefill_thread",
        "close_selected_prefill_bank_save",
        "_check_selected_prefill_native_save",
        "submit_selected_prefill_write",
        "wait_for_layerwise_prefill_load",
        "supports_batched_from_gpu_group",
        "batched_from_gpu_group",
    }
    cls.body = [node for node in cls.body if getattr(node, "name", None) in methods]
    cls.bases = []
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + [cls],
        type_ignores=[],
    )
    env = {
        "torch": SimpleNamespace(npu=SimpleNamespace(
            Event=Event, current_stream=lambda: compute, stream=on_stream,
        )),
        "prefill_start_timing_enabled": lambda: False,
        "current_thread": current_thread,
        "_DENSE_DIRECT_STORE_DISABLE": False,
        "_DENSE_DIRECT_GROUP_STORE_DISABLE": False,
    }
    exec(compile(module, "<prefill-writer>", "exec"), env)
    conn = env[cls.name]()
    conn._layerwise_prefill_dma = True
    conn._selected_prefill_writer_thread = current_thread()
    conn._selected_prefill_submitted_saves = {}
    conn._selected_prefill_closed_saves = set()
    conn._layerwise_prefill_bank_counts = {0: 2, 1: 2}
    conn._layerwise_prefill_dma_streams = {(0, 0): Stream("bank0")}
    conn._layerwise_prefill_load_owners = {}
    conn._layerwise_prefill_load_request_events = {}
    conn._prefill_dma_bound_loads = {}
    conn.kv_device = device
    conn.load_stream = Stream("load")
    conn.store_stream = Stream("store")
    conn._expected_group_layers = lambda _group: 4
    return conn, log, Event, Owner


def test_writer_joins_native_ready_and_retains_host_owner(harness):
    conn, log, Event, Owner = harness
    conn.record_layerwise_prefill_bank_use(0, 0, Event("final-use"))
    source = Owner()
    with pytest.raises(RuntimeError, match="still open"):
        conn.submit_selected_prefill_write(
            2, 0, {"selected": (source,)}, (Event("topk"),),
            lambda: pytest.fail("writer ran before save"),
        )
    with pytest.raises(RuntimeError, match="does not match"):
        conn.close_selected_prefill_bank_save(0, 0, ("dense", "selected"))
    conn._check_selected_prefill_native_save(0, "dense")
    log.append(("save", "bank0"))
    conn._selected_prefill_submitted_saves[0] = {"dense"}
    with pytest.raises(RuntimeError, match="does not match"):
        conn.close_selected_prefill_bank_save(0, 0, ("dense", "selected"))
    conn.close_selected_prefill_bank_save(0, 0, ("dense",))
    with pytest.raises(RuntimeError, match="window closed"):
        conn._check_selected_prefill_native_save(0, "selected")
    load_done = conn._layerwise_prefill_transfer_state()[3]
    load_done[(0, 2)] = {"dense": (0, Event("dense"))}
    complete = conn.submit_selected_prefill_write(
        2, 0, {"selected": (source, source)}, (Event("topk"),),
        lambda: log.append(("enqueue", "bank0")),
    )
    assert log == [
        ("wait", "bank0", "final-use"), ("save", "bank0"),
        ("retain",), ("enter", "bank0"), ("wait", "bank0", "topk"),
        ("enqueue", "bank0"), ("record", "bank0", "selected"),
    ]
    with pytest.raises(RuntimeError, match="precedes"):
        conn.record_layerwise_prefill_bank_use(2, 0, Event("consumer"))
    with pytest.raises(RuntimeError, match="outside|Invalid"):
        conn.wait_for_layerwise_prefill_load(2, 0, request_ids=("dense",))
    conn.wait_for_layerwise_prefill_load(2, 0, request_ids=("dense", "selected"))
    assert log[-2:] == [
        ("wait", "compute", "dense"), ("wait", "compute", "selected"),
    ]
    assert conn._layerwise_prefill_load_request_events["selected"] == [complete]
    with pytest.raises(RuntimeError, match="already submitted"):
        conn.submit_selected_prefill_write(
            2, 0, {"selected": (source,)}, (Event("topk"),), lambda: None,
        )
    conn.release_layerwise_prefill_dma_cache("selected")
    assert log[-2:] == [("sync-event", "selected"), ("release",)]


def test_wrong_generation_device_and_missing_handoff_do_not_enqueue(harness):
    conn, log, Event, Owner = harness
    sources = {"req": (Owner(),)}
    enqueue = lambda: pytest.fail("unexpected write")
    assert conn.submit_selected_prefill_write(0, 0, {}, (), enqueue) is None
    with pytest.raises(RuntimeError, match="reset step"):
        conn.submit_selected_prefill_write(0, 1, sources, (Event(),), enqueue)
    with pytest.raises(ValueError, match="KV device"):
        conn.submit_selected_prefill_write(
            0, 0, sources, (Event(event_device=None),), enqueue,
        )
    with pytest.raises(RuntimeError, match="Missing final-access"):
        conn.submit_selected_prefill_write(2, 0, sources, (Event(),), enqueue)
    assert not log


def test_partial_submission_must_fence_before_abort_releases_owner(harness):
    conn, log, Event, Owner = harness

    def enqueue():
        log.append(("partial enqueue",))
        raise RuntimeError("kernel enqueue failed")

    with pytest.raises(RuntimeError, match="kernel enqueue failed"):
        conn.submit_selected_prefill_write(
            0, 0, {"req": (Owner(),)}, (Event("topk"),), enqueue,
        )
    with pytest.raises(RuntimeError, match="Invalid"):
        conn.wait_for_layerwise_prefill_load(0, 0, request_ids=("req",))
    with pytest.raises(RuntimeError, match="without synchronization"):
        conn.reset_layerwise_prefill_transfer_state(synchronize=False)

    stream = conn._layerwise_prefill_dma_streams[(0, 0)]
    stream.fail_sync = True
    with pytest.raises(RuntimeError, match="stream fence failed"):
        conn.release_layerwise_prefill_dma_cache("req")
    assert ("release",) not in log

    stream.fail_sync = False
    conn.release_layerwise_prefill_dma_cache("req")
    assert log[-2:] == [("sync", "bank0"), ("release",)]
    conn.reset_layerwise_prefill_transfer_state()
    assert conn._layerwise_prefill_transfer_generation(0) == 1


def test_passive_rank_closes_save_window_without_event(harness):
    conn, _, Event, Owner = harness
    conn.record_layerwise_prefill_bank_use(0, 0, Event("final-use"))
    conn.close_selected_prefill_bank_save(0, 0, ())
    with pytest.raises(RuntimeError, match="window closed"):
        conn._check_selected_prefill_native_save(0, "late")
    conn.submit_selected_prefill_write(
        2, 0, {"req": (Owner(),)}, (Event("topk"),), lambda: None,
    )


def test_cross_thread_cleanup_is_rejected_before_releasing_source(harness):
    conn, log, Event, Owner = harness
    conn.submit_selected_prefill_write(
        0, 0, {"req": (Owner(),)}, (Event("topk"),), lambda: None,
    )
    errors = []

    def on_other_thread():
        for operation in (
            lambda: conn.release_layerwise_prefill_dma_cache("req"),
            conn.reset_layerwise_prefill_transfer_state,
        ):
            try:
                operation()
            except RuntimeError as error:
                errors.append(str(error))

    thread = Thread(target=on_other_thread)
    thread.start()
    thread.join()
    assert len(errors) == 2
    assert all("connector worker thread" in error for error in errors)
    assert not any(item[0].startswith("sync") or item == ("release",) for item in log)
    conn.release_layerwise_prefill_dma_cache("req")
    assert log[-2:] == [("sync-event", "selected"), ("release",)]


def test_grouped_store_capability_respects_selected_main_bank(harness):
    conn, _, _, _ = harness
    layout_checks = []

    def supports_layout(group):
        layout_checks.append(group)
        return group in (0, 1)

    conn._is_mla_dsa_format = supports_layout
    assert not conn.supports_batched_from_gpu_group(0)
    assert not layout_checks
    with pytest.raises(RuntimeError, match="requires banked main saves"):
        conn.batched_from_gpu_group([], [], [], kv_group=0)
    assert conn.supports_batched_from_gpu_group(1)
    assert layout_checks == [1]

    del conn._selected_prefill_writer_thread
    assert conn.supports_batched_from_gpu_group()
    assert conn.supports_batched_from_gpu_group(1)
    assert not conn.supports_batched_from_gpu_group(2)
    assert layout_checks == [1, 0, 1, 2]
