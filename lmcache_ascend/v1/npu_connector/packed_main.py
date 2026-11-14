# SPDX-License-Identifier: Apache-2.0
"""Opaque byte contract for the packed main KV group.

Packed main rows are already final storage records: NOPE, RoPE and FP32
scales are inline in one 656-byte row. The MLA/DSA connector paths split a
layer into typed key/value/scale planes, so they must never consume these
tensors. This module provides the explicit recognition gate and an
exact-byte gather/scatter adapter for save/restore without a
``KVCacheFormat`` change (the enum maps to the native ABI).

Integration point: ``VLLMPagedMemLayerwiseNPUConnector`` recognizes packed
main in ``_initialize_packed_main_buffer`` only when the caller passes
``packed_main_schema=PackedMainKVLayout.SCHEMA`` and ``kv_group == 0``, and
its ``get_shape`` returns :meth:`PackedMainByteAdapter.chunk_shape`.

Still not wired end-to-end: CPU ``MemoryObj`` creation/metadata for the
opaque format, per-slot staging allocation, NPU D2H/H2D kernel selection,
and routing of the connector ``batched_from_gpu``/``batched_to_gpu`` entry
points to this adapter. Those entry points fail closed with
``NotImplementedError`` while the schema is active.
"""

from typing import Optional, Tuple

import torch

from lmcache_ascend.v1.kv_format import PackedMainKVLayout

Layer = Tuple[torch.Tensor, ...]


def recognize_packed_main(
    kv_caches: list,
    *,
    kv_group: int,
    schema: Optional[str],
) -> Optional[PackedMainKVLayout]:
    """Return the packed layout only when the caller explicitly opts in.

    ``schema is None`` leaves the regular ``KVCacheFormat.detect`` path
    untouched. Any non-null schema is validated by
    :meth:`PackedMainKVLayout.from_group`, which requires group 0 and the
    exact schema marker; shape alone never selects packed main.
    """
    if schema is None:
        return None
    return PackedMainKVLayout.from_group(
        kv_caches, kv_group=kv_group, layout=schema
    )


class PackedMainByteAdapter:
    """Exact-byte chunk access for a recognized packed main group.

    A chunk is one contiguous ``uint8`` buffer of
    ``num_tokens * PackedMainKVLayout.ROW_BYTES`` bytes. Rows are addressed
    by linear token slot, matching the connector ``slot_mapping`` contract.
    """

    __slots__ = ("layout",)

    def __init__(self, layout: PackedMainKVLayout) -> None:
        self.layout = layout

    @property
    def token_bytes(self) -> int:
        """Bytes in one token row."""
        return PackedMainKVLayout.ROW_BYTES

    @property
    def layer_bytes(self) -> int:
        """Bytes in one complete packed layer."""
        return self.layout.blocks * PackedMainKVLayout.PAGE_TOKENS * self.token_bytes

    def chunk_shape(self, num_tokens: int) -> torch.Size:
        """Return the one-dimensional byte buffer shape for ``num_tokens``."""
        return self.layout.chunk_shape(num_tokens)

    def layer_view(self, layer: Layer) -> torch.Tensor:
        """Return a zero-copy flat uint8 view of one packed layer."""
        return self.layout.storage_rows(layer).view(-1)

    def gather(self, layer: Layer, slots: torch.Tensor) -> torch.Tensor:
        """Copy the rows named by ``slots`` into one contiguous byte chunk."""
        rows = self.layout.storage_rows(layer)
        indices = self._slot_indices(slots, rows)
        return rows.index_select(0, indices).reshape(-1)

    def scatter(
        self,
        layer: Layer,
        slots: torch.Tensor,
        payload: torch.Tensor,
    ) -> None:
        """Write one exact byte chunk to the rows named by ``slots``."""
        rows = self.layout.storage_rows(layer)
        indices = self._slot_indices(slots, rows)
        expected = indices.numel() * self.token_bytes
        valid = (
            isinstance(payload, torch.Tensor)
            and payload.dtype == torch.uint8
            and payload.is_contiguous()
            and payload.numel() == expected
        )
        if not valid:
            raise ValueError(
                "Packed main chunk must be "
                f"{expected} contiguous uint8 bytes for "
                f"{indices.numel()} slots, got {_describe(payload)}"
            )
        rows.index_copy_(0, indices, payload.view(-1, self.token_bytes))

    def _slot_indices(
        self, slots: torch.Tensor, rows: torch.Tensor
    ) -> torch.Tensor:
        if (
            not isinstance(slots, torch.Tensor)
            or slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "Packed main slots must be a one-dimensional integer tensor"
            )
        # The connector validates NPU slot maps before dispatch. Only check
        # host tensors here so the adapter does not force a device sync.
        if slots.device.type == "cpu" and slots.numel():
            if int(slots.min()) < 0 or int(slots.max()) >= rows.shape[0]:
                raise ValueError(
                    "Packed main slot is outside the registered block capacity"
                )
            if slots.unique().numel() != slots.numel():
                raise ValueError("Packed main slots must be unique")
        return slots.to(device=rows.device, dtype=torch.int64)


def _describe(value: object) -> str:
    if not isinstance(value, torch.Tensor):
        return type(value).__name__
    return (
        f"dtype={value.dtype} contiguous={value.is_contiguous()} "
        f"numel={value.numel()}"
    )
