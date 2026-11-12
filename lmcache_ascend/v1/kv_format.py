# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, List, Tuple, Union

# Third Party
from lmcache.logging import init_logger
import torch

logger = init_logger(__name__)


@dataclass(frozen=True)
class PackedMainKVLayout:
    """Python-only layout for the packed main group.

    Do not send this layout to native APIs that expect KVCacheFormat.
    Packed rows are opaque bytes in storage, with NOPE, RoPE, and scales
    inline. There is no separate key, value, or scale plane.
    """

    SCHEMA: ClassVar[str] = "dsa_packed_main_c8_bf16_rope_fp32_scale_v1"
    PAGE_TOKENS: ClassVar[int] = 128
    ROW_BYTES: ClassVar[int] = 512 + 64 * 2 + 4 * 4
    STORAGE_DTYPE: ClassVar[torch.dtype] = torch.uint8

    layers: int
    blocks: int
    device: torch.device

    @classmethod
    def from_group(
        cls,
        kvcaches: List[Tuple[torch.Tensor, ...]],
        *,
        kv_group: int,
        layout: str,
    ) -> "PackedMainKVLayout":
        """Recognize packed main only with an explicit main group and schema."""
        if type(kv_group) is not int or kv_group != 0 or layout != cls.SCHEMA:
            raise ValueError("Packed main KV requires group 0 and its explicit schema")
        if not kvcaches:
            raise ValueError("Packed main KV requires at least one layer")
        first = cls._packed_tensor(kvcaches[0])
        result = cls(layers=len(kvcaches), blocks=first.shape[0], device=first.device)
        for layer in kvcaches[1:]:
            result._check_layer(layer)
        return result

    @classmethod
    def _packed_tensor(cls, layer: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not isinstance(layer, tuple) or len(layer) != 1:
            raise ValueError("Packed main KV requires one tensor per layer")
        (tensor,) = layer
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.int8
            or tensor.ndim != 4
            or tensor.shape[0] <= 0
            or tensor.shape[1:] != (cls.PAGE_TOKENS, 1, cls.ROW_BYTES)
            or not tensor.is_contiguous()
        ):
            raise ValueError("Packed main KV requires contiguous INT8 [blocks, 128, 1, 656]")
        return tensor

    def _check_layer(self, layer: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        tensor = self._packed_tensor(layer)
        if tensor.shape[0] != self.blocks or tensor.device != self.device:
            raise ValueError("Packed main KV layers must agree on capacity and device")
        return tensor

    def storage_rows(self, layer: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Expose one layer's rows as uint8 with identical bytes and storage."""
        return self._check_layer(layer).view(self.STORAGE_DTYPE).view(-1, self.ROW_BYTES)

    def chunk_shape(self, num_tokens: int) -> torch.Size:
        """Return the one-dimensional byte buffer shape for one layer."""
        if type(num_tokens) is not int or num_tokens < 0:
            raise ValueError("Packed main KV chunk length must be a nonnegative integer")
        return torch.Size((num_tokens * self.ROW_BYTES,))

    def payload_schema(self, chunk_size: int) -> dict:
        """Describe serialized bytes independently of runtime HBM addresses."""
        return {
            "format": self.SCHEMA,
            "layers": self.layers,
            "chunk_size": chunk_size,
            "storage_dtype": str(self.STORAGE_DTYPE),
            "token_bytes": self.ROW_BYTES,
            "page_bytes": self.layers * self.chunk_shape(chunk_size).numel(),
        }


class KVCacheFormat(Enum):
    """
    The storage format enumeration of KV cache is used to distinguish
    the KV cache data structures of different versions of vLLM.

    The order of enum values MUST match the KVCacheFormat
    definition in kernels/types.h to ensure correct interoperability
    between Python and C++ code.
    """

    UNDEFINED = 0

    MERGED_KV = auto()
    """Merge format (eg: vLLM 0.9.2 ...)
    layer: [num_kv, num_blocks, block_size, num_heads, head_dim]
    """

    SEPARATE_KV = auto()
    """Separation format (eg: vLLM 0.11.0+ ...)
    layer: tuple: (K_tensor, V_tensor)
    - K_tensor.shape = [num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [num_blocks, block_size, num_heads, head_dim]

    eg: kvcaches[0] = (K, V)

    SGLang NPU Layer-Concatenated
    kvcaches = [K_all_layers, V_all_layers]
    - K_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    """

    MLA_KV = auto()
    """MLA format for DeepSeek V2/V3 models
    layer: tuple: (k_cache, v_cache) where K and V have different dimensions
    - k_cache.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - v_cache.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]

    This format is used when K/V shapes differ (detected automatically).
    """

    DSA_KV = auto()
    """DSA (Deep Sparse Attention) format for DeepSeek V3.2 sparse models
    layer: tuple: (k_cache, v_cache, dsa_k_cache)
    - k_cache.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - v_cache.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]
    - dsa_k_cache.shape = [num_blocks, block_size, 1, 128]

    This format is used for sparse attention with lightning indexer.
    Legacy bundled path: latent and indexer packed into one 3-tuple.
    """

    MLA_LATENT = auto()
    """MLA latent-only format for the two-group DSA path.
    layer: tuple: (k_nope, k_pe) where the two have different dimensions
    - k_nope.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - k_pe.shape   = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]

    The indexer key is stored separately (see DSA_INDEX). CPU chunk layout is
    1D stacked planes: [k_nope plane: N*kH | k_pe plane: N*vH].
    """

    DSA_INDEX = auto()
    """DSA indexer-only format for the two-group DSA path.
    layer: tuple: (indexer_k,) — a single tensor
    - indexer_k.shape = [num_blocks, block_size, 1, index_head_dim]

    CPU chunk layout is 1D single plane: [indexer plane: N*dsaH].
    Naturally append-friendly (single plane, no V-plane shifting).
    """

    def is_separate_format(self) -> bool:
        return self == KVCacheFormat.SEPARATE_KV

    def is_merged_format(self) -> bool:
        return self == KVCacheFormat.MERGED_KV

    def is_mla_format(self) -> bool:
        return self in (KVCacheFormat.MLA_KV, KVCacheFormat.MLA_LATENT)

    def is_mla_latent_format(self) -> bool:
        return self == KVCacheFormat.MLA_LATENT

    def is_dsa_format(self) -> bool:
        return self in (KVCacheFormat.DSA_KV, KVCacheFormat.DSA_INDEX)

    def is_dsa_index_format(self) -> bool:
        return self == KVCacheFormat.DSA_INDEX

    def is_tuple_format(self) -> bool:
        return self in (
            KVCacheFormat.SEPARATE_KV,
            KVCacheFormat.MLA_KV,
            KVCacheFormat.DSA_KV,
            KVCacheFormat.MLA_LATENT,
            KVCacheFormat.DSA_INDEX,
        )

    def get_kv_size(self) -> int:
        if self == KVCacheFormat.DSA_KV:
            return 3
        elif self in (KVCacheFormat.SEPARATE_KV,
                      KVCacheFormat.MLA_KV,
                      KVCacheFormat.MLA_LATENT):
            return 2
        elif self in (KVCacheFormat.MERGED_KV, KVCacheFormat.DSA_INDEX):
            return 1
        return 0

    @property
    def kv_group(self) -> int:
        """0 = latent group, 1 = indexer group. Used in LayerCacheEngineKey."""
        if self == KVCacheFormat.DSA_INDEX:
            return 1
        return 0

    @staticmethod
    def detect(
        kvcaches: List[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        use_mla: bool = False,
        dsa_two_groups: bool = False,
    ) -> "KVCacheFormat":
        """
        Automatically detect KV cache format based on data structure.

        Detection logic:
        1. DSA_KV: tuple with 3 elements (k_cache, v_cache, dsa_k_cache) — legacy bundled
        2. DSA_INDEX: 1-tuple (indexer_k,) — two-group indexer (only when dsa_two_groups)
        3. MLA_LATENT: tuple with 2 elements in MLA two-group mode
        4. MLA_KV: tuple with 2 elements in MLA mode
        5. SEPARATE_KV: tuple with 2 elements where K/V shapes are same and
           use_mla is false
        6. MERGED_KV: single tensor with specific shape patterns
        """
        if not kvcaches:
            return KVCacheFormat.UNDEFINED

        first_cache = kvcaches[0]

        # SGLang NPU: kvcaches = [K_tensor, V_tensor]
        if isinstance(kvcaches, list) and len(kvcaches) == 2:
            if isinstance(first_cache, torch.Tensor) and first_cache.ndim == 5:
                return KVCacheFormat.SEPARATE_KV

        if isinstance(first_cache, tuple):
            tuple_len = len(first_cache)

            # DSA_KV: tuple with 3 elements (k_cache, v_cache, dsa_k_cache) — legacy bundled
            if tuple_len == 3:
                k_cache, v_cache, dsa_k_cache = first_cache
                if all(isinstance(t, torch.Tensor) for t in first_cache):
                    if k_cache.shape != v_cache.shape:
                        logger.debug(
                            f"Detected DSA_KV format: k_shape={k_cache.shape}, "
                            f"v_shape={v_cache.shape}, dsa_k_shape={dsa_k_cache.shape}"
                        )
                        return KVCacheFormat.DSA_KV

            # DSA_INDEX: 1-tuple (indexer_k,) — two-group indexer
            if tuple_len == 1 and dsa_two_groups:
                indexer_k = first_cache[0]
                if isinstance(indexer_k, torch.Tensor):
                    logger.debug(
                        f"Detected DSA_INDEX format: "
                        f"indexer_k_shape={indexer_k.shape}"
                    )
                    return KVCacheFormat.DSA_INDEX

            # MLA_LATENT / MLA_KV / SEPARATE_KV: tuple with 2 elements
            if tuple_len == 2:
                k_cache, v_cache = first_cache
                if isinstance(k_cache, torch.Tensor) and isinstance(
                    v_cache, torch.Tensor
                ):
                    # MLA_LATENT or MLA_KV: the tensor pair is semantically MLA.
                    # Shape alone is not enough: at some TP degrees the
                    # sharded latent width can equal the PE width.
                    if use_mla or k_cache.shape != v_cache.shape:
                        if dsa_two_groups:
                            logger.debug(
                                f"Detected MLA_LATENT format (two-group): "
                                f"k_shape={k_cache.shape}, "
                                f"v_shape={v_cache.shape}"
                            )
                            return KVCacheFormat.MLA_LATENT
                        logger.debug(
                            f"Detected MLA_KV format: k_shape={k_cache.shape}, "
                            f"v_shape={v_cache.shape}"
                        )
                        return KVCacheFormat.MLA_KV
                    # SEPARATE_KV: K/V shapes are same
                    return KVCacheFormat.SEPARATE_KV

            return KVCacheFormat.SEPARATE_KV

        elif isinstance(first_cache, torch.Tensor):
            ndim = first_cache.ndim
            shape = first_cache.shape

            # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
            if ndim == 5 and shape[0] == 2:
                return KVCacheFormat.MERGED_KV

            # Flash Infer: [num_blocks, 2, block_size, num_heads, head_size]
            if ndim == 5 and shape[1] == 2:
                return KVCacheFormat.MERGED_KV

        return KVCacheFormat.UNDEFINED
