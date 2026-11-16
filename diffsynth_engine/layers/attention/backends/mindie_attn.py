from functools import lru_cache

import torch

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from diffsynth_engine.layers.attention.backends.ascend_attn import AscendFusedAttentionImpl
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


@lru_cache(maxsize=1)
def _use_mindie_impl() -> bool:
    """Whether MindIE's kernel can actually run here.

    Probing allocates an NPU tensor and runs one attention pass, and the answer
    is asked for every time a ``USPAttention`` is constructed, so cache it.
    """
    from diffsynth_engine.platforms.ascend import is_mindie_attention_available

    try:
        available = bool(is_mindie_attention_available())
    except Exception:
        available = False
    if not available:
        # Say so rather than silently switching kernels: "mindie" then means
        # "the Ascend fused kernel", which is also the only path that can drive ring.
        logger.warning(
            "MindIE-SD kernels are unavailable; the mindie attention backend falls back to "
            "torch_npu's fused attention (ring attention stays available)."
        )
    return available


class MindieAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def build(self, **kwargs) -> AttentionMetadata:
        return AttentionMetadata()


class MindieAttentionBackend(AttentionBackend):
    """Ascend attention backend: MindIE when it is usable, the fused kernel otherwise.

    MindIE-SD is preferred because it is the vendor-tuned path for this device.
    It is not a hard requirement: ``get_impl_cls`` falls back to the fused kernel
    when MindIE is missing or fails its probe, which keeps the backend usable on a
    plain torch_npu install. The fallback is not only a degradation -- MindIE's
    ``attention_forward`` returns no softmax statistics, so the fused kernel is
    what makes ring attention possible on Ascend.
    """

    @staticmethod
    def check_availability() -> None:
        from diffsynth_engine.platforms.ascend import AscendPlatform

        if not AscendPlatform.is_available():
            error_msg = "The mindie attention backend requires an available Ascend NPU device."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def get_type() -> str:
        return str(AttentionType.MINDIE)

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return MindieAttentionImpl if _use_mindie_impl() else AscendFusedAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return MindieAttentionMetadataBuilder

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []

    @classmethod
    def supports_ring_attention(cls) -> bool:
        # Ring merges per-shard softmax statistics, so it follows whichever impl
        # is selected rather than the backend name.
        return cls.get_impl_cls() is AscendFusedAttentionImpl


class MindieAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        **extra_impl_args,
    ) -> None:
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.num_heads = num_heads
        self.head_size = head_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from mindiesd.layers.flash_attn.attention_forward import attention_forward

        # MindIE attention_forward 没有 causal 参数，只接受 attn_mask（布尔张量，
        # True 表示保留/参与计算，False 表示被屏蔽）。因此当请求 causal 且调用方未显式
        # 传入 attn_mask 时，手动构造下三角 causal mask 并通过 attn_mask 传入。
        # layout 为 "BSND"，故 q_seqlen = query.shape[1]、kv_seqlen = key.shape[1]。
        if self.causal and attn_mask is None:
            q_seqlen = query.shape[1]
            kv_seqlen = key.shape[1]
            attn_mask = torch.tril(torch.ones(q_seqlen, kv_seqlen, dtype=torch.bool, device=query.device))

        return attention_forward(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            scale=self.softmax_scale,
            fused=True,
            head_first=False,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout="BSND",
        )
