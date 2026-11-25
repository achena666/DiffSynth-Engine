import torch

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


class MindieAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def build(self, **kwargs) -> AttentionMetadata:
        return AttentionMetadata()


class MindieAttentionBackend(AttentionBackend):
    @staticmethod
    def check_availability() -> None:
        from diffsynth_engine.platforms.ascend import AscendPlatform, is_mindie_attention_available

        if not AscendPlatform.is_available():
            error_msg = "MindIE attention requires an available Ascend NPU device."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        if not is_mindie_attention_available():
            error_msg = (
                "MindIE attention backend is not available. "
                "Install MindIE-SD 3.x matching the current torch_npu and CANN versions, "
                "and ensure mindiesd.layers.flash_attn.attention_forward works on NPU."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def get_type() -> str:
        return str(AttentionType.MINDIE)

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return MindieAttentionImpl

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
        return True


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

    def forward_with_lse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        if self.causal and attn_mask is None:
            q_seqlen = query.shape[1]
            kv_seqlen = key.shape[1]
            attn_mask = torch.tril(torch.ones(q_seqlen, kv_seqlen, dtype=torch.bool, device=query.device))

        scale = self.softmax_scale if self.softmax_scale is not None else query.shape[-1] ** -0.5
        drop_mask = _prepare_attention_mask(query, key, attn_mask)
        result = torch_npu.npu_fusion_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            atten_mask=drop_mask,
            input_layout="BSND",
            scale=scale,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            head_num=query.shape[2],
        )
        return result[0], _lse_from_softmax_stats(result)


def _prepare_attention_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Convert a keep-mask to the bool drop-mask expected by torch_npu."""
    if attn_mask is None:
        return None

    if attn_mask.dtype == torch.bool:
        drop_mask = ~attn_mask
    else:
        drop_mask = attn_mask < 0

    batch_size, q_seqlen = query.shape[:2]
    kv_seqlen = key.shape[1]
    if drop_mask.dim() == 2:
        if tuple(drop_mask.shape) == (batch_size, kv_seqlen):
            return drop_mask[:, None, None, :]
        if tuple(drop_mask.shape) == (q_seqlen, kv_seqlen):
            return drop_mask[None, None, :, :]
        raise ValueError(
            "2D attn_mask must be key-padding [B, S_kv] or pairwise "
            f"[S_q, S_kv], got {tuple(drop_mask.shape)}"
        )
    if drop_mask.dim() != 4:
        raise ValueError(
            "attn_mask must be [B, S_kv], [S_q, S_kv], or [B, H, S_q, S_kv], "
            f"got {tuple(drop_mask.shape)}"
        )
    return drop_mask


def _lse_from_softmax_stats(result) -> torch.Tensor:
    softmax_max, softmax_sum = result[1], result[2]
    if not torch.is_tensor(softmax_sum) or softmax_sum.numel() == 0:
        raise RuntimeError(
            "npu_fusion_attention returned no softmax statistics, so ring attention cannot merge shards."
        )
    return torch.log(softmax_sum[..., 0].float().clamp(min=1e-30)) + softmax_max[..., 0].float()
