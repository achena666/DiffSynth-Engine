"""Fused-kernel attention implementation for Ascend.

Selected by the ``mindie`` backend when MindIE-SD is not installed.

Why it exists: MindIE's ``attention_forward`` cannot return softmax statistics,
so it cannot drive ring attention (``sp_ring_degree > 1``). The fused kernel
behind ``npu_fusion_attention`` does return ``softmax_max`` / ``softmax_sum``,
and ``log(sum) + max`` is exactly the log-sum-exp the ring loop needs in order
to merge per-shard partial results. Measured against a float32 reference on
Ascend950PR: output matches SDPA exactly, LSE matches to ~1e-6.

The kernel call itself lives in ``platforms/ascend.py``; this module only adapts
the ``AttentionImpl`` interface to it, so it holds no vendor import.
"""

import torch

from diffsynth_engine.layers.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
)
from diffsynth_engine.platforms import fused_attention

__all__ = ["AscendFusedAttentionImpl"]


class AscendFusedAttentionImpl(AttentionImpl):
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
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_groups = num_heads // num_kv_heads
        self.causal = causal
        # Resolve the default here instead of passing None down: the kernel
        # wants a concrete float and does not apply the 1/sqrt(d) itself.
        self.softmax_scale = softmax_scale if softmax_scale is not None else head_size**-0.5

    def _expand_kv(self, key: torch.Tensor, value: torch.Tensor):
        """GQA/MQA: repeat K/V heads up to the query head count.

        The kernel does accept ``num_key_value_heads``, but repeating is what the
        sdpa backend does and it keeps one code path for every head layout.
        Qwen-Image's DiT is plain MHA, so this is a no-op there.
        """
        if self.num_kv_groups == 1:
            return key, value
        # BSND layout -> head dim is 2 (sdpa transposes to BNSD first, hence dim 1)
        return (
            torch.repeat_interleave(key, self.num_kv_groups, dim=2),
            torch.repeat_interleave(value, self.num_kv_groups, dim=2),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        key, value = self._expand_kv(key, value)
        return fused_attention(
            query,
            key,
            value,
            # after the Ulysses all-to-all the head count is the per-rank one,
            # so read it off the tensor rather than from the constructor
            num_heads=query.shape[2],
            scale=self.softmax_scale,
            causal=self.causal,
            attn_mask=attn_mask,
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
        """Attention plus log-sum-exp, for ring attention to merge shards.

        Returns:
            output: ``[B, S, H, D]``
            lse:    ``[B, H, S]`` -- the layout ``ring.py`` expects (it transposes
                    to ``[B, S, H, 1]`` internally).
        """
        key, value = self._expand_kv(key, value)
        return fused_attention(
            query,
            key,
            value,
            num_heads=query.shape[2],
            scale=self.softmax_scale,
            causal=self.causal,
            attn_mask=attn_mask,
            return_lse=True,
        )
