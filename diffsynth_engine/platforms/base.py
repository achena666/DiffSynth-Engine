from __future__ import annotations

from abc import ABC
from typing import Any

import torch


class PlatformBackend(ABC):
    name = "unknown"
    device_type = "cpu"

    op_fusion = False

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def normalize_device(cls, device: str | torch.device) -> torch.device:
        return torch.device(device)

    @classmethod
    def get_device(cls, local_rank: int) -> torch.device:
        return torch.device(cls.device_type)

    @classmethod
    def set_device(cls, index: int | str | torch.device) -> None:
        return None

    @classmethod
    def device_count(cls) -> int:
        return 1

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        return torch.float8_e4m3fn

    @classmethod
    def synchronize(cls) -> None:
        return None

    @classmethod
    def empty_cache(cls) -> None:
        return None

    @classmethod
    def pin_memory(cls, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.pin_memory()

    @classmethod
    def distributed_backend(cls) -> str:
        return "gloo"

    @classmethod
    def compile_backend(cls) -> Any | None:
        return None

    @classmethod
    def compile_kwargs(cls) -> dict[str, Any]:
        backend = cls.compile_backend()
        return {} if backend is None else {"backend": backend}

    # ------------------------------------------------------------------
    # Compute ops.
    #
    # These are the kernels whose implementation differs per platform. Callers
    # in layers/ go through them instead of importing a vendor library, so a
    # backend never has to ask which device it is running on. Platforms that do
    # not provide an op keep the default and the caller falls back.
    # ------------------------------------------------------------------

    @classmethod
    def supports_fused_attention(cls) -> bool:
        """Whether ``fused_attention`` is implemented for this platform."""
        return False

    @classmethod
    def fused_attention(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        num_heads: int,
        scale: float,
        causal: bool = False,
        attn_mask: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Attention through the platform's fused kernel.

        Args:
            query, key, value: ``[B, S, H, D]``.
            num_heads: number of query heads (``query.shape[2]``).
            scale: softmax scale.
            causal: build a causal mask inside the kernel when ``attn_mask`` is None.
            attn_mask: boolean mask, True = keep.
            return_lse: also return the log-sum-exp that ring attention needs in
                order to merge per-shard partial results.

        Returns:
            ``output``, or ``(output, lse)`` with ``lse`` shaped ``[B, H, S]``.
        """
        raise NotImplementedError(f"{cls.name} does not provide a fused attention kernel.")

    @classmethod
    def dynamic_quant(cls, x: torch.Tensor, *, fp8: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token online quantization of activations.

        Returns:
            ``(quantized, scale)`` with ``scale`` in float32, one value per row.
        """
        raise NotImplementedError(f"{cls.name} does not implement dynamic quantization.")

    @classmethod
    def quant_matmul(
        cls,
        x_q: torch.Tensor,
        weight_q: torch.Tensor,
        weight_scale: torch.Tensor,
        act_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """8-bit matmul with per-channel weight scale and per-token activation scale."""
        raise NotImplementedError(f"{cls.name} does not implement quantized matmul.")
