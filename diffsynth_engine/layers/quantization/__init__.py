"""8-bit quantization layers.

Named without a device so the same entry points serve every backend; the kernels
come from ``diffsynth_engine.platforms``.
"""

from diffsynth_engine.layers.quantization.fp8_linear import (
    enable_fp8_linear,
    is_quantized,
    quantized_linear,
)

__all__ = ["enable_fp8_linear", "is_quantized", "quantized_linear"]
