"""MindIE FP8 attention backend tests without requiring NPU hardware."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

import diffsynth_engine.layers.attention.backends.mindie_attn as mindie_attn
from diffsynth_engine.layers.attention.backends.abstract import AttentionType


@pytest.fixture
def quant_attention(monkeypatch):
    quant_attention = Mock()
    modules = {
        "mindiesd": ModuleType("mindiesd"),
        "mindiesd.layers": ModuleType("mindiesd.layers"),
        "mindiesd.layers.flash_attn": ModuleType("mindiesd.layers.flash_attn"),
        "mindiesd.layers.flash_attn.quant_flash_attn": ModuleType("mindiesd.layers.flash_attn.quant_flash_attn"),
    }
    for module in modules.values():
        module.__path__ = []
    modules["mindiesd.layers.flash_attn.quant_flash_attn"].quant_attention = quant_attention
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return quant_attention


def _impl() -> mindie_attn.MindieAttentionFp8Impl:
    return mindie_attn.MindieAttentionFp8Impl(
        num_heads=2,
        head_size=4,
        num_kv_heads=1,
        softmax_scale=0.25,
    )


def test_fp8_backend_is_non_ring_and_uses_fp8_impl():
    assert mindie_attn.MindieAttentionFp8Backend.get_type() == str(AttentionType.MINDIE_FP8)
    assert mindie_attn.MindieAttentionFp8Backend.get_impl_cls() is mindie_attn.MindieAttentionFp8Impl
    assert mindie_attn.MindieAttentionFp8Backend.supports_ring_attention() is False
    assert not hasattr(mindie_attn.MindieAttentionFp8Impl, "forward_with_lse")


def test_forward_calls_quant_attention_with_gqa(quant_attention):
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 1, 4)
    value = torch.randn(1, 3, 1, 4)
    expected = torch.randn_like(query)
    quant_attention.return_value = expected

    output = _impl().forward(query, key, value)

    assert output is expected
    quant_attention.assert_called_once_with(
        query,
        key,
        value,
        precision="fp8",
        layout="BSND",
        scale=0.25,
    )


def test_default_quant_attention_options_are_omitted(quant_attention):
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 1, 4)
    value = torch.randn(1, 3, 1, 4)
    expected = torch.randn_like(query)
    quant_attention.return_value = expected
    impl = mindie_attn.MindieAttentionFp8Impl(num_heads=2, head_size=4, num_kv_heads=1)

    assert impl.forward(query, key, value) is expected
    quant_attention.assert_called_once_with(
        query,
        key,
        value,
        precision="fp8",
        layout="BSND",
    )


def test_attention_metadata_forwards_fp8_mode_and_rotation(quant_attention):
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 1, 4)
    value = torch.randn(1, 3, 1, 4)
    expected = torch.randn_like(query)
    quant_attention.return_value = expected
    metadata = mindie_attn.MindieAttentionMetadata(
        precision="fp8",
        fp8_fa_mode="C8V16_TILING512",
        rotation_mode="hadamard",
    )

    assert _impl().forward(query, key, value, attn_metadata=metadata) is expected
    assert quant_attention.call_args.kwargs["fp8_fa_mode"] == "C8V16_TILING512"
    assert quant_attention.call_args.kwargs["rotation_mode"] == "hadamard"


def test_mask_is_rejected_before_kernel(quant_attention):
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 1, 4)
    value = torch.randn(1, 3, 1, 4)

    with pytest.raises(ValueError, match="does not support attn_mask"):
        _impl().forward(query, key, value, attn_mask=torch.ones(1, 3, dtype=torch.bool))

    quant_attention.assert_not_called()


def test_causal_uses_native_next_token_window(quant_attention):
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 1, 4)
    value = torch.randn(1, 3, 1, 4)
    expected = torch.randn_like(query)
    quant_attention.return_value = expected
    impl = mindie_attn.MindieAttentionFp8Impl(
        num_heads=2,
        head_size=4,
        num_kv_heads=1,
        causal=True,
    )

    assert impl.forward(query, key, value) is expected
    assert quant_attention.call_args.kwargs["next_tokens"] == 0
    assert "pre_tokens" not in quant_attention.call_args.kwargs


@pytest.mark.parametrize("precision", ["fp8", "mxfp8"])
def test_non_causal_unequal_lengths_are_allowed(quant_attention, precision):
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 5, 1, 4)
    value = torch.randn(1, 5, 1, 4)
    expected = torch.randn_like(query)
    quant_attention.return_value = expected
    impl = mindie_attn.MindieAttentionFp8Impl(num_heads=2, head_size=4, num_kv_heads=1)
    metadata = mindie_attn.MindieAttentionMetadata(precision=precision)

    assert impl.forward(query, key, value, attn_metadata=metadata) is expected


def test_causal_unequal_lengths_are_rejected():
    impl = mindie_attn.MindieAttentionFp8Impl(
        num_heads=2,
        head_size=4,
        num_kv_heads=1,
        causal=True,
    )
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 5, 1, 4)
    value = torch.randn(1, 5, 1, 4)

    with pytest.raises(ValueError, match="causal mapping requires equal lengths"):
        impl.forward(query, key, value)


def test_mxfp8_precision_is_forwarded(quant_attention):
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 1, 4)
    value = torch.randn(1, 3, 1, 4)
    expected = torch.randn_like(query)
    quant_attention.return_value = expected
    impl = mindie_attn.MindieAttentionFp8Impl(num_heads=2, head_size=4, num_kv_heads=1)
    metadata = mindie_attn.MindieAttentionMetadata(precision="mxfp8")

    assert impl.forward(query, key, value, attn_metadata=metadata) is expected
    assert quant_attention.call_args.kwargs["precision"] == "mxfp8"


def test_fp8_batch_is_explicitly_rejected():
    impl = _impl()
    query = torch.randn(2, 3, 2, 4)
    key = torch.randn(2, 3, 1, 4)
    value = torch.randn(2, 3, 1, 4)

    with pytest.raises(ValueError, match="block-FP8 attention requires batch size 1"):
        impl.forward(query, key, value)


def test_mxfp8_accepts_multiple_batches(quant_attention):
    query = torch.randn(2, 3, 2, 4)
    key = torch.randn(2, 3, 1, 4)
    value = torch.randn(2, 3, 1, 4)
    expected = torch.randn_like(query)
    quant_attention.return_value = expected
    impl = mindie_attn.MindieAttentionFp8Impl(num_heads=2, head_size=4, num_kv_heads=1)
    metadata = mindie_attn.MindieAttentionMetadata(precision="mxfp8")

    assert impl.forward(query, key, value, attn_metadata=metadata) is expected
    quant_attention.assert_called_once()
