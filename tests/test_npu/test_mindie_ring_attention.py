"""MindIE ring-attention tests without requiring NPU hardware."""

import sys
from unittest.mock import Mock, patch

import torch

import diffsynth_engine.layers.attention.backends.mindie_attn as mindie_attn


def test_mindie_backend_uses_ring_capable_impl():
    assert mindie_attn.MindieAttentionBackend.get_impl_cls() is mindie_attn.MindieAttentionImpl
    assert mindie_attn.MindieAttentionBackend.supports_ring_attention() is True


def test_forward_with_lse_calls_torch_npu_directly():
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 2, 4)
    value = torch.randn(1, 3, 2, 4)
    keep_mask = torch.tensor([[True, False, True]])
    output = torch.randn_like(query)
    softmax_max = torch.zeros(1, 2, 3, 8)
    softmax_sum = torch.ones(1, 2, 3, 8)
    torch_npu = Mock()
    torch_npu.npu_fusion_attention.return_value = (output, softmax_max, softmax_sum)
    impl = mindie_attn.MindieAttentionImpl(num_heads=2, head_size=4)

    with patch.dict(sys.modules, {"torch_npu": torch_npu}):
        actual_output, lse = impl.forward_with_lse(query, key, value, attn_mask=keep_mask)

    assert actual_output is output
    assert torch.equal(lse, torch.zeros(1, 2, 3))
    call_kwargs = torch_npu.npu_fusion_attention.call_args.kwargs
    assert torch.equal(call_kwargs["atten_mask"], (~keep_mask)[:, None, None, :])
    assert call_kwargs["scale"] == 0.5
    assert call_kwargs["head_num"] == 2
