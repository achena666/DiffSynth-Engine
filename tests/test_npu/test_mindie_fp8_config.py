"""Configuration plumbing tests for the MindIE FP8 attention backend."""

from diffsynth_engine.configs import MindieFp8AttentionParams
from diffsynth_engine.layers.attention.backends.mindie_attn import (
    MindieAttentionFp8Backend,
    MindieAttentionMetadata,
    MindieAttentionMetadataBuilder,
)


def test_attention_params_build_metadata():
    params = MindieFp8AttentionParams(
        precision="mxfp8",
        rotation_mode="hadamard",
    )

    metadata = MindieAttentionMetadataBuilder().build(**params.to_dict())

    assert metadata == MindieAttentionMetadata(precision="mxfp8", rotation_mode="hadamard")
    assert MindieAttentionFp8Backend.get_metadata_cls() is MindieAttentionMetadata
