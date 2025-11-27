# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test configuration fixtures."""

from gemma import config as gemma_config


def get_tiny_config(architecture=gemma_config.Architecture.GEMMA_1) -> gemma_config.GemmaConfig:
    """Get a tiny config for fast testing."""
    return gemma_config.GemmaConfig(
        architecture=architecture,
        vocab_size=256,
        max_position_embeddings=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
        head_dim=32,
        rms_norm_eps=1e-6,
        dtype='float32',
        quant=False,
        tokenizer=None,
    )


def get_tiny_gemma2_config() -> gemma_config.GemmaConfig:
    """Get a tiny Gemma2 config for testing."""
    return gemma_config.GemmaConfig(
        architecture=gemma_config.Architecture.GEMMA_2,
        vocab_size=256,
        max_position_embeddings=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
        head_dim=32,
        rms_norm_eps=1e-6,
        dtype='float32',
        quant=False,
        tokenizer=None,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        attn_types=[gemma_config.AttentionType.LOCAL_SLIDING, gemma_config.AttentionType.GLOBAL],
        sliding_window_size=64,
    )


def get_tiny_gemma3_config() -> gemma_config.GemmaConfig:
    """Get a tiny Gemma3 config for testing."""
    return gemma_config.GemmaConfig(
        architecture=gemma_config.Architecture.GEMMA_3,
        vocab_size=256,
        max_position_embeddings=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=128,
        intermediate_size=256,
        head_dim=32,
        rms_norm_eps=1e-6,
        dtype='float32',
        quant=False,
        tokenizer=None,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        use_qk_norm=True,
        attn_types=(
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.GLOBAL,
        ),
        sliding_window_size=64,
        rope_wave_length={
            gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
            gemma_config.AttentionType.GLOBAL: 1_000_000,
        },
    )
