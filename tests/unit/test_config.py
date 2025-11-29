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
"""Unit tests for Gemma configuration."""

import pytest
import torch

from gemma import config as gemma_config


class TestGemmaConfig:
    """Tests for GemmaConfig dataclass."""

    def test_default_config(self):
        """Test that default config can be created."""
        cfg = gemma_config.GemmaConfig()
        assert cfg.vocab_size == 256000
        assert cfg.architecture == gemma_config.Architecture.GEMMA_1

    def test_dtype_conversion(self):
        """Test dtype string to torch dtype conversion."""
        cfg = gemma_config.GemmaConfig(dtype='float32')
        assert cfg.get_dtype() == torch.float32

        cfg = gemma_config.GemmaConfig(dtype='float16')
        assert cfg.get_dtype() == torch.float16

        cfg = gemma_config.GemmaConfig(dtype='bfloat16')
        assert cfg.get_dtype() == torch.bfloat16

    def test_invalid_dtype(self):
        """Test that invalid dtype returns None."""
        cfg = gemma_config.GemmaConfig(dtype='invalid_dtype')
        assert cfg.get_dtype() is None


class TestModelVariants:
    """Tests for different model variant configurations."""

    @pytest.mark.parametrize('variant,expected_layers', [
        ('2b', 18),
        ('7b', 28),
        ('2b-v2', 26),
        ('9b', 42),
        ('27b', 46),
        ('1b', 26),
    ])
    def test_variant_layer_count(self, variant, expected_layers):
        """Test that each variant has the correct number of layers."""
        cfg = gemma_config.get_model_config(variant, dtype='float32')
        assert cfg.num_hidden_layers == expected_layers

    def test_2b_config(self, config_2b):
        """Test Gemma 2B configuration."""
        assert config_2b.num_hidden_layers == 18
        assert config_2b.num_attention_heads == 8
        assert config_2b.num_key_value_heads == 1
        assert config_2b.hidden_size == 2048
        assert config_2b.architecture == gemma_config.Architecture.GEMMA_1

    def test_7b_config(self, config_7b):
        """Test Gemma 7B configuration."""
        assert config_7b.num_hidden_layers == 28
        assert config_7b.num_attention_heads == 16
        assert config_7b.architecture == gemma_config.Architecture.GEMMA_1

    def test_2b_v2_config(self, config_2b_v2):
        """Test Gemma 2B v2 configuration."""
        assert config_2b_v2.architecture == gemma_config.Architecture.GEMMA_2
        assert config_2b_v2.use_pre_ffw_norm is True
        assert config_2b_v2.use_post_ffw_norm is True
        assert config_2b_v2.final_logit_softcapping == 30.0
        assert config_2b_v2.attn_logit_softcapping == 50.0
        assert config_2b_v2.sliding_window_size == 4096

    def test_9b_config(self, config_9b):
        """Test Gemma 9B configuration."""
        assert config_9b.architecture == gemma_config.Architecture.GEMMA_2
        assert config_9b.num_hidden_layers == 42
        assert config_9b.hidden_size == 3584

    def test_1b_config(self, config_1b):
        """Test Gemma 1B (Gemma 3) configuration."""
        assert config_1b.architecture == gemma_config.Architecture.GEMMA_3
        assert config_1b.vocab_size == 262_144
        assert config_1b.max_position_embeddings == 32_768
        assert config_1b.use_qk_norm is True
        assert config_1b.rope_wave_length is not None
        assert config_1b.sliding_window_size == 512

    def test_4b_config(self, config_4b):
        """Test Gemma 4B (Gemma 3) configuration."""
        assert config_4b.architecture == gemma_config.Architecture.GEMMA_3
        assert config_4b.vision_config is not None
        assert config_4b.rope_scaling_factor == 8

    def test_invalid_variant(self):
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="Invalid variant"):
            gemma_config.get_model_config('invalid_variant')


class TestAttentionTypes:
    """Tests for attention type configurations."""

    def test_gemma1_no_attention_types(self, config_2b):
        """Test that Gemma 1 models have no attention types specified."""
        assert config_2b.attn_types is None

    def test_gemma2_attention_types(self, config_2b_v2):
        """Test that Gemma 2 models have alternating attention types."""
        assert config_2b_v2.attn_types is not None
        # Should alternate between LOCAL_SLIDING and GLOBAL
        expected = [
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.GLOBAL
        ] * 13
        assert config_2b_v2.attn_types == expected

    def test_gemma3_attention_pattern(self, config_1b):
        """Test Gemma 3 attention pattern (5 local, 1 global)."""
        assert config_1b.attn_types is not None
        expected = (
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.LOCAL_SLIDING,
            gemma_config.AttentionType.GLOBAL,
        )
        assert config_1b.attn_types == expected


class TestArchitecture:
    """Tests for architecture enumeration."""

    def test_architecture_enum_values(self):
        """Test that all architecture values are defined."""
        assert gemma_config.Architecture.GEMMA_1 is not None
        assert gemma_config.Architecture.GEMMA_2 is not None
        assert gemma_config.Architecture.GEMMA_3 is not None

    @pytest.mark.parametrize('variant,expected_arch', [
        ('2b', gemma_config.Architecture.GEMMA_1),
        ('7b', gemma_config.Architecture.GEMMA_1),
        ('2b-v2', gemma_config.Architecture.GEMMA_2),
        ('9b', gemma_config.Architecture.GEMMA_2),
        ('27b', gemma_config.Architecture.GEMMA_2),
        ('1b', gemma_config.Architecture.GEMMA_3),
        ('4b', gemma_config.Architecture.GEMMA_3),
    ])
    def test_variant_architecture(self, variant, expected_arch):
        """Test that each variant has the correct architecture."""
        cfg = gemma_config.get_model_config(variant, dtype='float32')
        assert cfg.architecture == expected_arch
