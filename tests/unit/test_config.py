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
"""Tests for Gemma configuration."""

import pytest
import torch
from gemma import config as gemma_config


@pytest.mark.unit
class TestGemmaConfig:
    """Test GemmaConfig class."""

    def test_get_dtype_float16(self):
        """Test dtype conversion for float16."""
        cfg = gemma_config.GemmaConfig(dtype='float16')
        assert cfg.get_dtype() == torch.float16

    def test_get_dtype_float32(self):
        """Test dtype conversion for float32."""
        cfg = gemma_config.GemmaConfig(dtype='float32')
        assert cfg.get_dtype() == torch.float32

    def test_get_dtype_bfloat16(self):
        """Test dtype conversion for bfloat16."""
        cfg = gemma_config.GemmaConfig(dtype='bfloat16')
        assert cfg.get_dtype() == torch.bfloat16

    def test_get_dtype_invalid(self):
        """Test invalid dtype returns None."""
        cfg = gemma_config.GemmaConfig(dtype='invalid')
        assert cfg.get_dtype() is None

    def test_default_config(self):
        """Test default configuration values."""
        cfg = gemma_config.GemmaConfig()
        assert cfg.architecture == gemma_config.Architecture.GEMMA_1
        assert cfg.vocab_size == 256000
        assert cfg.num_hidden_layers == 28
        assert cfg.num_attention_heads == 16
        assert cfg.quant is False


@pytest.mark.unit
class TestModelVariants:
    """Test model variant configuration functions."""

    def test_get_config_for_2b(self):
        """Test 2B model configuration."""
        cfg = gemma_config.get_config_for_2b()
        assert cfg.num_hidden_layers == 18
        assert cfg.num_attention_heads == 8
        assert cfg.num_key_value_heads == 1
        assert cfg.hidden_size == 2048
        assert cfg.intermediate_size == 16384
        assert cfg.architecture == gemma_config.Architecture.GEMMA_1

    def test_get_config_for_7b(self):
        """Test 7B model configuration."""
        cfg = gemma_config.get_config_for_7b()
        assert cfg.num_hidden_layers == 28
        assert cfg.num_attention_heads == 16
        assert cfg.architecture == gemma_config.Architecture.GEMMA_1

    def test_get_config_for_2b_v2(self):
        """Test 2B v2 model configuration."""
        cfg = gemma_config.get_config_for_2b_v2()
        assert cfg.architecture == gemma_config.Architecture.GEMMA_2
        assert cfg.num_hidden_layers == 26
        assert cfg.use_pre_ffw_norm is True
        assert cfg.use_post_ffw_norm is True
        assert cfg.final_logit_softcapping == 30.0
        assert cfg.attn_logit_softcapping == 50.0
        assert cfg.sliding_window_size == 4096

    def test_get_config_for_9b(self):
        """Test 9B model configuration."""
        cfg = gemma_config.get_config_for_9b()
        assert cfg.architecture == gemma_config.Architecture.GEMMA_2
        assert cfg.num_hidden_layers == 42
        assert cfg.num_attention_heads == 16

    def test_get_config_for_27b(self):
        """Test 27B model configuration."""
        cfg = gemma_config.get_config_for_27b()
        assert cfg.architecture == gemma_config.Architecture.GEMMA_2
        assert cfg.num_hidden_layers == 46
        assert cfg.num_attention_heads == 32
        assert cfg.query_pre_attn_scalar == 144

    def test_get_config_for_1b(self):
        """Test 1B Gemma3 model configuration."""
        cfg = gemma_config.get_config_for_1b('float32')
        assert cfg.architecture == gemma_config.Architecture.GEMMA_3
        assert cfg.num_hidden_layers == 26
        assert cfg.use_qk_norm is True
        assert cfg.rope_wave_length is not None
        assert cfg.vocab_size == 262_144

    def test_get_config_for_4b(self):
        """Test 4B Gemma3 model configuration."""
        cfg = gemma_config.get_config_for_4b('float32')
        assert cfg.architecture == gemma_config.Architecture.GEMMA_3
        assert cfg.num_hidden_layers == 34
        assert cfg.vision_config is not None
        assert cfg.rope_scaling_factor == 8

    def test_get_config_for_12b(self):
        """Test 12B Gemma3 model configuration."""
        cfg = gemma_config.get_config_for_12b('float32')
        assert cfg.architecture == gemma_config.Architecture.GEMMA_3
        assert cfg.num_hidden_layers == 48
        assert cfg.max_position_embeddings == 131_072

    def test_get_config_for_27b_v3(self):
        """Test 27B v3 Gemma3 model configuration."""
        cfg = gemma_config.get_config_for_27b_v3('float32')
        assert cfg.architecture == gemma_config.Architecture.GEMMA_3
        assert cfg.num_hidden_layers == 62
        assert cfg.query_pre_attn_scalar == 5376 // 32

    def test_get_model_config_2b(self):
        """Test get_model_config with 2b variant."""
        cfg = gemma_config.get_model_config('2b')
        assert cfg.num_hidden_layers == 18

    def test_get_model_config_2b_v2(self):
        """Test get_model_config with 2b-v2 variant."""
        cfg = gemma_config.get_model_config('2b-v2')
        assert cfg.architecture == gemma_config.Architecture.GEMMA_2

    def test_get_model_config_7b(self):
        """Test get_model_config with 7b variant."""
        cfg = gemma_config.get_model_config('7b')
        assert cfg.num_hidden_layers == 28

    def test_get_model_config_9b(self):
        """Test get_model_config with 9b variant."""
        cfg = gemma_config.get_model_config('9b')
        assert cfg.num_hidden_layers == 42

    def test_get_model_config_27b(self):
        """Test get_model_config with 27b variant."""
        cfg = gemma_config.get_model_config('27b')
        assert cfg.num_hidden_layers == 46

    def test_get_model_config_1b(self):
        """Test get_model_config with 1b variant."""
        cfg = gemma_config.get_model_config('1b')
        assert cfg.architecture == gemma_config.Architecture.GEMMA_3

    def test_get_model_config_4b(self):
        """Test get_model_config with 4b variant."""
        cfg = gemma_config.get_model_config('4b')
        assert cfg.num_hidden_layers == 34

    def test_get_model_config_12b(self):
        """Test get_model_config with 12b variant."""
        cfg = gemma_config.get_model_config('12b')
        assert cfg.num_hidden_layers == 48

    def test_get_model_config_27b_v3(self):
        """Test get_model_config with 27b_v3 variant."""
        cfg = gemma_config.get_model_config('27b_v3')
        assert cfg.num_hidden_layers == 62

    def test_get_model_config_invalid_variant(self):
        """Test get_model_config with invalid variant raises error."""
        with pytest.raises(ValueError, match="Invalid variant"):
            gemma_config.get_model_config('invalid')

    def test_dtype_propagation(self):
        """Test that dtype parameter is properly propagated."""
        cfg = gemma_config.get_model_config('2b', dtype='float16')
        assert cfg.dtype == 'float16'
        assert cfg.get_dtype() == torch.float16


@pytest.mark.unit
class TestAttentionType:
    """Test AttentionType enum."""

    def test_attention_types(self):
        """Test attention type enum values."""
        assert gemma_config.AttentionType.GLOBAL.value == 1
        assert gemma_config.AttentionType.LOCAL_SLIDING.value == 2


@pytest.mark.unit
class TestArchitecture:
    """Test Architecture enum."""

    def test_architecture_types(self):
        """Test architecture enum values."""
        assert gemma_config.Architecture.GEMMA_1.value == 1
        assert gemma_config.Architecture.GEMMA_2.value == 2
        assert gemma_config.Architecture.GEMMA_3.value == 3
