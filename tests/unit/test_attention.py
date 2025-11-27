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
"""Tests for attention mechanisms."""

import pytest
import torch
from gemma.model import GemmaAttention, GemmaMLP, precompute_freqs_cis
from gemma import config as gemma_config
from tests.fixtures.configs import get_tiny_config


@pytest.mark.unit
class TestGemmaAttention:
    """Test GemmaAttention layer."""

    def test_initialization(self, tiny_config):
        """Test attention initialization."""
        attn = GemmaAttention(tiny_config, gemma_config.AttentionType.GLOBAL)

        assert attn.num_heads == tiny_config.num_attention_heads
        assert attn.num_kv_heads == tiny_config.num_key_value_heads
        assert attn.head_dim == tiny_config.head_dim
        assert attn.hidden_size == tiny_config.hidden_size

    def test_forward_shape(self, tiny_config):
        """Test forward pass maintains correct shape."""
        batch_size, input_len, max_seq_len = 2, 5, 128
        attn = GemmaAttention(tiny_config, gemma_config.AttentionType.GLOBAL)

        hidden_states = torch.randn(batch_size, input_len, tiny_config.hidden_size)
        freqs_cis = precompute_freqs_cis(tiny_config.head_dim, max_seq_len)[:input_len]
        kv_write_indices = torch.arange(input_len)

        k_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
        v_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(batch_size, 1, input_len, max_seq_len)

        output = attn(hidden_states, freqs_cis, kv_write_indices, kv_cache, mask)

        assert output.shape == (batch_size, input_len, tiny_config.hidden_size)

    def test_kv_cache_write(self, tiny_config):
        """Test that KV cache is correctly updated."""
        batch_size, input_len, max_seq_len = 2, 5, 128
        attn = GemmaAttention(tiny_config, gemma_config.AttentionType.GLOBAL)

        hidden_states = torch.randn(batch_size, input_len, tiny_config.hidden_size)
        freqs_cis = precompute_freqs_cis(tiny_config.head_dim, max_seq_len)[:input_len]
        kv_write_indices = torch.arange(input_len)

        k_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
        v_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(batch_size, 1, input_len, max_seq_len)

        # Before forward pass, cache should be zero
        assert torch.all(k_cache[:, :input_len] == 0)
        assert torch.all(v_cache[:, :input_len] == 0)

        attn(hidden_states, freqs_cis, kv_write_indices, kv_cache, mask)

        # After forward pass, cache should be updated
        assert not torch.all(k_cache[:, :input_len] == 0)
        assert not torch.all(v_cache[:, :input_len] == 0)

    def test_grouped_query_attention(self):
        """Test grouped query attention (GQA) with fewer KV heads."""
        config = get_tiny_config()
        config.num_attention_heads = 8
        config.num_key_value_heads = 2  # GQA: 8 query heads, 2 KV heads

        batch_size, input_len, max_seq_len = 2, 5, 128
        attn = GemmaAttention(config, gemma_config.AttentionType.GLOBAL)

        hidden_states = torch.randn(batch_size, input_len, config.hidden_size)
        freqs_cis = precompute_freqs_cis(config.head_dim, max_seq_len)[:input_len]
        kv_write_indices = torch.arange(input_len)

        k_cache = torch.zeros(batch_size, max_seq_len, config.num_key_value_heads, config.head_dim)
        v_cache = torch.zeros(batch_size, max_seq_len, config.num_key_value_heads, config.head_dim)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(batch_size, 1, input_len, max_seq_len)

        output = attn(hidden_states, freqs_cis, kv_write_indices, kv_cache, mask)

        assert output.shape == (batch_size, input_len, config.hidden_size)

    def test_no_nan_or_inf(self, tiny_config):
        """Test that attention produces no NaN or Inf values."""
        batch_size, input_len, max_seq_len = 2, 5, 128
        attn = GemmaAttention(tiny_config, gemma_config.AttentionType.GLOBAL)

        hidden_states = torch.randn(batch_size, input_len, tiny_config.hidden_size)
        freqs_cis = precompute_freqs_cis(tiny_config.head_dim, max_seq_len)[:input_len]
        kv_write_indices = torch.arange(input_len)

        k_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
        v_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(batch_size, 1, input_len, max_seq_len)

        output = attn(hidden_states, freqs_cis, kv_write_indices, kv_cache, mask)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.unit
class TestGemmaMLP:
    """Test GemmaMLP layer."""

    def test_initialization(self, tiny_config):
        """Test MLP initialization."""
        mlp = GemmaMLP(
            hidden_size=tiny_config.hidden_size,
            intermediate_size=tiny_config.intermediate_size,
            quant=False,
        )

        assert mlp.gate_proj.weight.shape == (tiny_config.intermediate_size, tiny_config.hidden_size)
        assert mlp.up_proj.weight.shape == (tiny_config.intermediate_size, tiny_config.hidden_size)
        assert mlp.down_proj.weight.shape == (tiny_config.hidden_size, tiny_config.intermediate_size)

    def test_forward_shape(self, tiny_config):
        """Test forward pass maintains shape."""
        batch_size, seq_len = 2, 10
        mlp = GemmaMLP(
            hidden_size=tiny_config.hidden_size,
            intermediate_size=tiny_config.intermediate_size,
            quant=False,
        )

        x = torch.randn(batch_size, seq_len, tiny_config.hidden_size)
        output = mlp(x)

        assert output.shape == x.shape

    def test_forward_with_quant(self, tiny_config):
        """Test forward pass with quantization."""
        batch_size, seq_len = 2, 10
        mlp = GemmaMLP(
            hidden_size=tiny_config.hidden_size,
            intermediate_size=tiny_config.intermediate_size,
            quant=True,
        )

        # Initialize scalers
        mlp.gate_proj.weight_scaler.data.fill_(0.1)
        mlp.up_proj.weight_scaler.data.fill_(0.1)
        mlp.down_proj.weight_scaler.data.fill_(0.1)

        x = torch.randn(batch_size, seq_len, tiny_config.hidden_size)
        output = mlp(x)

        assert output.shape == x.shape

    def test_no_nan_or_inf(self, tiny_config):
        """Test that MLP produces no NaN or Inf values."""
        batch_size, seq_len = 2, 10
        mlp = GemmaMLP(
            hidden_size=tiny_config.hidden_size,
            intermediate_size=tiny_config.intermediate_size,
            quant=False,
        )

        x = torch.randn(batch_size, seq_len, tiny_config.hidden_size)
        output = mlp(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gelu_activation(self, tiny_config):
        """Test that GELU activation is applied."""
        mlp = GemmaMLP(
            hidden_size=tiny_config.hidden_size,
            intermediate_size=tiny_config.intermediate_size,
            quant=False,
        )

        # Zero input should produce zero or near-zero output
        x = torch.zeros(1, 1, tiny_config.hidden_size)
        output = mlp(x)

        # Output should be close to zero for zero input
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-2)
