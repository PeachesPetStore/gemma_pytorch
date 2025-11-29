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
"""Unit tests for attention mechanisms."""

import pytest
import torch

from gemma import config as gemma_config
from gemma import model as gemma_model


class TestGemmaAttention:
    """Tests for GemmaAttention module."""

    @pytest.fixture
    def attention_config(self):
        """Create a small config for testing attention."""
        return gemma_config.GemmaConfig(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
            head_dim=64,
            quant=False,
        )

    @pytest.fixture
    def attention_global(self, attention_config):
        """Create a global attention module."""
        return gemma_model.GemmaAttention(
            config=attention_config,
            attn_type=gemma_config.AttentionType.GLOBAL
        )

    @pytest.fixture
    def attention_local(self, attention_config):
        """Create a local sliding window attention module."""
        attention_config.sliding_window_size = 128
        return gemma_model.GemmaAttention(
            config=attention_config,
            attn_type=gemma_config.AttentionType.LOCAL_SLIDING
        )

    def test_attention_initialization(self, attention_global):
        """Test attention module initialization."""
        assert attention_global.num_heads == 4
        assert attention_global.num_kv_heads == 2
        assert attention_global.head_dim == 64
        assert attention_global.num_queries_per_kv == 2

    def test_attention_qkv_projection_shapes(self, attention_global, attention_config):
        """Test QKV projection weight shapes."""
        expected_out_features = (
            attention_config.num_attention_heads * attention_config.head_dim +
            2 * attention_config.num_key_value_heads * attention_config.head_dim
        )
        assert attention_global.qkv_proj.weight.shape == (
            expected_out_features,
            attention_config.hidden_size
        )

    def test_attention_forward_shape(self, attention_global, cpu_device):
        """Test attention forward pass output shape."""
        attention_global = attention_global.to(cpu_device)
        batch_size = 2
        seq_len = 8
        hidden_size = 256
        head_dim = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(cpu_device)
        freqs_cis = gemma_model.precompute_freqs_cis(head_dim, seq_len).to(cpu_device)
        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        # Initialize KV cache
        k_cache = torch.zeros(batch_size, seq_len, 2, head_dim).to(cpu_device)
        v_cache = torch.zeros(batch_size, seq_len, 2, head_dim).to(cpu_device)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        output = attention_global(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask
        )

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_attention_with_causal_mask(self, attention_global, cpu_device):
        """Test attention with causal masking."""
        attention_global = attention_global.to(cpu_device)
        batch_size = 2
        seq_len = 8
        hidden_size = 256
        head_dim = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(cpu_device)
        freqs_cis = gemma_model.precompute_freqs_cis(head_dim, seq_len).to(cpu_device)
        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        k_cache = torch.zeros(batch_size, seq_len, 2, head_dim).to(cpu_device)
        v_cache = torch.zeros(batch_size, seq_len, 2, head_dim).to(cpu_device)
        kv_cache = (k_cache, v_cache)

        # Create causal mask
        mask = torch.triu(
            torch.full((1, 1, seq_len, seq_len), -2.3819763e38),
            diagonal=1
        ).to(cpu_device)

        output = attention_global(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask
        )

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_attention_kv_cache_update(self, attention_global, cpu_device):
        """Test that KV cache is properly updated."""
        attention_global = attention_global.to(cpu_device)
        batch_size = 2
        seq_len = 8
        hidden_size = 256
        head_dim = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(cpu_device)
        freqs_cis = gemma_model.precompute_freqs_cis(head_dim, seq_len).to(cpu_device)
        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        k_cache = torch.zeros(batch_size, seq_len, 2, head_dim).to(cpu_device)
        v_cache = torch.zeros(batch_size, seq_len, 2, head_dim).to(cpu_device)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        # Cache should be zero before forward
        assert torch.all(k_cache == 0)
        assert torch.all(v_cache == 0)

        _ = attention_global(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask
        )

        # Cache should be updated after forward
        assert not torch.all(k_cache == 0)
        assert not torch.all(v_cache == 0)

    def test_attention_with_qk_norm(self, cpu_device):
        """Test attention with query-key normalization."""
        config = gemma_config.GemmaConfig(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
            head_dim=64,
            use_qk_norm=True,
            quant=False,
        )

        attention = gemma_model.GemmaAttention(
            config=config,
            attn_type=gemma_config.AttentionType.GLOBAL
        ).to(cpu_device)

        assert attention.query_norm is not None
        assert attention.key_norm is not None

        batch_size = 2
        seq_len = 8

        hidden_states = torch.randn(batch_size, seq_len, 256).to(cpu_device)
        freqs_cis = gemma_model.precompute_freqs_cis(64, seq_len).to(cpu_device)
        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        k_cache = torch.zeros(batch_size, seq_len, 2, 64).to(cpu_device)
        v_cache = torch.zeros(batch_size, seq_len, 2, 64).to(cpu_device)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        output = attention(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask
        )

        assert output.shape == (batch_size, seq_len, 256)

    def test_attention_with_softcapping(self, cpu_device):
        """Test attention with logit softcapping."""
        config = gemma_config.GemmaConfig(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
            head_dim=64,
            attn_logit_softcapping=50.0,
            quant=False,
        )

        attention = gemma_model.GemmaAttention(
            config=config,
            attn_type=gemma_config.AttentionType.GLOBAL
        ).to(cpu_device)

        assert attention.attn_logit_softcapping == 50.0

        batch_size = 2
        seq_len = 8

        hidden_states = torch.randn(batch_size, seq_len, 256).to(cpu_device)
        freqs_cis = gemma_model.precompute_freqs_cis(64, seq_len).to(cpu_device)
        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        k_cache = torch.zeros(batch_size, seq_len, 2, 64).to(cpu_device)
        v_cache = torch.zeros(batch_size, seq_len, 2, 64).to(cpu_device)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        output = attention(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask
        )

        assert output.shape == (batch_size, seq_len, 256)

    def test_attention_scaling(self, attention_config):
        """Test that attention scaling is computed correctly."""
        # Without query_pre_attn_scalar
        attn = gemma_model.GemmaAttention(
            config=attention_config,
            attn_type=gemma_config.AttentionType.GLOBAL
        )
        assert attn.scaling == attention_config.head_dim ** -0.5

        # With query_pre_attn_scalar
        attention_config.query_pre_attn_scalar = 144
        attn_with_scalar = gemma_model.GemmaAttention(
            config=attention_config,
            attn_type=gemma_config.AttentionType.GLOBAL
        )
        assert attn_with_scalar.scaling == 144 ** -0.5

    def test_multi_query_attention(self, cpu_device):
        """Test multi-query attention (num_kv_heads=1)."""
        config = gemma_config.GemmaConfig(
            num_attention_heads=8,
            num_key_value_heads=1,  # Multi-query attention
            hidden_size=256,
            head_dim=64,
            quant=False,
        )

        attention = gemma_model.GemmaAttention(
            config=config,
            attn_type=gemma_config.AttentionType.GLOBAL
        ).to(cpu_device)

        assert attention.num_queries_per_kv == 8

        batch_size = 2
        seq_len = 8

        hidden_states = torch.randn(batch_size, seq_len, 256).to(cpu_device)
        freqs_cis = gemma_model.precompute_freqs_cis(64, seq_len).to(cpu_device)
        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        k_cache = torch.zeros(batch_size, seq_len, 1, 64).to(cpu_device)
        v_cache = torch.zeros(batch_size, seq_len, 1, 64).to(cpu_device)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        output = attention(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask
        )

        assert output.shape == (batch_size, seq_len, 256)

    def test_local_sliding_attention(self, attention_local, cpu_device):
        """Test local sliding window attention."""
        attention_local = attention_local.to(cpu_device)
        batch_size = 2
        seq_len = 16
        hidden_size = 256
        head_dim = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(cpu_device)
        freqs_cis = gemma_model.precompute_freqs_cis(head_dim, seq_len).to(cpu_device)
        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        k_cache = torch.zeros(batch_size, seq_len, 2, head_dim).to(cpu_device)
        v_cache = torch.zeros(batch_size, seq_len, 2, head_dim).to(cpu_device)
        kv_cache = (k_cache, v_cache)

        # Create masks
        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)
        local_mask = torch.triu(
            torch.full((1, 1, seq_len, seq_len), -2.3819763e38),
            diagonal=1
        ).to(cpu_device)
        local_mask += torch.tril(
            torch.full((1, 1, seq_len, seq_len), -2.3819763e38, device=cpu_device),
            diagonal=-128  # Sliding window size
        )

        output = attention_local(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask
        )

        assert output.shape == (batch_size, seq_len, hidden_size)

    @pytest.mark.parametrize('batch_size,seq_len', [
        (1, 4),
        (2, 8),
        (4, 16),
    ])
    def test_attention_various_batch_seq(self, attention_global, cpu_device, batch_size, seq_len):
        """Test attention with various batch sizes and sequence lengths."""
        attention_global = attention_global.to(cpu_device)
        hidden_size = 256
        head_dim = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(cpu_device)
        freqs_cis = gemma_model.precompute_freqs_cis(head_dim, seq_len).to(cpu_device)
        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        k_cache = torch.zeros(batch_size, seq_len, 2, head_dim).to(cpu_device)
        v_cache = torch.zeros(batch_size, seq_len, 2, head_dim).to(cpu_device)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        output = attention_global(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask
        )

        assert output.shape == (batch_size, seq_len, hidden_size)
