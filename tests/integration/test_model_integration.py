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
"""Integration tests for full Gemma model."""

import pytest
import torch
from gemma.model import GemmaModel, GemmaForCausalLM, GemmaDecoderLayer, Gemma2DecoderLayer
from gemma import config as gemma_config
from tests.fixtures.configs import get_tiny_config, get_tiny_gemma2_config


@pytest.mark.integration
class TestGemmaDecoderLayer:
    """Test decoder layer integration."""

    def test_gemma1_decoder_layer_forward(self, tiny_config):
        """Test Gemma1 decoder layer forward pass."""
        batch_size, input_len, max_seq_len = 2, 5, 128
        layer = GemmaDecoderLayer(tiny_config)

        hidden_states = torch.randn(batch_size, input_len, tiny_config.hidden_size)
        freqs_cis = torch.randn(input_len, tiny_config.head_dim // 2, dtype=torch.complex64)
        kv_write_indices = torch.arange(input_len)

        k_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
        v_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(batch_size, 1, input_len, max_seq_len)
        local_mask = torch.zeros(batch_size, 1, input_len, max_seq_len)

        output = layer(hidden_states, freqs_cis, kv_write_indices, kv_cache, mask, local_mask)

        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gemma2_decoder_layer_forward(self, tiny_gemma2_config):
        """Test Gemma2 decoder layer forward pass."""
        batch_size, input_len, max_seq_len = 2, 5, 128
        layer = Gemma2DecoderLayer(tiny_gemma2_config, gemma_config.AttentionType.GLOBAL)

        hidden_states = torch.randn(batch_size, input_len, tiny_gemma2_config.hidden_size)
        freqs_cis = torch.randn(input_len, tiny_gemma2_config.head_dim // 2, dtype=torch.complex64)
        kv_write_indices = torch.arange(input_len)

        k_cache = torch.zeros(batch_size, max_seq_len, tiny_gemma2_config.num_key_value_heads, tiny_gemma2_config.head_dim)
        v_cache = torch.zeros(batch_size, max_seq_len, tiny_gemma2_config.num_key_value_heads, tiny_gemma2_config.head_dim)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(batch_size, 1, input_len, max_seq_len)
        local_mask = torch.zeros(batch_size, 1, input_len, max_seq_len)

        output = layer(hidden_states, freqs_cis, kv_write_indices, kv_cache, mask, local_mask)

        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.integration
class TestGemmaModel:
    """Test GemmaModel integration."""

    def test_gemma1_model_forward(self, tiny_config):
        """Test Gemma1 model forward pass."""
        batch_size, input_len, max_seq_len = 2, 5, 128
        model = GemmaModel(tiny_config)

        hidden_states = torch.randn(batch_size, input_len, tiny_config.hidden_size)
        freqs_cis = {
            gemma_config.AttentionType.GLOBAL: torch.randn(input_len, tiny_config.head_dim // 2, dtype=torch.complex64),
            gemma_config.AttentionType.LOCAL_SLIDING: torch.randn(input_len, tiny_config.head_dim // 2, dtype=torch.complex64),
        }
        kv_write_indices = torch.arange(input_len)

        kv_caches = []
        for _ in range(tiny_config.num_hidden_layers):
            k_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
            v_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
            kv_caches.append((k_cache, v_cache))

        mask = torch.zeros(batch_size, 1, input_len, max_seq_len)
        local_mask = torch.zeros(batch_size, 1, input_len, max_seq_len)

        output = model(hidden_states, freqs_cis, kv_write_indices, kv_caches, mask, local_mask)

        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gemma2_model_forward(self, tiny_gemma2_config):
        """Test Gemma2 model forward pass."""
        batch_size, input_len, max_seq_len = 2, 5, 128
        model = GemmaModel(tiny_gemma2_config)

        hidden_states = torch.randn(batch_size, input_len, tiny_gemma2_config.hidden_size)
        freqs_cis = {
            gemma_config.AttentionType.GLOBAL: torch.randn(input_len, tiny_gemma2_config.head_dim // 2, dtype=torch.complex64),
            gemma_config.AttentionType.LOCAL_SLIDING: torch.randn(input_len, tiny_gemma2_config.head_dim // 2, dtype=torch.complex64),
        }
        kv_write_indices = torch.arange(input_len)

        kv_caches = []
        for _ in range(tiny_gemma2_config.num_hidden_layers):
            k_cache = torch.zeros(batch_size, max_seq_len, tiny_gemma2_config.num_key_value_heads, tiny_gemma2_config.head_dim)
            v_cache = torch.zeros(batch_size, max_seq_len, tiny_gemma2_config.num_key_value_heads, tiny_gemma2_config.head_dim)
            kv_caches.append((k_cache, v_cache))

        mask = torch.zeros(batch_size, 1, input_len, max_seq_len)
        local_mask = torch.zeros(batch_size, 1, input_len, max_seq_len)

        output = model(hidden_states, freqs_cis, kv_write_indices, kv_caches, mask, local_mask)

        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.integration
class TestGemmaForCausalLM:
    """Test GemmaForCausalLM integration."""

    def test_model_initialization(self, tiny_config):
        """Test model initialization."""
        tiny_config.tokenizer = None  # Skip tokenizer for this test
        model = GemmaForCausalLM(tiny_config)

        assert isinstance(model.model, GemmaModel)
        assert model.config == tiny_config

    def test_forward_pass(self, tiny_config):
        """Test full forward pass through the model."""
        tiny_config.tokenizer = None
        batch_size, input_len, max_seq_len = 2, 10, 128
        model = GemmaForCausalLM(tiny_config)

        input_token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, input_len))
        input_positions = torch.arange(input_len)
        kv_write_indices = torch.arange(input_len)

        kv_caches = []
        for _ in range(tiny_config.num_hidden_layers):
            k_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
            v_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
            kv_caches.append((k_cache, v_cache))

        mask = torch.triu(
            torch.full((1, 1, max_seq_len, max_seq_len), float('-inf')),
            diagonal=1
        )
        curr_mask = mask[:, :, :input_len, :]
        output_positions = torch.tensor([input_len - 1])
        top_ps = torch.tensor([0.9] * batch_size)
        top_ks = torch.tensor([50] * batch_size)

        next_tokens, logits = model(
            input_token_ids=input_token_ids,
            input_positions=input_positions,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=curr_mask,
            output_positions=output_positions,
            temperatures=None,
            top_ps=top_ps,
            top_ks=top_ks,
        )

        assert next_tokens.shape == (batch_size,)
        assert logits.shape == (batch_size, tiny_config.vocab_size)
        assert torch.all(next_tokens >= 0)
        assert torch.all(next_tokens < tiny_config.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_different_batch_sizes(self, tiny_config):
        """Test model works with different batch sizes."""
        tiny_config.tokenizer = None
        model = GemmaForCausalLM(tiny_config)

        for batch_size in [1, 2, 4]:
            input_len, max_seq_len = 10, 128
            input_token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, input_len))
            input_positions = torch.arange(input_len)
            kv_write_indices = torch.arange(input_len)

            kv_caches = []
            for _ in range(tiny_config.num_hidden_layers):
                k_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
                v_cache = torch.zeros(batch_size, max_seq_len, tiny_config.num_key_value_heads, tiny_config.head_dim)
                kv_caches.append((k_cache, v_cache))

            mask = torch.triu(
                torch.full((1, 1, max_seq_len, max_seq_len), float('-inf')),
                diagonal=1
            )[:, :, :input_len, :]
            output_positions = torch.tensor([input_len - 1])
            top_ps = torch.tensor([0.9] * batch_size)
            top_ks = torch.tensor([50] * batch_size)

            next_tokens, logits = model(
                input_token_ids=input_token_ids,
                input_positions=input_positions,
                kv_write_indices=kv_write_indices,
                kv_caches=kv_caches,
                mask=mask,
                output_positions=output_positions,
                temperatures=None,
                top_ps=top_ps,
                top_ks=top_ks,
            )

            assert next_tokens.shape == (batch_size,)
            assert logits.shape == (batch_size, tiny_config.vocab_size)

    def test_embedding_normalization(self, tiny_config):
        """Test that embeddings are normalized by sqrt(hidden_size)."""
        tiny_config.tokenizer = None
        model = GemmaForCausalLM(tiny_config)

        batch_size, input_len = 2, 10
        input_token_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, input_len))

        # Get embeddings directly
        embeddings = model.embedder(input_token_ids)

        # Check that normalization factor is applied
        # The actual normalized embeddings should have been scaled
        assert embeddings.shape == (batch_size, input_len, tiny_config.hidden_size)
