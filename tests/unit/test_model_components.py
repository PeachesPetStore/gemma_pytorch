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
"""Unit tests for Gemma model components."""

import pytest
import torch

from gemma import config as gemma_config
from gemma import model as gemma_model


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass."""
        dim = 256
        batch_size = 2
        seq_len = 8

        norm = gemma_model.RMSNorm(dim=dim, eps=1e-6, add_unit_offset=True)
        x = torch.randn(batch_size, seq_len, dim)

        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_rmsnorm_without_unit_offset(self):
        """Test RMSNorm without unit offset."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim, eps=1e-6, add_unit_offset=False)
        x = torch.randn(2, 8, dim)

        output = norm(x)
        assert output.shape == x.shape

    def test_rmsnorm_preserves_device(self):
        """Test that RMSNorm preserves input device."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim)
        x = torch.randn(2, 8, dim)

        output = norm(x)
        assert output.device == x.device


class TestLinear:
    """Tests for custom Linear layer."""

    def test_linear_no_quant(self):
        """Test Linear layer without quantization."""
        in_features = 256
        out_features = 512
        batch_size = 2

        linear = gemma_model.Linear(in_features, out_features, quant=False)
        x = torch.randn(batch_size, in_features)

        output = linear(x)

        assert output.shape == (batch_size, out_features)
        assert not linear.quant

    def test_linear_with_quant(self):
        """Test Linear layer with quantization."""
        in_features = 256
        out_features = 512

        linear = gemma_model.Linear(in_features, out_features, quant=True)

        assert linear.quant
        assert linear.weight.dtype == torch.int8
        assert hasattr(linear, 'weight_scaler')

    def test_linear_quant_forward(self):
        """Test quantized Linear forward pass."""
        in_features = 256
        out_features = 512
        batch_size = 2

        linear = gemma_model.Linear(in_features, out_features, quant=True)
        # Initialize weights for testing
        linear.weight.data = torch.randint(-127, 127, (out_features, in_features), dtype=torch.int8)
        linear.weight_scaler.data = torch.randn(out_features)

        x = torch.randn(batch_size, in_features)
        output = linear(x)

        assert output.shape == (batch_size, out_features)


class TestEmbedding:
    """Tests for custom Embedding layer."""

    def test_embedding_no_quant(self):
        """Test Embedding layer without quantization."""
        num_embeddings = 1000
        embedding_dim = 256

        embedding = gemma_model.Embedding(num_embeddings, embedding_dim, quant=False)
        indices = torch.randint(0, num_embeddings, (2, 8))

        output = embedding(indices)

        assert output.shape == (2, 8, embedding_dim)
        assert not embedding.quant

    def test_embedding_with_quant(self):
        """Test Embedding layer with quantization."""
        num_embeddings = 1000
        embedding_dim = 256

        embedding = gemma_model.Embedding(num_embeddings, embedding_dim, quant=True)

        assert embedding.quant
        assert embedding.weight.dtype == torch.int8
        assert hasattr(embedding, 'weight_scaler')


class TestRotaryEmbedding:
    """Tests for rotary position embeddings."""

    def test_precompute_freqs_cis(self):
        """Test precomputation of frequency tensor."""
        dim = 256
        end = 1024

        freqs_cis = gemma_model.precompute_freqs_cis(dim, end)

        assert freqs_cis.shape == (end, dim // 2)
        assert freqs_cis.dtype == torch.complex64

    def test_precompute_freqs_cis_with_scaling(self):
        """Test precomputation with rope scaling factor."""
        dim = 256
        end = 1024
        scaling_factor = 8

        freqs_cis = gemma_model.precompute_freqs_cis(
            dim, end, rope_scaling_factor=scaling_factor
        )

        assert freqs_cis.shape == (end, dim // 2)

    def test_apply_rotary_emb(self):
        """Test applying rotary embeddings."""
        batch_size = 2
        seq_len = 8
        num_heads = 4
        head_dim = 64

        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        freqs_cis = gemma_model.precompute_freqs_cis(head_dim, seq_len)

        output = gemma_model.apply_rotary_emb(x, freqs_cis)

        assert output.shape == x.shape


class TestGemmaMLP:
    """Tests for Gemma MLP layer."""

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        hidden_size = 256
        intermediate_size = 1024
        batch_size = 2
        seq_len = 8

        mlp = gemma_model.GemmaMLP(hidden_size, intermediate_size, quant=False)
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = mlp(x)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_mlp_with_quant(self):
        """Test MLP with quantization."""
        hidden_size = 256
        intermediate_size = 1024

        mlp = gemma_model.GemmaMLP(hidden_size, intermediate_size, quant=True)

        assert mlp.gate_proj.quant
        assert mlp.up_proj.quant
        assert mlp.down_proj.quant


class TestSampler:
    """Tests for Sampler class."""

    def test_sampler_greedy(self):
        """Test greedy sampling (temperature=None)."""
        vocab_size = 1000
        hidden_size = 256
        batch_size = 2

        cfg = gemma_config.GemmaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )
        sampler = gemma_model.Sampler(vocab_size, cfg)

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=None,
            top_ps=torch.tensor([1.0, 1.0]),
            top_ks=torch.tensor([vocab_size, vocab_size])
        )

        assert next_tokens.shape == (batch_size,)
        assert logits.shape == (batch_size, vocab_size)

    def test_sampler_with_temperature(self):
        """Test sampling with temperature."""
        vocab_size = 1000
        hidden_size = 256
        batch_size = 2

        cfg = gemma_config.GemmaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )
        sampler = gemma_model.Sampler(vocab_size, cfg)

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=torch.tensor([1.0, 1.0]),
            top_ps=torch.tensor([0.95, 0.95]),
            top_ks=torch.tensor([50, 50])
        )

        assert next_tokens.shape == (batch_size,)
        assert logits.shape == (batch_size, vocab_size)

    def test_sampler_with_softcapping(self):
        """Test sampler with final logit softcapping."""
        vocab_size = 1000
        hidden_size = 256

        cfg = gemma_config.GemmaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            final_logit_softcapping=30.0
        )
        sampler = gemma_model.Sampler(vocab_size, cfg)

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(1, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=None,
            top_ps=torch.tensor([1.0]),
            top_ks=torch.tensor([vocab_size])
        )

        # Logits should be capped
        assert torch.all(torch.abs(logits) <= 30.0 * 1.1)  # Small margin for tanh
