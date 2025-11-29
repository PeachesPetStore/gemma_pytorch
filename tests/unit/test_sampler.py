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
"""Unit tests for Sampler."""

import pytest
import torch

from gemma import config as gemma_config
from gemma import model as gemma_model


class TestSampler:
    """Tests for Sampler class."""

    @pytest.fixture
    def sampler_config(self):
        """Create a config for sampler testing."""
        return gemma_config.GemmaConfig(
            vocab_size=1000,
            hidden_size=256,
        )

    @pytest.fixture
    def sampler(self, sampler_config):
        """Create a sampler instance."""
        return gemma_model.Sampler(
            vocab_size=sampler_config.vocab_size,
            config=sampler_config
        )

    def test_sampler_initialization(self, sampler, sampler_config):
        """Test sampler initialization."""
        assert sampler.vocab_size == sampler_config.vocab_size
        assert sampler.config == sampler_config

    def test_greedy_sampling(self, sampler):
        """Test greedy sampling (temperature=None)."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

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
        assert next_tokens.dtype == torch.long

    def test_temperature_sampling(self, sampler):
        """Test sampling with temperature."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=torch.tensor([1.0, 1.0]),
            top_ps=torch.tensor([1.0, 1.0]),
            top_ks=torch.tensor([vocab_size, vocab_size])
        )

        assert next_tokens.shape == (batch_size,)
        assert logits.shape == (batch_size, vocab_size)

    def test_high_temperature(self, sampler):
        """Test sampling with high temperature (more random)."""
        batch_size = 4
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        # High temperature should produce more diverse samples
        next_tokens, _ = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=torch.tensor([2.0] * batch_size),
            top_ps=torch.tensor([1.0] * batch_size),
            top_ks=torch.tensor([vocab_size] * batch_size)
        )

        assert next_tokens.shape == (batch_size,)

    def test_low_temperature(self, sampler):
        """Test sampling with low temperature (more deterministic)."""
        batch_size = 4
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, _ = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=torch.tensor([0.1] * batch_size),
            top_ps=torch.tensor([1.0] * batch_size),
            top_ks=torch.tensor([vocab_size] * batch_size)
        )

        assert next_tokens.shape == (batch_size,)

    def test_top_p_sampling(self, sampler):
        """Test top-p (nucleus) sampling."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, _ = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=torch.tensor([1.0, 1.0]),
            top_ps=torch.tensor([0.9, 0.95]),
            top_ks=torch.tensor([vocab_size, vocab_size])
        )

        assert next_tokens.shape == (batch_size,)

    def test_top_k_sampling(self, sampler):
        """Test top-k sampling."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, _ = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=torch.tensor([1.0, 1.0]),
            top_ps=torch.tensor([1.0, 1.0]),
            top_ks=torch.tensor([50, 100])
        )

        assert next_tokens.shape == (batch_size,)

    def test_combined_top_p_top_k(self, sampler):
        """Test combined top-p and top-k sampling."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, _ = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=torch.tensor([1.0, 1.0]),
            top_ps=torch.tensor([0.9, 0.95]),
            top_ks=torch.tensor([50, 100])
        )

        assert next_tokens.shape == (batch_size,)

    def test_sampler_with_softcapping(self):
        """Test sampler with final logit softcapping."""
        config = gemma_config.GemmaConfig(
            vocab_size=1000,
            hidden_size=256,
            final_logit_softcapping=30.0
        )
        sampler = gemma_model.Sampler(vocab_size=1000, config=config)

        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

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

        # Logits should be capped (with some margin for tanh)
        assert torch.all(torch.abs(logits) <= 30.0 * 1.1)

    def test_sampler_with_embedding_bias(self, sampler):
        """Test sampler with embedding bias."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])
        embedding_bias = torch.randn(vocab_size)

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=None,
            top_ps=torch.tensor([1.0, 1.0]),
            top_ks=torch.tensor([vocab_size, vocab_size]),
            embedding_bias=embedding_bias
        )

        assert next_tokens.shape == (batch_size,)
        assert logits.shape == (batch_size, vocab_size)

    def test_output_position_selection(self, sampler):
        """Test that output position correctly selects tokens."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256
        seq_len = 5

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Select the last position
        output_positions = torch.tensor([seq_len - 1])

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=None,
            top_ps=torch.tensor([1.0, 1.0]),
            top_ks=torch.tensor([vocab_size, vocab_size])
        )

        assert next_tokens.shape == (batch_size,)

    @pytest.mark.parametrize('temperature', [0.5, 1.0, 1.5, 2.0])
    def test_various_temperatures(self, sampler, temperature):
        """Test sampling with various temperature values."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, _ = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=torch.tensor([temperature, temperature]),
            top_ps=torch.tensor([1.0, 1.0]),
            top_ks=torch.tensor([vocab_size, vocab_size])
        )

        assert next_tokens.shape == (batch_size,)

    @pytest.mark.parametrize('top_p', [0.5, 0.7, 0.9, 0.95, 1.0])
    def test_various_top_p(self, sampler, top_p):
        """Test sampling with various top-p values."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, _ = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=torch.tensor([1.0, 1.0]),
            top_ps=torch.tensor([top_p, top_p]),
            top_ks=torch.tensor([vocab_size, vocab_size])
        )

        assert next_tokens.shape == (batch_size,)

    @pytest.mark.parametrize('top_k', [1, 10, 50, 100, 500])
    def test_various_top_k(self, sampler, top_k):
        """Test sampling with various top-k values."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        next_tokens, _ = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=torch.tensor([1.0, 1.0]),
            top_ps=torch.tensor([1.0, 1.0]),
            top_ks=torch.tensor([top_k, top_k])
        )

        assert next_tokens.shape == (batch_size,)

    def test_greedy_deterministic(self, sampler):
        """Test that greedy sampling is deterministic."""
        batch_size = 1
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        # Run greedy sampling multiple times
        results = []
        for _ in range(3):
            next_tokens, _ = sampler(
                embedding=embedding,
                hidden_states=hidden_states,
                output_positions=output_positions,
                temperatures=None,
                top_ps=torch.tensor([1.0]),
                top_ks=torch.tensor([vocab_size])
            )
            results.append(next_tokens.item())

        # All results should be identical
        assert len(set(results)) == 1

    def test_logits_output(self, sampler):
        """Test that logits are returned correctly."""
        batch_size = 2
        vocab_size = 1000
        hidden_size = 256

        embedding = torch.randn(vocab_size, hidden_size)
        hidden_states = torch.randn(batch_size, 1, hidden_size)
        output_positions = torch.tensor([0])

        _, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=None,
            top_ps=torch.tensor([1.0, 1.0]),
            top_ks=torch.tensor([vocab_size, vocab_size])
        )

        assert logits.dtype in [torch.float32, torch.float16, torch.bfloat16]
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
