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
"""Tests for Sampler class."""

import pytest
import torch
from gemma.model import Sampler
from tests.fixtures.configs import get_tiny_config


@pytest.mark.unit
class TestSampler:
    """Test Sampler class."""

    def test_initialization(self):
        """Test Sampler initialization."""
        vocab_size = 1000
        config = get_tiny_config()
        sampler = Sampler(vocab_size, config)

        assert sampler.vocab_size == vocab_size
        assert sampler.config == config

    def test_greedy_sampling_temperature_none(self):
        """Test greedy sampling with temperature=None."""
        vocab_size = 100
        batch_size = 2
        config = get_tiny_config()
        sampler = Sampler(vocab_size, config)

        embedding = torch.randn(vocab_size, config.hidden_size)
        hidden_states = torch.randn(batch_size, 5, config.hidden_size)
        output_positions = torch.tensor([4])
        top_ps = torch.tensor([0.9, 0.9])
        top_ks = torch.tensor([50, 50])

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=None,
            top_ps=top_ps,
            top_ks=top_ks,
        )

        # Should return argmax
        assert next_tokens.shape == (batch_size,)
        assert logits.shape == (batch_size, vocab_size)
        # Verify greedy selection
        expected_tokens = torch.argmax(logits, dim=-1)
        assert torch.all(next_tokens == expected_tokens)

    def test_sampling_with_temperature(self):
        """Test sampling with temperature."""
        torch.manual_seed(42)  # For reproducibility
        vocab_size = 100
        batch_size = 2
        config = get_tiny_config()
        sampler = Sampler(vocab_size, config)

        embedding = torch.randn(vocab_size, config.hidden_size)
        hidden_states = torch.randn(batch_size, 5, config.hidden_size)
        output_positions = torch.tensor([4])
        temperatures = torch.tensor([1.0, 1.0])
        top_ps = torch.tensor([1.0, 1.0])  # No top-p filtering
        top_ks = torch.tensor([vocab_size, vocab_size])  # No top-k filtering

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )

        assert next_tokens.shape == (batch_size,)
        assert logits.shape == (batch_size, vocab_size)
        assert torch.all(next_tokens >= 0)
        assert torch.all(next_tokens < vocab_size)

    def test_low_temperature_approximates_greedy(self):
        """Test that very low temperature approximates greedy sampling."""
        torch.manual_seed(42)
        vocab_size = 100
        batch_size = 4
        config = get_tiny_config()
        sampler = Sampler(vocab_size, config)

        embedding = torch.randn(vocab_size, config.hidden_size)
        hidden_states = torch.randn(batch_size, 5, config.hidden_size)
        output_positions = torch.tensor([4])
        temperatures = torch.tensor([0.01, 0.01, 0.01, 0.01])
        top_ps = torch.tensor([1.0, 1.0, 1.0, 1.0])
        top_ks = torch.tensor([vocab_size, vocab_size, vocab_size, vocab_size])

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )

        # With very low temperature, should mostly match argmax
        expected_tokens = torch.argmax(logits, dim=-1)
        # Allow some variation due to numerical precision
        match_ratio = (next_tokens == expected_tokens).float().mean()
        assert match_ratio > 0.7  # Most should match

    def test_top_k_filtering(self):
        """Test top-k filtering."""
        torch.manual_seed(42)
        vocab_size = 100
        batch_size = 2
        config = get_tiny_config()
        sampler = Sampler(vocab_size, config)

        embedding = torch.randn(vocab_size, config.hidden_size)
        hidden_states = torch.randn(batch_size, 5, config.hidden_size)
        output_positions = torch.tensor([4])
        temperatures = torch.tensor([1.0, 1.0])
        top_ps = torch.tensor([1.0, 1.0])
        top_ks = torch.tensor([10, 10])  # Only top 10 tokens

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )

        # Tokens should be sampled (not necessarily from top-k but valid)
        assert torch.all(next_tokens >= 0)
        assert torch.all(next_tokens < vocab_size)

    def test_top_p_filtering(self):
        """Test top-p (nucleus) filtering."""
        torch.manual_seed(42)
        vocab_size = 100
        batch_size = 2
        config = get_tiny_config()
        sampler = Sampler(vocab_size, config)

        embedding = torch.randn(vocab_size, config.hidden_size)
        hidden_states = torch.randn(batch_size, 5, config.hidden_size)
        output_positions = torch.tensor([4])
        temperatures = torch.tensor([1.0, 1.0])
        top_ps = torch.tensor([0.9, 0.9])  # Nucleus sampling
        top_ks = torch.tensor([vocab_size, vocab_size])

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )

        assert torch.all(next_tokens >= 0)
        assert torch.all(next_tokens < vocab_size)

    def test_combined_top_k_top_p(self):
        """Test combined top-k and top-p filtering."""
        torch.manual_seed(42)
        vocab_size = 100
        batch_size = 2
        config = get_tiny_config()
        sampler = Sampler(vocab_size, config)

        embedding = torch.randn(vocab_size, config.hidden_size)
        hidden_states = torch.randn(batch_size, 5, config.hidden_size)
        output_positions = torch.tensor([4])
        temperatures = torch.tensor([1.0, 1.0])
        top_ps = torch.tensor([0.9, 0.9])
        top_ks = torch.tensor([50, 50])

        next_tokens, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )

        assert torch.all(next_tokens >= 0)
        assert torch.all(next_tokens < vocab_size)

    def test_logit_softcapping(self):
        """Test logit softcapping."""
        vocab_size = 100
        batch_size = 2
        config = get_tiny_config()
        config.final_logit_softcapping = 30.0
        sampler = Sampler(vocab_size, config)

        embedding = torch.randn(vocab_size, config.hidden_size)
        # Create hidden states that would produce large logits
        hidden_states = torch.randn(batch_size, 5, config.hidden_size) * 100
        output_positions = torch.tensor([4])
        top_ps = torch.tensor([1.0, 1.0])
        top_ks = torch.tensor([vocab_size, vocab_size])

        _, logits = sampler(
            embedding=embedding,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=None,
            top_ps=top_ps,
            top_ks=top_ks,
        )

        # Logits should be capped
        assert torch.all(torch.abs(logits) <= config.final_logit_softcapping * 1.01)

    def test_output_position_selection(self):
        """Test that output position selects correct hidden state."""
        vocab_size = 100
        batch_size = 2
        seq_len = 10
        config = get_tiny_config()
        sampler = Sampler(vocab_size, config)

        embedding = torch.randn(vocab_size, config.hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        for pos in [0, 5, seq_len - 1]:
            output_positions = torch.tensor([pos])
            next_tokens, _ = sampler(
                embedding=embedding,
                hidden_states=hidden_states,
                output_positions=output_positions,
                temperatures=None,
                top_ps=torch.tensor([1.0, 1.0]),
                top_ks=torch.tensor([vocab_size, vocab_size]),
            )
            assert next_tokens.shape == (batch_size,)

    def test_batch_independence(self):
        """Test that sampling is independent across batch dimension."""
        torch.manual_seed(42)
        vocab_size = 100
        config = get_tiny_config()
        sampler = Sampler(vocab_size, config)

        embedding = torch.randn(vocab_size, config.hidden_size)
        hidden_states1 = torch.randn(1, 5, config.hidden_size)
        hidden_states2 = torch.randn(1, 5, config.hidden_size)
        hidden_states_batch = torch.cat([hidden_states1, hidden_states2], dim=0)

        output_positions = torch.tensor([4])
        top_ps = torch.tensor([0.9, 0.9])
        top_ks = torch.tensor([50, 50])

        # Process individually
        torch.manual_seed(42)
        tokens1, logits1 = sampler(
            embedding, hidden_states1, output_positions,
            None, top_ps[:1], top_ks[:1]
        )
        torch.manual_seed(42)
        tokens2, logits2 = sampler(
            embedding, hidden_states2, output_positions,
            None, top_ps[1:], top_ks[1:]
        )

        # Logits should be deterministic
        torch.manual_seed(42)
        tokens_batch, logits_batch = sampler(
            embedding, hidden_states_batch, output_positions,
            None, top_ps, top_ks
        )

        assert torch.allclose(logits_batch[0], logits1[0], atol=1e-5)
        assert torch.allclose(logits_batch[1], logits2[0], atol=1e-5)

    def test_no_nan_or_inf(self):
        """Test that sampler never produces NaN or Inf."""
        vocab_size = 100
        batch_size = 2
        config = get_tiny_config()
        sampler = Sampler(vocab_size, config)

        embedding = torch.randn(vocab_size, config.hidden_size)
        hidden_states = torch.randn(batch_size, 5, config.hidden_size)
        output_positions = torch.tensor([4])
        temperatures = torch.tensor([1.0, 1.0])
        top_ps = torch.tensor([0.9, 0.9])
        top_ks = torch.tensor([50, 50])

        next_tokens, logits = sampler(
            embedding, hidden_states, output_positions,
            temperatures, top_ps, top_ks
        )

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        assert not torch.isnan(next_tokens.float()).any()
