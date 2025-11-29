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
"""Integration tests for text generation."""

import os
import pytest
import torch

from gemma import config as gemma_config
from gemma import model as gemma_model


@pytest.mark.integration
@pytest.mark.requires_checkpoint
class TestGenerationWithCheckpoint:
    """Tests requiring actual model checkpoints."""

    def test_generate_with_checkpoint(self, skip_if_no_checkpoint):
        """Test generation with actual checkpoint."""
        # This would test with real checkpoint
        pytest.skip("Requires actual checkpoint - implement when checkpoint available")


@pytest.mark.integration
class TestGenerationPipeline:
    """Tests for generation pipeline without requiring checkpoints."""

    @pytest.fixture
    def small_model_config(self):
        """Create a very small model config for fast testing."""
        return gemma_config.GemmaConfig(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=128,
            intermediate_size=512,
            head_dim=64,
            max_position_embeddings=128,
            vocab_size=256,
            quant=False,
            tokenizer="tokenizer/tokenizer.model",
        )

    @pytest.mark.slow
    def test_model_forward_pass(self, small_model_config, cpu_device):
        """Test complete forward pass through model."""
        if not os.path.exists(small_model_config.tokenizer):
            pytest.skip("Tokenizer not available")

        model = gemma_model.GemmaForCausalLM(small_model_config)
        model = model.to(cpu_device).eval()

        batch_size = 1
        seq_len = 8

        # Prepare inputs
        input_token_ids = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len)).to(cpu_device)
        input_positions = torch.arange(seq_len).to(cpu_device)
        output_positions = torch.tensor([seq_len - 1]).to(cpu_device)

        # Prepare KV caches
        kv_caches = []
        for _ in range(small_model_config.num_hidden_layers):
            k_cache = torch.zeros(
                batch_size, seq_len,
                small_model_config.num_key_value_heads,
                small_model_config.head_dim
            ).to(cpu_device)
            v_cache = torch.zeros(
                batch_size, seq_len,
                small_model_config.num_key_value_heads,
                small_model_config.head_dim
            ).to(cpu_device)
            kv_caches.append((k_cache, v_cache))

        # Prepare mask
        mask = torch.triu(
            torch.full((1, 1, seq_len, seq_len), -2.3819763e38),
            diagonal=1
        ).to(cpu_device)

        # Forward pass
        next_tokens, logits = model(
            input_token_ids=input_token_ids,
            input_positions=input_positions,
            kv_write_indices=None,
            kv_caches=kv_caches,
            mask=mask,
            output_positions=output_positions,
            temperatures=torch.tensor([1.0]).to(cpu_device),
            top_ps=torch.tensor([0.95]).to(cpu_device),
            top_ks=torch.tensor([50]).to(cpu_device),
        )

        assert next_tokens.shape == (batch_size,)
        assert logits.shape == (batch_size, small_model_config.vocab_size)

    @pytest.mark.slow
    def test_generation_basic(self, small_model_config, cpu_device):
        """Test basic generation."""
        if not os.path.exists(small_model_config.tokenizer):
            pytest.skip("Tokenizer not available")

        model = gemma_model.GemmaForCausalLM(small_model_config)
        model = model.to(cpu_device).eval()

        prompt = "Hello"
        output = model.generate(
            prompts=prompt,
            device=cpu_device,
            output_len=5,
            temperature=1.0,
            top_p=0.95,
            top_k=50,
        )

        assert isinstance(output, str)
        assert len(output) > 0

    @pytest.mark.slow
    def test_generation_batch(self, small_model_config, cpu_device):
        """Test batch generation."""
        if not os.path.exists(small_model_config.tokenizer):
            pytest.skip("Tokenizer not available")

        model = gemma_model.GemmaForCausalLM(small_model_config)
        model = model.to(cpu_device).eval()

        prompts = ["Hello", "Hi there"]
        outputs = model.generate(
            prompts=prompts,
            device=cpu_device,
            output_len=5,
            temperature=1.0,
            top_p=0.95,
            top_k=50,
        )

        assert isinstance(outputs, list)
        assert len(outputs) == len(prompts)
        assert all(isinstance(o, str) for o in outputs)

    @pytest.mark.slow
    def test_generation_greedy(self, small_model_config, cpu_device):
        """Test greedy generation (temperature=None)."""
        if not os.path.exists(small_model_config.tokenizer):
            pytest.skip("Tokenizer not available")

        model = gemma_model.GemmaForCausalLM(small_model_config)
        model = model.to(cpu_device).eval()

        prompt = "Hello"
        output = model.generate(
            prompts=prompt,
            device=cpu_device,
            output_len=5,
            temperature=None,  # Greedy
        )

        assert isinstance(output, str)

    @pytest.mark.slow
    def test_generation_deterministic(self, small_model_config, cpu_device):
        """Test that greedy generation is deterministic."""
        if not os.path.exists(small_model_config.tokenizer):
            pytest.skip("Tokenizer not available")

        model = gemma_model.GemmaForCausalLM(small_model_config)
        model = model.to(cpu_device).eval()

        prompt = "Hello"

        # Generate twice with greedy sampling
        output1 = model.generate(
            prompts=prompt,
            device=cpu_device,
            output_len=5,
            temperature=None,
        )

        output2 = model.generate(
            prompts=prompt,
            device=cpu_device,
            output_len=5,
            temperature=None,
        )

        # Should be identical
        assert output1 == output2

    @pytest.mark.slow
    @pytest.mark.parametrize('output_len', [1, 5, 10])
    def test_generation_various_lengths(self, small_model_config, cpu_device, output_len):
        """Test generation with various output lengths."""
        if not os.path.exists(small_model_config.tokenizer):
            pytest.skip("Tokenizer not available")

        model = gemma_model.GemmaForCausalLM(small_model_config)
        model = model.to(cpu_device).eval()

        prompt = "Hello"
        output = model.generate(
            prompts=prompt,
            device=cpu_device,
            output_len=output_len,
            temperature=1.0,
        )

        assert isinstance(output, str)

    @pytest.mark.slow
    def test_generation_with_different_prompts(self, small_model_config, cpu_device):
        """Test generation with different prompt types."""
        if not os.path.exists(small_model_config.tokenizer):
            pytest.skip("Tokenizer not available")

        model = gemma_model.GemmaForCausalLM(small_model_config)
        model = model.to(cpu_device).eval()

        prompts = [
            "Hello",
            "The capital of France is",
            "Once upon a time",
        ]

        for prompt in prompts:
            output = model.generate(
                prompts=prompt,
                device=cpu_device,
                output_len=5,
            )
            assert isinstance(output, str)
            assert len(output) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestKVCacheManagement:
    """Tests for KV cache management during generation."""

    @pytest.fixture
    def tiny_config(self):
        """Create tiny config for cache testing."""
        return gemma_config.GemmaConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=64,
            intermediate_size=256,
            head_dim=32,
            max_position_embeddings=64,
            vocab_size=100,
            quant=False,
        )

    def test_kv_cache_initialization(self, tiny_config, cpu_device):
        """Test KV cache initialization."""
        batch_size = 2
        max_seq_len = 32
        num_layers = tiny_config.num_hidden_layers

        kv_caches = []
        for _ in range(num_layers):
            k_cache = torch.zeros(
                batch_size, max_seq_len,
                tiny_config.num_key_value_heads,
                tiny_config.head_dim
            ).to(cpu_device)
            v_cache = torch.zeros(
                batch_size, max_seq_len,
                tiny_config.num_key_value_heads,
                tiny_config.head_dim
            ).to(cpu_device)
            kv_caches.append((k_cache, v_cache))

        assert len(kv_caches) == num_layers
        for k_cache, v_cache in kv_caches:
            assert k_cache.shape == (batch_size, max_seq_len, 1, 32)
            assert v_cache.shape == (batch_size, max_seq_len, 1, 32)

    def test_kv_cache_update(self, tiny_config, cpu_device):
        """Test that KV cache is updated during forward pass."""
        model = gemma_model.GemmaModel(tiny_config).to(cpu_device)

        batch_size = 1
        seq_len = 4

        hidden_states = torch.randn(batch_size, seq_len, tiny_config.hidden_size).to(cpu_device)

        freqs_cis = {
            gemma_config.AttentionType.GLOBAL: gemma_model.precompute_freqs_cis(
                tiny_config.head_dim, seq_len
            ).to(cpu_device)
        }

        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        # Initialize empty caches
        kv_caches = []
        for _ in range(tiny_config.num_hidden_layers):
            k_cache = torch.zeros(
                batch_size, seq_len,
                tiny_config.num_key_value_heads,
                tiny_config.head_dim
            ).to(cpu_device)
            v_cache = torch.zeros(
                batch_size, seq_len,
                tiny_config.num_key_value_heads,
                tiny_config.head_dim
            ).to(cpu_device)
            kv_caches.append((k_cache, v_cache))

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        # Caches should be zero before
        assert torch.all(kv_caches[0][0] == 0)

        # Forward pass
        _ = model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=None,
        )

        # Caches should be updated after
        assert not torch.all(kv_caches[0][0] == 0)


@pytest.mark.integration
@pytest.mark.gpu
class TestGPUGeneration:
    """Tests for GPU-accelerated generation."""

    def test_generation_on_gpu(self, skip_if_no_gpu):
        """Test generation on GPU."""
        config = gemma_config.GemmaConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=64,
            intermediate_size=256,
            head_dim=32,
            vocab_size=100,
        )

        device = torch.device("cuda")
        model = gemma_model.GemmaModel(config).to(device)

        # Verify model is on GPU
        for param in model.parameters():
            assert param.device.type == "cuda"
