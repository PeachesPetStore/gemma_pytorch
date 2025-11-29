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
"""Integration tests for Gemma models."""

import pytest
import torch

from gemma import config as gemma_config
from gemma import model as gemma_model


class TestModelForward:
    """Integration tests for model forward pass."""

    @pytest.mark.slow
    def test_gemma_decoder_layer_forward(self, config_2b, cpu_device):
        """Test GemmaDecoderLayer forward pass."""
        layer = gemma_model.GemmaDecoderLayer(config_2b)
        layer = layer.to(cpu_device)

        batch_size = 2
        seq_len = 8
        hidden_size = config_2b.hidden_size

        hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(cpu_device)
        freqs_cis = gemma_model.precompute_freqs_cis(
            config_2b.head_dim,
            seq_len
        ).to(cpu_device)
        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        # Initialize KV caches
        k_cache = torch.zeros(
            batch_size, seq_len,
            config_2b.num_key_value_heads,
            config_2b.head_dim
        ).to(cpu_device)
        v_cache = torch.zeros(
            batch_size, seq_len,
            config_2b.num_key_value_heads,
            config_2b.head_dim
        ).to(cpu_device)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        output = layer(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=None
        )

        assert output.shape == (batch_size, seq_len, hidden_size)

    @pytest.mark.slow
    def test_gemma2_decoder_layer_forward(self, config_2b_v2, cpu_device):
        """Test Gemma2DecoderLayer forward pass."""
        layer = gemma_model.Gemma2DecoderLayer(
            config_2b_v2,
            attn_type=gemma_config.AttentionType.GLOBAL
        )
        layer = layer.to(cpu_device)

        batch_size = 2
        seq_len = 8
        hidden_size = config_2b_v2.hidden_size

        hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(cpu_device)
        freqs_cis = gemma_model.precompute_freqs_cis(
            config_2b_v2.head_dim,
            seq_len
        ).to(cpu_device)
        kv_write_indices = torch.arange(seq_len).to(cpu_device)

        k_cache = torch.zeros(
            batch_size, seq_len,
            config_2b_v2.num_key_value_heads,
            config_2b_v2.head_dim
        ).to(cpu_device)
        v_cache = torch.zeros(
            batch_size, seq_len,
            config_2b_v2.num_key_value_heads,
            config_2b_v2.head_dim
        ).to(cpu_device)
        kv_cache = (k_cache, v_cache)

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)
        local_mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        output = layer(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask
        )

        assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.requires_checkpoint
class TestModelLoading:
    """Integration tests for model checkpoint loading."""

    def test_load_weights_file(self, config_2b, temp_dir):
        """Test loading weights from a single file."""
        # This test requires actual checkpoint files
        # Skip for now as it's an example
        pytest.skip("Requires actual checkpoint file")

    def test_load_weights_sharded(self, config_2b, temp_dir):
        """Test loading weights from sharded files."""
        # This test requires actual checkpoint files
        # Skip for now as it's an example
        pytest.skip("Requires actual sharded checkpoint files")


@pytest.mark.gpu
class TestGPUInference:
    """Integration tests for GPU inference."""

    def test_model_on_gpu(self, config_2b, skip_if_no_gpu):
        """Test that model can be moved to GPU."""
        device = torch.device("cuda")

        # Use a smaller config for faster testing
        small_config = gemma_config.GemmaConfig(
            num_hidden_layers=2,
            hidden_size=256,
            intermediate_size=1024,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64
        )

        model = gemma_model.GemmaModel(small_config)
        model = model.to(device)

        # Verify model is on GPU
        for param in model.parameters():
            assert param.device.type == "cuda"
