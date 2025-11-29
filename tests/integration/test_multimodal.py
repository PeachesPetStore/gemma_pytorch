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
"""Integration tests for multimodal models."""

import pytest
import torch

from gemma import config as gemma_config
from gemma import gemma3_model
from gemma.siglip_vision import siglip_vision_model, config as siglip_config


@pytest.mark.integration
@pytest.mark.multimodal
class TestSiglipVisionModel:
    """Tests for Siglip vision encoder."""

    @pytest.fixture
    def vision_config(self):
        """Create a small vision config for testing."""
        return siglip_config.SiglipVisionModelConfig(
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            image_size=224,
            patch_size=16,
            embedding_dim=256,
        )

    def test_siglip_initialization(self, vision_config):
        """Test Siglip vision model initialization."""
        model = siglip_vision_model.SiglipVisionModel(vision_config)

        assert model.config == vision_config

    @pytest.mark.slow
    def test_siglip_forward(self, vision_config, cpu_device):
        """Test Siglip vision model forward pass."""
        model = siglip_vision_model.SiglipVisionModel(vision_config).to(cpu_device)

        batch_size = 2
        # Input: BxCxHxW
        images = torch.randn(
            batch_size,
            vision_config.num_channels,
            vision_config.image_size,
            vision_config.image_size
        ).to(cpu_device)

        output = model(images)

        # Output should be BxNxD where N is number of patches
        expected_num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
        assert output.shape[0] == batch_size
        assert output.shape[1] == expected_num_patches
        assert output.shape[2] == vision_config.embedding_dim

    @pytest.mark.slow
    def test_siglip_different_batch_sizes(self, vision_config, cpu_device):
        """Test Siglip with different batch sizes."""
        model = siglip_vision_model.SiglipVisionModel(vision_config).to(cpu_device)

        for batch_size in [1, 2, 4]:
            images = torch.randn(
                batch_size,
                vision_config.num_channels,
                vision_config.image_size,
                vision_config.image_size
            ).to(cpu_device)

            output = model(images)

            assert output.shape[0] == batch_size


@pytest.mark.integration
@pytest.mark.multimodal
@pytest.mark.slow
class TestGemma3Multimodal:
    """Tests for Gemma 3 multimodal model."""

    @pytest.fixture
    def multimodal_config(self):
        """Create a small multimodal config for testing."""
        vision_config = siglip_config.SiglipVisionModelConfig(
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_channels=3,
            image_size=224,
            patch_size=16,
            embedding_dim=128,
        )

        return gemma_config.GemmaConfig(
            architecture=gemma_config.Architecture.GEMMA_3,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=128,
            intermediate_size=512,
            head_dim=64,
            max_position_embeddings=512,
            vocab_size=262144,
            quant=False,
            tokenizer="tokenizer/gemma3_cleaned_262144_v2.spiece.model",
            vision_config=vision_config,
            rope_wave_length={
                gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
                gemma_config.AttentionType.GLOBAL: 1_000_000,
            },
            attn_types=(
                gemma_config.AttentionType.LOCAL_SLIDING,
                gemma_config.AttentionType.GLOBAL,
            ),
            use_qk_norm=True,
            rope_scaling_factor=8,
        )

    def test_gemma3_multimodal_initialization(self, multimodal_config):
        """Test Gemma 3 multimodal model initialization."""
        import os
        if not os.path.exists(multimodal_config.tokenizer):
            pytest.skip("Gemma 3 tokenizer not available")

        model = gemma3_model.Gemma3ForMultimodalLM(multimodal_config)

        assert model.config == multimodal_config
        assert model.siglip_vision_model is not None
        assert model.mm_soft_embedding_norm is not None
        assert model.mm_input_projection is not None

    def test_gemma3_multimodal_forward(self, multimodal_config, cpu_device):
        """Test Gemma 3 multimodal forward pass."""
        import os
        if not os.path.exists(multimodal_config.tokenizer):
            pytest.skip("Gemma 3 tokenizer not available")

        model = gemma3_model.Gemma3ForMultimodalLM(multimodal_config).to(cpu_device)

        batch_size = 1
        seq_len = 16
        num_images = 1

        # Prepare inputs
        input_token_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(cpu_device)
        image_patches = torch.randn(
            batch_size, num_images, 3, 896, 896
        ).to(cpu_device)
        image_presence_mask = torch.ones(batch_size, num_images).to(cpu_device)

        input_positions = torch.arange(seq_len).to(cpu_device)
        output_positions = torch.tensor([seq_len - 1]).to(cpu_device)

        # Prepare KV caches
        kv_caches = []
        for _ in range(multimodal_config.num_hidden_layers):
            k_cache = torch.zeros(
                batch_size, seq_len,
                multimodal_config.num_key_value_heads,
                multimodal_config.head_dim
            ).to(cpu_device)
            v_cache = torch.zeros(
                batch_size, seq_len,
                multimodal_config.num_key_value_heads,
                multimodal_config.head_dim
            ).to(cpu_device)
            kv_caches.append((k_cache, v_cache))

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        # Forward pass
        next_tokens, logits = model(
            input_token_ids=input_token_ids,
            image_patches=image_patches,
            image_presence_mask=image_presence_mask,
            input_positions=input_positions,
            kv_caches=kv_caches,
            mask=mask,
            output_positions=output_positions,
            temperatures=torch.tensor([1.0]).to(cpu_device),
            top_ps=torch.tensor([0.95]).to(cpu_device),
            top_ks=torch.tensor([50]).to(cpu_device),
        )

        assert next_tokens.shape == (batch_size,)
        assert logits.shape == (batch_size, multimodal_config.vocab_size)

    def test_gemma3_without_images(self, multimodal_config, cpu_device):
        """Test Gemma 3 multimodal with no images (text-only)."""
        import os
        if not os.path.exists(multimodal_config.tokenizer):
            pytest.skip("Gemma 3 tokenizer not available")

        model = gemma3_model.Gemma3ForMultimodalLM(multimodal_config).to(cpu_device)

        batch_size = 1
        seq_len = 16

        input_token_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(cpu_device)
        input_positions = torch.arange(seq_len).to(cpu_device)
        output_positions = torch.tensor([seq_len - 1]).to(cpu_device)

        kv_caches = []
        for _ in range(multimodal_config.num_hidden_layers):
            k_cache = torch.zeros(
                batch_size, seq_len,
                multimodal_config.num_key_value_heads,
                multimodal_config.head_dim
            ).to(cpu_device)
            v_cache = torch.zeros(
                batch_size, seq_len,
                multimodal_config.num_key_value_heads,
                multimodal_config.head_dim
            ).to(cpu_device)
            kv_caches.append((k_cache, v_cache))

        mask = torch.zeros(1, 1, seq_len, seq_len).to(cpu_device)

        # Forward pass without images
        next_tokens, logits = model(
            input_token_ids=input_token_ids,
            image_patches=None,
            image_presence_mask=torch.zeros(batch_size, 0).to(cpu_device),
            input_positions=input_positions,
            kv_caches=kv_caches,
            mask=mask,
            output_positions=output_positions,
            temperatures=torch.tensor([1.0]).to(cpu_device),
            top_ps=torch.tensor([0.95]).to(cpu_device),
            top_ks=torch.tensor([50]).to(cpu_device),
        )

        assert next_tokens.shape == (batch_size,)

    def test_populate_image_embeddings(self, multimodal_config, cpu_device):
        """Test image embedding population."""
        import os
        if not os.path.exists(multimodal_config.tokenizer):
            pytest.skip("Gemma 3 tokenizer not available")

        model = gemma3_model.Gemma3ForMultimodalLM(multimodal_config).to(cpu_device)

        batch_size = 2
        seq_len = 10
        num_images = 1
        num_image_tokens = 4

        hidden_states = torch.randn(batch_size, seq_len, multimodal_config.hidden_size).to(cpu_device)
        image_embeddings = torch.randn(
            batch_size * num_images, num_image_tokens, multimodal_config.hidden_size
        ).to(cpu_device)

        # Create input_token_ids with image placeholder tokens
        input_token_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(cpu_device)
        # Set some positions as image placeholders
        input_token_ids[:, 2:2+num_image_tokens] = model.tokenizer.image_token_placeholder_id

        image_presence_mask = torch.ones(batch_size, num_images).to(cpu_device)

        output = model.populate_image_embeddings(
            hidden_states,
            image_embeddings,
            input_token_ids,
            image_presence_mask
        )

        assert output.shape == hidden_states.shape


@pytest.mark.integration
@pytest.mark.multimodal
class TestVisionModelComponents:
    """Tests for individual vision model components."""

    def test_average_pool_2d(self, cpu_device):
        """Test AveragePool2D layer."""
        from gemma.siglip_vision.siglip_vision_model import AveragePool2D

        config = siglip_config.SiglipVisionModelConfig()
        pool = AveragePool2D(config).to(cpu_device)

        batch_size = 2
        seq_len = 64 * 64  # Must be perfect square
        channels = 256

        x = torch.randn(batch_size, seq_len, channels).to(cpu_device)
        output = pool(x)

        # Output should be downsampled by 4x4 = 16x
        expected_seq_len = seq_len // 16
        assert output.shape == (batch_size, expected_seq_len, channels)

    def test_siglip_attention(self, cpu_device):
        """Test SiglipAttention module."""
        from gemma.siglip_vision.siglip_vision_model import SiglipAttention

        dim = 256
        num_heads = 4
        head_dim = 64

        attention = SiglipAttention(dim, num_heads, head_dim).to(cpu_device)

        batch_size = 2
        seq_len = 16

        x = torch.randn(batch_size, seq_len, dim).to(cpu_device)
        output = attention(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_siglip_mlp(self, cpu_device):
        """Test SiglipMLP module."""
        from gemma.siglip_vision.siglip_vision_model import SiglipMLP

        hidden_size = 256
        intermediate_size = 1024

        mlp = SiglipMLP(hidden_size, intermediate_size).to(cpu_device)

        batch_size = 2
        seq_len = 16

        x = torch.randn(batch_size, seq_len, hidden_size).to(cpu_device)
        output = mlp(x)

        assert output.shape == (batch_size, seq_len, hidden_size)
