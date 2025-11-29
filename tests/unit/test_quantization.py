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
"""Unit tests for quantization functionality."""

import pytest
import torch

from gemma import config as gemma_config
from gemma import model as gemma_model


@pytest.mark.quantization
class TestQuantizedLinear:
    """Tests for quantized Linear layer."""

    def test_quantized_linear_initialization(self):
        """Test quantized linear layer initialization."""
        in_features = 256
        out_features = 512

        linear = gemma_model.Linear(in_features, out_features, quant=True)

        assert linear.quant is True
        assert linear.weight.dtype == torch.int8
        assert hasattr(linear, 'weight_scaler')
        assert linear.weight_scaler.shape == (out_features,)

    def test_quantized_linear_weight_shape(self):
        """Test that quantized weights have correct shape."""
        in_features = 256
        out_features = 512

        linear = gemma_model.Linear(in_features, out_features, quant=True)

        assert linear.weight.shape == (out_features, in_features)

    def test_quantized_linear_forward(self):
        """Test quantized linear forward pass."""
        in_features = 256
        out_features = 512
        batch_size = 2

        linear = gemma_model.Linear(in_features, out_features, quant=True)

        # Initialize weights
        linear.weight.data = torch.randint(-127, 127, (out_features, in_features), dtype=torch.int8)
        linear.weight_scaler.data = torch.randn(out_features)

        x = torch.randn(batch_size, in_features)
        output = linear(x)

        assert output.shape == (batch_size, out_features)
        assert output.dtype != torch.int8  # Output should be float

    def test_quantized_linear_vs_regular(self):
        """Test that quantized linear produces reasonable output compared to regular."""
        in_features = 256
        out_features = 512
        batch_size = 2

        # Create both versions
        linear_quant = gemma_model.Linear(in_features, out_features, quant=True)
        linear_regular = gemma_model.Linear(in_features, out_features, quant=False)

        # Initialize quantized version
        linear_quant.weight.data = torch.randint(-127, 127, (out_features, in_features), dtype=torch.int8)
        linear_quant.weight_scaler.data = torch.ones(out_features) * 0.01

        # Initialize regular version with similar scaled weights
        linear_regular.weight.data = linear_quant.weight.float() * linear_quant.weight_scaler.unsqueeze(-1)

        x = torch.randn(batch_size, in_features)

        output_quant = linear_quant(x)
        output_regular = linear_regular(x)

        # Shapes should match
        assert output_quant.shape == output_regular.shape

        # Values should be similar (with some tolerance for quantization)
        assert torch.allclose(output_quant, output_regular, rtol=0.1, atol=1e-2)

    def test_quantized_linear_weight_range(self):
        """Test that quantized weights are within int8 range."""
        in_features = 256
        out_features = 512

        linear = gemma_model.Linear(in_features, out_features, quant=True)
        linear.weight.data = torch.randint(-127, 127, (out_features, in_features), dtype=torch.int8)

        assert torch.all(linear.weight >= -128)
        assert torch.all(linear.weight <= 127)


@pytest.mark.quantization
class TestQuantizedEmbedding:
    """Tests for quantized Embedding layer."""

    def test_quantized_embedding_initialization(self):
        """Test quantized embedding layer initialization."""
        num_embeddings = 1000
        embedding_dim = 256

        embedding = gemma_model.Embedding(num_embeddings, embedding_dim, quant=True)

        assert embedding.quant is True
        assert embedding.weight.dtype == torch.int8
        assert hasattr(embedding, 'weight_scaler')
        assert embedding.weight_scaler.shape == (num_embeddings,)

    def test_quantized_embedding_weight_shape(self):
        """Test that quantized embedding weights have correct shape."""
        num_embeddings = 1000
        embedding_dim = 256

        embedding = gemma_model.Embedding(num_embeddings, embedding_dim, quant=True)

        assert embedding.weight.shape == (num_embeddings, embedding_dim)

    def test_quantized_embedding_forward(self):
        """Test quantized embedding forward pass."""
        num_embeddings = 1000
        embedding_dim = 256
        batch_size = 2
        seq_len = 8

        embedding = gemma_model.Embedding(num_embeddings, embedding_dim, quant=True)

        # Initialize weights
        embedding.weight.data = torch.randint(-127, 127, (num_embeddings, embedding_dim), dtype=torch.int8)
        embedding.weight_scaler.data = torch.randn(num_embeddings)

        indices = torch.randint(0, num_embeddings, (batch_size, seq_len))
        output = embedding(indices)

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert output.dtype != torch.int8  # Output should be float

    def test_quantized_embedding_weight_range(self):
        """Test that quantized embedding weights are within int8 range."""
        num_embeddings = 1000
        embedding_dim = 256

        embedding = gemma_model.Embedding(num_embeddings, embedding_dim, quant=True)
        embedding.weight.data = torch.randint(-127, 127, (num_embeddings, embedding_dim), dtype=torch.int8)

        assert torch.all(embedding.weight >= -128)
        assert torch.all(embedding.weight <= 127)


@pytest.mark.quantization
class TestQuantizedMLP:
    """Tests for quantized MLP."""

    def test_quantized_mlp_initialization(self):
        """Test quantized MLP initialization."""
        hidden_size = 256
        intermediate_size = 1024

        mlp = gemma_model.GemmaMLP(hidden_size, intermediate_size, quant=True)

        assert mlp.gate_proj.quant is True
        assert mlp.up_proj.quant is True
        assert mlp.down_proj.quant is True

    def test_quantized_mlp_forward(self):
        """Test quantized MLP forward pass."""
        hidden_size = 256
        intermediate_size = 1024
        batch_size = 2
        seq_len = 8

        mlp = gemma_model.GemmaMLP(hidden_size, intermediate_size, quant=True)

        # Initialize weights
        mlp.gate_proj.weight.data = torch.randint(-127, 127, (intermediate_size, hidden_size), dtype=torch.int8)
        mlp.gate_proj.weight_scaler.data = torch.ones(intermediate_size) * 0.01

        mlp.up_proj.weight.data = torch.randint(-127, 127, (intermediate_size, hidden_size), dtype=torch.int8)
        mlp.up_proj.weight_scaler.data = torch.ones(intermediate_size) * 0.01

        mlp.down_proj.weight.data = torch.randint(-127, 127, (hidden_size, intermediate_size), dtype=torch.int8)
        mlp.down_proj.weight_scaler.data = torch.ones(hidden_size) * 0.01

        x = torch.randn(batch_size, seq_len, hidden_size)
        output = mlp(x)

        assert output.shape == (batch_size, seq_len, hidden_size)


@pytest.mark.quantization
class TestQuantizedModel:
    """Tests for fully quantized model configurations."""

    def test_quantized_config_creation(self):
        """Test creating a quantized model config."""
        config = gemma_config.get_config_for_2b()
        config.quant = True

        assert config.quant is True

    def test_quantized_attention(self, cpu_device):
        """Test quantized attention mechanism."""
        config = gemma_config.GemmaConfig(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
            head_dim=64,
            quant=True,
        )

        attention = gemma_model.GemmaAttention(
            config=config,
            attn_type=gemma_config.AttentionType.GLOBAL
        ).to(cpu_device)

        # Check that projections are quantized
        assert attention.qkv_proj.quant is True
        assert attention.o_proj.quant is True

    def test_quantized_decoder_layer(self, cpu_device):
        """Test quantized decoder layer."""
        config = gemma_config.GemmaConfig(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
            intermediate_size=1024,
            head_dim=64,
            quant=True,
        )

        layer = gemma_model.GemmaDecoderLayer(config).to(cpu_device)

        # Check that components are quantized
        assert layer.self_attn.qkv_proj.quant is True
        assert layer.mlp.gate_proj.quant is True


@pytest.mark.quantization
class TestQuantizationAccuracy:
    """Tests for quantization accuracy and numerical stability."""

    def test_weight_scaler_effect(self):
        """Test that weight scaler properly scales quantized weights."""
        in_features = 256
        out_features = 512

        linear = gemma_model.Linear(in_features, out_features, quant=True)

        # Set known values
        linear.weight.data.fill_(1)  # All weights = 1
        linear.weight_scaler.data.fill_(0.1)  # Scaler = 0.1

        # Effective weight should be 0.1 * 1 = 0.1
        x = torch.ones(1, in_features)
        output = linear(x)

        # Each output element should be sum of (0.1 * 256) = 25.6
        expected = torch.ones(1, out_features) * (0.1 * in_features)
        assert torch.allclose(output, expected, rtol=1e-5)

    def test_quantization_preserves_dtype(self):
        """Test that quantization doesn't affect output dtype."""
        in_features = 256
        out_features = 512

        linear = gemma_model.Linear(in_features, out_features, quant=True)
        linear.weight.data = torch.randint(-127, 127, (out_features, in_features), dtype=torch.int8)
        linear.weight_scaler.data = torch.randn(out_features)

        # Test with different input dtypes
        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(2, in_features, dtype=dtype)
            output = linear(x)

            assert output.dtype == dtype

    def test_quantization_numerical_stability(self):
        """Test numerical stability of quantized operations."""
        in_features = 256
        out_features = 512

        linear = gemma_model.Linear(in_features, out_features, quant=True)
        linear.weight.data = torch.randint(-127, 127, (out_features, in_features), dtype=torch.int8)
        linear.weight_scaler.data = torch.randn(out_features)

        # Test with extreme input values
        x_large = torch.randn(2, in_features) * 1000
        x_small = torch.randn(2, in_features) * 1e-5

        output_large = linear(x_large)
        output_small = linear(x_small)

        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()
