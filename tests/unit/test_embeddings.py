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
"""Tests for custom Linear and Embedding layers."""

import pytest
import torch
from gemma.model import Linear, Embedding


@pytest.mark.unit
class TestLinear:
    """Test custom Linear layer."""

    def test_initialization_no_quant(self):
        """Test Linear initialization without quantization."""
        in_features, out_features = 128, 256
        layer = Linear(in_features, out_features, quant=False)

        assert layer.weight.shape == (out_features, in_features)
        assert layer.weight.dtype == torch.float32
        assert layer.quant is False
        assert not hasattr(layer, 'weight_scaler')

    def test_initialization_with_quant(self):
        """Test Linear initialization with quantization."""
        in_features, out_features = 128, 256
        layer = Linear(in_features, out_features, quant=True)

        assert layer.weight.shape == (out_features, in_features)
        assert layer.weight.dtype == torch.int8
        assert layer.quant is True
        assert hasattr(layer, 'weight_scaler')
        assert layer.weight_scaler.shape == (out_features,)

    def test_forward_no_quant_shape(self):
        """Test forward pass without quantization maintains shape."""
        in_features, out_features = 128, 256
        batch_size, seq_len = 2, 10
        layer = Linear(in_features, out_features, quant=False)

        x = torch.randn(batch_size, seq_len, in_features)
        output = layer(x)

        assert output.shape == (batch_size, seq_len, out_features)

    def test_forward_with_quant_shape(self):
        """Test forward pass with quantization maintains shape."""
        in_features, out_features = 128, 256
        batch_size, seq_len = 2, 10
        layer = Linear(in_features, out_features, quant=True)
        # Initialize weight_scaler
        layer.weight_scaler.data.fill_(0.1)

        x = torch.randn(batch_size, seq_len, in_features)
        output = layer(x)

        assert output.shape == (batch_size, seq_len, out_features)

    def test_forward_no_quant_computation(self):
        """Test forward pass computation without quantization."""
        in_features, out_features = 4, 8
        layer = Linear(in_features, out_features, quant=False)
        layer.weight.data = torch.eye(out_features, in_features)

        x = torch.ones(1, 1, in_features)
        output = layer(x)

        # With identity-like weights, output should match input dimensions
        assert output.shape == (1, 1, out_features)
        assert not torch.isnan(output).any()

    def test_forward_with_quant_scaling(self):
        """Test forward pass with quantization applies scaling correctly."""
        in_features, out_features = 4, 8
        layer = Linear(in_features, out_features, quant=True)
        layer.weight.data = torch.ones(out_features, in_features, dtype=torch.int8)
        layer.weight_scaler.data = torch.ones(out_features) * 0.5

        x = torch.ones(1, 1, in_features)
        output = layer(x)

        # Output should be weight (1) * scaler (0.5) * input (1) * in_features (4) = 2.0
        expected = torch.ones(1, 1, out_features) * 2.0
        assert torch.allclose(output, expected, atol=1e-5)

    def test_no_gradient_computation(self):
        """Test that weights have requires_grad=False."""
        layer = Linear(128, 256, quant=False)
        assert layer.weight.requires_grad is False


@pytest.mark.unit
class TestEmbedding:
    """Test custom Embedding layer."""

    def test_initialization_no_quant(self):
        """Test Embedding initialization without quantization."""
        num_embeddings, embedding_dim = 1000, 128
        layer = Embedding(num_embeddings, embedding_dim, quant=False)

        assert layer.weight.shape == (num_embeddings, embedding_dim)
        assert layer.weight.dtype == torch.float32
        assert layer.quant is False
        assert not hasattr(layer, 'weight_scaler')

    def test_initialization_with_quant(self):
        """Test Embedding initialization with quantization."""
        num_embeddings, embedding_dim = 1000, 128
        layer = Embedding(num_embeddings, embedding_dim, quant=True)

        assert layer.weight.shape == (num_embeddings, embedding_dim)
        assert layer.weight.dtype == torch.int8
        assert layer.quant is True
        assert hasattr(layer, 'weight_scaler')
        assert layer.weight_scaler.shape == (num_embeddings,)

    def test_forward_no_quant_shape(self):
        """Test forward pass without quantization maintains shape."""
        num_embeddings, embedding_dim = 1000, 128
        batch_size, seq_len = 2, 10
        layer = Embedding(num_embeddings, embedding_dim, quant=False)

        x = torch.randint(0, num_embeddings, (batch_size, seq_len))
        output = layer(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_forward_with_quant_shape(self):
        """Test forward pass with quantization maintains shape."""
        num_embeddings, embedding_dim = 1000, 128
        batch_size, seq_len = 2, 10
        layer = Embedding(num_embeddings, embedding_dim, quant=True)
        layer.weight_scaler.data.fill_(0.1)

        x = torch.randint(0, num_embeddings, (batch_size, seq_len))
        output = layer(x)

        assert output.shape == (batch_size, seq_len, embedding_dim)

    def test_forward_no_quant_lookup(self):
        """Test forward pass correctly looks up embeddings."""
        num_embeddings, embedding_dim = 10, 4
        layer = Embedding(num_embeddings, embedding_dim, quant=False)
        # Set specific embeddings
        layer.weight.data = torch.arange(num_embeddings * embedding_dim).reshape(
            num_embeddings, embedding_dim
        ).float()

        x = torch.tensor([[0, 1, 2]])
        output = layer(x)

        expected = layer.weight[[0, 1, 2]].unsqueeze(0)
        assert torch.allclose(output, expected)

    def test_forward_with_quant_scaling(self):
        """Test forward pass with quantization applies scaling."""
        num_embeddings, embedding_dim = 10, 4
        layer = Embedding(num_embeddings, embedding_dim, quant=True)
        layer.weight.data = torch.ones(num_embeddings, embedding_dim, dtype=torch.int8)
        layer.weight_scaler.data = torch.arange(num_embeddings).float()

        x = torch.tensor([[0, 1, 5]])
        output = layer(x)

        # First embedding should be scaled by 0, second by 1, sixth by 5
        assert output.shape == (1, 3, embedding_dim)
        assert torch.allclose(output[0, 0], torch.zeros(embedding_dim))
        assert torch.allclose(output[0, 1], torch.ones(embedding_dim))
        assert torch.allclose(output[0, 2], torch.ones(embedding_dim) * 5)

    def test_no_gradient_computation(self):
        """Test that weights have requires_grad=False."""
        layer = Embedding(1000, 128, quant=False)
        assert layer.weight.requires_grad is False

    def test_boundary_indices(self):
        """Test embedding lookup with boundary indices."""
        num_embeddings, embedding_dim = 100, 128
        layer = Embedding(num_embeddings, embedding_dim, quant=False)

        # Test first and last valid indices
        x = torch.tensor([[0, num_embeddings - 1]])
        output = layer(x)

        assert output.shape == (1, 2, embedding_dim)
        assert not torch.isnan(output).any()
