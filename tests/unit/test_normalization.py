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
"""Tests for normalization layers."""

import pytest
import torch
from gemma.model import RMSNorm


@pytest.mark.unit
class TestRMSNorm:
    """Test RMSNorm layer."""

    def test_initialization(self):
        """Test RMSNorm initialization."""
        dim = 128
        norm = RMSNorm(dim)
        assert norm.weight.shape == (dim,)
        assert torch.all(norm.weight == 0)
        assert norm.eps == 1e-6
        assert norm.add_unit_offset is True

    def test_initialization_custom_eps(self):
        """Test RMSNorm with custom epsilon."""
        norm = RMSNorm(128, eps=1e-5)
        assert norm.eps == 1e-5

    def test_initialization_no_unit_offset(self):
        """Test RMSNorm without unit offset."""
        norm = RMSNorm(128, add_unit_offset=False)
        assert norm.add_unit_offset is False

    def test_forward_shape(self):
        """Test forward pass maintains shape."""
        batch_size, seq_len, dim = 2, 10, 128
        norm = RMSNorm(dim)
        x = torch.randn(batch_size, seq_len, dim)
        output = norm(x)
        assert output.shape == x.shape

    def test_forward_with_unit_offset(self):
        """Test forward pass with unit offset."""
        dim = 128
        norm = RMSNorm(dim, add_unit_offset=True)
        x = torch.randn(2, 10, dim)
        output = norm(x)

        # Check output is not NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_without_unit_offset(self):
        """Test forward pass without unit offset."""
        dim = 128
        norm = RMSNorm(dim, add_unit_offset=False)
        # Set weight to ones for predictable behavior
        norm.weight.data.fill_(1.0)
        x = torch.randn(2, 10, dim)
        output = norm(x)

        # Check output is not NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_normalization_reduces_variance(self):
        """Test that RMSNorm reduces variance."""
        dim = 128
        norm = RMSNorm(dim)
        norm.weight.data.fill_(0.0)  # Set to 0 so (1 + 0) = 1
        x = torch.randn(2, 10, dim) * 10  # High variance input
        output = norm(x)

        # Output variance should be approximately 1
        output_var = output.pow(2).mean(-1)
        assert torch.allclose(output_var, torch.ones_like(output_var), atol=1e-4)

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        dim = 128
        norm = RMSNorm(dim)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(2, 10, dim, dtype=dtype)
            output = norm(x)
            assert output.dtype == dtype

    def test_zero_input(self):
        """Test behavior with zero input."""
        dim = 128
        norm = RMSNorm(dim)
        x = torch.zeros(2, 10, dim)
        output = norm(x)

        # Should not crash and should produce zeros
        assert not torch.isnan(output).any()

    def test_small_values_stability(self):
        """Test numerical stability with very small values."""
        dim = 128
        norm = RMSNorm(dim, eps=1e-6)
        x = torch.randn(2, 10, dim) * 1e-10
        output = norm(x)

        # Should not produce NaN or Inf due to epsilon
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_values_stability(self):
        """Test numerical stability with very large values."""
        dim = 128
        norm = RMSNorm(dim)
        x = torch.randn(2, 10, dim) * 1e10
        output = norm(x)

        # Check output is not NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_batch_independence(self):
        """Test that normalization is independent across batch dimension."""
        dim = 128
        norm = RMSNorm(dim)

        x1 = torch.randn(1, 10, dim)
        x2 = torch.randn(1, 10, dim)
        x_batch = torch.cat([x1, x2], dim=0)

        out1 = norm(x1)
        out2 = norm(x2)
        out_batch = norm(x_batch)

        # Outputs should match when processed separately vs in batch
        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through RMSNorm."""
        dim = 128
        norm = RMSNorm(dim)
        x = torch.randn(2, 10, dim, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        assert norm.weight.grad is not None
