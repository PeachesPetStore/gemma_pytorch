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
"""Unit tests for normalization layers."""

import pytest
import torch

from gemma import model as gemma_model


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_rmsnorm_initialization(self):
        """Test RMSNorm initialization."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim, eps=1e-6, add_unit_offset=True)

        assert norm.eps == 1e-6
        assert norm.add_unit_offset is True
        assert norm.weight.shape == (dim,)

    def test_rmsnorm_forward_shape(self):
        """Test RMSNorm forward pass output shape."""
        dim = 256
        batch_size = 2
        seq_len = 8

        norm = gemma_model.RMSNorm(dim=dim)
        x = torch.randn(batch_size, seq_len, dim)

        output = norm(x)

        assert output.shape == x.shape

    def test_rmsnorm_forward_dtype(self):
        """Test that RMSNorm preserves input dtype."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim)

        # Test with float32
        x_float32 = torch.randn(2, 8, dim, dtype=torch.float32)
        output_float32 = norm(x_float32)
        assert output_float32.dtype == torch.float32

        # Test with float16
        x_float16 = torch.randn(2, 8, dim, dtype=torch.float16)
        output_float16 = norm(x_float16)
        assert output_float16.dtype == torch.float16

    def test_rmsnorm_with_unit_offset(self):
        """Test RMSNorm with unit offset enabled."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim, add_unit_offset=True)
        x = torch.randn(2, 8, dim)

        output = norm(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_rmsnorm_without_unit_offset(self):
        """Test RMSNorm with unit offset disabled."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim, add_unit_offset=False)
        x = torch.randn(2, 8, dim)

        output = norm(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_rmsnorm_zero_input(self):
        """Test RMSNorm with zero input."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim, eps=1e-6)
        x = torch.zeros(2, 8, dim)

        output = norm(x)

        assert output.shape == x.shape
        # Output should be zero (or very close to zero)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-5)

    def test_rmsnorm_normalization_effect(self):
        """Test that RMSNorm actually normalizes the input."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim, eps=1e-6, add_unit_offset=False)

        # Initialize weight to ones for this test
        norm.weight.data.fill_(1.0)

        x = torch.randn(2, 8, dim) * 10  # Scaled input

        output = norm(x)

        # Compute RMS of output
        output_rms = torch.sqrt((output ** 2).mean(dim=-1))

        # RMS should be approximately 1 (with some tolerance due to weight)
        assert torch.allclose(output_rms, torch.ones_like(output_rms), rtol=0.5)

    def test_rmsnorm_eps_prevents_div_by_zero(self):
        """Test that epsilon prevents division by zero."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim, eps=1e-6)

        # Very small input
        x = torch.ones(2, 8, dim) * 1e-10

        output = norm(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_rmsnorm_different_eps_values(self):
        """Test RMSNorm with different epsilon values."""
        dim = 256
        x = torch.randn(2, 8, dim)

        for eps in [1e-8, 1e-6, 1e-5]:
            norm = gemma_model.RMSNorm(dim=dim, eps=eps)
            output = norm(x)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()

    def test_rmsnorm_gradient_flow(self):
        """Test that gradients flow through RMSNorm."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim)

        x = torch.randn(2, 8, dim, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_rmsnorm_weight_gradient(self):
        """Test that weight parameter receives gradients."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim)

        x = torch.randn(2, 8, dim)
        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert norm.weight.grad is not None
        assert not torch.isnan(norm.weight.grad).any()

    @pytest.mark.parametrize('batch_size,seq_len,dim', [
        (1, 4, 128),
        (2, 8, 256),
        (4, 16, 512),
        (8, 32, 1024),
    ])
    def test_rmsnorm_various_shapes(self, batch_size, seq_len, dim):
        """Test RMSNorm with various input shapes."""
        norm = gemma_model.RMSNorm(dim=dim)
        x = torch.randn(batch_size, seq_len, dim)

        output = norm(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_rmsnorm_batch_independence(self):
        """Test that RMSNorm processes each sample independently."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim)

        # Create two different inputs
        x1 = torch.randn(1, 8, dim)
        x2 = torch.randn(1, 8, dim)

        # Process separately
        out1 = norm(x1)
        out2 = norm(x2)

        # Process together
        x_batch = torch.cat([x1, x2], dim=0)
        out_batch = norm(x_batch)

        # Results should be identical
        assert torch.allclose(out_batch[0:1], out1, rtol=1e-5)
        assert torch.allclose(out_batch[1:2], out2, rtol=1e-5)

    def test_rmsnorm_device_placement(self):
        """Test RMSNorm on different devices."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim)

        # Test on CPU
        x_cpu = torch.randn(2, 8, dim)
        output_cpu = norm(x_cpu)
        assert output_cpu.device == x_cpu.device

        # Test on CUDA if available
        if torch.cuda.is_available():
            norm_cuda = norm.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = norm_cuda(x_cuda)
            assert output_cuda.device == x_cuda.device

    def test_rmsnorm_eval_mode(self):
        """Test RMSNorm in eval mode."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim)
        norm.eval()

        x = torch.randn(2, 8, dim)
        output = norm(x)

        assert output.shape == x.shape

    def test_rmsnorm_large_values(self):
        """Test RMSNorm with large input values."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim, eps=1e-6)

        x = torch.randn(2, 8, dim) * 1000

        output = norm(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_rmsnorm_small_values(self):
        """Test RMSNorm with small input values."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim, eps=1e-6)

        x = torch.randn(2, 8, dim) * 1e-5

        output = norm(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_rmsnorm_consistency(self):
        """Test that RMSNorm produces consistent results."""
        dim = 256
        norm = gemma_model.RMSNorm(dim=dim)

        x = torch.randn(2, 8, dim)

        # Run multiple times
        output1 = norm(x)
        output2 = norm(x)

        # Should be identical
        assert torch.allclose(output1, output2)
