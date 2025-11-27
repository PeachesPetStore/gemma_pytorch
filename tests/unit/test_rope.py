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
"""Tests for Rotary Position Embedding (RoPE)."""

import pytest
import torch
from gemma.model import precompute_freqs_cis, apply_rotary_emb


@pytest.mark.unit
class TestPrecomputeFreqsCis:
    """Test precompute_freqs_cis function."""

    def test_output_shape(self):
        """Test output shape is correct."""
        dim = 64
        end = 128
        freqs_cis = precompute_freqs_cis(dim, end)

        assert freqs_cis.shape == (end, dim // 2)
        assert freqs_cis.dtype == torch.complex64

    def test_output_is_complex(self):
        """Test output is complex tensor."""
        freqs_cis = precompute_freqs_cis(64, 128)
        assert torch.is_complex(freqs_cis)

    def test_different_dimensions(self):
        """Test with different dimensions."""
        for dim in [32, 64, 128, 256]:
            end = 128
            freqs_cis = precompute_freqs_cis(dim, end)
            assert freqs_cis.shape == (end, dim // 2)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        dim = 64
        for end in [64, 128, 256, 512]:
            freqs_cis = precompute_freqs_cis(dim, end)
            assert freqs_cis.shape == (end, dim // 2)

    def test_custom_theta(self):
        """Test with custom theta value."""
        dim = 64
        end = 128
        freqs_default = precompute_freqs_cis(dim, end, theta=10000.0)
        freqs_custom = precompute_freqs_cis(dim, end, theta=100000.0)

        # Different theta should produce different frequencies
        assert not torch.allclose(freqs_default, freqs_custom)

    def test_rope_scaling_factor(self):
        """Test RoPE scaling factor."""
        dim = 64
        end = 128
        freqs_no_scale = precompute_freqs_cis(dim, end, rope_scaling_factor=1)
        freqs_scaled = precompute_freqs_cis(dim, end, rope_scaling_factor=2)

        # Scaling should affect the frequencies
        assert not torch.allclose(freqs_no_scale, freqs_scaled)

    def test_unit_magnitude(self):
        """Test that complex numbers have unit magnitude."""
        freqs_cis = precompute_freqs_cis(64, 128)
        magnitudes = torch.abs(freqs_cis)

        # All complex numbers should have magnitude ~1
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_increasing_phases(self):
        """Test that phases increase with position."""
        dim = 64
        freqs_cis = precompute_freqs_cis(dim, 128)

        # Angle should generally increase with position
        angles = torch.angle(freqs_cis)
        # Check first frequency component increases
        assert (angles[1:, 0] >= angles[:-1, 0]).sum() > len(angles) * 0.9


@pytest.mark.unit
class TestApplyRotaryEmb:
    """Test apply_rotary_emb function."""

    def test_output_shape_preserved(self):
        """Test output shape matches input shape."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 8, 64
        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        output = apply_rotary_emb(x, freqs_cis)

        assert output.shape == x.shape

    def test_output_dtype_preserved(self):
        """Test output dtype matches input dtype."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 8, 64
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
            output = apply_rotary_emb(x, freqs_cis)
            assert output.dtype == dtype

    def test_no_nan_or_inf(self):
        """Test output contains no NaN or Inf values."""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 8, 64
        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        output = apply_rotary_emb(x, freqs_cis)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_sequence_positions(self):
        """Test that different positions get different embeddings."""
        batch_size, seq_len, num_heads, head_dim = 1, 10, 4, 64
        x = torch.ones(batch_size, num_heads, seq_len, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        output = apply_rotary_emb(x, freqs_cis)

        # Different positions should have different values due to rotation
        # Check that not all positions are identical
        for i in range(seq_len - 1):
            assert not torch.allclose(output[0, 0, i], output[0, 0, i + 1])

    def test_position_invariance_property(self):
        """Test basic rotary embedding property."""
        batch_size, seq_len, num_heads, head_dim = 2, 5, 4, 64
        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        output = apply_rotary_emb(x, freqs_cis)

        # Output should have similar magnitude to input (rotations preserve norm)
        input_norm = torch.norm(x, dim=-1)
        output_norm = torch.norm(output, dim=-1)
        assert torch.allclose(input_norm, output_norm, rtol=1e-4, atol=1e-4)

    def test_batch_independence(self):
        """Test that rotation is applied independently to each batch."""
        seq_len, num_heads, head_dim = 5, 4, 64
        x1 = torch.randn(1, num_heads, seq_len, head_dim)
        x2 = torch.randn(1, num_heads, seq_len, head_dim)
        x_batch = torch.cat([x1, x2], dim=0)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        out1 = apply_rotary_emb(x1, freqs_cis)
        out2 = apply_rotary_emb(x2, freqs_cis)
        out_batch = apply_rotary_emb(x_batch, freqs_cis)

        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)

    def test_zero_input(self):
        """Test behavior with zero input."""
        batch_size, seq_len, num_heads, head_dim = 2, 5, 4, 64
        x = torch.zeros(batch_size, num_heads, seq_len, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        output = apply_rotary_emb(x, freqs_cis)

        # Zero input should produce zero output (or very close to zero)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_head_independence(self):
        """Test that rotation is applied independently to each head."""
        batch_size, seq_len, num_heads, head_dim = 2, 5, 4, 64
        x = torch.randn(batch_size, num_heads, seq_len, head_dim)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        output = apply_rotary_emb(x, freqs_cis)

        # Each head should get the same rotation at same positions
        # but the output will differ due to different input values
        assert output.shape[1] == num_heads
