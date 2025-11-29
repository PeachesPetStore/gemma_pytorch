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
"""Integration tests for XLA/TPU functionality."""

import pytest
import sys

# Try to import torch_xla
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

pytestmark = pytest.mark.skipif(not HAS_XLA, reason="torch_xla not available")


@pytest.mark.integration
@pytest.mark.tpu
class TestXLABasic:
    """Basic tests for XLA functionality."""

    def test_xla_import(self):
        """Test that XLA can be imported."""
        assert HAS_XLA
        import torch_xla
        assert torch_xla is not None

    def test_xla_device(self):
        """Test XLA device availability."""
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        assert device is not None

    def test_tensor_on_xla_device(self):
        """Test creating tensor on XLA device."""
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        x = torch.randn(2, 3, device=device)

        assert x.device.type == 'xla'


@pytest.mark.integration
@pytest.mark.tpu
@pytest.mark.slow
class TestXLAModel:
    """Tests for running models on XLA."""

    def test_model_on_xla(self):
        """Test moving model to XLA device."""
        import torch
        import torch_xla.core.xla_model as xm
        from gemma import config as gemma_config
        from gemma import model as gemma_model

        device = xm.xla_device()

        # Create small model
        config = gemma_config.GemmaConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=64,
            intermediate_size=256,
            head_dim=32,
            vocab_size=100,
        )

        model = gemma_model.GemmaModel(config).to(device)

        # Verify model is on XLA device
        for param in model.parameters():
            assert param.device.type == 'xla'

    def test_forward_pass_on_xla(self):
        """Test forward pass on XLA device."""
        import torch
        import torch_xla.core.xla_model as xm
        from gemma import config as gemma_config
        from gemma import model as gemma_model

        device = xm.xla_device()

        config = gemma_config.GemmaConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=64,
            intermediate_size=256,
            head_dim=32,
            vocab_size=100,
        )

        model = gemma_model.GemmaModel(config).to(device)

        batch_size = 1
        seq_len = 4

        # Create inputs on XLA device
        hidden_states = torch.randn(batch_size, seq_len, 64, device=device)

        freqs_cis = {
            gemma_config.AttentionType.GLOBAL: gemma_model.precompute_freqs_cis(
                32, seq_len
            ).to(device)
        }

        kv_write_indices = torch.arange(seq_len, device=device)

        kv_caches = []
        k_cache = torch.zeros(batch_size, seq_len, 1, 32, device=device)
        v_cache = torch.zeros(batch_size, seq_len, 1, 32, device=device)
        kv_caches.append((k_cache, v_cache))

        mask = torch.zeros(1, 1, seq_len, seq_len, device=device)

        # Forward pass
        output = model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=None,
        )

        # Synchronize XLA
        xm.mark_step()

        assert output.shape == (batch_size, seq_len, 64)
        assert output.device.type == 'xla'


@pytest.mark.integration
@pytest.mark.tpu
class TestXLAModelParallel:
    """Tests for XLA model parallelism."""

    def test_xla_model_parallel_import(self):
        """Test that XLA model parallel module can be imported."""
        try:
            from gemma import xla_model_parallel
            assert xla_model_parallel is not None
        except ImportError:
            pytest.skip("XLA model parallel not available")

    def test_model_parallel_config(self):
        """Test model parallel configuration."""
        try:
            from gemma import xla_model_parallel
            # Basic test to ensure module loads
            assert hasattr(xla_model_parallel, '__name__')
        except ImportError:
            pytest.skip("XLA model parallel not available")


@pytest.mark.integration
@pytest.mark.tpu
class TestXLAPerformance:
    """Tests for XLA performance features."""

    def test_xla_compilation(self):
        """Test that XLA compilation works."""
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        # Simple operation that should be compiled
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)

        z = torch.matmul(x, y)

        # Mark step to trigger compilation
        xm.mark_step()

        assert z.device.type == 'xla'

    def test_xla_mark_step(self):
        """Test XLA mark_step functionality."""
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        x = torch.randn(5, 5, device=device)
        y = x + 1

        # Mark step should not raise
        xm.mark_step()

        assert y.device.type == 'xla'


@pytest.mark.integration
@pytest.mark.tpu
class TestXLADataTypes:
    """Tests for XLA data type handling."""

    def test_xla_float32(self):
        """Test float32 on XLA."""
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        x = torch.randn(5, 5, dtype=torch.float32, device=device)

        assert x.dtype == torch.float32
        assert x.device.type == 'xla'

    def test_xla_bfloat16(self):
        """Test bfloat16 on XLA."""
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        try:
            x = torch.randn(5, 5, dtype=torch.bfloat16, device=device)
            assert x.dtype == torch.bfloat16
            assert x.device.type == 'xla'
        except RuntimeError:
            pytest.skip("bfloat16 not supported on this XLA device")


@pytest.mark.integration
@pytest.mark.tpu
class TestXLAMemory:
    """Tests for XLA memory management."""

    def test_xla_memory_allocation(self):
        """Test memory allocation on XLA device."""
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        # Allocate some tensors
        tensors = []
        for _ in range(10):
            tensors.append(torch.randn(100, 100, device=device))

        xm.mark_step()

        # All tensors should be on XLA device
        for t in tensors:
            assert t.device.type == 'xla'

    def test_xla_tensor_cleanup(self):
        """Test that XLA tensors can be cleaned up."""
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        x = torch.randn(100, 100, device=device)
        xm.mark_step()

        # Delete tensor
        del x

        # Should not raise
        xm.mark_step()


@pytest.mark.integration
@pytest.mark.tpu
class TestXLASpecificOps:
    """Tests for XLA-specific operations."""

    def test_xla_reduce_sum(self):
        """Test reduce operations on XLA."""
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        x = torch.randn(10, 10, device=device)
        result = x.sum()

        xm.mark_step()

        assert result.device.type == 'xla'

    def test_xla_matmul(self):
        """Test matrix multiplication on XLA."""
        import torch
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        x = torch.randn(10, 20, device=device)
        y = torch.randn(20, 15, device=device)

        result = torch.matmul(x, y)

        xm.mark_step()

        assert result.shape == (10, 15)
        assert result.device.type == 'xla'

    def test_xla_softmax(self):
        """Test softmax on XLA."""
        import torch
        import torch.nn.functional as F
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()

        x = torch.randn(10, 100, device=device)
        result = F.softmax(x, dim=-1)

        xm.mark_step()

        assert result.shape == x.shape
        assert result.device.type == 'xla'


# Note: These tests will be skipped if torch_xla is not available
# To run XLA tests, ensure torch_xla is installed and PJRT_DEVICE is set appropriately
