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
"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest
import torch
import numpy as np

from gemma import config as gemma_config


@pytest.fixture(scope="session")
def test_data_dir():
    """Returns the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir():
    """Creates a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def random_seed():
    """Returns a fixed random seed for reproducible tests."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seeds(random_seed):
    """Sets random seeds for reproducibility across all tests."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


@pytest.fixture
def device():
    """Returns the appropriate torch device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Returns CPU device."""
    return torch.device("cpu")


# Config fixtures for different model variants

@pytest.fixture
def config_2b():
    """Returns a Gemma 2B config."""
    return gemma_config.get_config_for_2b()


@pytest.fixture
def config_7b():
    """Returns a Gemma 7B config."""
    return gemma_config.get_config_for_7b()


@pytest.fixture
def config_2b_v2():
    """Returns a Gemma 2B v2 config."""
    return gemma_config.get_config_for_2b_v2()


@pytest.fixture
def config_9b():
    """Returns a Gemma 9B config."""
    return gemma_config.get_config_for_9b()


@pytest.fixture
def config_1b():
    """Returns a Gemma 1B (Gemma 3) config."""
    return gemma_config.get_config_for_1b('float32')


@pytest.fixture
def config_4b():
    """Returns a Gemma 4B (Gemma 3) config."""
    return gemma_config.get_config_for_4b('float32')


@pytest.fixture(params=['2b', '7b'])
def config_gemma1(request):
    """Parametrized fixture for Gemma 1 configs."""
    variant = request.param
    return gemma_config.get_model_config(variant, dtype='float32')


@pytest.fixture(params=['2b-v2', '9b'])
def config_gemma2(request):
    """Parametrized fixture for Gemma 2 configs."""
    variant = request.param
    return gemma_config.get_model_config(variant, dtype='float32')


@pytest.fixture(params=['1b', '4b'])
def config_gemma3(request):
    """Parametrized fixture for Gemma 3 configs."""
    variant = request.param
    return gemma_config.get_model_config(variant, dtype='float32')


# Test data fixtures

@pytest.fixture
def sample_text():
    """Returns sample text for testing."""
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture
def sample_prompts():
    """Returns a list of sample prompts for batch testing."""
    return [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]


@pytest.fixture
def mock_tokenizer_model_path(temp_dir):
    """Creates a mock tokenizer model file path."""
    # Note: This is just a path, actual tokenizer tests will need a real model
    model_path = temp_dir / "tokenizer.model"
    return str(model_path)


@pytest.fixture
def sample_token_ids():
    """Returns sample token IDs for testing."""
    return torch.tensor([[1, 2, 3, 4, 5], [1, 6, 7, 8, 9]])


@pytest.fixture
def sample_batch_size():
    """Returns a small batch size for testing."""
    return 2


@pytest.fixture
def sample_seq_len():
    """Returns a small sequence length for testing."""
    return 8


@pytest.fixture
def sample_hidden_size():
    """Returns a small hidden size for testing."""
    return 256


@pytest.fixture
def sample_vocab_size():
    """Returns a small vocab size for testing."""
    return 1000


# Marker-based test configuration

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "tpu: mark test as requiring TPU/XLA"
    )
    config.addinivalue_line(
        "markers", "multimodal: mark test as testing multimodal features"
    )
    config.addinivalue_line(
        "markers", "quantization: mark test as testing quantization"
    )
    config.addinivalue_line(
        "markers", "requires_checkpoint: mark test as requiring model checkpoint"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Auto-mark tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Auto-mark tests in integration/ directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark slow tests (tests taking > 1 second)
        # This can be refined based on actual test timing
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture
def skip_if_no_checkpoint():
    """Skip test if checkpoint is not available."""
    # This can be customized based on where checkpoints are stored
    checkpoint_env = os.getenv("GEMMA_CHECKPOINT_PATH")
    if not checkpoint_env or not os.path.exists(checkpoint_env):
        pytest.skip("Model checkpoint not available")
