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
"""Pytest configuration and fixtures."""

import pytest
import torch
from tests.fixtures.configs import get_tiny_config, get_tiny_gemma2_config, get_tiny_gemma3_config


@pytest.fixture
def device():
    """Return the device to use for testing."""
    return torch.device('cpu')


@pytest.fixture
def tiny_config():
    """Return a tiny Gemma config for testing."""
    return get_tiny_config()


@pytest.fixture
def tiny_gemma2_config():
    """Return a tiny Gemma2 config for testing."""
    return get_tiny_gemma2_config()


@pytest.fixture
def tiny_gemma3_config():
    """Return a tiny Gemma3 config for testing."""
    return get_tiny_gemma3_config()


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "numerical: mark test as a numerical correctness test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
