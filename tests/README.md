# Gemma PyTorch Test Suite

This directory contains the test suite for the Gemma PyTorch implementation.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── unit/                    # Unit tests for individual components
│   ├── test_config.py       # Configuration tests
│   └── test_model_components.py  # Model component tests
├── integration/             # Integration tests
│   └── test_model_integration.py  # Model integration tests
└── fixtures/                # Test data and mock files
```

## Running Tests

### Prerequisites

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

**Unit tests only:**
```bash
pytest tests/unit -v
```

**Integration tests only:**
```bash
pytest tests/integration -v
```

**Exclude slow tests:**
```bash
pytest -m "not slow"
```

**Exclude GPU/TPU tests:**
```bash
pytest -m "not (gpu or tpu)"
```

**Run only tests that don't require checkpoints:**
```bash
pytest -m "not requires_checkpoint"
```

### Run Tests with Coverage

```bash
pytest --cov=gemma --cov-report=html --cov-report=term-missing
```

View the coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Tests in Parallel

```bash
pytest -n auto  # Uses all available CPU cores
pytest -n 4     # Uses 4 workers
```

## Test Markers

Tests can be marked with the following markers to categorize them:

- `@pytest.mark.unit` - Unit tests (auto-applied to tests in `unit/` directory)
- `@pytest.mark.integration` - Integration tests (auto-applied to tests in `integration/` directory)
- `@pytest.mark.slow` - Tests that take a long time to run
- `@pytest.mark.gpu` - Tests requiring GPU/CUDA
- `@pytest.mark.tpu` - Tests requiring TPU/XLA
- `@pytest.mark.multimodal` - Tests for multimodal models
- `@pytest.mark.quantization` - Tests for quantized models
- `@pytest.mark.requires_checkpoint` - Tests requiring model checkpoint files

Example usage:

```python
import pytest

@pytest.mark.slow
@pytest.mark.gpu
def test_large_model_inference():
    # Test code here
    pass
```

## Shared Fixtures

The `conftest.py` file provides shared fixtures available to all tests:

### Device Fixtures
- `device` - Returns cuda if available, else cpu
- `cpu_device` - Returns CPU device
- `skip_if_no_gpu` - Skip test if GPU not available

### Config Fixtures
- `config_2b`, `config_7b` - Gemma 1 configs
- `config_2b_v2`, `config_9b` - Gemma 2 configs
- `config_1b`, `config_4b` - Gemma 3 configs
- `config_gemma1`, `config_gemma2`, `config_gemma3` - Parametrized configs

### Data Fixtures
- `sample_text` - Sample text string
- `sample_prompts` - List of sample prompts
- `sample_token_ids` - Sample token ID tensors
- `random_seed` - Fixed seed for reproducibility

### Utility Fixtures
- `temp_dir` - Temporary directory for test outputs
- `test_data_dir` - Path to test fixtures directory

## Writing New Tests

### Unit Test Example

```python
# tests/unit/test_my_component.py
import pytest
import torch
from gemma import model

class TestMyComponent:
    """Tests for MyComponent."""

    def test_forward_pass(self):
        """Test forward pass."""
        component = model.MyComponent()
        x = torch.randn(2, 8, 256)
        output = component(x)
        assert output.shape == x.shape

    @pytest.mark.parametrize('batch_size', [1, 2, 4])
    def test_different_batch_sizes(self, batch_size):
        """Test with different batch sizes."""
        component = model.MyComponent()
        x = torch.randn(batch_size, 8, 256)
        output = component(x)
        assert output.shape[0] == batch_size
```

### Integration Test Example

```python
# tests/integration/test_end_to_end.py
import pytest
from gemma import model, config

@pytest.mark.integration
@pytest.mark.slow
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_model_generation(self, config_2b, cpu_device):
        """Test complete generation pipeline."""
        # This would require a checkpoint, so we skip in this example
        pytest.skip("Requires checkpoint")
```

## CI/CD

Tests are automatically run on:
- Every push to `main`/`master`
- Every pull request

The CI pipeline runs:
1. Linting (flake8)
2. Type checking (mypy)
3. Unit tests
4. Integration tests (excluding GPU/TPU/checkpoint-required tests)
5. Code coverage report

See `.github/workflows/test.yml` for the complete CI configuration.

## Code Quality

### Run Linting

```bash
# Check for errors
flake8 gemma --count --select=E9,F63,F7,F82 --show-source --statistics

# Check all issues
flake8 gemma --count --max-complexity=10 --max-line-length=127 --statistics
```

### Run Type Checking

```bash
mypy gemma --ignore-missing-imports
```

### Format Code

```bash
# Check formatting
black --check gemma tests

# Apply formatting
black gemma tests

# Sort imports
isort gemma tests
```

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on other tests
2. **Use Fixtures**: Leverage shared fixtures in `conftest.py` for common setup
3. **Mark Tests Appropriately**: Use markers to categorize tests (slow, gpu, etc.)
4. **Test Edge Cases**: Include tests for boundary conditions and error cases
5. **Keep Tests Fast**: Mock expensive operations when possible
6. **Descriptive Names**: Use clear, descriptive test names that explain what is being tested
7. **One Assert Per Test**: Generally prefer one logical assertion per test
8. **Use Parametrize**: Use `@pytest.mark.parametrize` for testing multiple inputs

## Coverage Goals

Target coverage levels:
- **Unit tests**: > 80% code coverage
- **Critical paths**: 100% coverage for:
  - Configuration validation
  - Tokenization
  - Core model components (attention, MLP, normalization)
  - Weight loading

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Add appropriate markers
4. Update this README if adding new test categories
5. Aim for > 80% coverage of new code

## Troubleshooting

### Tests are slow
- Use `pytest -n auto` for parallel execution
- Mark slow tests with `@pytest.mark.slow` and exclude them during development
- Use smaller test fixtures when possible

### GPU tests failing
- Ensure CUDA is properly installed
- Check `torch.cuda.is_available()`
- Use `@pytest.mark.gpu` and `skip_if_no_gpu` fixture

### Import errors
- Ensure you've installed the package: `pip install -e .`
- Check that you're running from the project root directory

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/latest/goodpractices.html)
- [Testing PyTorch Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
