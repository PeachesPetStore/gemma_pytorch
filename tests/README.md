# Gemma PyTorch Test Suite

Comprehensive test suite for the Gemma PyTorch implementation.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
│   ├── test_config.py           # Configuration tests
│   ├── test_normalization.py    # RMSNorm tests
│   ├── test_embeddings.py       # Linear/Embedding layer tests
│   ├── test_rope.py             # Rotary position embedding tests
│   ├── test_sampler.py          # Sampling strategy tests
│   └── test_attention.py        # Attention mechanism tests
├── integration/       # Integration tests
│   └── test_model_integration.py  # Full model tests
├── numerical/         # Numerical correctness tests
├── fixtures/          # Test fixtures and utilities
│   ├── configs.py               # Test configurations
│   └── sample_data.py           # Sample data generators
└── conftest.py        # Pytest configuration
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run unit tests only
```bash
pytest tests/unit -v
```

### Run integration tests only
```bash
pytest tests/integration -v
```

### Run with coverage
```bash
pytest --cov=gemma --cov-report=html --cov-report=term-missing
```

### Run in parallel
```bash
pytest -n auto
```

### Run specific test file
```bash
pytest tests/unit/test_config.py -v
```

### Run specific test
```bash
pytest tests/unit/test_config.py::TestGemmaConfig::test_get_dtype_float16 -v
```

## Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.numerical` - Numerical correctness tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.gpu` - Tests requiring GPU

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

### Skip slow tests
```bash
pytest -m "not slow"
```

## Coverage Goals

Current coverage targets:
- **Overall**: 70%+ coverage
- **Core model components**: 80%+ coverage
- **Critical paths**: 90%+ coverage

View coverage report:
```bash
pytest --cov=gemma --cov-report=html
open htmlcov/index.html
```

## Test Coverage by Module

### Covered Components

✅ **Configuration** (`gemma/config.py`)
- All model variants (1B, 2B, 7B, 9B, 12B, 27B)
- dtype conversion
- Architecture enums

✅ **Normalization** (`gemma/model.py:RMSNorm`)
- Forward pass
- Numerical stability
- dtype preservation
- Gradient flow

✅ **Custom Layers** (`gemma/model.py:Linear, Embedding`)
- Quantized and non-quantized modes
- Weight scaling
- Shape preservation

✅ **RoPE** (`gemma/model.py:precompute_freqs_cis, apply_rotary_emb`)
- Frequency computation
- Rotary embedding application
- Position invariance
- Scaling factors

✅ **Sampler** (`gemma/model.py:Sampler`)
- Greedy sampling
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) filtering
- Logit softcapping

✅ **Attention** (`gemma/model.py:GemmaAttention`)
- Self-attention
- Grouped query attention (GQA)
- KV cache management
- Sliding window attention

✅ **MLP** (`gemma/model.py:GemmaMLP`)
- GELU activation
- Quantization support

✅ **Model Integration**
- Decoder layers (Gemma1, Gemma2)
- Full model forward pass
- Batch processing

### Areas for Future Enhancement

- Tokenizer tests (requires mock tokenizer file)
- Gemma3 multimodal tests (vision components)
- Numerical correctness tests (golden outputs)
- GPU-specific tests
- Performance benchmarks
- Fuzz testing for edge cases

## Writing New Tests

### Test Template

```python
import pytest
import torch
from gemma.model import YourComponent

@pytest.mark.unit
class TestYourComponent:
    """Test YourComponent."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        component = YourComponent(...)
        output = component(...)
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
```

### Best Practices

1. **Use fixtures** - Define reusable test data in `conftest.py` or `fixtures/`
2. **Test edge cases** - Zero inputs, boundary values, extreme values
3. **Test shapes** - Verify output shapes match expectations
4. **Test dtypes** - Ensure dtype preservation
5. **Test numerical stability** - Check for NaN/Inf values
6. **Test independence** - Verify batch/sequence independence where applicable
7. **Use descriptive names** - Test names should describe what they test
8. **Add docstrings** - Explain what each test validates

## Continuous Integration

Tests run automatically on:
- Push to main/master branch
- Pull requests
- Push to claude/** branches

See `.github/workflows/tests.yml` for CI configuration.

## Troubleshooting

### Tests failing locally but passing in CI
- Check Python version matches CI
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Clear pytest cache: `pytest --cache-clear`

### Coverage not updating
- Clear coverage data: `coverage erase`
- Ensure using `--cov` flag: `pytest --cov=gemma`

### Slow tests
- Run in parallel: `pytest -n auto`
- Skip slow tests: `pytest -m "not slow"`
