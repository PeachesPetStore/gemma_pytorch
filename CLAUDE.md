# CLAUDE.md - AI Assistant Guide for Gemma PyTorch

## Project Overview

This is the **official PyTorch implementation of Google's Gemma models** - a family of lightweight, state-of-the-art open models built from research and technology used to create Google Gemini models. The project supports both text-only and multimodal decoder-only large language models with open weights, pre-trained variants, and instruction-tuned variants.

**Key Facts:**
- **License:** Apache 2.0
- **Python Requirements:** >= 3.11
- **Core Dependencies:** PyTorch 2.6.0, NumPy 2.2.3, SentencePiece 0.2.0, Pillow 11.1.0, absl-py 2.1.0
- **Primary Use Case:** Inference-only implementation (not training)
- **Device Support:** CPU, GPU (CUDA), TPU (via PyTorch/XLA)

## Repository Structure

```
gemma_pytorch/
├── gemma/                          # Main package
│   ├── __init__.py                # Package initialization (empty)
│   ├── config.py                  # Model configurations for all variants
│   ├── model.py                   # Main Gemma model (text-only, Gemma 1 & 2)
│   ├── gemma3_model.py            # Gemma 3 multimodal implementation
│   ├── gemma3_preprocessor.py     # Preprocessor for Gemma 3 multimodal inputs
│   ├── model_xla.py               # XLA-optimized model for TPU
│   ├── xla_model_parallel.py      # Model parallelism for XLA
│   ├── tokenizer.py               # SentencePiece tokenizer wrapper
│   └── siglip_vision/             # Vision encoder for multimodal models
│       ├── __init__.py
│       ├── config.py              # Vision model configuration
│       ├── siglip_vision_model.py # SigLIP vision transformer
│       ├── preprocessor.py        # Image preprocessing
│       └── pan_and_scan.py        # Image tiling/scanning utilities
├── scripts/                        # Inference scripts
│   ├── run.py                     # Text-only inference (CPU/GPU)
│   ├── run_multimodal.py          # Multimodal inference (CPU/GPU)
│   ├── run_xla.py                 # XLA inference (CPU/GPU/TPU)
│   └── images/                    # Sample images for multimodal demos
├── tokenizer/                      # Tokenizer models
│   ├── tokenizer.model            # Gemma 1/2 SentencePiece model (256k vocab)
│   └── gemma3_cleaned_262144_v2.spiece.model  # Gemma 3 tokenizer (262k vocab)
├── docker/                         # Docker configurations
│   ├── Dockerfile                 # PyTorch CPU/GPU
│   ├── xla.Dockerfile             # PyTorch/XLA for CPU/TPU
│   └── xla_gpu.Dockerfile         # PyTorch/XLA for GPU
├── setup.py                        # Package setup
├── requirements.txt                # Python dependencies
├── README.md                       # User-facing documentation
├── CONTRIBUTING.md                 # Contribution guidelines
└── .gitignore                      # Git ignore rules
```

## Model Architectures

### Gemma Generations

The repository supports three generations of Gemma models:

1. **Gemma 1** (Architecture.GEMMA_1)
   - Variants: 2b, 7b
   - Standard transformer architecture
   - Global attention only

2. **Gemma 2** (Architecture.GEMMA_2)
   - Variants: 2b-v2, 9b, 27b
   - Alternating local sliding window and global attention
   - Pre/post feedforward normalization
   - Logit softcapping (final and attention)
   - Head dimension: 256 (27b uses 128)

3. **Gemma 3** (Architecture.GEMMA_3)
   - Variants: 1b (text-only), 4b, 12b, 27b_v3 (multimodal)
   - Pattern: 5 local sliding + 1 global attention (repeating)
   - QK normalization in attention blocks
   - Different RoPE wavelengths for local vs global layers
   - SigLIP vision encoder for multimodal variants
   - Extended vocabulary: 262,144 tokens

### Model Configurations

All configurations are defined in `gemma/config.py`:

```python
# Example: Getting a model config
from gemma import config
model_config = config.get_model_config('4b')  # Returns GemmaConfig for 4b variant
```

**Key Configuration Parameters:**
- `architecture`: GEMMA_1, GEMMA_2, or GEMMA_3
- `vocab_size`: 256000 (Gemma 1/2) or 262144 (Gemma 3)
- `num_hidden_layers`: Transformer layers count
- `num_attention_heads`: Number of attention heads
- `num_key_value_heads`: KV heads for grouped-query attention
- `hidden_size`: Hidden dimension size
- `intermediate_size`: FFN intermediate dimension
- `head_dim`: Dimension per attention head
- `attn_types`: List/tuple of AttentionType (LOCAL_SLIDING or GLOBAL)
- `sliding_window_size`: Window size for local attention
- `final_logit_softcapping`: Softcap value for output logits
- `attn_logit_softcapping`: Softcap value for attention logits
- `use_qk_norm`: Whether to normalize queries and keys
- `vision_config`: Vision encoder config (multimodal models only)

## File-Specific Details

### gemma/model.py

**Primary Classes:**
- `Sampler`: Handles sampling strategies (top-k, top-p, temperature)
- `GemmaAttention`: Self-attention with RoPE, supports local sliding and global
- `GemmaMLP`: Feedforward network with gating (GELU)
- `GemmaDecoderLayer`: Single transformer layer
- `GemmaModel`: Core transformer model
- `GemmaForCausalLM`: Main inference class with `.generate()` method

**Key Functions:**
- `precompute_freqs_cis()`: Generates RoPE frequency embeddings
- `apply_rotary_emb()`: Applies rotary positional embeddings
- `build_attn_mask()`: Creates causal + sliding window masks

**Device Handling:**
- Supports CPU and CUDA devices
- Use `model.to(device)` for device placement
- KV cache is device-aware

### gemma/gemma3_model.py

**Extends model.py for multimodal:**
- `Gemma3ForMultimodalLM`: Main multimodal class
- Integrates SigLIP vision encoder
- Handles interleaved text and image inputs
- Uses `gemma3_preprocessor.py` for input preparation

**Input Format:**
```python
# Text only
inputs = [["<start_of_turn>user What is AI?<end_of_turn>\n<start_of_turn>model"]]

# With images (PIL Image objects)
inputs = [[
    "<start_of_turn>user\n",
    image_object,  # PIL.Image
    "Describe this image.<end_of_turn>\n<start_of_turn>model"
]]
```

### gemma/config.py

**Configuration Functions:**
- `get_config_for_2b()`, `get_config_for_7b()`: Gemma 1 configs
- `get_config_for_2b_v2()`, `get_config_for_9b()`, `get_config_for_27b()`: Gemma 2
- `get_config_for_1b()`, `get_config_for_4b()`, `get_config_for_12b()`, `get_config_for_27b_v3()`: Gemma 3
- `get_model_config(variant, dtype)`: Universal config getter

**Valid Variants:**
`'1b'`, `'2b'`, `'2b-v2'`, `'4b'`, `'7b'`, `'9b'`, `'12b'`, `'27b'`, `'27b_v3'`

### scripts/run.py

**Text-Only Inference Script**

Valid variants: `'2b'`, `'2b-v2'`, `'7b'`, `'9b'`, `'27b'`, `'1b'`

**Usage:**
```bash
python scripts/run.py \
  --ckpt=/path/to/checkpoint \
  --variant=7b \
  --device=cuda \
  --prompt="What is machine learning?" \
  --output_len=100 \
  --quant=False
```

### scripts/run_multimodal.py

**Multimodal Inference Script**

Valid variants: `'4b'`, `'12b'`, `'27b_v3'` (Gemma 3 multimodal only)

**Key Features:**
- Loads and displays images using PIL
- Supports text-only, single-image, and multi-image inputs
- Images are interleaved with text in prompts

### scripts/run_xla.py

**XLA-Optimized Inference**

Supports CPU, TPU, and GPU via PyTorch/XLA. Uses `model_xla.py` implementation.

**Environment Variables:**
- `PJRT_DEVICE=CPU|TPU|CUDA`: Select device
- `USE_CUDA=1`: Enable CUDA for XLA (GPU only)

## Development Workflows

### Setting Up Development Environment

```bash
# Clone repository
git clone <repo-url>
cd gemma_pytorch

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Running Inference Locally (No Docker)

```bash
# CPU inference
python scripts/run.py \
  --ckpt=/path/to/checkpoint \
  --variant=2b \
  --device=cpu \
  --output_len=50

# GPU inference
python scripts/run.py \
  --ckpt=/path/to/checkpoint \
  --variant=7b \
  --device=cuda \
  --output_len=100
```

### Docker Workflows

**Build PyTorch Docker Image:**
```bash
DOCKER_URI=gemma:${USER}
docker build -f docker/Dockerfile ./ -t ${DOCKER_URI}
```

**Run with GPU:**
```bash
docker run -t --rm --gpus all \
  -v ${CKPT_PATH}:/tmp/ckpt \
  ${DOCKER_URI} \
  python scripts/run_multimodal.py \
  --device=cuda \
  --ckpt=/tmp/ckpt \
  --variant=4b
```

**Build and Run XLA (TPU/CPU):**
```bash
# Build
docker build -f docker/xla.Dockerfile ./ -t gemma_xla:${USER}

# Run on TPU
docker run -t --rm --shm-size 4gb \
  -e PJRT_DEVICE=TPU \
  -v ${CKPT_PATH}:/tmp/ckpt \
  gemma_xla:${USER} \
  python scripts/run_xla.py \
  --ckpt=/tmp/ckpt \
  --variant=4b
```

### Testing Changes

**Manual Testing:**
1. Modify code in `gemma/` directory
2. Run inference script with small model and short output
3. Verify output quality and check for errors

**Example Quick Test:**
```bash
python scripts/run.py \
  --ckpt=/path/to/2b/checkpoint \
  --variant=2b \
  --device=cpu \
  --output_len=10 \
  --prompt="Hello"
```

## Code Conventions

### Python Style

- **Copyright Header:** All files start with Apache 2.0 license header
- **Docstrings:** Google-style docstrings for classes and complex functions
- **Type Hints:** Used throughout (Python 3.11+)
- **Imports:** Standard library → Third-party → Local imports
- **Line Length:** Generally follows PEP 8 (within reason)

### Naming Conventions

- **Classes:** PascalCase (`GemmaForCausalLM`, `SigLIPVisionModel`)
- **Functions/Methods:** snake_case (`precompute_freqs_cis`, `apply_rotary_emb`)
- **Constants:** UPPER_SNAKE_CASE (`_BEGIN_IMAGE_TOKEN = 255999`)
- **Private/Internal:** Leading underscore (`_assert_file_exists`, `_set_default_tensor_type`)
- **Config Functions:** `get_config_for_<variant>()` pattern

### PyTorch Patterns

**Model Definition:**
```python
class MyLayer(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        # Initialize layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        return output
```

**Device Handling:**
```python
# Move tensors to correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = tensor.to(device)

# Or match existing tensor device
new_tensor = torch.zeros_like(existing_tensor)  # Inherits device
```

**No Gradients (Inference Only):**
```python
@torch.no_grad()
def generate(self, ...):
    # Inference code
```

### Attention Implementation Notes

1. **RoPE (Rotary Position Embeddings):**
   - Precomputed with `precompute_freqs_cis()`
   - Different wavelengths for local vs global in Gemma 3
   - Applied in complex domain via `apply_rotary_emb()`

2. **Sliding Window Attention:**
   - Uses `build_attn_mask()` with `sliding_window_size`
   - Mask shape: `[batch, seq_len, seq_len]`
   - Combines causal mask + sliding window constraint

3. **Grouped-Query Attention:**
   - `num_key_value_heads` < `num_attention_heads`
   - KV heads are repeated to match query heads

4. **Logit Softcapping:**
   - `logits = tanh(logits / cap) * cap`
   - Applied to attention logits and final output logits

### Tokenizer Notes

**Reserved Tokens:**
- 99 unused tokens: `<unused0>` to `<unused98>` (IDs 7-104)
- Reserved for efficient fine-tuning
- Gemma 3 uses extended vocab (262,144 vs 256,000)

**Special Tokens:**
- `bos_id`: Beginning of sequence
- `eos_id`: End of sequence
- `pad_id`: Padding token
- `boi_id`: Begin image (255999)
- `eoi_id`: End image (256000)

## Device-Specific Considerations

### CPU Inference

- Use `dtype='float32'` for better CPU compatibility
- Smaller models (1b, 2b) recommended for CPU
- No special setup required

### GPU (CUDA) Inference

- Use `dtype='bfloat16'` or `dtype='float16'` for efficiency
- Ensure sufficient VRAM for model size
- Use `--device=cuda` flag
- Docker: Add `--gpus all` flag

### TPU Inference

- Requires PyTorch/XLA
- Use `xla.Dockerfile` or `xla_gpu.Dockerfile`
- Set `PJRT_DEVICE=TPU`
- Models are defined in `model_xla.py` with XLA-specific optimizations

### Quantization

- INT8 quantization supported via `--quant` flag
- Reduces memory footprint
- Slight performance degradation
- Useful for larger models on limited hardware

## Common Patterns for AI Assistants

### 1. Adding a New Model Variant

If a new Gemma variant is released:

1. **Update `gemma/config.py`:**
   ```python
   def get_config_for_<variant>(dtype: str) -> GemmaConfig:
       return GemmaConfig(
           architecture=Architecture.GEMMA_X,
           # ... variant-specific parameters
       )
   ```

2. **Update `get_model_config()` function:**
   ```python
   elif variant == '<new_variant>':
       return get_config_for_<new_variant>(dtype)
   ```

3. **Update validator lists in run scripts:**
   ```python
   _VALID_MODEL_VARIANTS = ['2b', '7b', ..., '<new_variant>']
   ```

### 2. Debugging Attention Masks

Attention masks are critical. Check:
- Shape: `[batch_size, num_heads, seq_len, seq_len]`
- Values: `0.0` (attend) or `-2.3819763e38` (mask)
- Causal property: Upper triangle is masked
- Sliding window: Only recent tokens visible for local attention

**Debug Pattern:**
```python
# In GemmaAttention.forward()
print(f"Attention mask shape: {attn_mask.shape}")
print(f"Mask min: {attn_mask.min()}, max: {attn_mask.max()}")
```

### 3. Adding Device Support

When adding support for a new device:
1. Update device validators in scripts
2. Ensure tensor creation uses device parameter
3. Test KV cache device placement
4. Verify mask tensors are on correct device

### 4. Modifying Generation Logic

The `.generate()` method in `GemmaForCausalLM`:
- Uses KV caching for efficiency
- Handles variable-length inputs via padding
- Supports batch generation

**Modification Points:**
- Sampling strategy: `Sampler` class
- Stopping criteria: Check for `eos_token` in generation loop
- Decoding: `tokenizer.decode()`

### 5. Working with Multimodal Inputs

**Image Preprocessing:**
- Images are resized and normalized in `siglip_vision/preprocessor.py`
- Pan-and-scan strategy for high-resolution images
- Vision tokens are inserted at image placeholder positions

**Text-Image Interleaving:**
- Images can appear anywhere in the prompt
- Use `gemma3_preprocessor.py` to handle mixed inputs
- Vision encoder processes images → embeddings → merged with text tokens

## Git and Contribution Workflow

### Branch Strategy

Recent commits show:
- Main development on `main` branch
- Feature branches merged via PRs (e.g., `gemma3`, `fix-xla-downcast`)
- Clear commit messages (e.g., "Add Gemma3", "Supporting Gemma 2 2b")

### Commit Message Style

Based on git history:
- Concise, descriptive summaries
- Examples:
  - "Add Gemma3"
  - "Supporting Gemma 2 2b"
  - "Remove unused imports"
  - "Using head_dim instead of override scalar value for the 9b model"

### Contributing

Per `CONTRIBUTING.md`:
1. **Sign CLA:** Required for all contributions
2. **Code Review:** All PRs require review
3. **Community Guidelines:** Follow Google's Open Source Community Guidelines

### Recent Development Focus

Based on recent commits:
- Gemma 3 support added
- GPU compatibility fixes (e.g., 1B model on GPUs)
- XLA optimization improvements
- Parameter corrections (head_dim usage)

## Important Notes for AI Assistants

### 1. Inference-Only Codebase

This repository is **inference-only**. Do not attempt to:
- Add training loops
- Implement gradient-based optimization
- Add backward pass logic

### 2. Model Weights Not Included

Checkpoints must be downloaded separately from:
- Kaggle: https://www.kaggle.com/models/google/gemma-3/
- Hugging Face: https://huggingface.co/models?other=gemma_torch

### 3. Architecture-Specific Code Paths

Different architectures use different implementations:
- **Gemma 1/2 (text-only):** `model.py` → `GemmaForCausalLM`
- **Gemma 3 (multimodal):** `gemma3_model.py` → `Gemma3ForMultimodalLM`
- **XLA (all):** `model_xla.py` → XLA-optimized versions

### 4. Variant Validation

Always validate variant strings against supported lists:
- Text-only: 1b, 2b, 2b-v2, 7b, 9b, 27b
- Multimodal: 4b, 12b, 27b_v3

### 5. Device Placement Consistency

Critical pattern:
```python
# All tensors in a computation must be on the same device
device = torch.device('cuda')
model = model.to(device)
input_ids = input_ids.to(device)
mask = mask.to(device)
# etc.
```

### 6. Tokenizer Path Configuration

Tokenizer paths are relative to checkout directory:
- `tokenizer/tokenizer.model` (Gemma 1/2)
- `tokenizer/gemma3_cleaned_262144_v2.spiece.model` (Gemma 3)

When loading models, ensure tokenizer path is accessible.

### 7. Memory Management

For large models:
- Use quantization (`--quant`)
- Use smaller batch sizes
- Use bfloat16/float16 on GPU
- Consider gradient checkpointing (if adding training)

### 8. Attention Type Patterns

Gemma 2/3 use alternating attention:
```python
# Gemma 2: Simple alternation
attn_types = [AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * (num_layers // 2)

# Gemma 3: 5 local + 1 global pattern
attn_types = (LOCAL_SLIDING,) * 5 + (GLOBAL,)
```

### 9. Docker Preferences

For reproducible environments, prefer Docker workflows:
- Development: Use local Python environment
- Testing/Deployment: Use Docker containers
- TPU: XLA Docker is mandatory

### 10. File Modification Guidelines

**Safe to Modify:**
- `scripts/*.py` - Inference scripts
- `gemma/config.py` - Add new configs (append only)

**Modify with Caution:**
- `gemma/model.py`, `gemma/gemma3_model.py` - Core model logic
- `gemma/tokenizer.py` - Tokenization logic

**Do Not Modify:**
- Tokenizer model files (`.model`, `.spiece.model`)
- `setup.py` (without testing package installation)

## Quick Reference

### File Location Guide

| Task | File/Directory |
|------|----------------|
| Add model config | `gemma/config.py` |
| Modify text model | `gemma/model.py` |
| Modify multimodal model | `gemma/gemma3_model.py` |
| Change inference script | `scripts/run*.py` |
| Update vision encoder | `gemma/siglip_vision/` |
| Modify tokenizer | `gemma/tokenizer.py` |
| Add Docker support | `docker/*.Dockerfile` |

### Command Patterns

```bash
# Quick CPU test (text-only)
python scripts/run.py --ckpt=<path> --variant=2b --device=cpu --output_len=10

# GPU inference (multimodal)
python scripts/run_multimodal.py --ckpt=<path> --variant=4b --device=cuda --output_len=50

# Docker build and run
docker build -f docker/Dockerfile ./ -t gemma:test
docker run -t --rm --gpus all -v <ckpt>:/tmp/ckpt gemma:test python scripts/run.py --ckpt=/tmp/ckpt --variant=7b --device=cuda

# XLA on TPU
docker run -t --rm --shm-size 4gb -e PJRT_DEVICE=TPU -v <ckpt>:/tmp/ckpt gemma_xla:latest python scripts/run_xla.py --ckpt=/tmp/ckpt --variant=4b
```

## Troubleshooting

### Common Issues

1. **Device Mismatch Errors:**
   - Ensure all tensors are on the same device
   - Check model, inputs, masks, and KV cache

2. **Out of Memory:**
   - Use smaller model variant
   - Enable quantization (`--quant`)
   - Reduce `output_len`
   - Use float16/bfloat16 instead of float32

3. **Tokenizer Not Found:**
   - Verify tokenizer path in config
   - Ensure `tokenizer/` directory exists
   - Check relative path from script execution directory

4. **Invalid Variant:**
   - Check variant string against supported list
   - Ensure text-only scripts don't use multimodal variants (and vice versa)

5. **XLA Compilation Slow:**
   - First run triggers XLA compilation (expected)
   - Subsequent runs should be faster
   - Use smaller models for development

## Summary

This repository provides a clean, well-structured implementation of Gemma models optimized for inference across multiple device types. The codebase is organized around three architecture generations (Gemma 1, 2, 3), with clear separation between text-only and multimodal implementations. Key patterns include configuration-driven model instantiation, device-agnostic tensor operations, and specialized attention mechanisms (sliding window, grouped-query, RoPE).

When modifying or extending this codebase:
1. Follow existing patterns (especially in `config.py`)
2. Maintain device placement consistency
3. Test across CPU and GPU
4. Preserve inference-only nature (no training code)
5. Update validators when adding new variants
6. Use Docker for reproducible environments

The codebase is designed for clarity and correctness over performance optimizations, making it an excellent reference implementation for understanding Gemma architectures.
