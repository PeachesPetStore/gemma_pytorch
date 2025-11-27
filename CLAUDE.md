# CLAUDE.md - AI Assistant Guide for gemma_pytorch

## Project Overview

**gemma_pytorch** is the official PyTorch implementation of Google's Gemma family of large language models. This repository provides inference implementations for text-only and multimodal models across three generations (Gemma 1, 2, and 3) with support for CPU, GPU, and TPU execution.

**Key Facts:**
- License: Apache 2.0
- Python Version: >=3.11
- Primary Dependencies: PyTorch 2.6.0, sentencepiece, numpy, pillow, absl-py
- Model Sizes: 1B, 2B, 4B, 7B, 9B, 12B, 27B parameters
- Model Types: Text-only (Gemma 1, 2) and Multimodal (Gemma 3)

## Repository Structure

```
gemma_pytorch/
├── gemma/                          # Core library module
│   ├── config.py                   # Model configurations (341 lines)
│   ├── model.py                    # PyTorch implementation (756 lines)
│   ├── model_xla.py                # XLA/TPU optimized (725 lines)
│   ├── gemma3_model.py             # Gemma 3 multimodal (334 lines)
│   ├── gemma3_preprocessor.py      # Input preprocessing (207 lines)
│   ├── tokenizer.py                # SentencePiece wrapper (54 lines)
│   ├── xla_model_parallel.py       # Model parallelization (727 lines)
│   └── siglip_vision/              # Vision encoder for multimodal
│       ├── siglip_vision_model.py  # SigLIP vision transformer
│       ├── config.py               # Vision model config
│       ├── preprocessor.py         # Image preprocessing
│       └── pan_and_scan.py         # Image cropping strategy
├── scripts/                        # Inference entry points
│   ├── run.py                      # Text-only inference (CPU/GPU)
│   ├── run_multimodal.py           # Multimodal inference (CPU/GPU)
│   ├── run_xla.py                  # XLA inference (TPU/GPU)
│   └── images/                     # Sample test images
├── tokenizer/                      # Tokenizer model files
│   ├── tokenizer.model             # Gemma 1/2 tokenizer (256K vocab)
│   └── gemma3_cleaned_262144_v2.spiece.model  # Gemma 3 tokenizer
├── docker/                         # Containerization
│   ├── Dockerfile                  # Standard PyTorch + CUDA
│   ├── xla.Dockerfile              # PyTorch/XLA for TPU
│   └── xla_gpu.Dockerfile          # PyTorch/XLA for GPU
├── setup.py                        # Package installation
├── requirements.txt                # Dependencies
├── README.md                       # User documentation
├── CONTRIBUTING.md                 # Contribution guidelines
└── .gitignore                      # Git ignore patterns
```

## Model Architecture Overview

### Three Model Generations

#### Gemma 1 (Original, Text-only)
- **Variants:** 2B, 7B
- **Architecture:** Standard decoder-only transformer
- **Attention:** Global attention only
- **Context Length:** 8,192 tokens
- **Vocab:** 256,000 tokens
- **Normalization:** Pre-norm only (RMSNorm)
- **Key Files:** `model.py`, `model_xla.py`

#### Gemma 2 (Enhanced, Text-only)
- **Variants:** 2B-v2, 9B, 27B
- **Architecture:** Improved decoder with sliding window attention
- **Attention:** Alternating LOCAL_SLIDING (4096 window) + GLOBAL
- **Context Length:** 8,192 tokens
- **Vocab:** 256,000 tokens
- **Normalization:** Pre+post FFN norm
- **Special Features:** Logit softcapping (attn=50.0, final=30.0)
- **Key Files:** `model.py`, `model_xla.py`

#### Gemma 3 (Multimodal)
- **Variants:** 1B (text), 4B, 12B, 27B_v3 (multimodal)
- **Architecture:** Decoder + SigLIP vision encoder
- **Attention:** 5:1 pattern of LOCAL_SLIDING:GLOBAL
- **Context Length:** 32,768 (1B) or 131,072 (12B, 27B_v3)
- **Vocab:** 262,144 tokens (new tokenizer)
- **Special Features:** QK normalization, different RoPE wavelengths, RoPE scaling (8x)
- **Vision:** 896x896 images → 256 tokens (1152-dim)
- **Key Files:** `gemma3_model.py`, `gemma3_preprocessor.py`, `siglip_vision/`

### Key Architectural Components

#### Model Implementations (model.py)
- **`Sampler`**: Top-p/top-k sampling with temperature and logit softcapping
- **`Embedding` & `Linear`**: Custom layers with optional INT8 quantization
- **`RMSNorm`**: Root Mean Square normalization with optional unit offset
- **`GemmaMLP`**: Gated FFN (gate * up → down) with GELU activation
- **`GemmaAttention`**: Multi-query/Grouped-query attention with RoPE, sliding window support
- **`GemmaDecoderLayer`**: Gemma 1 decoder (pre-norm only)
- **`Gemma2DecoderLayer`**: Gemma 2/3 decoder (pre+post norm, logit softcapping)
- **`GemmaModel`**: Stacks decoder layers
- **`GemmaForCausalLM`**: Complete model with embeddings, generation, and weight loading

#### Multimodal Components (gemma3_model.py)
- **`Gemma3ForMultimodalLM`**: Extends text model with vision
- **`SiglipVisionModel`**: Vision transformer encoder (896x896 → 256 tokens)
- **`mm_soft_embedding_norm`**: RMSNorm for vision embeddings
- **`mm_input_projection`**: Projects vision to text embedding space
- **Pan-and-scan**: Smart cropping for non-square images (2-4 crops)

#### XLA Parallelization (xla_model_parallel.py)
- **`ColumnParallelLinear` / `RowParallelLinear`**: Sharded linear layers
- **`ParallelEmbedding`**: Sharded embeddings
- **Custom autograd functions**: For distributed collectives
- **INT8 quantization utilities**: Per-channel quantization

## Configuration System (config.py)

All model hyperparameters are defined in `gemma/config.py` using the `GemmaConfig` dataclass.

### Key Configuration Fields
```python
@dataclasses.dataclass
class GemmaConfig:
    architecture: Architecture          # GEMMA_1, GEMMA_2, or GEMMA_3
    vocab_size: int                     # 256000 or 262144
    max_position_embeddings: int        # 8192, 32768, or 131072
    num_hidden_layers: int              # Number of transformer blocks
    num_attention_heads: int            # Number of attention heads
    num_key_value_heads: int            # For multi-query/grouped-query
    hidden_size: int                    # Model dimension
    intermediate_size: int              # FFN dimension
    head_dim: int                       # 256 or 128
    attn_types: Sequence[AttentionType] # LOCAL_SLIDING or GLOBAL
    sliding_window_size: int            # For local attention (512, 1024, or 4096)
    final_logit_softcapping: float      # Gemma 2/3: 30.0
    attn_logit_softcapping: float       # Gemma 2/3: 50.0
    use_qk_norm: bool                   # Gemma 3 only
    vision_config: SiglipVisionModelConfig  # Gemma 3 multimodal only
    rope_wave_length: dict              # Gemma 3: different per attention type
    rope_scaling_factor: int            # Gemma 3: 8
```

### Getting Configurations
```python
from gemma import config

# Method 1: Use helper function
model_config = config.get_model_config('4b', dtype='bfloat16')

# Method 2: Use specific factory
model_config = config.get_config_for_4b('bfloat16')
```

### Valid Variants
- Gemma 1: `'2b'`, `'7b'`
- Gemma 2: `'2b-v2'`, `'9b'`, `'27b'`
- Gemma 3: `'1b'`, `'4b'`, `'12b'`, `'27b_v3'`

## Development Workflows

### Installation

```bash
# From source
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Running Inference

#### Text-only (CPU/GPU)
```bash
# CPU
python scripts/run.py \
    --ckpt=/path/to/checkpoint \
    --variant=2b \
    --device=cpu \
    --prompt="Explain quantum computing"

# GPU with quantization
python scripts/run.py \
    --ckpt=/path/to/checkpoint \
    --variant=7b \
    --device=cuda \
    --quant
```

#### Multimodal (Gemma 3)
```bash
python scripts/run_multimodal.py \
    --ckpt=/path/to/checkpoint \
    --variant=4b \
    --device=cuda \
    --output_len=256
```

#### XLA/TPU
```bash
# TPU
PJRT_DEVICE=TPU python scripts/run_xla.py \
    --ckpt=/path/to/checkpoint \
    --variant=9b

# GPU with XLA
PJRT_DEVICE=CUDA USE_CUDA=1 python scripts/run_xla.py \
    --ckpt=/path/to/checkpoint \
    --variant=27b
```

### Docker Workflow

```bash
# Build standard container
DOCKER_URI=gemma:${USER}
docker build -f docker/Dockerfile ./ -t ${DOCKER_URI}

# Run inference
docker run -t --rm \
    --gpus all \
    -v ${CKPT_PATH}:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run.py --device=cuda --ckpt=/tmp/ckpt --variant=7b

# Build XLA container for TPU
docker build -f docker/xla.Dockerfile ./ -t gemma_xla:${USER}

# Build XLA container for GPU
docker build -f docker/xla_gpu.Dockerfile ./ -t gemma_xla_gpu:${USER}
```

## Code Conventions and Patterns

### File Organization
- **Copyright headers**: All files must have Apache 2.0 license header
- **Imports**: Standard library → third-party → local, alphabetically sorted
- **Type hints**: Required for function signatures
- **Docstrings**: Required for public APIs

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `GemmaAttention`, `SiglipVisionModel`)
- **Functions/methods**: `snake_case` (e.g., `precompute_freqs_cis`, `load_state_dict`)
- **Private functions**: Prefix with `_` (e.g., `_set_default_tensor_type`)
- **Constants**: `UPPERCASE` (e.g., `USE_CUDA`, `FLAGS`)
- **Config fields**: `snake_case` (e.g., `num_hidden_layers`, `attn_logit_softcapping`)

### Architecture Patterns

#### 1. Configuration-Driven Design
All model variants are defined through configuration factories:
```python
# Don't hardcode model parameters
# Instead, use config system
config = config.get_model_config(variant, dtype)
model = GemmaForCausalLM(config)
```

#### 2. KV Cache Management
Pre-allocated caches indexed by position, shared across layers:
```python
# Cache structure: [batch, num_kv_heads, max_seq_len, head_dim]
k_cache = torch.zeros(batch, num_kv_heads, max_len, head_dim)
v_cache = torch.zeros(batch, num_kv_heads, max_len, head_dim)

# Update at position
k_cache.index_copy_(2, positions, key)
v_cache.index_copy_(2, positions, value)
```

#### 3. Rotary Position Embeddings (RoPE)
```python
# Precompute frequency tables
freqs = precompute_freqs_cis(dim, max_len, theta=10000.0, rope_scaling_factor=1)

# Apply in attention
query = apply_rotary_emb(query, freqs)
key = apply_rotary_emb(key, freqs)
```

#### 4. Quantization Pattern
INT8 quantization with per-channel scales:
```python
# Weight stored as int8, scale as fp32
weight: torch.Tensor  # int8
weight_scaler: torch.Tensor  # fp32

# Apply at runtime
dequantized = weight * weight_scaler.unsqueeze(-1)
```

#### 5. Attention Masking
```python
# Causal mask: upper triangular with -inf
mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

# Sliding window mask (Gemma 2/3)
if attn_type == AttentionType.LOCAL_SLIDING:
    mask = build_sliding_window_mask(seq_len, window_size)

# Bidirectional for images (Gemma 3)
if image_patches:
    mask[image_start:image_end, image_start:image_end] = 0
```

#### 6. Weight Loading Strategies
```python
# Single checkpoint file
state_dict = torch.load('model.ckpt')['model_state_dict']
model.load_state_dict(state_dict, strict=False)

# Sharded checkpoints
with open('pytorch_model.bin.index.json') as f:
    index = json.load(f)
# Load each shard and merge
```

#### 7. XLA Optimization
```python
# Mark compilation boundaries
xm.mark_step()

# Use sharded linear layers
layer = ColumnParallelLinear(in_features, out_features, ...)
```

### Common Pitfalls

#### 1. Device Placement
```python
# WRONG: Mixing CPU/GPU tensors
tensor_cpu = torch.zeros(10)
tensor_gpu = torch.zeros(10).to('cuda')
result = tensor_cpu + tensor_gpu  # Error!

# RIGHT: Ensure consistent device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor1 = torch.zeros(10, device=device)
tensor2 = torch.zeros(10, device=device)
result = tensor1 + tensor2
```

#### 2. Attention Type Indexing (Gemma 3)
```python
# Attention types repeat in a pattern
# For 26 layers with pattern [LOCAL, LOCAL, LOCAL, LOCAL, LOCAL, GLOBAL]:
# Layer 0-4: LOCAL, Layer 5: GLOBAL, Layer 6-10: LOCAL, Layer 11: GLOBAL, etc.

# Get attention type for layer i
attn_type = config.attn_types[i % len(config.attn_types)]
```

#### 3. Multimodal Input Format
```python
# WRONG: Flat list
inputs = ["text", image, "more text"]

# RIGHT: Nested list
inputs = [["text", image, "more text"]]  # Batch of 1
```

#### 4. Tokenizer Selection
```python
# Gemma 1/2: Use tokenizer.model
tokenizer_path = 'tokenizer/tokenizer.model'

# Gemma 3: Use new tokenizer
tokenizer_path = 'tokenizer/gemma3_cleaned_262144_v2.spiece.model'

# Better: Let config decide
tokenizer_path = os.path.join(ckpt_path, config.tokenizer)
```

#### 5. Vision Model Integration
```python
# Don't forget to normalize vision embeddings
vision_embeds = self.vision_model(images)
vision_embeds = self.mm_soft_embedding_norm(vision_embeds)  # Critical!
vision_embeds = self.mm_input_projection(vision_embeds)
```

## Testing Guidelines

### Current State
- No formal test suite exists in the repository
- Scripts include hardcoded demo prompts for validation
- Sample images provided in `scripts/images/` for multimodal testing

### Manual Testing Approach
1. **Text-only models**: Use `scripts/run.py` with known prompts
2. **Multimodal models**: Use `scripts/run_multimodal.py` with test images
3. **XLA/TPU**: Use `scripts/run_xla.py` with PJRT_DEVICE environment variable

### Validation Checklist
When making changes, verify:
- [ ] Model loads weights without errors
- [ ] Generated text is coherent and relevant
- [ ] Quantized models produce similar outputs to full precision
- [ ] CPU and GPU produce consistent results
- [ ] Multimodal models correctly process images
- [ ] XLA models work on TPU and GPU

## Common Tasks for AI Assistants

### Task 1: Adding a New Model Variant

1. **Add configuration in `config.py`:**
```python
def get_config_for_NEW_VARIANT(dtype: str) -> GemmaConfig:
    return GemmaConfig(
        dtype=dtype,
        architecture=Architecture.GEMMA_3,
        num_hidden_layers=...,
        # ... other parameters
    )
```

2. **Update `get_model_config()` function:**
```python
elif variant == 'new_variant':
    return get_config_for_NEW_VARIANT(dtype)
```

3. **Update error message with new variant name**

4. **Update README.md with new variant information**

### Task 2: Debugging Model Loading Issues

Common issues and solutions:

```python
# Issue: Missing keys in state_dict
# Solution: Use strict=False and check missing/unexpected keys
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing: {missing}")
print(f"Unexpected: {unexpected}")

# Issue: Shape mismatch
# Solution: Check config matches checkpoint
config = get_model_config(variant)  # Must match checkpoint variant

# Issue: Sharded checkpoint not loading
# Solution: Verify index.json and shard files exist
assert os.path.exists('pytorch_model.bin.index.json')
```

### Task 3: Adding Vision Support to a Text Model

Follow the pattern in `gemma3_model.py`:

1. **Add vision config to model config**
2. **Integrate `SiglipVisionModel`**
3. **Add normalization and projection layers**
4. **Implement `populate_image_embeddings()`**
5. **Create custom attention mask with bidirectional regions**
6. **Update preprocessing to handle images**

### Task 4: Optimizing Performance

```python
# 1. Enable quantization
config.quant = True  # INT8 weights

# 2. Use appropriate dtype
config.dtype = 'bfloat16'  # Better than float32 on modern GPUs

# 3. For XLA, use model parallelization
from gemma import model_xla
model = model_xla.GemmaForCausalLM(config)

# 4. Adjust batch size for throughput
# Larger batches = better GPU utilization
```

### Task 5: Implementing New Sampling Strategies

Modify `Sampler.forward()` in `model.py`:

```python
class Sampler(nn.Module):
    def forward(self, ...):
        # 1. Compute logits
        logits = torch.matmul(hidden_states, embedding.t())

        # 2. Apply softcapping (Gemma 2/3)
        if self.config.final_logit_softcapping:
            logits = torch.tanh(logits / cap) * cap

        # 3. Apply temperature
        logits = logits / temperature

        # 4. Add your custom sampling logic here
        # e.g., top-p, top-k, beam search, etc.

        # 5. Sample next token
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token, logits
```

## Recent Changes and Active Development

Based on recent commit history (as of 2025-11):

1. **Gemma 3 support added** - Multimodal models with vision encoder
2. **GPU support for 1B model** - Recent fix for device placement
3. **Gemma 2 2B support** - New 2B-v2 variant with improved architecture
4. **XLA optimization** - Ongoing improvements for TPU/GPU
5. **Bug fixes** - Device placement, type downcasting, unused imports

### Active Areas
- GPU optimization for smaller models (1B)
- Multimodal capabilities expansion
- XLA performance improvements
- Model parallelization enhancements

## Important Notes for AI Assistants

### 1. Model Selection
- **Text-only tasks**: Use Gemma 1 (2B, 7B) or Gemma 2 (2B-v2, 9B, 27B)
- **Multimodal tasks**: Use Gemma 3 (4B, 12B, 27B_v3)
- **Small/fast**: 1B, 2B variants
- **High quality**: 27B variants

### 2. Hardware Requirements
- **1B-2B models**: Can run on CPU or single GPU
- **7B-12B models**: Require GPU with 16GB+ VRAM
- **27B models**: Require multiple GPUs or TPU
- **XLA**: Best for TPU, also supports CUDA GPUs

### 3. File Locations
- Model implementations: `gemma/model.py`, `gemma/gemma3_model.py`, `gemma/model_xla.py`
- Configurations: `gemma/config.py`
- Inference scripts: `scripts/run.py`, `scripts/run_multimodal.py`, `scripts/run_xla.py`
- Vision encoder: `gemma/siglip_vision/`

### 4. When to Use Which Implementation
- **`model.py`**: Standard PyTorch, CPU/GPU, all variants except multimodal
- **`gemma3_model.py`**: Multimodal inference (imports from model.py)
- **`model_xla.py`**: TPU/distributed GPU, Gemma 1 & 2 only (no Gemma 3 yet)

### 5. Key Differences Between Generations
| Feature | Gemma 1 | Gemma 2 | Gemma 3 |
|---------|---------|---------|---------|
| Attention | Global | Sliding + Global | Sliding + Global (5:1 ratio) |
| Normalization | Pre-norm | Pre+post norm | Pre+post norm |
| Logit softcapping | No | Yes (30/50) | No explicit |
| QK normalization | No | No | Yes |
| RoPE wavelength | 10,000 | 10,000 | 10K/1M (dual) |
| Max context | 8K | 8K | 32K-131K |
| Vision support | No | No | Yes (4B+) |
| Vocab size | 256K | 256K | 262K |

### 6. Contributing Requirements
- Sign Google CLA (Contributor License Agreement)
- Follow Google's Open Source Community Guidelines
- All PRs require code review
- Maintain Apache 2.0 license headers
- Add type hints and docstrings for public APIs

## Useful Commands

```bash
# Check Python version
python --version  # Should be >= 3.11

# List available GPUs
python -c "import torch; print(torch.cuda.device_count())"

# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Run quick inference test
python scripts/run.py --ckpt=/path/to/ckpt --variant=2b --device=cpu --prompt="Hello world"

# Monitor GPU usage
nvidia-smi -l 1

# Check model size
du -sh /path/to/checkpoint/*
```

## Resources

- **Official Gemma site**: https://ai.google.dev/gemma
- **Model checkpoints (Kaggle)**: https://www.kaggle.com/models/google/gemma-3/pytorch
- **Model checkpoints (HuggingFace)**: https://huggingface.co/models?other=gemma_torch
- **Documentation**: https://ai.google.dev/gemma/docs/pytorch_gemma
- **GitHub repository**: https://github.com/google/gemma_pytorch
- **Issues tracker**: https://github.com/google/gemma_pytorch/issues

---

**Last Updated**: 2025-11-27
**Repository Version**: Gemma 3 support with GPU optimizations
