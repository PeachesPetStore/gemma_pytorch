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
"""Sample data for testing."""

import torch
from PIL import Image
import numpy as np


def get_sample_prompts():
    """Get sample text prompts for testing."""
    return [
        "What are large language models?",
        "Explain quantum computing.",
        "Hello, world!",
        "",  # Empty prompt edge case
        "A" * 1000,  # Long prompt
    ]


def get_sample_token_ids(vocab_size=256, seq_len=10, batch_size=2):
    """Generate sample token IDs for testing."""
    return torch.randint(0, vocab_size, (batch_size, seq_len))


def get_sample_image(height=224, width=224):
    """Generate a sample image for testing."""
    # Create a simple gradient image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img_array[i, j] = [i % 256, j % 256, (i + j) % 256]
    return Image.fromarray(img_array)


def get_sample_embeddings(batch_size=2, seq_len=10, hidden_size=128):
    """Generate sample embeddings for testing."""
    return torch.randn(batch_size, seq_len, hidden_size)


def get_sample_attention_mask(batch_size=2, seq_len=10):
    """Generate a sample causal attention mask."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def get_sample_kv_cache(batch_size=2, seq_len=128, num_heads=4, head_dim=32):
    """Generate sample KV cache for testing."""
    k_cache = torch.zeros(batch_size, seq_len, num_heads, head_dim)
    v_cache = torch.zeros(batch_size, seq_len, num_heads, head_dim)
    return (k_cache, v_cache)
