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
"""Unit tests for Gemma tokenizer."""

import os
import pytest

from gemma import tokenizer


class TestTokenizer:
    """Tests for Tokenizer class."""

    @pytest.fixture
    def tokenizer_path(self):
        """Get path to tokenizer model."""
        return "tokenizer/tokenizer.model"

    @pytest.fixture
    def gemma3_tokenizer_path(self):
        """Get path to Gemma 3 tokenizer model."""
        return "tokenizer/gemma3_cleaned_262144_v2.spiece.model"

    @pytest.fixture
    def tok(self, tokenizer_path):
        """Create a tokenizer instance."""
        if not os.path.exists(tokenizer_path):
            pytest.skip(f"Tokenizer model not found at {tokenizer_path}")
        return tokenizer.Tokenizer(tokenizer_path)

    @pytest.fixture
    def tok_gemma3(self, gemma3_tokenizer_path):
        """Create a Gemma 3 tokenizer instance."""
        if not os.path.exists(gemma3_tokenizer_path):
            pytest.skip(f"Gemma 3 tokenizer model not found at {gemma3_tokenizer_path}")
        return tokenizer.Tokenizer(gemma3_tokenizer_path)

    def test_tokenizer_initialization(self, tok):
        """Test that tokenizer initializes correctly."""
        assert tok.sp_model is not None
        assert tok.n_words > 0
        assert tok.bos_id is not None
        assert tok.eos_id is not None
        assert tok.pad_id is not None

    def test_special_token_ids(self, tok):
        """Test that special token IDs are set correctly."""
        assert tok.boi_id == 255999
        assert tok.eoi_id == 256000
        assert tok.image_token_placeholder_id == tok.pad_id

    def test_encode_basic(self, tok):
        """Test basic encoding."""
        text = "Hello world"
        tokens = tok.encode(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
        # Should include BOS by default
        assert tokens[0] == tok.bos_id

    def test_encode_without_bos(self, tok):
        """Test encoding without BOS token."""
        text = "Hello world"
        tokens = tok.encode(text, bos=False)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        # Should not start with BOS
        assert tokens[0] != tok.bos_id

    def test_encode_with_eos(self, tok):
        """Test encoding with EOS token."""
        text = "Hello world"
        tokens = tok.encode(text, eos=True)

        assert isinstance(tokens, list)
        # Should end with EOS
        assert tokens[-1] == tok.eos_id

    def test_encode_with_bos_and_eos(self, tok):
        """Test encoding with both BOS and EOS tokens."""
        text = "Hello world"
        tokens = tok.encode(text, bos=True, eos=True)

        assert tokens[0] == tok.bos_id
        assert tokens[-1] == tok.eos_id

    def test_decode_basic(self, tok):
        """Test basic decoding."""
        text = "Hello world"
        tokens = tok.encode(text, bos=False, eos=False)
        decoded = tok.decode(tokens)

        assert isinstance(decoded, str)
        # Decoded text should contain the original words
        assert "Hello" in decoded or "hello" in decoded.lower()
        assert "world" in decoded.lower()

    def test_encode_decode_roundtrip(self, tok):
        """Test encode-decode round trip."""
        original_text = "The quick brown fox jumps over the lazy dog."
        tokens = tok.encode(original_text, bos=False, eos=False)
        decoded_text = tok.decode(tokens)

        # Allow for minor differences in whitespace/tokenization
        assert decoded_text.strip().lower() == original_text.strip().lower() or \
               decoded_text.replace(" ", "").lower() == original_text.replace(" ", "").lower()

    def test_encode_empty_string(self, tok):
        """Test encoding empty string."""
        tokens = tok.encode("", bos=True)
        # Should at least have BOS token
        assert len(tokens) >= 1
        assert tokens[0] == tok.bos_id

    def test_encode_whitespace_only(self, tok):
        """Test encoding whitespace-only string."""
        tokens = tok.encode("   ", bos=False)
        assert isinstance(tokens, list)

    def test_encode_special_characters(self, tok):
        """Test encoding special characters."""
        text = "Hello! How are you? I'm fine, thanks."
        tokens = tok.encode(text, bos=False)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_encode_unicode(self, tok):
        """Test encoding Unicode characters."""
        text = "Hello 世界 🌍"
        tokens = tok.encode(text, bos=False)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_encode_long_text(self, tok):
        """Test encoding long text."""
        text = " ".join(["word"] * 1000)
        tokens = tok.encode(text, bos=False)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_decode_empty_list(self, tok):
        """Test decoding empty token list."""
        decoded = tok.decode([])
        assert isinstance(decoded, str)
        assert decoded == ""

    def test_encode_assertion_non_string(self, tok):
        """Test that encoding non-string raises assertion error."""
        with pytest.raises(AssertionError):
            tok.encode(123)

    def test_different_texts_different_encodings(self, tok):
        """Test that different texts produce different encodings."""
        text1 = "Hello world"
        text2 = "Goodbye world"

        tokens1 = tok.encode(text1, bos=False)
        tokens2 = tok.encode(text2, bos=False)

        assert tokens1 != tokens2

    def test_same_text_same_encoding(self, tok):
        """Test that same text produces same encoding."""
        text = "Hello world"

        tokens1 = tok.encode(text)
        tokens2 = tok.encode(text)

        assert tokens1 == tokens2

    @pytest.mark.parametrize('text', [
        "Hello world",
        "The quick brown fox",
        "Machine learning is fascinating",
        "Python programming",
    ])
    def test_encode_various_texts(self, tok, text):
        """Test encoding various texts."""
        tokens = tok.encode(text, bos=False)
        decoded = tok.decode(tokens)

        assert isinstance(tokens, list)
        assert isinstance(decoded, str)
        assert len(tokens) > 0

    def test_vocab_size(self, tok):
        """Test that vocabulary size is reasonable."""
        assert tok.n_words > 1000  # Should have substantial vocabulary
        assert tok.n_words < 1000000  # But not unreasonably large

    def test_special_token_ids_valid(self, tok):
        """Test that special token IDs are valid."""
        assert tok.bos_id >= 0
        assert tok.eos_id >= 0
        assert tok.pad_id >= 0
        assert tok.boi_id >= 0
        assert tok.eoi_id >= 0

    def test_gemma3_tokenizer_initialization(self, tok_gemma3):
        """Test Gemma 3 tokenizer initialization."""
        assert tok_gemma3.sp_model is not None
        assert tok_gemma3.n_words > 0
        # Gemma 3 has a larger vocabulary
        assert tok_gemma3.n_words >= 260000

    def test_gemma3_encode_decode(self, tok_gemma3):
        """Test Gemma 3 tokenizer encode/decode."""
        text = "Hello, this is a test for Gemma 3."
        tokens = tok_gemma3.encode(text, bos=False, eos=False)
        decoded = tok_gemma3.decode(tokens)

        assert isinstance(tokens, list)
        assert isinstance(decoded, str)
        assert len(tokens) > 0


class TestTokenizerErrors:
    """Tests for tokenizer error handling."""

    def test_invalid_tokenizer_path(self):
        """Test that invalid path raises assertion error."""
        with pytest.raises(AssertionError):
            tokenizer.Tokenizer("nonexistent/path.model")

    def test_none_tokenizer_path(self):
        """Test that None path raises error."""
        with pytest.raises((AssertionError, TypeError)):
            tokenizer.Tokenizer(None)
