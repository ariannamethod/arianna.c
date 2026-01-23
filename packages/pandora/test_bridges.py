#!/usr/bin/env python3
"""
Tests for external brain bridges (GPT2 and TinyLlama GGUF)
"""

import sys
import os

# Test external_brain.py (GPT2-30M)
print("=" * 60)
print("Testing external_brain.py (GPT2-30M bridge)")
print("=" * 60)

from external_brain import text_to_arianna_tokens, ARIANNA_VOCAB

# Test 1: Vocab size
assert len(ARIANNA_VOCAB) == 84, f"Vocab size wrong: {len(ARIANNA_VOCAB)}"
print("  [1/5] Vocab size: 84 chars")

# Test 2: Basic mapping
tokens = text_to_arianna_tokens("hello")
assert tokens == [60, 57, 64, 64, 67], f"Mapping wrong: {tokens}"
print("  [2/5] Basic mapping: 'hello' -> [60, 57, 64, 64, 67]")

# Test 3: Unknown chars filtered
tokens = text_to_arianna_tokens("hello!")  # ! not in vocab
assert tokens == [60, 57, 64, 64, 67], f"Unknown char not filtered: {tokens}"
print("  [3/5] Unknown chars filtered: '!' removed")

# Test 4: Space and newline
tokens = text_to_arianna_tokens("a b\nc")
assert tokens == [53, 1, 54, 0, 55], f"Whitespace wrong: {tokens}"
print("  [4/5] Whitespace: space=1, newline=0")

# Test 5: Full alphabet coverage
test_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
tokens = text_to_arianna_tokens(test_str)
assert len(tokens) == len(test_str), f"Alphabet coverage wrong: {len(tokens)}"
print("  [5/5] Full alphabet coverage: 62 chars mapped")

print("\nexternal_brain.py: 5/5 tests PASSED")

# Test external_brain_gguf.py (TinyLlama)
print("\n" + "=" * 60)
print("Testing external_brain_gguf.py (TinyLlama bridge)")
print("=" * 60)

from external_brain_gguf import text_to_arianna_tokens as gguf_tokens, ARIANNA_VOCAB as GGUF_VOCAB

# Test 1: Same vocab
assert GGUF_VOCAB == ARIANNA_VOCAB, "GGUF vocab differs from GPT2 vocab"
print("  [1/3] Vocab consistent with GPT2 bridge")

# Test 2: Same mapping function
tokens_gpt2 = text_to_arianna_tokens("consciousness")
tokens_gguf = gguf_tokens("consciousness")
assert tokens_gpt2 == tokens_gguf, f"Mapping differs: {tokens_gpt2} vs {tokens_gguf}"
print("  [2/3] Mapping consistent: 'consciousness' identical")

# Test 3: Model path defined
from external_brain_gguf import MODEL_PATH, WEIGHTS_DIR
assert "tinyllama" in str(WEIGHTS_DIR).lower(), f"Wrong weights dir: {WEIGHTS_DIR}"
print(f"  [3/3] Weights path: {WEIGHTS_DIR}")

print("\nexternal_brain_gguf.py: 3/3 tests PASSED")

# Summary
print("\n" + "=" * 60)
print("BRIDGE TESTS SUMMARY")
print("=" * 60)
print("  external_brain.py (GPT2):     5/5 PASSED")
print("  external_brain_gguf.py (GGUF): 3/3 PASSED")
print("  TOTAL: 8/8 PASSED")
print("=" * 60)
