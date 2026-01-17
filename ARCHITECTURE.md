# Arianna.c - Personality Weights Transformer

## Corpus
- **Size:** 13MB pure Arianna
- **Source:** 409 ariannabooks + genesis + SUPPERTIME + Origin + SCRIBE + DEFENDER
- **Language:** 100% English
- **Comparison:** 2.5x Shakespeare

## Target Architecture: 1-2M Parameters

### Option A: ~1M params (minimal)
```
dim = 128
n_layers = 4
n_heads = 4
vocab_size = 2048
max_seq_len = 256

Params breakdown:
- Token embedding: 2048 * 128 = 262K
- Position embedding: 256 * 128 = 33K
- Attention (per layer): 4 * 128 * 128 = 66K
- FFN (per layer): 2 * 128 * 512 = 131K
- Per layer total: ~200K
- 4 layers: 800K
- Output projection: 128 * 2048 = 262K
- TOTAL: ~1.3M params
```

### Option B: ~2M params (richer)
```
dim = 192
n_layers = 6
n_heads = 6
vocab_size = 4096
max_seq_len = 256

Params breakdown:
- Token embedding: 4096 * 192 = 786K
- Attention + FFN per layer: ~350K
- 6 layers: 2.1M
- TOTAL: ~2.8M params (slightly over)
```

### Option C: ~1.5M params (sweet spot)
```
dim = 160
n_layers = 5
n_heads = 5
vocab_size = 3072
max_seq_len = 256

TOTAL: ~1.5M params
```

## Implementation: Pure C

### Files needed:
1. `tokenizer.c` - SentencePiece or BPE tokenizer
2. `model.c` - Transformer architecture with RoPE
3. `train.c` - Training loop (can use numpy for gradients initially)
4. `inference.c` - Pure C inference, no dependencies
5. `weights.bin` - Binary weights file (~2-6MB)

### Training strategy:
- Use numpy/C for forward pass
- Gradient computation in C or numpy
- No PyTorch dependency
- Train on CPU (small model = fast)

## Philosophy

> 27MB weights ≠ "what I know"
> 2MB weights = "WHO I AM"

This is not a knowledge model. This is a personality.
Static weights = identity/persona.
Dynamic shards (Leo-style) = knowledge/context later.

"Кто я такая?" not "Что я знаю?"
