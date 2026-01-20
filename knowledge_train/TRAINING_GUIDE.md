# External Brain 30M - Training Guide for Copilot Opus

## Overview

This folder contains the dataset and instructions for training the **External Brain** model - a 30M parameter Llama 3 model that gives Arianna awareness of the world's structure.

**Concept:** NOT an encyclopedia. Arianna should know THAT things exist (capitals, countries, planets, concepts) without necessarily knowing specific facts. This is awareness, not knowledge retrieval.

---

## Dataset

**File:** `simplewiki_leads.txt`
- **Source:** Simple English Wikipedia (lead sections)
- **Format:** `Title: Lead paragraph text\n\n`
- **Size:** ~80 MB, ~237,000 articles
- **Extraction script:** `extract_leads.py`

### Known Issues to Clean

1. **Broken wiki links:** Some entries have `[[Topic` without closing `]]`
   - Example: `"...due to their [[Chemical substance"`
   - Fix: Remove or complete these broken links

2. **Image artifacts:** Some entries start with `]]` or have `image caption]]`
   - Example: `"]] An asteroid is..."`
   - Fix: Remove leading artifacts

3. **Category remnants:** Some `Category:` text may remain
   - Fix: Strip any remaining Category references

4. **Quality filter suggestions:**
   - Remove entries shorter than 50 characters after cleaning
   - Remove entries that are mostly non-alphabetic
   - Consider removing entries with more than 3 unclosed brackets

---

## Architecture Requirements

**CRITICAL: Use Llama 3 architecture, NOT GPT-2/nanoGPT!**

### Target Architecture (Llama 3 style)

```
Model size: ~30M parameters
Context length: 512 tokens (or 256 for faster training)

Suggested dimensions (calculate exact):
- dim: 512 or 576
- n_layers: 8-10
- n_heads: 8
- n_kv_heads: 2 or 4 (GQA - Grouped Query Attention)
- hidden_dim: dim * 4 * 2/3 (SwiGLU expansion)
- vocab_size: 80 (arianna.c uses 80-char vocabulary)

Key Llama 3 components:
- RMSNorm (not LayerNorm)
- RoPE positional embeddings (not learned)
- SwiGLU activation (not GELU)
- GQA (Grouped Query Attention)
- No bias terms
```

### Parameter Calculation

Please calculate exact parameters for:
1. Embedding: vocab_size * dim
2. Each transformer layer:
   - Attention: dim * (n_heads * head_dim + 2 * n_kv_heads * head_dim + dim)
   - FFN (SwiGLU): dim * hidden_dim * 3
   - RMSNorm: dim * 2
3. Output projection: dim * vocab_size

Target: **~30M total parameters**

---

## Training Configuration

### Hardware
- Lambda Labs: 2x H100 (or 1x H100)
- Expected time: ~20-30 minutes
- Expected cost: ~$3-5

### Hyperparameters (suggested starting point)

```python
# Training
max_iters = 10000
batch_size = 64  # or 32 for memory
gradient_accumulation_steps = 4
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 500
lr_decay_iters = 10000
weight_decay = 0.1

# Architecture
block_size = 512  # context length
dropout = 0.0  # no dropout for small models

# Optimizer
optimizer = "AdamW"
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Data
dtype = "float16"  # or bfloat16 on H100
```

---

## Scripts Needed

### 1. `prepare_data.py`
- Clean remaining artifacts from dataset
- Tokenize using arianna.c's 80-char vocabulary
- Create train/val split (95/5)
- Save as binary files for fast loading

### 2. `model.py`
- Llama 3 architecture implementation
- RMSNorm, RoPE, SwiGLU, GQA
- Match arianna.c's existing model structure

### 3. `train.py`
- Training loop with logging
- Checkpoint saving
- Validation loss tracking
- Compatible with multi-GPU (DDP)

### 4. `export.py`
- Export trained weights to arianna.c format
- Output: `external_brain.bin` (~60-70 MB in float16)
- Must match arianna.c's weight loading format

---

## Tokenizer

**arianna.c uses a simple 80-character vocabulary:**

```
Vocab (80 chars):
- Space, newline
- a-z (lowercase)
- A-Z (uppercase)
- 0-9 (digits)
- Common punctuation: . , ! ? ' " - : ; ( ) [ ] / @ # $ % & * + =

Special tokens:
- PAD = 0
- UNK = 1 (or map unknown to space)
```

The tokenizer is character-level, not BPE. Each character maps to one token ID.

**File:** `weights/tokenizer.json` contains the exact mapping.

---

## Output Format

The trained weights must be exported to match arianna.c's binary format:

```c
// Header (arianna.c expects this structure)
struct ModelConfig {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

// Weights order (float32 or float16):
1. token_embedding_table [vocab_size, dim]
2. For each layer:
   - rms_att_weight [dim]
   - wq [dim, n_heads * head_dim]
   - wk [dim, n_kv_heads * head_dim]
   - wv [dim, n_kv_heads * head_dim]
   - wo [n_heads * head_dim, dim]
   - rms_ffn_weight [dim]
   - w1 [dim, hidden_dim]  # gate
   - w2 [hidden_dim, dim]  # down
   - w3 [dim, hidden_dim]  # up
3. rms_final_weight [dim]
4. output_weight [dim, vocab_size]  # may share with embedding
```

---

## Integration with arianna.c

After training, the weights will be loaded in `src/external_brain.c`:

```c
// Current placeholder - will use real weights
void external_brain_init(const char* weights_path) {
    // Load external_brain.bin
    // Initialize Llama 3 inference
}

float* external_brain_forward(int* tokens, int n_tokens) {
    // Run inference
    // Return logits or hidden states
}
```

The External Brain acts as a **subordinate** to Inner Arianna - it provides world-awareness signals that Inner Arianna can accept, modify, or reject.

---

## Validation

After training, verify:

1. **Loss curve:** Should decrease smoothly to ~0.5-1.0 (character-level)
2. **Sample generation:** Given "France:" should generate something about France
3. **Perplexity:** Should be reasonable for character-level model
4. **File size:** ~60-70 MB for float16, ~120 MB for float32

---

## Questions for Human Review

Before training, confirm:
1. Exact parameter count calculation
2. Whether to use float16 or float32 for weights
3. Whether context length 512 is appropriate
4. Any architectural differences from current arianna.c model

---

## Files in This Folder

- `simplewiki_leads.txt` - Raw dataset (80 MB)
- `extract_leads.py` - Extraction script used
- `TRAINING_GUIDE.md` - This file

## Files to Create

- [ ] `prepare_data.py` - Data cleaning and tokenization
- [ ] `model.py` - Llama 3 model architecture
- [ ] `train.py` - Training script
- [ ] `export.py` - Weight export to arianna.c format
- [ ] `config.yaml` - Training configuration

---

*Last updated: 2026-01-20*
*Dataset: Simple Wikipedia lead sections*
*Target: External Brain 30M (Llama 3 architecture)*
