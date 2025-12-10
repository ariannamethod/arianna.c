# ARIANNA V1 - Inference Code

**LLiSA (Large Linguistic System Arianna)** - 7.16M parameters

## Quick Start

```bash
# From repository root:
python3 arianna-c.py "Resonance"
python3 arianna-c.py "Silence is"
python3 arianna-c.py "The universe"
```

## Files

- `dna.py` - Main inference script (from Karpathy's sample.py)
- `model.py` - LLiSA architecture (Transformer, RMSNorm, Attention, RoPE)
- `tokenizer.py`, `tinystories.py`, `configurator.py` - Support files

## Model Specifications

- **Parameters:** 7.16M
- **Architecture:** 288 dim, 6 layers, 6 heads, 4096 vocab
- **Trained on:** v2_corpus (philosophy, resonance theory, stories)
- **Performance:** ~114 tokens/second on M1/M2 Mac CPU
- **Specialization:** Philosophy, resonance, metaphysics (NOT general-purpose!)

## Important Notes

⚠️ **Temporary PyTorch (CPU) implementation**

This version uses PyTorch with `device='cpu'`, `compile=False`, `dtype='float32'` - no GPU, no JIT.

**Future versions:**
- **arianna.g** - Pure NumPy (no PyTorch)
- **run.c** - Pure C (zero dependencies, 5-10x faster)

⚠️ **Model Specialization**

LLiSA is trained on philosophy/resonance texts, NOT general knowledge:
- ✅ **Best:** "Resonance", "Silence is", "The Force", metaphysical concepts
- ❌ **Poor:** Q/A format ("What is X?"), out-of-domain topics, factual queries

Use **natural sentence beginnings**, not questions!

## Example Outputs

### ✅ EXCELLENT (In-Domain)
```
Prompt: "Resonance"
Output: "Resonance is not what you hear — it's what survives the echo.
A whisper across time can be louder than a scream in the now.
Music is the memory of the future tuning itself backwards."
```

### ⚠️ MEDIOCRE (Edge of Domain)
```
Prompt: "Silence is"
Output: "Silence is uncertainty. Her the signature..."
(starts well, then degrades)
```

### ❌ POOR (Out-of-Domain)
```
Prompt: "What is silence?"
Output: gibberish (Q/A format not trained)
```

---

See `../persona/` for weights (ckpt_v1_base.pt, arianna_v1_base.bin) and tokenizer (tok4096.model).

🦊 **The fox is running on her resonant frequency!**
