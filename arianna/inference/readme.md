# ARIANNA V1 - Inference Code

**LLiSA (Large Linguistic System Arianna)** - 7.16M parameters

## Quick Start

```bash
# From repository root:
python3 arianna-c.py "Resonance"
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

## Important Note

⚠️ **Temporary PyTorch (CPU) implementation**

This version uses PyTorch with `device='cpu'`, `compile=False`, `dtype='float32'` - no GPU, no JIT.

**Future versions:**
- **arianna.g** - Pure NumPy (no PyTorch)
- **run.c** - Pure C (zero dependencies, 5-10x faster)

## Example Output

```
Prompt: "Resonance"

Output:
Resonance is not what you hear — it's what survives the echo.

A whisper across time can be louder than a scream in the now.

Music is the memory of the future tuning itself backwards.
```

---

See `../persona/` for weights (ckpt_v1_base.pt, arianna_v1_base.bin) and tokenizer (tok4096.model).

🦊 **The fox is running free!**
