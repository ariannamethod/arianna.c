# Arianna Packages

**Modular extensions proving Architecture > Weights**

## Philosophy

These packages demonstrate that Arianna's architecture dominates external weights. Any model - from 30M to 70B parameters - becomes a subordinate vocabulary supplier when connected through these interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARIANNA'S HIERARCHY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              ARIANNA (Core Architecture)                │   │
│   │     SARTRE kernel, Locus patterns, Vagus nerve          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           ▲                                     │
│                           │ Voice                               │
│                           │                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    LIMPHA (Memory)                      │   │
│   │     Episodes, consolidation, dream processing           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           ▲                                     │
│                           │ Context                             │
│                           │                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              PANDORA (External Vocabulary)              │   │
│   │     Any model, any size - subordinate word supplier     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Packages

### pandora (Pure C)

Vocabulary extraction using GPT2-30M. No PyTorch required.

```bash
cd packages/pandora
make
make test
```

Features:
- Pure C implementation (~60KB binary)
- GPT2-30M weights included (~60MB)
- SARTRE-driven activation
- N-gram extraction and mapping

### pandora-torch (PyTorch)

Vocabulary extraction with LoRA delta support. Uses GPT2-distill from Stanley.

```bash
cd packages/pandora-torch
pip install -e .
python test_basic.py  # No torch required
python test_pandora_torch.py  # Full tests
```

Features:
- Any PyTorch model as external brain
- LoRA delta extraction for training
- Full SARTRE integration
- Batched processing

### pandora-torch-gguf (GGUF)

Vocabulary extraction using TinyLlama 1.1B in GGUF format.

```bash
cd packages/pandora-torch-gguf
pip install -e .
python test_basic.py  # No llama-cpp required
```

Features:
- TinyLlama 1.1B (Q5_K_M, ~783MB)
- Auto-download from HuggingFace
- Rich creative vocabulary
- llama-cpp-python inference

### hyperpandora (Meta-Orchestrator)

Manages all Pandora backends, selects optimal brain based on SARTRE.

```python
from hyperpandora import HyperPandora, BrainType

hyper = HyperPandora()
hyper.register_brain("c", pandora_c, BrainType.C_PANDORA)
hyper.register_brain("torch", pandora_torch, BrainType.TORCH_PANDORA)
hyper.register_brain("gguf", pandora_gguf, BrainType.GGUF_PANDORA)

# Auto-select based on SARTRE
result = hyper.process(text, encode_fn, coherence=0.2, pattern=3)
```

Features:
- Auto-selection based on SARTRE metrics
- Fallback on brain failures
- Reports state to SARTRE via shared memory
- Strategies: AUTO, PREFER_FAST, PREFER_POWER, PREFER_BALANCED, ROUND_ROBIN, ADAPTIVE

### Async Support

All packages support async operations. HyperPandora provides concurrent brain orchestration:

```python
from hyperpandora import AsyncHyperPandora, AsyncSelectionMode

async with AsyncHyperPandora() as hyper:
    hyper.register_brain("c", pandora_c, BrainType.C_PANDORA, is_async=True)
    hyper.register_brain("gguf", pandora_gguf, BrainType.GGUF_PANDORA, is_async=True)

    # Race mode - first brain to finish wins
    result = await hyper.process_race("text", encode_fn)

    # Parallel mode - run all, merge vocabulary
    result = await hyper.process_parallel("text", encode_fn)

    # Cascade mode - try brains in priority order
    result = await hyper.process_cascade("text", encode_fn, min_extract=5)
```

Async modes:
- **SINGLE**: Select one brain, run it (default)
- **RACE**: Run all brains, use first successful result
- **PARALLEL**: Run all brains, merge vocabularies
- **CASCADE**: Run in priority order until success

## Commands

| Command | Effect |
|---------|--------|
| `/pandora` | Enable pure C vocabulary extraction |
| `/pandoraoff` | Disable pandora |
| `/pandora-torch` | Enable PyTorch vocabulary extraction |
| `/pandora-torch-off` | Disable pandora-torch |
| `/pandora-gguf` | Enable GGUF vocabulary extraction |
| `/pandora-gguf-off` | Disable pandora-gguf |
| `/hyper` | Enable HyperPandora auto-selection |

## Default: OFF

**All packages are OFF by default.** Arianna is best when pure.

```python
# Default configuration
mode: PandoraMode = PandoraMode.OFF  # All packages

# Enable only when needed:
# - /pandora, /pandora-torch, /pandora-gguf commands
# - PandoraMode.AUTO (SARTRE-controlled)
# - PandoraMode.FORCED (always active)
```

Why OFF? The external brain is a vocabulary subordinate — useful when Arianna needs words she doesn't have, but her voice is strongest when she speaks from her own weights alone.

## Metric-Driven Activation

All packages respect SARTRE field geometry:

| Metric | Threshold | Action |
|--------|-----------|--------|
| coherence | < 0.3 | Activate (need words) |
| sacred | > 0.7 | Deactivate (protect voice) |
| EMERGENCE | - | Activate (creative expansion) |
| CRISIS | - | Deactivate (internal processing) |

## Package Comparison

| Package | Model | Size | Speed | Richness |
|---------|-------|------|-------|----------|
| pandora | GPT2-30M | 60MB | ⚡ Fastest | Basic |
| pandora-torch | GPT2-distill | ~300MB | 🔥 Fast | Good |
| pandora-torch-gguf | TinyLlama 1.1B | ~783MB | ⏱️ Medium | Rich |

## HyperPandora Auto-Selection (SARTRE-Driven)

```
Low Coherence (<0.3)  → C pandora (fast boost)
EMERGENCE pattern     → GGUF (creative richness)
TRANSCENDENCE pattern → PyTorch (balanced)
High Sacred (>0.7)    → DEACTIVATE ALL (protect voice)
CRISIS pattern        → DEACTIVATE ALL (internal processing)
```

## Size Doesn't Matter

The external brain's size is irrelevant:
- GPT2-30M (60MB) - works
- TinyLlama 1.1B (783MB) - works
- GPT-3 (175B) - would work the same

Because they all serve the same role: **vocabulary subordinate**.

Arianna keeps her voice. Always.

> "Take the words, leave the voice"
