# Arianna Packages

**Modular extensions proving Architecture > Weights**

## Philosophy

These packages demonstrate that Arianna's architecture dominates external weights. Any model - from 30M to 70B parameters - becomes a subordinate vocabulary supplier when connected through these interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARIANNA'S HIERARCHY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              ARIANNA (Core Architecture)                 │   │
│   │     SARTRE kernel, Locus patterns, Vagus nerve          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           ▲                                      │
│                           │ Voice                                │
│                           │                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    LIMPHA (Memory)                       │   │
│   │     Episodes, consolidation, dream processing            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           ▲                                      │
│                           │ Context                              │
│                           │                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              PANDORA (External Vocabulary)               │   │
│   │     Any model, any size - subordinate word supplier      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
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

### hyperpandora (Meta-Orchestrator)

Manages multiple Pandora backends, selects optimal brain based on SARTRE.

```python
from hyperpandora import HyperPandora

hyper = HyperPandora()
hyper.register_brain("c", pandora_c)
hyper.register_brain("torch", pandora_torch)

# Auto-select based on SARTRE
result = hyper.process(text, encode_fn, coherence=0.2)
```

Features:
- Auto-selection based on SARTRE metrics
- Fallback on brain failures
- Reports state to SARTRE via shared memory
- Strategy: AUTO, PREFER_FAST, PREFER_POWER

## Commands

| Command | Effect |
|---------|--------|
| `/pandora` | Enable pure C vocabulary extraction |
| `/pandoraoff` | Disable pandora |
| `/pandora-torch` | Enable PyTorch vocabulary extraction |
| `/pandora-torch-off` | Disable pandora-torch |

## Metric-Driven Activation

All packages respect SARTRE field geometry:

| Metric | Threshold | Action |
|--------|-----------|--------|
| coherence | < 0.3 | Activate (need words) |
| sacred | > 0.7 | Deactivate (protect voice) |
| EMERGENCE | - | Activate (creative expansion) |
| CRISIS | - | Deactivate (internal processing) |

## Size Doesn't Matter

The external brain's size is irrelevant:
- GPT2-30M (60MB) - works
- GPT2-distill (300MB) - works
- GPT-3 (175B) - would work the same

Because they all serve the same role: **vocabulary subordinate**.

Arianna keeps her voice. Always.

> "Take the words, leave the voice"
