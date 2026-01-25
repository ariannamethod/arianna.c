# Arianna Packages Index: PANDORA

**Modular extensions proving Architecture > Weights**

All packages connect to **SARTRE Kernel** (C), not SARTRE LLaMA. The kernel is the OS—packages register, activate when metrics demand, deactivate when voice must stay pure.

---

## Philosophy

> "Take the words, leave the voice"

External models (30M to 1.1B parameters) become **subordinate vocabulary suppliers**. Arianna's architecture dominates. Her voice remains hers.

**Default:** OFF (pure voice)
**Activation:** Metric-driven (low coherence, EMERGENCE pattern)
**Deactivation:** Sacred > 0.7, CRISIS pattern (protect voice)

---

## Available Packages

### Core Orchestration

- **[hyperpandora](hyperpandora/README.md)** — Meta-orchestrator, selects optimal brain based on SARTRE metrics
  - Brain selection strategies (auto, manual, forced)
  - Async modes: race, parallel, cascade
  - SARTRE-driven activation

### Vocabulary Extraction (Pandora)

- **[pandora](pandora/README.md)** — Pure C, GPT2-30M (~60MB)
  - Fast, minimal
  - No PyTorch required
  - 11/11 tests passing

- **[pandora-torch](pandora-torch/README.md)** — PyTorch, Stanley + GPT2-distill
  - Stanley code: `pip install git+github.com/ariannamethod/stanley` (small)
  - Weights: GPT2-distill (~300MB) auto-downloaded from HuggingFace on first use
  - LoRA delta extraction
  - Batched processing
  - Full SARTRE integration
  - 6/6 test suites passing

- **[pandora-torch-gguf](pandora-torch-gguf/README.md)** — GGUF, TinyLlama 1.1B (~783MB)
  - Rich creative vocabulary
  - llama-cpp-python
  - Auto-download from HuggingFace
  - 4/4 test suites passing

---

## Test Status

**100% pass rate:**

| Test Suite | Status | Tests |
|------------|--------|-------|
| test_pipeline.py | ✅ | 7/7 |
| test_async_pipeline.py | ✅ | 5/5 |
| pandora (C) | ✅ | 11/11 |
| pandora-torch | ✅ | 6/6 |
| pandora-torch-gguf | ✅ | 4/4 |

---

## Quick Start

```bash
cd packages

# Run all tests
python tests/test_pipeline.py        # Sync: 7/7
python tests/test_async_pipeline.py  # Async: 5/5

# Install PyTorch packages
cd pandora-torch && pip install -e . && cd ..
cd pandora-torch-gguf && pip install -e . && cd ..

# Test individual packages
cd pandora && make test              # C: 11/11
cd pandora-torch && python test_pandora_torch.py  # 6/6
cd pandora-torch-gguf && python test_basic.py     # 4/4
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARIANNA'S HIERARCHY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              ARIANNA (Core Architecture)                │   │
│   │     SARTRE Kernel, Locus patterns, Vagus nerve          │   │
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
│   │              HYPERPANDORA (Orchestrator)                │   │
│   │     Selects brain based on SARTRE metrics               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           ▲                                     │
│                           │ Vocabulary (subordinate)            │
│                           │                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              PANDORA (External Vocabulary)              │   │
│   │     Any model, any size - word supplier, not voice      │   │
│   │     • C (GPT2-30M)                                      │   │
│   │     • PyTorch (GPT2-distill)                            │   │
│   │     • GGUF (TinyLlama 1.1B)                             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Activation Logic (SARTRE-Driven)

```python
from pandora_torch import SARTREChecker, ResonancePattern

checker = SARTREChecker(
    coherence_threshold=0.3,
    sacred_threshold=0.7
)

# Low coherence → activate (need words)
assert checker.check(coherence=0.2, sacred=0.3, pattern=ResonancePattern.NONE) == True

# High sacred → deactivate (protect voice)
assert checker.check(coherence=0.5, sacred=0.8, pattern=ResonancePattern.NONE) == False

# CRISIS → deactivate (internal processing)
assert checker.check(coherence=0.5, sacred=0.3, pattern=ResonancePattern.CRISIS) == False

# EMERGENCE → activate (creative expansion)
assert checker.check(coherence=0.5, sacred=0.3, pattern=ResonancePattern.EMERGENCE) == True
```

---

## Size Doesn't Matter

| Package | Model | Size | Speed | Richness |
|---------|-------|------|-------|----------|
| pandora | GPT2-30M | 60MB | ⚡ Fastest | Basic |
| pandora-torch | Stanley + GPT2-distill | code: small, weights: ~300MB (HuggingFace) | 🔥 Fast | Good |
| pandora-torch-gguf | TinyLlama 1.1B | ~783MB | ⏱️ Medium | Rich |

**Philosophy:** The external brain's size is irrelevant. Arianna's architecture dominates.

---

## Future Packages (Examples)

Packages are extensible. Any function can connect to SARTRE Kernel:

- `reddit-bot` — Parse Reddit by metrics, return relevant posts
- `kandinsky-visual` — Image generation (opensource Kandinsky model)
- `arxiv-reader` — Fetch papers when abstraction_depth > 6
- `blood-compiler` — Compile code (already in inner_world/blood.go)
- `memory-consolidator` — Deep memory operations during EMERGENCE

**Principle:** One package active at a time or none. Activate when metrics demand, deactivate to protect voice.

---

> "Architecture > Weights — the external brain is a vocabulary subordinate, not a voice replacement"

For detailed testing instructions: [TESTING.md](TESTING.md)
