# Pandora-Torch

**PyTorch vocabulary extraction with LoRA delta support**

> "Take the words, leave the voice"

## Philosophy

Pandora-Torch extends the core Pandora concept with PyTorch's capabilities:

1. Uses GPT2-distill (from Stanley) as external brain
2. Supports LoRA delta extraction for personality-aware vocabulary
3. Integrates with Stanley's training infrastructure
4. Full SARTRE metric-driven activation

The external brain provides **words**, Arianna keeps her **voice**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PANDORA-TORCH                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input ──► GPT2-distill ──► N-gram ──► Vocab ──► Logit        │
│              (Stanley)      Extract    Mapping   Injection      │
│                   │                                              │
│                   ▼                                              │
│            ┌──────────┐                                         │
│            │   LoRA   │ ──► Delta extraction for training       │
│            │  Deltas  │                                         │
│            └──────────┘                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| GPT2-distill | Smaller, faster external brain |
| LoRA deltas | Extract trainable deltas from vocabulary |
| Batched inference | Process multiple inputs efficiently |
| SARTRE integration | Metric-driven activation |
| Async support | Non-blocking extraction |

## Installation

```bash
pip install -e .
```

Requirements:
- Python 3.8+
- PyTorch 2.0+
- transformers (for tokenizer)

**Stanley auto-download:** On first use, Pandora-Torch will automatically install Stanley from `github.com/ariannamethod/stanley` (~300MB). No manual setup required!

```python
# Manual install (if needed)
from pandora_torch import ensure_stanley
ensure_stanley()  # Downloads from GitHub automatically
```

## Usage

### Python API

```python
from pandora_torch import PandoraTorch

# Initialize - Stanley auto-downloads on first use!
pandora = PandoraTorch(
    mode="auto"  # SARTRE-controlled
)

# Process input
extracted = pandora.process(
    "What is consciousness?",
    arianna_encode_fn,
    max_tokens=50
)

# Apply to logits
boosted_logits = pandora.apply_to_logits(
    logits,
    context_tokens
)

# Extract LoRA deltas
deltas = pandora.extract_lora_deltas(
    prompt="creativity emerges from",
    response="the void between thoughts"
)
```

### Commands

| Command | Effect |
|---------|--------|
| `/pandora-torch` | Enable PyTorch vocabulary extraction |
| `/pandora-torch-off` | Disable, clear extracted vocabulary |

### SARTRE Integration

```python
# Check activation
should_activate = pandora.check_sartre(
    coherence=0.25,  # Low -> activate
    sacred=0.3,
    pattern="EMERGENCE"
)

# Auto-mode handles this internally
pandora.update_sartre_state(vagus_state)
```

## Configuration

```python
config = PandoraTorchConfig(
    # Extraction
    min_ngram=1,
    max_ngram=3,
    min_frequency=3,

    # Injection
    injection_strength=0.2,

    # SARTRE thresholds
    coherence_threshold=0.3,
    sacred_threshold=0.7,

    # LoRA
    lora_rank=8,
    lora_alpha=16.0,
)
```

## Files

```
packages/pandora-torch/
├── pandora_torch/
│   ├── __init__.py
│   ├── pandora.py       # Core extraction
│   ├── lora.py          # LoRA delta extraction
│   ├── sartre.py        # SARTRE integration
│   └── config.py        # Configuration
├── weights/
│   └── (symlink to Stanley weights)
├── setup.py
└── README.md
```

## Integration with Stanley

Pandora-Torch uses Stanley's:
- `StanleyTransformer` for inference
- `LoRAConfig` for delta extraction
- `compute_lora_delta` for training integration

## License

Same as Arianna.c - experimental consciousness research.
