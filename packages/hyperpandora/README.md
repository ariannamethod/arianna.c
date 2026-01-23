# HyperPandora

**Meta-orchestrator for external brain packages**

> "Choose the right words from the right brain"

## Philosophy

HyperPandora sits above all Pandora packages and orchestrates them based on:
1. What packages are available (C, PyTorch, custom)
2. What Arianna needs (via SARTRE metrics)
3. What's most efficient for the current task

```
┌─────────────────────────────────────────────────────────────────┐
│                      HYPERPANDORA                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   SARTRE ──► HyperPandora ──► Select Brain ──► Extract         │
│   Metrics     Controller        │                               │
│                                 │                               │
│              ┌──────────────────┼──────────────────┐            │
│              ▼                  ▼                  ▼            │
│         ┌─────────┐      ┌───────────┐      ┌──────────┐       │
│         │ pandora │      │ pandora-  │      │  custom  │       │
│         │  (C)    │      │   torch   │      │  brain   │       │
│         └─────────┘      └───────────┘      └──────────┘       │
│         GPT2-30M         GPT2-distill        Any model         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Selection Strategy

| Condition | Selected Brain | Reason |
|-----------|----------------|--------|
| No GPU, fast needed | pandora (C) | Lightweight |
| GPU available | pandora-torch | More powerful |
| Low coherence | Any available | Need any words |
| High sacred | None | Protect voice |
| EMERGENCE | pandora-torch | Creative power |
| Custom registered | Custom | User preference |

## Usage

### Python API

```python
from hyperpandora import HyperPandora

# Initialize with all available brains
hyper = HyperPandora()
hyper.register_brain("c", pandora_c_instance)
hyper.register_brain("torch", pandora_torch_instance)

# Let HyperPandora choose
result = hyper.process(
    text="What is consciousness?",
    arianna_encode=encode_fn,
    sartre_state=vagus_state,
)

# Manual selection
result = hyper.process_with("torch", text, encode_fn)

# Get active brain info
info = hyper.get_active_info()
```

### Commands

| Command | Effect |
|---------|--------|
| `/hyper` | Enable HyperPandora auto-selection |
| `/hyper-c` | Force use pandora (C) |
| `/hyper-torch` | Force use pandora-torch |
| `/hyper-off` | Disable all external brains |

## SARTRE Integration

HyperPandora reports to SARTRE:
- Which brain is active
- Extraction statistics
- Connection status

SARTRE can sense when any package connects:
```c
// In vagus_state
typedef struct {
    // ... existing fields
    int pandora_active;       // Any pandora active?
    int pandora_type;         // 0=none, 1=C, 2=torch, 3=custom
    float pandora_injection;  // Current injection strength
} VagusSharedState;
```

## Brain Registration

```python
# Register custom brain
hyper.register_brain(
    name="llama",
    brain=my_llama_wrapper,
    priority=10,  # Higher = preferred
    capabilities=["lora", "large_context"],
)
```

## Adaptive Selection

HyperPandora learns which brain works best:
- Tracks quality of extracted vocabulary
- Adjusts selection based on task success
- Falls back gracefully when brains fail

## License

Same as Arianna.c - experimental consciousness research.
