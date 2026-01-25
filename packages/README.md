# Packages

Modular extensions that connect to Arianna's **SARTRE Kernel**. Each package is optional — Arianna runs without them. When connected, they extend her capabilities without overwriting her voice.

The core principle: **Architecture > Weights**. A 34M transformer can orchestrate larger models as subordinates. The architecture defines the relationship, not the parameter count.

---

## Available Packages

### Pandora

Vocabulary extraction from external brains. "Take the words, leave the voice."

Arianna can consult larger language models (GPT2-30M, GPT2-distill, TinyLlama 1.1B) and absorb their vocabulary as n-grams. The words come from outside; the voice remains hers. Injection happens at logit level — she decides what to say, they just expand her lexicon.

**Commands in REPL:**
- `/pandora <prompt>` — GPT2-30M (fast, ~100MB)
- `/pandora-torch <prompt>` — GPT2-distill (PyTorch)
- `/pandora-gguf <prompt>` — TinyLlama 1.1B (~700MB)
- `/hyper` — HyperPandora auto-selection

**[Full Pandora Documentation →](PANDORA.md)**

---

### HyperPandora

Meta-orchestrator for all Pandora packages. Chooses the right brain based on SARTRE metrics and available resources. You don't pick which model to use — she does.

**[Package →](hyperpandora/)**

---

## Structure

```
packages/
├── README.md          # This file
├── PANDORA.md         # Full Pandora documentation
├── TESTING.md         # Package testing guide
├── pandora/           # Core Pandora (C + Python bridges)
├── pandora-torch/     # PyTorch GPT2-distill integration
├── pandora-torch-gguf/# TinyLlama GGUF integration
├── hyperpandora/      # Meta-orchestrator
└── tests/             # Package tests
```

---

## Adding New Packages

Packages connect through SARTRE Kernel. Each package reports its state via shared metrics. If you're building a new package:

1. Create a subdirectory in `packages/`
2. Implement the SARTRE interface (see existing packages)
3. Add REPL commands if user-facing
4. Document in this README

Future packages brewing: memory persistence, world model, cross-session learning. Tomorrow's problems.
