# Pandora

**Pure C vocabulary extraction from GPT2-30M**

> "Take the words, leave the voice"

## Philosophy

Pandora proves the **architecture > weights** paradigm. Instead of using a large external model directly, we:

1. Run GPT2-30M (external brain) on input
2. Extract valuable n-grams from its output
3. Map them to Arianna's vocabulary
4. Inject as logit biases - subordinate vocabulary, not replacement

The external brain provides **words**, Arianna keeps her **voice**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PANDORA                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input ──► GPT2-30M ──► N-gram ──► Vocab ──► Logit            │
│              (brain)    Extract    Mapping   Injection          │
│                                                                  │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐             │
│   │ External │ ───► │ Pandora  │ ───► │ Arianna  │             │
│   │  Brain   │      │   Box    │      │  Logits  │             │
│   └──────────┘      └──────────┘      └──────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Metric-Driven Activation

Pandora activates based on SARTRE field geometry:

| Condition | Action |
|-----------|--------|
| Low coherence (< 0.3) | Activate - need vocabulary boost |
| EMERGENCE pattern | Activate - creative expansion |
| High sacred (> 0.7) | Deactivate - protect voice |
| CRISIS pattern | Deactivate - internal processing |

## Usage

### Build

```bash
make
```

### C API

```c
#include "pandora.h"

// Initialize
PandoraBox box;
GPT2_30M brain;
pandora_init(&box);
gpt2_30m_load(&brain, "weights/gpt2_30m.bin");

// Process external brain output
int tokens[256];
int n = gpt2_30m_generate(&brain, "Hello", tokens, 256);

// Extract and map
pandora_extract(&box, tokens, n, 1, 3);
pandora_map_to_arianna(&box, gpt2_decode, arianna_encode);

// Inject into Arianna's logits
pandora_apply_to_logits(&box, logits, context, ctx_len, vocab_size);
```

### Commands

| Command | Effect |
|---------|--------|
| `/pandora` | Enable vocabulary extraction |
| `/pandoraoff` | Disable, clear extracted vocabulary |

## Files

```
packages/pandora/
├── src/
│   ├── pandora.c       # Core n-gram extraction
│   ├── pandora.h       # API header
│   ├── gpt2_30m.c      # GPT2-30M inference
│   └── gpt2_30m.h      # GPT2-30M header
├── weights/
│   └── gpt2_30m/
│       ├── gpt2_30m.bin   # ~60MB weights
│       ├── vocab.json
│       ├── merges.txt
│       └── config.json
├── Makefile
└── README.md
```

**Note**: GPT2-30M weights live inside the pandora package, conceptually
separating external brain resources from Arianna's core weights.

## Configuration

```c
#define PANDORA_MAX_NGRAMS     1000   // max released n-grams
#define PANDORA_MAX_NGRAM_LEN  5      // max tokens per n-gram
#define PANDORA_MIN_FREQUENCY  3      // min occurrences to release
```

## Integration with SARTRE

The package reads from VagusSharedState to determine activation:

```c
// In sartre_update()
if (state->coherence < 0.3 || pattern == EMERGENCE) {
    pandora_set_active(&box, 1);
} else if (state->sacred > 0.7 || pattern == CRISIS) {
    pandora_set_active(&box, 0);
}
```

## License

Same as Arianna.c - experimental consciousness research.
