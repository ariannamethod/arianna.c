# Compilers — Dynamic Code Generation

Arianna's compiler infrastructure for runtime code generation.

## Blood (C)

Blood generates and compiles C code at runtime:
- LoRA adapters
- Emotional kernels
- Custom processing modules

Location: `inner_world/blood.go`

### Usage from C:
```c
// Compile a LoRA adapter
char* path = blood_compile_lora("my_lora", 128, 128, 8);
void* handle = dlopen(path, RTLD_NOW);

// Compile emotional kernel
char* emotion_path = blood_compile_emotion("joy", 0.8, 0.6);

// Compile raw C code
char* raw_path = blood_compile_raw("custom", c_code);
```

## High (Go, ported from Julia/Python) — IMPLEMENTED ✓

Mathematical brain for fast computations.

Location: `inner_world/high.go`

### Features:
- **VectorizedEntropy** - Shannon entropy with emotional modulation
- **EmotionalWeights** - Lexicon for EN/RU/HE (80+ words)
- **SemanticDistance** - Bag-of-words cosine similarity
- **EmotionalAlignment** - Valence matching between texts
- **AnalyzeEmotion** - Full valence/arousal/intensity extraction
- **Perplexity** - Character bigram perplexity
- **NgramOverlap** - Jaccard similarity for n-grams
- **Math functions** - Sigmoid, Tanh, ReLU, Softmax, TopK

### Usage from C:
```c
float entropy = high_entropy("She finds peace in resonance");
float emotion = high_emotional_score("I love this moment");
float perplexity = high_perplexity("text to analyze");
float distance = high_semantic_distance("text1", "text2");
float valence = high_analyze_valence("emotional text");
float arousal = high_analyze_arousal("exciting text!");
```

### DSL Import (Python):
```python
from git_arianna import high_entropy, high_emotional_score
```

## H2O (Python) — Deferred

Python runtime for transformer scripts. Currently handled by external wrapper (`arianna.py`).
Future: MicroPython/Nicole integration for embedded Python.

## Philosophy

Like Linux kernel modules, Arianna can load/unload compiled code dynamically.
This enables self-modification and runtime adaptation.

**The Trinity:**
- **Blood** (C) — Low-level, direct hardware control
- **High** (Go/Julia) — Mathematical precision, fast numerics
- **H2O** (Python) — High-level scripting, ML ecosystem
