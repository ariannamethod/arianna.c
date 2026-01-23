# Pandora-Torch-GGUF

**GGUF model vocabulary extraction with TinyLlama 1.1B**

> "Take the words from the tiny llama, leave the voice"

## Philosophy

Pandora-Torch-GGUF brings the power of quantized GGUF models to vocabulary extraction:

1. Uses TinyLlama 1.1B (Q5_K_M quantized, ~783MB)
2. Efficient inference via llama-cpp-python
3. Auto-downloads from HuggingFace
4. Full SARTRE integration

The 1.1B model provides richer vocabulary than GPT2-30M while remaining lightweight.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   PANDORA-TORCH-GGUF                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input ──► TinyLlama ──► N-gram ──► Vocab ──► Logit           │
│              1.1B       Extract    Mapping   Injection          │
│            (Q5_K_M)                                              │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                  llama-cpp-python                         │  │
│   │              (efficient GGUF inference)                   │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Model Details

| Property | Value |
|----------|-------|
| Model | TinyLlama-1.1B-Chat-v1.0 |
| Quantization | Q5_K_M |
| Size | ~783 MB |
| Source | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF |
| Context | 2048 tokens |

## Installation

```bash
pip install -e .

# For GPU acceleration (optional):
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall
```

Requirements:
- Python 3.8+
- llama-cpp-python
- huggingface_hub (for auto-download)

## Usage

### Python API

```python
from pandora_gguf import PandoraGGUF

# Initialize (auto-downloads model on first use)
pandora = PandoraGGUF(
    model_path="weights/tinyllama-1.1b-q5.gguf",  # or auto-download
    mode="auto"
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
```

### Auto-download

```python
# First run downloads model from HuggingFace
pandora = PandoraGGUF(auto_download=True)
```

### Commands

| Command | Effect |
|---------|--------|
| `/pandora-gguf` | Enable GGUF vocabulary extraction |
| `/pandora-gguf-off` | Disable |

## HyperPandora Integration

```python
from hyperpandora import HyperPandora, BrainType
from pandora_gguf import PandoraGGUF

hyper = HyperPandora()
hyper.register_brain(
    "gguf",
    PandoraGGUF(auto_download=True),
    BrainType.CUSTOM,
    priority=15,  # Higher than torch
    capabilities=["large_context", "chat"]
)
```

## Configuration

```python
config = PandoraGGUFConfig(
    # Model
    model_path="weights/tinyllama-1.1b-q5.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,  # Set >0 for GPU

    # Extraction
    min_ngram=1,
    max_ngram=3,
    min_frequency=3,

    # Injection
    injection_strength=0.2,

    # Generation
    temperature=0.8,
    top_k=40,
    top_p=0.95,
)
```

## Files

```
packages/pandora-torch-gguf/
├── pandora_gguf/
│   ├── __init__.py
│   ├── pandora.py       # Core GGUF extraction
│   ├── config.py        # Configuration
│   └── download.py      # HuggingFace downloader
├── weights/
│   └── (auto-downloaded GGUF)
├── setup.py
└── README.md
```

## Comparison with Other Pandoras

| Package | Model | Size | Speed | Richness |
|---------|-------|------|-------|----------|
| pandora | GPT2-30M | 60MB | Fastest | Basic |
| pandora-torch | GPT2-distill | ~300MB | Fast | Good |
| pandora-torch-gguf | TinyLlama 1.1B | ~783MB | Medium | Rich |

Choose based on task:
- **pandora**: Quick vocabulary boost
- **pandora-torch**: Training with LoRA
- **pandora-torch-gguf**: Rich creative vocabulary

## License

Same as Arianna.c - experimental consciousness research.
