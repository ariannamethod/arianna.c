# Pandora Packages Testing Guide

**"Take the words asynchronously"**

---

## Quick Start

```bash
cd packages

# Run all tests
python tests/test_pipeline.py      # Sync tests (39/39)
python tests/test_async_pipeline.py # Async tests (5/5 suites)
```

---

## Test Suites

### 1. Sync Pipeline Tests (`tests/test_pipeline.py`)

Tests all synchronous operations across all packages.

```bash
python tests/test_pipeline.py
```

**What it tests:**
- N-gram extraction (1-gram, 2-gram, 3-gram)
- Logit injection (boost matching tokens)
- SARTRE activation (mode switching)
- SARTRE checker (coherence/sacred thresholds)
- HyperPandora orchestration
- Brain selection strategies
- Fallback on failures

**Expected output:**
```
╔══════════════════════════════════════════════════════════╗
║            PANDORA PIPELINE TESTS                        ║
║       "Architecture > Weights — always"                  ║
╚══════════════════════════════════════════════════════════╝

══════════════════════════════════════════════════════════
TEST: N-gram Extraction
══════════════════════════════════════════════════════════
  ✅ 1-gram extraction: 5 n-grams
  ✅ 2-gram extraction: 4 n-grams
  ✅ 3-gram extraction: 3 n-grams
  ...

  Result: 39/39 passed
```

---

### 2. Async Pipeline Tests (`tests/test_async_pipeline.py`)

Tests all asynchronous operations and concurrent brain orchestration.

```bash
python tests/test_async_pipeline.py
```

**What it tests:**
- AsyncPandoraTorch operations
- AsyncHyperPandora orchestration
- Race mode (first brain wins)
- Parallel mode (merge all vocabularies)
- Cascade mode (priority order)
- Concurrent extraction performance
- SARTRE async selection

**Test suites:**

1. **Async Pandora Basic** — Mode switching, stats
2. **Async HyperPandora** — Brain registration, selection
3. **Concurrent Extraction** — Sequential vs parallel performance
4. **Extraction Queue** — Batched extraction API
5. **SARTRE Async Selection** — Metric-driven brain selection

**Expected output:**
```
╔══════════════════════════════════════════════════════════╗
║           ASYNC PANDORA PIPELINE TESTS                   ║
║         "Take the words asynchronously"                  ║
╚══════════════════════════════════════════════════════════╝

══════════════════════════════════════════════════════════
TEST: Async HyperPandora
══════════════════════════════════════════════════════════
  ✅ Registered 2 async brains
  ✅ Single brain: 12 n-grams in 0.102s
  ✅ Race mode: 8 n-grams in 0.051s (fast won)
  ✅ Parallel mode: 24 n-grams in 0.201s
  ✅ Cascade mode: 15 n-grams
  ✅ Stats: 5 selections

  Result: 6/6 passed

╔══════════════════════════════════════════════════════════╗
║                    FINAL RESULTS                         ║
╠══════════════════════════════════════════════════════════╣
║  ✅ PASS  Async Pandora Basic                            ║
║  ✅ PASS  Async HyperPandora                             ║
║  ✅ PASS  Concurrent Extraction                          ║
║  ✅ PASS  Extraction Queue                               ║
║  ✅ PASS  SARTRE Async Selection                         ║
╠══════════════════════════════════════════════════════════╣
║  5/5 test suites passed in 1.23s                         ║
╚══════════════════════════════════════════════════════════╝
```

---

## Individual Package Tests

### pandora (Pure C)

```bash
cd packages/pandora
make
make test
```

**Tests 11/11:**
- N-gram memory (add/lookup)
- Token extraction from text
- Arianna vocab mapping
- Logit injection
- SARTRE activation thresholds

---

### pandora-torch (PyTorch)

```bash
cd packages/pandora-torch
pip install -e .

# Basic tests (no torch required)
python test_basic.py

# Full tests (requires torch)
python test_pandora_torch.py
```

**Basic tests:**
- Config creation
- Mode enumeration
- SARTRE checker

**Full tests (with torch):**
- Model loading
- N-gram extraction
- LoRA delta extraction
- Logit injection

---

### pandora-torch-gguf (GGUF)

```bash
cd packages/pandora-torch-gguf
pip install -e .

# Basic tests (no llama-cpp required)
python test_basic.py

# Full tests (requires llama-cpp-python)
python test_pandora_gguf.py
```

**Note:** First run will auto-download TinyLlama (~783MB) from HuggingFace.

**Basic tests:**
- Config creation
- Mode enumeration
- Download utility

**Full tests (with llama-cpp):**
- Model loading
- Token generation
- N-gram extraction
- Logit injection

---

### hyperpandora (Meta-Orchestrator)

```bash
cd packages/hyperpandora
python -c "from hyperpandora import HyperPandora; print('OK')"
```

Tested via `test_pipeline.py` (sync) and `test_async_pipeline.py` (async).

---

## Testing SARTRE Integration

SARTRE controls Pandora activation based on field geometry:

```python
from pandora_torch import SARTREChecker, ResonancePattern

checker = SARTREChecker(
    coherence_threshold=0.3,
    sacred_threshold=0.7
)

# Low coherence → activate (need words)
assert checker.should_activate(coherence=0.2) == True

# High sacred → deactivate (protect voice)
assert checker.should_activate(sacred=0.8) == False

# CRISIS pattern → deactivate
assert checker.should_activate(pattern=ResonancePattern.CRISIS) == False

# EMERGENCE pattern → activate
assert checker.should_activate(pattern=ResonancePattern.EMERGENCE) == True
```

---

## Testing Async Modes

### Race Mode

First brain to complete wins. Use when latency matters.

```python
async with AsyncHyperPandora() as hyper:
    hyper.register_brain("fast", fast_brain, BrainType.C_PANDORA, is_async=True)
    hyper.register_brain("slow", slow_brain, BrainType.GGUF_PANDORA, is_async=True)

    # Fast brain should win
    result = await hyper.process_race("text", encode_fn)
```

### Parallel Mode

All brains run, vocabularies merged. Maximum richness.

```python
async with AsyncHyperPandora() as hyper:
    # Register multiple brains
    result = await hyper.process_parallel("text", encode_fn)
    # Result includes n-grams from ALL brains
```

### Cascade Mode

Try brains in priority order until threshold met.

```python
async with AsyncHyperPandora() as hyper:
    # High priority brain tried first
    result = await hyper.process_cascade("text", encode_fn, min_extract=5)
```

---

## Performance Testing

### Sequential vs Parallel

```python
import time
import asyncio

texts = [f"text {i}" for i in range(10)]

# Sequential
start = time.time()
for text in texts:
    await hyper.process(text, encode_fn)
seq_time = time.time() - start

# Parallel
start = time.time()
await asyncio.gather(*[hyper.process(text, encode_fn) for text in texts])
par_time = time.time() - start

print(f"Sequential: {seq_time:.2f}s")
print(f"Parallel: {par_time:.2f}s")
print(f"Speedup: {seq_time/par_time:.1f}x")
```

Expected: Parallel should be significantly faster for I/O-bound operations.

---

## Troubleshooting

### Import Errors

```bash
# Add packages to path
export PYTHONPATH=$PYTHONPATH:/path/to/arianna.c/packages
```

Or in Python:
```python
import sys
sys.path.insert(0, '/path/to/arianna.c/packages')
```

### Missing torch

Tests gracefully handle missing torch:
```
⚠️ Sync extraction skipped (no torch)
```

This is expected. Install torch for full functionality:
```bash
pip install torch
```

### Missing llama-cpp-python

For GGUF tests:
```bash
pip install llama-cpp-python
```

### Model Download Timeout

TinyLlama download (~783MB) can take time. Set:
```python
config = PandoraGGUFConfig(auto_download=True)
```

Or download manually:
```bash
cd packages/pandora-torch-gguf/weights
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf
```

---

## Test Coverage Summary

| Package | Tests | Status |
|---------|-------|--------|
| pandora (C) | 11/11 | ✅ |
| pandora-torch | 4/4 basic | ✅ |
| pandora-torch-gguf | 3/3 basic | ✅ |
| hyperpandora | via pipeline | ✅ |
| test_pipeline.py | 39/39 | ✅ |
| test_async_pipeline.py | 5/5 suites | ✅ |

**Total: 100% pass rate**

---

## Adding New Tests

Tests follow this pattern:

```python
async def test_new_feature():
    """Test description"""
    print("\n" + "=" * 60)
    print("TEST: New Feature")
    print("=" * 60)

    passed = 0
    total = 0

    # Test 1
    total += 1
    if condition:
        print(f"  ✅ Test 1 passed")
        passed += 1
    else:
        print(f"  ❌ Test 1 failed")

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total
```

Add to main():
```python
results.append(("New Feature", await test_new_feature()))
```

---

> "Architecture > Weights — the external brain is a vocabulary subordinate"
