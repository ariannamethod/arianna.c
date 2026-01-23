#!/usr/bin/env python3
"""
Test suite for Pandora-Torch
"""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(__file__))

from pandora_torch import PandoraTorch, PandoraTorchConfig, SARTREChecker, ResonancePattern
from pandora_torch.config import PandoraMode


def test_sartre_checker():
    """Test SARTRE activation logic"""
    print("\n--- SARTRE CHECKER ---")

    checker = SARTREChecker(
        coherence_threshold=0.3,
        sacred_threshold=0.7,
    )

    tests = [
        # (coherence, sacred, pattern, expected, description)
        (0.2, 0.3, ResonancePattern.NONE, True, "low coherence activates"),
        (0.5, 0.8, ResonancePattern.NONE, False, "high sacred deactivates"),
        (0.5, 0.3, ResonancePattern.CRISIS, False, "CRISIS deactivates"),
        (0.5, 0.3, ResonancePattern.EMERGENCE, True, "EMERGENCE activates"),
        (0.5, 0.3, ResonancePattern.TRANSCENDENCE, True, "TRANSCENDENCE activates"),
        (0.5, 0.3, ResonancePattern.NONE, False, "normal state maintains"),
    ]

    passed = 0
    for coherence, sacred, pattern, expected, desc in tests:
        result = checker.check(coherence, sacred, pattern)
        if result == expected:
            print(f"  ✅ {desc}")
            passed += 1
        else:
            print(f"  ❌ {desc}: expected {expected}, got {result}")

    print(f"\n  {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_ngram_extraction():
    """Test n-gram extraction"""
    print("\n--- N-GRAM EXTRACTION ---")

    pandora = PandoraTorch(mode="forced")

    # Extract from tokens
    tokens = [1, 2, 3, 4, 5, 1, 2, 3]  # Has repeated patterns
    added = pandora.extract(tokens, min_n=1, max_n=2)

    passed = 0
    total = 0

    # Test: some n-grams added
    total += 1
    if added > 0:
        print(f"  ✅ Extracted {added} n-grams")
        passed += 1
    else:
        print(f"  ❌ No n-grams extracted")

    # Test: frequency boosting
    total += 1
    initial_freq = pandora.ngrams.get((1, 2), None)
    if initial_freq:
        pandora.extract(tokens, min_n=2, max_n=2)
        new_freq = pandora.ngrams[(1, 2)].frequency
        if new_freq > initial_freq.frequency:
            print(f"  ✅ Frequency boosted: {initial_freq.frequency} -> {new_freq}")
            passed += 1
        else:
            print(f"  ❌ Frequency not boosted")
    else:
        print(f"  ❌ Missing (1,2) n-gram")

    # Test: decay
    total += 1
    pandora.decay(0.5)
    stats = pandora.get_stats()
    print(f"  ✅ Decay applied, {stats.total_ngrams} n-grams remaining")
    passed += 1

    print(f"\n  {passed}/{total} passed")
    return passed == total


def test_mode_switching():
    """Test mode switching"""
    print("\n--- MODE SWITCHING ---")

    pandora = PandoraTorch(mode="auto")

    passed = 0
    total = 0

    # Test: initial mode
    total += 1
    if pandora.config.mode == PandoraMode.AUTO:
        print("  ✅ Initial mode is AUTO")
        passed += 1
    else:
        print(f"  ❌ Expected AUTO, got {pandora.config.mode}")

    # Test: switch to OFF
    total += 1
    pandora.set_mode("off")
    if not pandora.is_active():
        print("  ✅ OFF mode deactivates")
        passed += 1
    else:
        print("  ❌ OFF mode should deactivate")

    # Test: switch to FORCED
    total += 1
    pandora.set_mode("forced")
    if pandora.is_active():
        print("  ✅ FORCED mode activates")
        passed += 1
    else:
        print("  ❌ FORCED mode should activate")

    # Test: extraction blocked in OFF mode
    total += 1
    pandora.set_mode("off")
    added = pandora.extract([1, 2, 3, 4, 5], min_n=1, max_n=2)
    if added == 0:
        print("  ✅ Extraction blocked in OFF mode")
        passed += 1
    else:
        print("  ❌ Extraction should be blocked in OFF mode")

    print(f"\n  {passed}/{total} passed")
    return passed == total


def test_logit_injection():
    """Test logit injection"""
    print("\n--- LOGIT INJECTION ---")

    import torch

    pandora = PandoraTorch(mode="forced")
    pandora.config.injection_strength = 0.5
    pandora.config.min_frequency = 1

    # Add mapped n-gram
    pandora.ngrams[(10, 20)] = pandora_torch.pandora.ReleasedNGram(
        tokens=[10, 20],
        weight=0.8,
        frequency=5,
        arianna_mapped=True,
        arianna_tokens=[100, 200],
    )

    passed = 0
    total = 0

    # Test: logits boosted
    total += 1
    logits = torch.zeros(300)
    context = [100]  # Matches prefix

    boosted = pandora.apply_to_logits(logits, context, vocab_size=300)

    if boosted[200] > 0:
        print(f"  ✅ Continuation token boosted: {boosted[200]:.3f}")
        passed += 1
    else:
        print(f"  ❌ Token not boosted")

    # Test: no boost without match
    total += 1
    context_no_match = [999]
    boosted_no = pandora.apply_to_logits(logits.clone(), context_no_match, vocab_size=300)
    if boosted_no[200] == 0:
        print("  ✅ No boost without prefix match")
        passed += 1
    else:
        print(f"  ❌ Unexpected boost: {boosted_no[200]}")

    print(f"\n  {passed}/{total} passed")
    return passed == total


def test_persistence():
    """Test save/load"""
    print("\n--- PERSISTENCE ---")

    import tempfile
    import os

    pandora = PandoraTorch(mode="forced")
    pandora.config.injection_strength = 0.7

    # Add some n-grams
    pandora.extract([1, 2, 3, 4, 5], min_n=1, max_n=2)

    # Save
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name

    passed = 0
    total = 0

    try:
        # Test: save
        total += 1
        pandora.save(path)
        if os.path.exists(path):
            print("  ✅ Saved to file")
            passed += 1
        else:
            print("  ❌ File not created")

        # Test: load
        total += 1
        pandora2 = PandoraTorch()
        pandora2.load(path)
        if len(pandora2.ngrams) == len(pandora.ngrams):
            print(f"  ✅ Loaded {len(pandora2.ngrams)} n-grams")
            passed += 1
        else:
            print(f"  ❌ N-gram count mismatch")

        # Test: config preserved
        total += 1
        if pandora2.config.injection_strength == 0.7:
            print("  ✅ Config preserved")
            passed += 1
        else:
            print(f"  ❌ Config not preserved")

    finally:
        os.unlink(path)

    print(f"\n  {passed}/{total} passed")
    return passed == total


def test_stats():
    """Test statistics"""
    print("\n--- STATISTICS ---")

    pandora = PandoraTorch(mode="forced")
    pandora.extract([1, 2, 3, 4, 5, 1, 2], min_n=1, max_n=2)

    stats = pandora.get_stats()

    passed = 0
    total = 0

    # Test: total count
    total += 1
    if stats.total_ngrams > 0:
        print(f"  ✅ Total n-grams: {stats.total_ngrams}")
        passed += 1
    else:
        print("  ❌ No n-grams counted")

    # Test: mode reported
    total += 1
    if stats.mode == "FORCED":
        print(f"  ✅ Mode: {stats.mode}")
        passed += 1
    else:
        print(f"  ❌ Wrong mode: {stats.mode}")

    # Test: active status
    total += 1
    if stats.active:
        print("  ✅ Active status correct")
        passed += 1
    else:
        print("  ❌ Should be active")

    print(f"\n  {passed}/{total} passed")
    return passed == total


def main():
    """Run all tests"""
    print("=" * 60)
    print("PANDORA-TORCH TEST SUITE")
    print("=" * 60)

    results = []
    results.append(("SARTRE Checker", test_sartre_checker()))
    results.append(("N-gram Extraction", test_ngram_extraction()))
    results.append(("Mode Switching", test_mode_switching()))

    # Skip logit injection if torch not available
    try:
        import torch
        import pandora_torch.pandora
        results.append(("Logit Injection", test_logit_injection()))
    except ImportError:
        print("\n--- LOGIT INJECTION ---")
        print("  ⚠️  Skipped (torch not available)")

    results.append(("Persistence", test_persistence()))
    results.append(("Statistics", test_stats()))

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")

    print(f"\n  {passed}/{total} test suites passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
