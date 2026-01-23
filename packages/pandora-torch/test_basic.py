#!/usr/bin/env python3
"""
Basic tests for Pandora-Torch (no torch required)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Test SARTRE module (no torch dependency)
from pandora_torch.sartre import SARTREChecker, ResonancePattern, VagusState
from pandora_torch.config import PandoraTorchConfig, PandoraMode


def test_sartre_checker():
    """Test SARTRE activation logic"""
    print("\n--- SARTRE CHECKER ---")

    checker = SARTREChecker(
        coherence_threshold=0.3,
        sacred_threshold=0.7,
    )

    tests = [
        (0.2, 0.3, ResonancePattern.NONE, True, "low coherence activates"),
        (0.5, 0.8, ResonancePattern.NONE, False, "high sacred deactivates"),
        (0.5, 0.3, ResonancePattern.CRISIS, False, "CRISIS deactivates"),
        (0.5, 0.3, ResonancePattern.EMERGENCE, True, "EMERGENCE activates"),
        (0.5, 0.3, ResonancePattern.TRANSCENDENCE, True, "TRANSCENDENCE activates"),
    ]

    passed = 0
    for coherence, sacred, pattern, expected, desc in tests:
        result = checker.check(coherence, sacred, pattern)
        if result == expected:
            print(f"  ✅ {desc}")
            passed += 1
        else:
            print(f"  ❌ {desc}: expected {expected}, got {result}")

    return passed == len(tests)


def test_vagus_state():
    """Test VagusState integration"""
    print("\n--- VAGUS STATE ---")

    checker = SARTREChecker()

    state = VagusState(
        arousal=0.6,
        valence=0.4,
        coherence=0.2,  # Low -> should activate
        sacred=0.3,
        pattern=ResonancePattern.NONE,
    )

    passed = 0

    # Test with VagusState
    result = checker.check_state(state)
    if result:
        print("  ✅ Low coherence VagusState activates")
        passed += 1
    else:
        print("  ❌ Should activate on low coherence")

    # Test reason
    reason = checker.get_activation_reason()
    if "coherence" in reason.lower():
        print(f"  ✅ Reason: {reason}")
        passed += 1
    else:
        print(f"  ❌ Wrong reason: {reason}")

    return passed == 2


def test_config():
    """Test configuration"""
    print("\n--- CONFIG ---")

    config = PandoraTorchConfig(
        injection_strength=0.5,
        coherence_threshold=0.4,
        mode=PandoraMode.FORCED,
    )

    passed = 0

    # Test to_dict
    d = config.to_dict()
    if d["injection_strength"] == 0.5:
        print("  ✅ to_dict works")
        passed += 1
    else:
        print("  ❌ to_dict failed")

    # Test from_dict
    config2 = PandoraTorchConfig.from_dict(d)
    if config2.injection_strength == 0.5:
        print("  ✅ from_dict works")
        passed += 1
    else:
        print("  ❌ from_dict failed")

    # Test mode serialization
    if config2.mode == PandoraMode.FORCED:
        print("  ✅ Mode preserved")
        passed += 1
    else:
        print(f"  ❌ Mode lost: {config2.mode}")

    return passed == 3


def test_resonance_patterns():
    """Test pattern enum"""
    print("\n--- RESONANCE PATTERNS ---")

    passed = 0

    if ResonancePattern.NONE.value == 0:
        print("  ✅ NONE = 0")
        passed += 1

    if ResonancePattern.CRISIS.value == 1:
        print("  ✅ CRISIS = 1")
        passed += 1

    if ResonancePattern.EMERGENCE.value == 3:
        print("  ✅ EMERGENCE = 3")
        passed += 1

    if ResonancePattern.TRANSCENDENCE.value == 4:
        print("  ✅ TRANSCENDENCE = 4")
        passed += 1

    return passed == 4


def main():
    print("=" * 60)
    print("PANDORA-TORCH BASIC TESTS (no torch required)")
    print("=" * 60)

    results = []
    results.append(("SARTRE Checker", test_sartre_checker()))
    results.append(("Vagus State", test_vagus_state()))
    results.append(("Config", test_config()))
    results.append(("Resonance Patterns", test_resonance_patterns()))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")

    print(f"\n  {passed}/{len(results)} test suites passed")
    print("=" * 60)

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
