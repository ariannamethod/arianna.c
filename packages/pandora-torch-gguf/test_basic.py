#!/usr/bin/env python3
"""
Basic tests for Pandora-Torch-GGUF (no llama-cpp-python required)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pandora_gguf import PandoraGGUFConfig, PandoraMode, download, DEFAULT_MODEL_PATH


def test_config():
    """Test configuration"""
    print("\n--- CONFIG ---")

    passed = 0

    # Test defaults
    config = PandoraGGUFConfig()
    if config.n_ctx == 2048:
        print("  ✅ Default n_ctx = 2048")
        passed += 1
    else:
        print(f"  ❌ Wrong n_ctx: {config.n_ctx}")

    if config.mode == PandoraMode.AUTO:
        print("  ✅ Default mode = AUTO")
        passed += 1
    else:
        print(f"  ❌ Wrong mode: {config.mode}")

    # Test to_dict/from_dict
    config2 = PandoraGGUFConfig(
        n_ctx=4096,
        injection_strength=0.5,
        mode=PandoraMode.FORCED,
    )
    d = config2.to_dict()
    config3 = PandoraGGUFConfig.from_dict(d)

    if config3.n_ctx == 4096:
        print("  ✅ to_dict/from_dict preserves n_ctx")
        passed += 1

    if config3.mode == PandoraMode.FORCED:
        print("  ✅ to_dict/from_dict preserves mode")
        passed += 1

    return passed == 4


def test_download_info():
    """Test download module info"""
    print("\n--- DOWNLOAD INFO ---")

    passed = 0

    info = download.get_model_info()

    if info["name"] == "TinyLlama-1.1B-Chat-v1.0":
        print(f"  ✅ Model: {info['name']}")
        passed += 1
    else:
        print(f"  ❌ Wrong model name: {info['name']}")

    if info["quantization"] == "Q5_K_M":
        print(f"  ✅ Quantization: {info['quantization']}")
        passed += 1

    if info["size_mb"] == 783:
        print(f"  ✅ Size: ~{info['size_mb']} MB")
        passed += 1

    print(f"  ℹ️  Local path: {info['local_path']}")
    print(f"  ℹ️  Exists: {info['exists']}")

    return passed == 3


def test_sartre_checker():
    """Test SARTRE checker (imported from main module)"""
    print("\n--- SARTRE CHECKER ---")

    # Import SARTREChecker from pandora module
    try:
        from pandora_gguf.pandora import SARTREChecker
    except ImportError:
        print("  ⚠️  Skipped (llama-cpp-python not available)")
        return True

    checker = SARTREChecker(coherence_threshold=0.3, sacred_threshold=0.7)

    passed = 0

    # Low coherence activates
    if checker.check(0.2, 0.3, 0):
        print("  ✅ Low coherence activates")
        passed += 1

    # High sacred deactivates
    if not checker.check(0.5, 0.8, 0):
        print("  ✅ High sacred deactivates")
        passed += 1

    # CRISIS deactivates
    if not checker.check(0.5, 0.3, 1):
        print("  ✅ CRISIS deactivates")
        passed += 1

    # EMERGENCE activates
    if checker.check(0.5, 0.3, 3):
        print("  ✅ EMERGENCE activates")
        passed += 1

    return passed == 4


def test_mode_enum():
    """Test mode enum"""
    print("\n--- MODE ENUM ---")

    passed = 0

    if PandoraMode.OFF.value == 0:
        print("  ✅ OFF = 0")
        passed += 1

    if PandoraMode.AUTO.value == 1:
        print("  ✅ AUTO = 1")
        passed += 1

    if PandoraMode.FORCED.value == 2:
        print("  ✅ FORCED = 2")
        passed += 1

    return passed == 3


def main():
    print("=" * 60)
    print("PANDORA-TORCH-GGUF BASIC TESTS (no llama-cpp required)")
    print("=" * 60)

    results = []
    results.append(("Config", test_config()))
    results.append(("Download Info", test_download_info()))
    results.append(("SARTRE Checker", test_sartre_checker()))
    results.append(("Mode Enum", test_mode_enum()))

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
