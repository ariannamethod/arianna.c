#!/usr/bin/env python3
"""
Test SARTRE inference - pure NumPy, no PyTorch
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dubrovsky import DubrovskyConfig, Dubrovsky, load_weights_from_bin
from tokenizer import DubrovskyTokenizer


def test_load_sartre():
    """Test loading SARTRE weights."""
    print("TEST 1: Loading SARTRE weights...")

    weights_dir = "../weights/sartre"
    config_path = os.path.join(weights_dir, 'sartre_config.json')
    weights_path = os.path.join(weights_dir, 'sartre.bin')
    tokenizer_path = os.path.join(weights_dir, 'tokenizer.json')

    config = DubrovskyConfig.load(config_path)
    assert config.dim == 416, f"Expected dim=416, got {config.dim}"
    assert config.n_layers == 7, f"Expected n_layers=7, got {config.n_layers}"
    assert config.vocab_size == 93, f"Expected vocab=93, got {config.vocab_size}"

    weights = load_weights_from_bin(weights_path, config)
    model = Dubrovsky(config, weights)
    tokenizer = DubrovskyTokenizer(tokenizer_path)

    assert tokenizer.vocab_size == 93

    print("✅ PASS: Weights loaded")
    return model, tokenizer


def test_tokenizer(tokenizer):
    """Test tokenizer encode/decode."""
    print("\nTEST 2: Tokenizer encode/decode...")

    test_str = "Q: Who are you?\nA: I see"
    tokens = tokenizer.encode(test_str)
    decoded = tokenizer.decode(tokens)

    assert test_str == decoded, f"Encode/decode mismatch: '{test_str}' != '{decoded}'"

    print(f"   Input: \"{test_str}\"")
    print(f"   Tokens: {tokens[:10]}...")
    print(f"   Decoded: \"{decoded}\"")
    print("✅ PASS: Tokenizer works")


def test_generation(model, tokenizer):
    """Test generation."""
    print("\nTEST 3: Generation...")

    prompts = [
        "Q: Who are you?\nA: ",
        "Q: What do you feel?\nA: ",
    ]

    for prompt in prompts:
        print(f"\n   Prompt: \"{prompt}\"")
        print("   SARTRE: ", end="", flush=True)

        tokens = tokenizer.encode(prompt)
        output_tokens = model.generate(
            tokens,
            max_new_tokens=100,
            temperature=0.7,
            top_k=40,
            stop_tokens=[tokenizer.char_to_id.get('\n', 0)]
        )

        output = tokenizer.decode(output_tokens)
        response = output[len(prompt):]

        print(response[:100])

        assert len(response) > 0, "Empty generation"
        # SARTRE prefix is OK (voice patterns from training)

    print("\n✅ PASS: Generation works")


def test_no_pytorch():
    """Test that PyTorch is not imported."""
    print("\nTEST 4: No PyTorch dependency...")

    assert 'torch' not in sys.modules, "PyTorch should not be imported!"

    print("✅ PASS: No PyTorch imported")


def main():
    print("="*60)
    print("🔮 SARTRE TEST SUITE")
    print("="*60)

    test_no_pytorch()
    model, tokenizer = test_load_sartre()
    test_tokenizer(tokenizer)
    test_generation(model, tokenizer)

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
