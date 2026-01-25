#!/usr/bin/env python3
"""
SARTRE - Pure NumPy Inference (NO PYTORCH!)

Interactive REPL for SARTRE using pure NumPy implementation.

Usage:
    python3 sartre_talk.py
    python3 sartre_talk.py --prompt "Q: Who are you?"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sartre_model import SartreConfig, Sartre, load_weights_from_bin
from tokenizer import SartreTokenizer


def load_sartre():
    """Load SARTRE weights (pure NumPy, NO PyTorch)."""
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')

    config_path = os.path.join(weights_dir, 'sartre_config.json')
    weights_path = os.path.join(weights_dir, 'sartre.bin')
    tokenizer_path = os.path.join(weights_dir, 'tokenizer.json')

    print("🔮 Loading SARTRE (pure NumPy)...")

    # Load config
    config = SartreConfig.load(config_path)
    print(f"   dim={config.dim}, layers={config.n_layers}, vocab={config.vocab_size}")

    # Load weights (binary, float32)
    weights = load_weights_from_bin(weights_path, config)

    # Create model
    model = Sartre(config, weights)

    # Load tokenizer
    tokenizer = SartreTokenizer(tokenizer_path)

    print(f"✅ SARTRE loaded ({config.dim * config.n_layers * 1000 / 1000:.1f}K params)\n")

    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.7, top_k=40):
    """Generate from SARTRE."""
    # Encode
    tokens = tokenizer.encode(prompt)

    # Generate
    output_tokens = model.generate(
        tokens,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=0.9,
        stop_tokens=[tokenizer.char_to_id.get('\n', 0)]
    )

    # Decode
    return tokenizer.decode(output_tokens)


def repl(model, tokenizer):
    """Interactive REPL."""
    print("="*60)
    print("🔮 SARTRE REPL (Pure NumPy)")
    print("="*60)
    print("Commands: 'exit', 'quit', 'reset'")
    print("="*60 + "\n")

    context = ""

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit']:
                print("\n👋")
                break

            if user_input.lower() == 'reset':
                context = ""
                print("🔄 Context reset\n")
                continue

            if not user_input.startswith("Q:"):
                user_input = f"Q: {user_input}"

            prompt = context + user_input + "\nA: "

            print("SARTRE: ", end="", flush=True)

            # Generate
            output = generate(model, tokenizer, prompt, max_tokens=300, temperature=0.7)

            # Extract answer
            response = output[len(prompt):]
            if '\nQ:' in response:
                response = response.split('\nQ:')[0].strip()

            print(response)
            print()

            # Update context (keep last 2 Q&A)
            context += user_input + "\nA: " + response + "\n"
            lines = context.split('\n')
            if len(lines) > 8:
                context = '\n'.join(lines[-8:])

        except KeyboardInterrupt:
            print("\n\n👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SARTRE Pure NumPy Inference')
    parser.add_argument('--prompt', type=str, help='Single prompt (non-interactive)')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=200)
    args = parser.parse_args()

    model, tokenizer = load_sartre()

    if args.prompt:
        # Single prompt mode
        print(f"Prompt: {args.prompt}")
        print("SARTRE: ", end="", flush=True)
        output = generate(model, tokenizer, args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        response = output[len(args.prompt):]
        print(response)
    else:
        # Interactive mode
        repl(model, tokenizer)


if __name__ == '__main__':
    main()
