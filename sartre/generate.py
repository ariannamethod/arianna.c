#!/usr/bin/env python3
"""
SARTRE GENERATOR

Pure Python inference - NO PYTORCH REQUIRED!
Just NumPy.

Usage:
    python generate.py --prompt "Q: What is life?"
    python generate.py --prompt "Q: " --max_tokens 200 --temperature 0.7
    python generate.py --interactive

Requirements: Only numpy!
"""

import argparse
import os
import sys
import time

# Import the model from sartre_model.py (pure NumPy implementation)
from sartre_model import (
    SartreConfig,
    Sartre,
    load_weights_from_bin,
)
from tokenizer import SartreTokenizer


def load_model_and_tokenizer(
    config_path: str = 'weights/sartre_config.json',
    weights_path: str = 'weights/sartre.bin',
    tokenizer_path: str = 'subtitles/tokenizer.json',
) -> tuple:
    """Load model and tokenizer from files."""
    print("🧠 Loading model...")
    
    # Load config
    config = SartreConfig.load(config_path)
    print(f"   dim={config.dim}, layers={config.n_layers}, vocab={config.vocab_size}")
    
    # Load weights
    weights = load_weights_from_bin(weights_path, config)
    print(f"   Loaded weights from {weights_path}")
    
    # Create model
    model = Sartre(config, weights)
    
    # Load tokenizer
    tokenizer = SartreTokenizer(tokenizer_path)
    print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
    
    return model, tokenizer


def generate_text(
    model: Sartre,
    tokenizer: SartreTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    stop_on_newline: bool = True,
    verbose: bool = True,
) -> str:
    """Generate text from prompt.
    
    Args:
        stop_on_newline: If True, stops generation when a newline is produced.
                        This is useful for Q&A format where each answer ends with newline.
    """
    # Auto-format: if prompt looks like a question without "A:", add it
    if prompt.strip().endswith('?') and 'A:' not in prompt:
        prompt = prompt.strip() + '\nA: '
    
    if verbose:
        print(f"\n📝 Prompt: {prompt}")
        print("=" * 60)
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    
    # Stop tokens: newline (token 0) marks end of answer in Q&A format
    stop_tokens = None
    if stop_on_newline:
        newline_token = tokenizer.char_to_id.get('\n', 0)
        stop_tokens = [newline_token]
    
    # Generate
    start_time = time.time()
    output_tokens = model.generate(
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_tokens=stop_tokens,
    )
    elapsed = time.time() - start_time
    
    # Decode
    output_text = tokenizer.decode(output_tokens)
    
    # Trim to last complete sentence to avoid mid-word cutoff
    # Find the last sentence-ending punctuation in the generated part
    generated_part = output_text[len(prompt):]
    sentence_endings = ['.', '!', '?']
    last_end = -1
    for ending in sentence_endings:
        pos = generated_part.rfind(ending)
        if pos > last_end:
            last_end = pos
    
    # If we found a sentence ending, trim there
    if last_end >= 0:
        output_text = prompt + generated_part[:last_end + 1]
    
    if verbose:
        new_tokens = len(output_tokens) - len(prompt_tokens)
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0
        print(output_text)
        print("=" * 60)
        print(f"⏱️  Generated {new_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
    
    return output_text


def interactive_mode(
    model: Sartre,
    tokenizer: SartreTokenizer,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
):
    """Interactive chat mode."""
    print("\n" + "=" * 60)
    print("🌀 DUBROVSKY INTERACTIVE MODE 🌀")
    print("=" * 60)
    print("Enter your questions. Sartre will enlighten you.")
    print("Commands: /quit, /temp <float>, /tokens <int>")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye! Sartre's consciousness returns to the void.")
            break
        
        if not user_input:
            continue
        
        # Commands
        if user_input.startswith('/'):
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd == '/quit':
                print("👋 Goodbye!")
                break
            elif cmd == '/temp' and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"🌡️  Temperature set to {temperature}")
                except ValueError:
                    print("❌ Invalid temperature value")
            elif cmd == '/tokens' and len(parts) > 1:
                try:
                    max_new_tokens = int(parts[1])
                    print(f"📊 Max tokens set to {max_new_tokens}")
                except ValueError:
                    print("❌ Invalid token count")
            else:
                print("❓ Unknown command")
            continue
        
        # Format as Q&A
        prompt = f"Q: {user_input}\nA: Sartre "
        
        # Generate
        output = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=False
        )
        
        # Print response
        response = output[len(prompt):]
        print(f"Sartre: {response}\n")


def benchmark(model: Sartre, tokenizer: SartreTokenizer, num_runs: int = 5):
    """Benchmark generation speed."""
    print("\n⚡ BENCHMARK MODE")
    print("=" * 60)
    
    prompts = [
        "Q: What is consciousness?\nA: ",
        "Q: Why do we exist?\nA: ",
        "Q: What is the meaning of life?\nA: ",
    ]
    
    total_tokens = 0
    total_time = 0
    
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        prompt_tokens = tokenizer.encode(prompt)
        
        start = time.time()
        output = model.generate(prompt_tokens, max_new_tokens=50, temperature=0.8)
        elapsed = time.time() - start
        
        new_tokens = len(output) - len(prompt_tokens)
        total_tokens += new_tokens
        total_time += elapsed
        
        print(f"Run {i+1}: {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s)")
    
    print("=" * 60)
    print(f"Average: {total_tokens/num_runs:.0f} tokens, {total_tokens/total_time:.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(
        description='Generate text with Sartre (Pure Python, No PyTorch!)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate.py --prompt "Q: What is life?"
    python generate.py --interactive
    python generate.py --benchmark
        """
    )
    
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt for generation')
    parser.add_argument('--interactive', action='store_true', help='Interactive chat mode')
    parser.add_argument('--benchmark', action='store_true', help='Run speed benchmark')
    
    parser.add_argument('--max_tokens', type=int, default=150, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--no_stop', action='store_true', help='Disable stop on newline (generate full max_tokens)')
    
    parser.add_argument('--config', type=str, default='weights/sartre_config.json', help='Config path')
    parser.add_argument('--weights', type=str, default='weights/sartre.bin', help='Weights path')
    parser.add_argument('--tokenizer', type=str, default='weights/tokenizer.json', help='Tokenizer path')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        print("   Please train the model first: python train.py")
        print("   Then export weights: python export_weights.py weights/sartre_final.pt weights/sartre.bin")
        sys.exit(1)

    if not os.path.exists(args.weights):
        print(f"Weights not found: {args.weights}")
        print("   Please export weights: python export_weights.py weights/sartre_final.pt weights/sartre.bin")
        sys.exit(1)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.config, args.weights, args.tokenizer)
    
    # Run mode
    if args.benchmark:
        benchmark(model, tokenizer)
    elif args.interactive:
        interactive_mode(model, tokenizer, args.max_tokens, args.temperature)
    elif args.prompt:
        generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_on_newline=not args.no_stop,
        )
    else:
        # Default: generate a sample
        prompts = [
            "Q: What is consciousness?\nA: ",
            "Q: Why does my code have bugs?\nA: ",
            "Q: What is the meaning of life?\nA: ",
        ]
        
        print("\n🌀 DUBROVSKY SAMPLE GENERATION 🌀")
        for prompt in prompts:
            generate_text(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print()


if __name__ == '__main__':
    main()
