#!/usr/bin/env python3
"""
arianna.py - Python wrapper for arianna.c REPL

Uses subprocess to communicate with compiled binary.
Batch mode for training, interactive mode for chat.

Usage:
    python arianna.py                    # Interactive mode
    python arianna.py --train-math 100   # Train MathBrain with N problems
"""

import subprocess
import sys
import os
import random


def run_arianna_batch(commands, weights="weights/arianna_34m.bin", tokenizer="weights/arianna_34m_tokenizer.json", max_tokens=50, temp=0.8):
    """Run arianna with a batch of commands, return output."""
    binary = "./bin/arianna_dynamic"

    if not os.path.exists(binary):
        raise FileNotFoundError(f"No arianna binary found. Run 'make dynamic' first.")

    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")

    if not os.path.exists(tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer}")

    cmd = [binary, weights, tokenizer, "--repl", str(max_tokens), str(temp)]

    # Add exit command
    input_text = "\n".join(commands) + "\nexit\n"

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    stdout, _ = proc.communicate(input_text, timeout=60)
    return stdout


def parse_math_result(output):
    """Parse MathBrain result from output."""
    for line in output.split("\n"):
        if "[resonance: correct]" in line:
            return True, line
        if "[truth:" in line:
            return False, line
    return None, None


class AriannaSession:
    """Session-based wrapper using batch calls."""

    def __init__(self, weights="weights/arianna_34m.bin", tokenizer="weights/arianna_34m_tokenizer.json", max_tokens=100, temp=0.8):
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temp = temp
        self.history = []

    def run(self, commands):
        """Run commands and return output."""
        if isinstance(commands, str):
            commands = [commands]

        output = run_arianna_batch(
            commands,
            weights=self.weights,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            temp=self.temp
        )

        self.history.extend(commands)
        return output

    def math(self, expr):
        """Run math expression."""
        return self.run([expr])

    def chat(self, message):
        """Chat with Arianna."""
        return self.run([message])

    def signals(self):
        """Get signals."""
        return self.run(["signals"])

    def body(self):
        """Get body state."""
        return self.run(["body"])

    def math_stats(self):
        """Get MathBrain stats."""
        return self.run(["math"])


def train_mathbrain(weights="weights/arianna_34m.bin", tokenizer="weights/arianna_34m_tokenizer.json", n_problems=100, batch_size=10):
    """Train MathBrain with arithmetic problems using batch mode."""

    print(f"\n=== Training MathBrain ({n_problems} problems) ===\n")

    # Curriculum: start simple
    phases = [
        (n_problems // 4, 1, 5),   # Easy: 1-5
        (n_problems // 4, 1, 10),  # Medium: 1-10
        (n_problems // 4, 1, 20),  # Harder: 1-20
        (n_problems // 4, 1, 30),  # Full: 1-30
    ]

    correct = 0
    total = 0

    for phase_problems, min_n, max_n in phases:
        print(f"Phase: numbers {min_n}-{max_n}")

        # Generate batch of problems
        commands = []
        expected = []

        for _ in range(phase_problems):
            a = random.randint(min_n, max_n)
            b = random.randint(min_n, max_n)
            op = random.choice(['+', '-'])
            commands.append(f"{a} {op} {b}")

            if op == '+':
                expected.append(a + b)
            else:
                expected.append(a - b)

        # Add math stats at end
        commands.append("math")

        # Run batch
        output = run_arianna_batch(commands, weights=weights, tokenizer=tokenizer, max_tokens=10)

        # Count correct
        for line in output.split("\n"):
            if "[resonance: correct]" in line:
                correct += 1
                total += 1
            elif "[truth:" in line:
                total += 1

        if total > 0:
            acc = correct / total * 100
            print(f"  Accuracy so far: {acc:.1f}% ({correct}/{total})")

    print(f"\n=== Training complete ===")
    print(f"Final accuracy: {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else "No data")


def interactive_mode(weights="weights/arianna_34m.bin", tokenizer="weights/arianna_34m_tokenizer.json", max_tokens=100, temp=0.8):
    """Run interactive REPL (each input is a batch call)."""
    print("\n" + "="*50)
    print("  Arianna.c Python Wrapper")
    print("  Each input runs a fresh arianna session")
    print("  Type 'quit' to exit")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("you> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye.")
                break

            # Run single command
            output = run_arianna_batch([user_input], weights=weights, tokenizer=tokenizer, max_tokens=max_tokens, temp=temp)

            # Extract relevant output (skip init messages)
            lines = output.split("\n")
            in_response = False
            for line in lines:
                if line.startswith("> "):
                    in_response = True
                    print(line[2:])  # Skip "> " prefix
                elif in_response and line.strip():
                    if "[exiting" in line:
                        break
                    print(line)

            print()

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except EOFError:
            break


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Arianna.c Python Wrapper")
    parser.add_argument("--weights", default="weights/arianna_34m.bin", help="Path to weights")
    parser.add_argument("--tokenizer", default="weights/arianna_34m_tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--tokens", type=int, default=100, help="Max tokens per response")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature")
    parser.add_argument("--train-math", type=int, metavar="N", help="Train MathBrain with N problems")

    args = parser.parse_args()

    try:
        if args.train_math:
            train_mathbrain(args.weights, args.tokenizer, args.train_math)
        else:
            interactive_mode(args.weights, args.tokenizer, args.tokens, args.temp)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
