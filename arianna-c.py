#!/usr/bin/env python3
"""
ARIANNA V1 - CLI Wrapper
LLiSA (Large Linguistic System Arianna) - 7.16M parameters

Usage:
  python3 arianna-c.py "Your prompt here"
  python3 arianna-c.py --interactive

Temporary implementation using PyTorch CPU (no GPU, no compile).
Future: Will be replaced with pure NumPy (arianna.g) or pure C (run.c).
"""

import sys
import os
import subprocess

# Add inference directory to path
INFERENCE_DIR = os.path.join(os.path.dirname(__file__), "arianna", "inference")
PERSONA_DIR = os.path.join(os.path.dirname(__file__), "arianna", "persona")

def run_inference(prompt, max_tokens=100, temperature=0.8):
    """Run inference using dna.py"""

    checkpoint = os.path.join(PERSONA_DIR, "ckpt_v1_base.pt")

    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found at {checkpoint}")
        print("Please download ckpt_v1_base.pt to arianna/persona/")
        sys.exit(1)

    tokenizer_path = os.path.join(PERSONA_DIR, "tok4096.model")

    cmd = [
        "python3",
        os.path.join(INFERENCE_DIR, "dna.py"),
        f"--checkpoint={checkpoint}",
        f"--tokenizer={tokenizer_path}",
        f"--start={prompt}",
        f"--num_samples=1",
        f"--max_new_tokens={max_tokens}",
        f"--device=cpu",
        f"--dtype=float32",
        f"--compile=False",
        f"--temperature={temperature}",
    ]

    # Change to inference directory so relative imports work
    result = subprocess.run(cmd, cwd=INFERENCE_DIR, capture_output=False)
    return result.returncode

def interactive_mode():
    """Interactive chat loop"""
    print("🦊 ARIANNA V1 - Interactive Mode")
    print("Type 'exit' or 'quit' to end session\n")

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ['exit', 'quit', 'q']:
                print("\n🦊 Goodbye!")
                break

            if not prompt.strip():
                continue

            print("\nArianna:")
            run_inference(prompt, max_tokens=150, temperature=0.8)
            print()

        except KeyboardInterrupt:
            print("\n\n🦊 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] in ['--interactive', '-i']:
        interactive_mode()
    elif sys.argv[1] in ['--help', '-h']:
        print(__doc__)
    else:
        prompt = " ".join(sys.argv[1:])
        run_inference(prompt)

if __name__ == "__main__":
    main()
