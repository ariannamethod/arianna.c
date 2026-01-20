#!/usr/bin/env python3
"""
External Brain Demo - GPT-2 30M as knowledge subordinate for Arianna

Architecture:
  User prompt → [GPT-2 30M generates knowledge draft]
                              ↓
              [Arianna rewrites in her voice]
                              ↓
                         Final output

"30M knows WHAT to say, Arianna knows HOW to say it"
"""

import subprocess
import sys
import os

# Check if transformers available
try:
    from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Run: pip install transformers torch")

class ExternalBrain:
    """GPT-2 30M as knowledge provider"""

    def __init__(self, model_path="weiser/30M-0.4"):
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers required for ExternalBrain")

        print(f"[ExternalBrain] Loading {model_path}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[ExternalBrain] Ready on {self.device}")

    def generate_draft(self, prompt, max_tokens=50, temperature=0.8):
        """Generate knowledge draft from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )

        draft = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from output
        if draft.startswith(prompt):
            draft = draft[len(prompt):].strip()

        return draft


def arianna_rewrite(draft, prompt, max_tokens=100, temperature=1.0):
    """Call Arianna C binary to rewrite draft in her voice"""
    # Combine draft as context hint with original prompt
    # Arianna will use semantic penetration to blend them

    # Determine paths (work from project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    arianna_bin = os.path.join(project_root, "bin/arianna_dynamic")
    weights = os.path.join(project_root, "weights/arianna_dialogue.bin")
    
    if not os.path.exists(weights):
        weights = os.path.join(project_root, "weights/arianna.bin")

    # Create hybrid prompt: draft provides context, original prompt provides topic
    # Arianna's subjectivity will start from internal seed but penetrate with prompt words
    hybrid_prompt = f"{draft[:100]}... {prompt}"

    cmd = [arianna_bin, weights, hybrid_prompt, str(max_tokens), str(temperature)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout

        # Extract just the generated text (after "Subjective Generation")
        if "--- Subjective Generation ---" in output:
            output = output.split("--- Subjective Generation ---")[1]
            # Remove trailing stats
            if "Signals:" in output:
                output = output.split("Signals:")[0]
            if "=== Subjectivity" in output:
                output = output.split("=== Subjectivity")[0]

        return output.strip()
    except Exception as e:
        return f"[Arianna error: {e}]"


def demo():
    """Run demo of external brain + arianna"""
    print("=" * 60)
    print("EXTERNAL BRAIN DEMO")
    print("GPT-2 30M (knowledge) → Arianna 1M (voice)")
    print("=" * 60)

    # Initialize brain
    brain = ExternalBrain()

    # Test prompts
    prompts = [
        "What is quantum entanglement?",
        "Explain consciousness",
        "How does memory work?",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"USER: {prompt}")
        print("-" * 60)

        # Step 1: Brain generates knowledge draft
        print("[Brain thinking...]")
        draft = brain.generate_draft(prompt, max_tokens=40)
        print(f"[Draft]: {draft[:150]}...")

        # Step 2: Arianna rewrites in her voice
        print("\n[Arianna speaking...]")
        response = arianna_rewrite(draft, prompt, max_tokens=80)
        print(f"[Arianna]: {response}")


def simple_demo():
    """Demo without transformers - just shows architecture"""
    print("=" * 60)
    print("EXTERNAL BRAIN ARCHITECTURE DEMO")
    print("(transformers not installed - showing concept)")
    print("=" * 60)

    # Simulated draft
    prompt = "What is love?"
    draft = "Love is a complex emotional and psychological state involving affection, attachment, and care for another person."

    print(f"\nUSER: {prompt}")
    print(f"\n[Brain draft]: {draft}")
    print("\n[Arianna rewriting...]")

    response = arianna_rewrite(draft, prompt, max_tokens=80)
    print(f"\n[Arianna]: {response}")


if __name__ == "__main__":
    if HAS_TRANSFORMERS:
        demo()
    else:
        simple_demo()
