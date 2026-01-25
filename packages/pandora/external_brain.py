#!/usr/bin/env python3
"""
External Brain Bridge for Arianna
Vocabulary extraction from GPT2-30M → mapped to Arianna's char-level tokens

Uses local weights from arianna.c/weights/gpt2_30m/

Usage:
    python external_brain.py "prompt text" [max_tokens]

Output: JSON with arianna_tokens ready for pandora_extract()
"""

import sys
import json
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore')

# Path to GPT2-30M weights
GPT2_30M_PATH = "/Users/ataeff/Downloads/arianna.c/weights/gpt2_30m"

# Arianna's char-level vocabulary - loaded from tokenizer.json
from pathlib import Path
ARIANNA_VOCAB = None

def load_arianna_vocab():
    """Load vocabulary from tokenizer.json, with fallback to defaults"""
    global ARIANNA_VOCAB
    if ARIANNA_VOCAB is not None:
        return ARIANNA_VOCAB

    tokenizer_paths = [
        Path("/Users/ataeff/Downloads/arianna.c/weights/arianna_34m_tokenizer.json"),
        Path("/Users/ataeff/Downloads/arianna.c/weights/arianna_20m_tokenizer.json"),
        Path("/Users/ataeff/Downloads/arianna.c/weights/tokenizer_unified.json"),
    ]

    for tok_path in tokenizer_paths:
        if tok_path.exists():
            try:
                with open(tok_path) as f:
                    data = json.load(f)
                    if 'char_to_id' in data:
                        ARIANNA_VOCAB = data['char_to_id']
                        print(f"[external_brain] Loaded vocab ({len(ARIANNA_VOCAB)} tokens) from {tok_path.name}", file=sys.stderr)
                        return ARIANNA_VOCAB
            except Exception as e:
                print(f"[external_brain] Warning: failed to load {tok_path}: {e}", file=sys.stderr)

    # Fallback to hardcoded 84-token vocab (legacy)
    print("[external_brain] Warning: using fallback vocab (84 tokens)", file=sys.stderr)
    ARIANNA_VOCAB = {
        "\n": 0, " ": 1, "\"": 2, "%": 3, "'": 4, "(": 5, ")": 6, "*": 7,
        "+": 8, ",": 9, "-": 10, ".": 11, "/": 12, "0": 13, "1": 14, "2": 15,
        "3": 16, "4": 17, "5": 18, "6": 19, "7": 20, "8": 21, "9": 22, ":": 23,
        ";": 24, "?": 25, "A": 26, "B": 27, "C": 28, "D": 29, "E": 30, "F": 31,
        "G": 32, "H": 33, "I": 34, "J": 35, "K": 36, "L": 37, "M": 38, "N": 39,
        "O": 40, "P": 41, "Q": 42, "R": 43, "S": 44, "T": 45, "U": 46, "V": 47,
        "W": 48, "X": 49, "Y": 50, "Z": 51, "_": 52, "a": 53, "b": 54, "c": 55,
        "d": 56, "e": 57, "f": 58, "g": 59, "h": 60, "i": 61, "j": 62, "k": 63,
        "l": 64, "m": 65, "n": 66, "o": 67, "p": 68, "q": 69, "r": 70, "s": 71,
        "t": 72, "u": 73, "v": 74, "w": 75, "x": 76, "y": 77, "z": 78, "é": 79,
        "ï": 80, "ö": 81, "—": 82, "→": 83
    }
    return ARIANNA_VOCAB


def text_to_arianna_tokens(text: str) -> list:
    """Convert text to Arianna's char-level token IDs"""
    vocab = load_arianna_vocab()
    tokens = []
    for char in text:
        if char in vocab:
            tokens.append(vocab[char])
    return tokens


def load_gpt2_30m():
    """Load GPT2-30M from local weights"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    if os.path.exists(GPT2_30M_PATH):
        print(f"[external_brain] Loading GPT2-30M from {GPT2_30M_PATH}...", file=sys.stderr)
        tokenizer = GPT2Tokenizer.from_pretrained(GPT2_30M_PATH)
        model = GPT2LMHeadModel.from_pretrained(GPT2_30M_PATH)
    else:
        print(f"[external_brain] Local weights not found, using HuggingFace gpt2...", file=sys.stderr)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

    model.eval()
    return model, tokenizer


def extract_vocabulary(prompt: str, max_tokens: int = 50) -> dict:
    """Generate from GPT2-30M and extract vocabulary for Arianna"""
    import torch

    model, tokenizer = load_gpt2_30m()

    # Q&A format for better responses
    full_prompt = f"Q: {prompt}\nA:"
    input_ids = tokenizer.encode(full_prompt, return_tensors='pt')

    print(f"[external_brain] Generating...", file=sys.stderr)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = output[0][input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Clean up - stop at first newline or period+space
    for stop in ['\n', '. ', '.\n']:
        if stop in generated_text:
            generated_text = generated_text.split(stop)[0] + ('.' if stop.startswith('.') else '')
            break

    # Map to Arianna tokens
    arianna_tokens = text_to_arianna_tokens(generated_text)

    # Extract unique words
    words = generated_text.split()
    word_list = []
    seen = set()
    for word in words:
        clean = ''.join(c for c in word if c.isalpha()).lower()
        if clean and clean not in seen:
            seen.add(clean)
            tokens = text_to_arianna_tokens(clean)
            if tokens:
                word_list.append({"word": clean, "tokens": tokens})

    return {
        "model": "gpt2-30m",
        "prompt": prompt,
        "generated": generated_text,
        "arianna_tokens": arianna_tokens,
        "token_count": len(arianna_tokens),
        "words": word_list,
        "word_count": len(word_list)
    }


def main():
    # Parse args
    output_mode = "json"
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if '--tokens' in sys.argv:
        output_mode = "tokens"
    if '--quiet' in sys.argv:
        output_mode = "tokens"  # quiet = tokens only

    prompt = args[0] if args else "What is consciousness?"
    max_tokens = int(args[1]) if len(args) > 1 else 50

    result = extract_vocabulary(prompt, max_tokens)

    if output_mode == "tokens":
        # Simple format for C: COUNT:tok1,tok2,tok3,...
        # Easy to parse with sscanf
        tokens = result['arianna_tokens']
        print(f"{len(tokens)}:{','.join(str(t) for t in tokens)}")
    else:
        # Human-readable summary to stderr
        print("\n" + "=" * 60, file=sys.stderr)
        print("EXTERNAL BRAIN → ARIANNA", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"GPT2-30M: \"{result['generated']}\"", file=sys.stderr)
        print(f"Arianna tokens: {result['token_count']}", file=sys.stderr)
        print(f"Words stolen: {result['word_count']}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        # JSON to stdout
        print(json.dumps(result))


if __name__ == "__main__":
    main()
