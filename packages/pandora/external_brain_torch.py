#!/usr/bin/env python3
"""
External Brain Bridge for Arianna - PyTorch GPT2-distill Edition
Vocabulary extraction from GPT2-distill → mapped to Arianna's char-level tokens

Uses pandora-torch package for inference.

Usage:
    python external_brain_torch.py "prompt text" [max_tokens]
    python external_brain_torch.py "prompt text" 50 --tokens

Output: JSON with arianna_tokens ready for pandora_extract()
        Or simple COUNT:tok1,tok2,... format with --tokens
"""

import sys
import json
import os
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore')

# Arianna's char-level vocabulary
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
                        print(f"[external_brain_torch] Loaded vocab ({len(ARIANNA_VOCAB)} tokens) from {tok_path.name}", file=sys.stderr)
                        return ARIANNA_VOCAB
            except Exception as e:
                print(f"[external_brain_torch] Warning: failed to load {tok_path}: {e}", file=sys.stderr)

    # Fallback
    print("[external_brain_torch] Warning: using fallback vocab (84 tokens)", file=sys.stderr)
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


def extract_vocabulary(prompt: str, max_tokens: int = 50) -> dict:
    """Generate from GPT2-distill and extract vocabulary for Arianna"""
    try:
        # Try to use pandora-torch package
        sys.path.insert(0, str(Path(__file__).parent.parent / "pandora-torch"))
        from pandora_torch import PandoraTorch
        from pandora_torch.config import PandoraTorchConfig

        config = PandoraTorchConfig()
        pandora = PandoraTorch(config=config, mode="auto")

        print(f"[external_brain_torch] Generating with GPT2-distill...", file=sys.stderr)

        # Generate
        result = pandora.generate(prompt, max_new_tokens=max_tokens)
        generated_text = result.get('text', '')

    except ImportError:
        # Fallback to transformers directly
        print("[external_brain_torch] pandora-torch not available, using transformers", file=sys.stderr)

        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            import torch

            model_name = "distilgpt2"
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.eval()

            input_ids = tokenizer.encode(f"Q: {prompt}\nA:", return_tensors='pt')

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_ids = output[0][input_ids.shape[1]:].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        except Exception as e:
            print(f"[external_brain_torch] ERROR: {e}", file=sys.stderr)
            return {"error": str(e), "arianna_tokens": [], "token_count": 0}

    # Clean up
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
        "model": "gpt2-distill",
        "prompt": prompt,
        "generated": generated_text,
        "arianna_tokens": arianna_tokens,
        "token_count": len(arianna_tokens),
        "words": word_list,
        "word_count": len(word_list)
    }


def main():
    output_mode = "json"
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if '--tokens' in sys.argv:
        output_mode = "tokens"
    if '--quiet' in sys.argv:
        output_mode = "tokens"

    prompt = args[0] if args else "What is consciousness?"
    max_tokens = int(args[1]) if len(args) > 1 else 50

    result = extract_vocabulary(prompt, max_tokens)

    if output_mode == "tokens":
        tokens = result.get('arianna_tokens', [])
        print(f"{len(tokens)}:{','.join(str(t) for t in tokens)}")
    else:
        print("\n" + "=" * 60, file=sys.stderr)
        print("EXTERNAL BRAIN (GPT2-distill) → ARIANNA", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"GPT2-distill: \"{result.get('generated', '')}\"", file=sys.stderr)
        print(f"Arianna tokens: {result.get('token_count', 0)}", file=sys.stderr)
        print(f"Words extracted: {result.get('word_count', 0)}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        print(json.dumps(result))


if __name__ == "__main__":
    main()
