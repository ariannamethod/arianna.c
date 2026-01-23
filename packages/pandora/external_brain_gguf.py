#!/usr/bin/env python3
"""
External Brain Bridge for Arianna - TinyLlama GGUF Edition
Vocabulary extraction from TinyLlama-1.1B → mapped to Arianna's char-level tokens

Uses llama-cpp-python for inference, auto-downloads from HuggingFace if needed.

Usage:
    python external_brain_gguf.py "prompt text" [max_tokens]
    python external_brain_gguf.py "prompt text" 50 --tokens

Output: JSON with arianna_tokens ready for pandora_extract()
        Or simple COUNT:tok1,tok2,... format with --tokens
"""

import sys
import json
import os
from pathlib import Path

# Suppress warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore')

# TinyLlama GGUF config
HF_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
HF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  # Q4 for speed, Q5 for quality
WEIGHTS_DIR = Path("/Users/ataeff/Downloads/arianna.c/weights/tinyllama")
MODEL_PATH = WEIGHTS_DIR / "tinyllama-1.1b-chat-q4.gguf"

# Arianna's char-level vocabulary (84 chars) - same as GPT2 bridge
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


def text_to_arianna_tokens(text: str) -> list:
    """Convert text to Arianna's char-level token IDs"""
    tokens = []
    for char in text:
        if char in ARIANNA_VOCAB:
            tokens.append(ARIANNA_VOCAB[char])
    return tokens


def download_model():
    """Download TinyLlama GGUF from HuggingFace if not present"""
    if MODEL_PATH.exists():
        return MODEL_PATH

    print(f"[external_brain_gguf] Downloading TinyLlama from HuggingFace...", file=sys.stderr)
    print(f"[external_brain_gguf] Repo: {HF_REPO}", file=sys.stderr)
    print(f"[external_brain_gguf] File: {HF_FILENAME}", file=sys.stderr)

    try:
        from huggingface_hub import hf_hub_download

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILENAME,
            local_dir=WEIGHTS_DIR,
            local_dir_use_symlinks=False
        )

        # Rename to our standard name
        downloaded = Path(downloaded_path)
        if downloaded.exists() and downloaded != MODEL_PATH:
            downloaded.rename(MODEL_PATH)

        print(f"[external_brain_gguf] Downloaded to {MODEL_PATH}", file=sys.stderr)
        return MODEL_PATH

    except ImportError:
        print("[external_brain_gguf] ERROR: huggingface_hub not installed", file=sys.stderr)
        print("[external_brain_gguf] Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[external_brain_gguf] ERROR downloading: {e}", file=sys.stderr)
        sys.exit(1)


def load_tinyllama():
    """Load TinyLlama GGUF model"""
    from llama_cpp import Llama

    model_path = download_model()

    print(f"[external_brain_gguf] Loading TinyLlama from {model_path}...", file=sys.stderr)

    llm = Llama(
        model_path=str(model_path),
        n_ctx=512,       # Context window
        n_threads=4,     # CPU threads
        verbose=False    # Suppress llama.cpp output
    )

    return llm


def extract_vocabulary(prompt: str, max_tokens: int = 50) -> dict:
    """Generate from TinyLlama and extract vocabulary for Arianna"""

    llm = load_tinyllama()

    # TinyLlama chat format
    chat_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"

    print(f"[external_brain_gguf] Generating...", file=sys.stderr)

    output = llm(
        chat_prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repeat_penalty=1.2,
        stop=["<|user|>", "<|assistant|>", "\n\n"]
    )

    generated_text = output['choices'][0]['text'].strip()

    # Clean up - stop at first double newline or excessive length
    if '\n\n' in generated_text:
        generated_text = generated_text.split('\n\n')[0]

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
        "model": "tinyllama-1.1b-gguf",
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
        output_mode = "tokens"
    if '--download' in sys.argv:
        # Just download, don't generate
        download_model()
        print("Model ready.")
        return

    prompt = args[0] if args else "What is consciousness?"
    max_tokens = int(args[1]) if len(args) > 1 else 50

    result = extract_vocabulary(prompt, max_tokens)

    if output_mode == "tokens":
        # Simple format for C: COUNT:tok1,tok2,tok3,...
        tokens = result['arianna_tokens']
        print(f"{len(tokens)}:{','.join(str(t) for t in tokens)}")
    else:
        # Human-readable summary to stderr
        print("\n" + "=" * 60, file=sys.stderr)
        print("EXTERNAL BRAIN (TinyLlama GGUF) → ARIANNA", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"TinyLlama: \"{result['generated']}\"", file=sys.stderr)
        print(f"Arianna tokens: {result['token_count']}", file=sys.stderr)
        print(f"Words stolen: {result['word_count']}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        # JSON to stdout
        print(json.dumps(result))


if __name__ == "__main__":
    main()
