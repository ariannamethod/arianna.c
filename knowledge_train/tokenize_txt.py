#!/usr/bin/env python3
"""
Simple tokenizer for pre-cleaned datasets.
No filtering - just tokenize txt -> bin.

Usage:
    python tokenize_txt.py external_brain_filtered.txt data_filtered
    python tokenize_txt.py external_brain_qa.txt data_qa
"""

import argparse
import json
import os
import numpy as np


def load_tokenizer(tokenizer_path: str) -> dict:
    """Load arianna.c tokenizer from JSON."""
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    char_to_id = {}
    for char, token_id in data['char_to_id'].items():
        char_to_id[char] = token_id

    print(f"Loaded tokenizer: {data.get('vocab_size', 80)} tokens")
    return char_to_id


def tokenize(text: str, char_to_id: dict, unk_id: int = 1) -> list:
    """Convert text to token IDs."""
    tokens = []
    for char in text:
        if char in char_to_id:
            tokens.append(char_to_id[char])
        else:
            tokens.append(unk_id)
    return tokens


def main():
    parser = argparse.ArgumentParser(description='Tokenize txt to bin (no filtering)')
    parser.add_argument('input', help='Input txt file')
    parser.add_argument('output_dir', help='Output directory for train.bin/val.bin')
    parser.add_argument('--tokenizer', default='../weights/tokenizer.json', help='Tokenizer JSON')
    parser.add_argument('--train-ratio', type=float, default=0.95, help='Train/val split')
    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}/")

    # Load tokenizer
    char_to_id = load_tokenizer(args.tokenizer)

    # Read input
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Input size: {len(text):,} bytes")

    # Tokenize entire text
    tokens = tokenize(text, char_to_id)
    tokens = np.array(tokens, dtype=np.uint8)

    print(f"Total tokens: {len(tokens):,}")

    # Split train/val
    np.random.seed(42)
    split_idx = int(len(tokens) * args.train_ratio)

    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"Train: {len(train_tokens):,} tokens")
    print(f"Val: {len(val_tokens):,} tokens")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, 'train.bin')
    val_path = os.path.join(args.output_dir, 'val.bin')

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"Saved: {train_path} ({os.path.getsize(train_path):,} bytes)")
    print(f"Saved: {val_path} ({os.path.getsize(val_path):,} bytes)")

    # Verify ratio
    total_bytes = len(train_tokens) + len(val_tokens)
    model_params = 29_583_872
    ratio = model_params / total_bytes
    print(f"\nParams/byte ratio: {ratio:.2f} (target: ~9-10)")


if __name__ == '__main__':
    main()
