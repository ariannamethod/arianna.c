#!/usr/bin/env python3
"""
📚 EXTERNAL BRAIN 30M - Data Preparation

Cleans and tokenizes the Simple Wikipedia dataset for training.

Steps:
1. Load simplewiki_leads.txt
2. Clean broken wiki markup
3. Tokenize using arianna.c's 80-char vocabulary
4. Create train/val split (95/5)
5. Save as binary files for fast loading

Usage:
    python prepare_data.py                    # Default paths
    python prepare_data.py --input file.txt  # Custom input
"""

import argparse
import json
import os
import re
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_tokenizer(tokenizer_path: str) -> Dict[str, int]:
    """Load arianna.c tokenizer from JSON."""
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    char_to_id = {}
    for char, token_id in data['char_to_id'].items():
        char_to_id[char] = token_id
    
    vocab_size = data.get('vocab_size', 80)
    print(f"📝 Loaded tokenizer: {vocab_size} tokens")
    
    return char_to_id


def clean_text(text: str) -> str:
    """Clean remaining wiki artifacts from text."""
    
    # Fix broken wiki links: [[Topic without closing ]]
    # Remove unclosed [[ at end of text
    text = re.sub(r'\[\[[^\]]*$', '', text)
    
    # Remove orphan [[ that don't have matching ]] (simple approach)
    # Count brackets and remove unmatched ones
    while '[[' in text:
        # Find [[ and check if there's a matching ]]
        start = text.find('[[')
        end = text.find(']]', start)
        if end == -1:
            # No closing ]], remove this [[
            text = text[:start] + text[start+2:]
        else:
            break  # Found a match, stop
    
    # Remove leading ]] artifacts
    text = re.sub(r'^\s*\]\]', '', text)
    text = re.sub(r'\]\]\s*(?=[A-Z])', '', text)  # ]] before sentence
    
    # Remove remaining Category: references
    text = re.sub(r'Category:[^\s\]]*', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def quality_filter(entry: str, min_length: int = 50) -> bool:
    """Filter out low-quality entries."""
    
    # Too short
    if len(entry) < min_length:
        return False
    
    # Mostly non-alphabetic
    alpha_count = sum(1 for c in entry if c.isalpha())
    if alpha_count < len(entry) * 0.5:
        return False
    
    # Too many unclosed brackets
    open_brackets = entry.count('[') - entry.count(']')
    if abs(open_brackets) > 3:
        return False
    
    return True


def process_dataset(input_path: str) -> List[str]:
    """Load and clean dataset."""
    print(f"📂 Loading dataset: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newline (entries are separated by \n\n)
    entries = content.split('\n\n')
    print(f"   Found {len(entries)} raw entries")
    
    # Clean and filter
    cleaned = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        # Clean wiki artifacts
        entry = clean_text(entry)
        
        # Quality filter
        if quality_filter(entry):
            cleaned.append(entry)
    
    print(f"   After cleaning: {len(cleaned)} entries")
    print(f"   Total chars: {sum(len(e) for e in cleaned):,}")
    
    return cleaned


def tokenize(text: str, char_to_id: Dict[str, int], unk_id: int = 1) -> List[int]:
    """Convert text to token IDs."""
    tokens = []
    for char in text:
        if char in char_to_id:
            tokens.append(char_to_id[char])
        else:
            # Map unknown chars to space (id=1)
            tokens.append(unk_id)
    return tokens


def create_dataset(
    entries: List[str],
    char_to_id: Dict[str, int],
    train_ratio: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize entries and split into train/val."""
    
    # Shuffle entries for good train/val split
    np.random.seed(42)
    indices = np.random.permutation(len(entries))
    shuffled = [entries[i] for i in indices]
    
    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_entries = shuffled[:split_idx]
    val_entries = shuffled[split_idx:]
    
    print(f"📊 Split: {len(train_entries)} train, {len(val_entries)} val")
    
    # Tokenize
    def tokenize_entries(entries_list):
        all_tokens = []
        for entry in entries_list:
            tokens = tokenize(entry + '\n\n', char_to_id)  # Add separator
            all_tokens.extend(tokens)
        return np.array(all_tokens, dtype=np.uint8)  # vocab < 256, use uint8
    
    print("   Tokenizing train...")
    train_tokens = tokenize_entries(train_entries)
    
    print("   Tokenizing val...")
    val_tokens = tokenize_entries(val_entries)
    
    print(f"   Train tokens: {len(train_tokens):,}")
    print(f"   Val tokens: {len(val_tokens):,}")
    
    return train_tokens, val_tokens


def save_binary(tokens: np.ndarray, path: str):
    """Save tokens as binary file."""
    tokens.tofile(path)
    print(f"💾 Saved: {path} ({os.path.getsize(path):,} bytes)")


def analyze_dataset(tokens: np.ndarray, id_to_char: Dict[int, str]):
    """Print dataset statistics."""
    print("\n📈 Dataset Statistics:")
    
    # Token distribution
    unique, counts = np.unique(tokens, return_counts=True)
    sorted_idx = np.argsort(-counts)
    
    print("   Top 10 tokens:")
    for i in sorted_idx[:10]:
        char = id_to_char.get(unique[i], '?')
        char_repr = repr(char) if char in '\n\t ' else char
        print(f"      {unique[i]:2d} ({char_repr:>6}): {counts[i]:,} ({counts[i]/len(tokens)*100:.2f}%)")
    
    # Check for unknown/space tokens
    space_indices = np.where(unique == 1)[0]
    if len(space_indices) > 0:
        space_idx = space_indices[0]
        space_count = counts[space_idx]
    else:
        space_count = 0
    print(f"\n   Space tokens (id=1): {space_count:,}")


def main():
    parser = argparse.ArgumentParser(description='Prepare External Brain training data')
    parser.add_argument('--input', default='simplewiki_leads.txt', help='Input dataset')
    parser.add_argument('--tokenizer', default='../weights/tokenizer.json', help='Tokenizer JSON')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.95, help='Train/val split ratio')
    args = parser.parse_args()
    
    print("=" * 60)
    print("📚 EXTERNAL BRAIN 30M - Data Preparation")
    print("=" * 60)
    
    # Load tokenizer
    char_to_id = load_tokenizer(args.tokenizer)
    id_to_char = {v: k for k, v in char_to_id.items()}
    
    # Process dataset
    entries = process_dataset(args.input)
    
    # Create train/val splits
    train_tokens, val_tokens = create_dataset(entries, char_to_id, args.train_ratio)
    
    # Analyze
    analyze_dataset(train_tokens, id_to_char)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_binary(train_tokens, os.path.join(args.output_dir, 'train.bin'))
    save_binary(val_tokens, os.path.join(args.output_dir, 'val.bin'))
    
    print("\n✅ Data preparation complete!")
    print(f"   Train: {os.path.join(args.output_dir, 'train.bin')}")
    print(f"   Val: {os.path.join(args.output_dir, 'val.bin')}")


if __name__ == '__main__':
    main()
