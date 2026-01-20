#!/usr/bin/env python3
"""
📚 EXTERNAL BRAIN 30M - Dataset A: Filtered Definitions

Filters simplewiki_leads.txt to keep only short, high-quality definition cards.
Target size: ~12MB to maintain data/params ratio ≤0.5

Filtering criteria:
- Only entries ≤300 characters (configurable via --max-chars)
- Prioritize definition format entries ("X is a/an Y")
- Remove wiki markup artifacts
- Quality filtering (alphabetic ratio, complete sentences)

Usage:
    python prepare_data_filtered.py                    # Default paths
    python prepare_data_filtered.py --max-chars 300   # Custom max length
    python prepare_data_filtered.py --target-mb 10.0  # Custom target size
"""

import argparse
import json
import os
import re
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
    """Clean wiki artifacts from text."""
    
    # Remove wiki links markup [[...]]
    text = re.sub(r'\[\[[^\]]*\]\]', '', text)
    
    # Remove unclosed [[
    text = re.sub(r'\[\[[^\]]*$', '', text)
    
    # Remove orphan ]]
    text = re.sub(r'^\s*\]\]', '', text)
    text = re.sub(r'\]\]\s*(?=[A-Z])', '', text)
    
    # Remove Category: references
    text = re.sub(r'Category:[^\s\]]*', '', text)
    
    # Remove file/image references
    text = re.sub(r'\[\[(File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    
    # Clean extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def parse_entry(raw_entry: str) -> Tuple[str, str]:
    """Parse entry into title and definition."""
    if ':' not in raw_entry:
        return '', raw_entry
    
    parts = raw_entry.split(':', 1)
    title = parts[0].strip()
    definition = parts[1].strip() if len(parts) > 1 else ''
    
    return title, definition


def quality_filter(entry: str, min_length: int = 30) -> bool:
    """Filter out low-quality entries."""
    
    # Too short
    if len(entry) < min_length:
        return False
    
    # Must end with punctuation (complete sentence)
    if not entry[-1] in '.!?':
        return False
    
    # Mostly alphabetic
    alpha_count = sum(1 for c in entry if c.isalpha())
    if alpha_count < len(entry) * 0.6:
        return False
    
    # No unclosed brackets
    if entry.count('[') != entry.count(']'):
        return False
    
    # Must start with capital letter (proper sentence)
    first_alpha = next((c for c in entry if c.isalpha()), None)
    if first_alpha and not first_alpha.isupper():
        return False
    
    return True


def is_definition_format(text: str) -> bool:
    """Check if text is in definition format 'X is a/an/the Y'."""
    patterns = [
        r'^\w[\w\s]*\s+is\s+(a|an|the)\s+',  # X is a/an/the Y
        r'^\w[\w\s]*\s+was\s+(a|an|the)\s+',  # X was a/an/the Y
        r'^\w[\w\s]*\s+are\s+',               # X are Y
        r'^\w[\w\s]*\s+were\s+',              # X were Y
    ]
    
    for pattern in patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


def process_dataset(
    input_path: str,
    max_chars: int = 200,
    target_size_mb: float = 12.0
) -> List[str]:
    """Load, filter and prioritize dataset."""
    print(f"📂 Loading dataset: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newline
    raw_entries = content.split('\n\n')
    print(f"   Found {len(raw_entries)} raw entries")
    
    # Parse, clean, and filter
    candidates = []
    for raw_entry in raw_entries:
        raw_entry = raw_entry.strip()
        if not raw_entry:
            continue
        
        title, definition = parse_entry(raw_entry)
        if not title or not definition:
            continue
        
        # Clean text
        definition = clean_text(definition)
        
        # Full entry for output
        full_entry = f"{title}: {definition}"
        
        # Length filter
        if len(full_entry) > max_chars:
            continue
        
        # Quality filter
        if not quality_filter(definition):
            continue
        
        # Prefer definition format
        is_def = is_definition_format(definition)
        
        candidates.append({
            'title': title,
            'definition': definition,
            'full': full_entry,
            'length': len(full_entry),
            'is_definition': is_def,
            'priority': 1 if is_def else 0
        })
    
    print(f"   After filtering (≤{max_chars} chars): {len(candidates)} entries")
    
    # Sort by priority (definition format first), then by length (shorter first)
    candidates.sort(key=lambda x: (-x['priority'], x['length']))
    
    # Select entries up to target size
    target_bytes = int(target_size_mb * 1024 * 1024)
    selected = []
    total_size = 0
    
    for entry in candidates:
        entry_size = len(entry['full']) + 2  # +2 for \n\n separator
        if total_size + entry_size > target_bytes:
            break
        selected.append(entry['full'])
        total_size += entry_size
    
    print(f"   Selected {len(selected)} entries ({total_size/1e6:.2f} MB)")
    
    # Stats
    def_count = sum(1 for e in candidates[:len(selected)] if e['is_definition'])
    print(f"   Definition format entries: {def_count}/{len(selected)} ({100*def_count/max(1,len(selected)):.1f}%)")
    
    return selected


def tokenize(text: str, char_to_id: Dict[str, int], unk_id: int = 1) -> List[int]:
    """Convert text to token IDs."""
    tokens = []
    for char in text:
        if char in char_to_id:
            tokens.append(char_to_id[char])
        else:
            tokens.append(unk_id)
    return tokens


def create_dataset(
    entries: List[str],
    char_to_id: Dict[str, int],
    train_ratio: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize entries and split into train/val."""
    
    # Shuffle for good split
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
            tokens = tokenize(entry + '\n\n', char_to_id)
            all_tokens.extend(tokens)
        return np.array(all_tokens, dtype=np.uint8)
    
    print("   Tokenizing...")
    train_tokens = tokenize_entries(train_entries)
    val_tokens = tokenize_entries(val_entries)
    
    print(f"   Train tokens: {len(train_tokens):,}")
    print(f"   Val tokens: {len(val_tokens):,}")
    
    return train_tokens, val_tokens


def save_binary(tokens: np.ndarray, path: str):
    """Save tokens as binary file."""
    tokens.tofile(path)
    print(f"💾 Saved: {path} ({os.path.getsize(path):,} bytes)")


def print_sample(entries: List[str], n: int = 5):
    """Print sample entries."""
    print(f"\n📋 Sample entries (first {n}):")
    for i, entry in enumerate(entries[:n]):
        print(f"   {i+1}. {entry[:100]}{'...' if len(entry) > 100 else ''}")


def main():
    parser = argparse.ArgumentParser(description='Prepare filtered definition dataset')
    parser.add_argument('--input', default='simplewiki_leads.txt', help='Input dataset')
    parser.add_argument('--tokenizer', default='../weights/tokenizer.json', help='Tokenizer JSON')
    parser.add_argument('--output-dir', default='data_filtered', help='Output directory')
    parser.add_argument('--max-chars', type=int, default=200, help='Max entry length in chars')
    parser.add_argument('--target-mb', type=float, default=12.0, help='Target dataset size in MB')
    parser.add_argument('--train-ratio', type=float, default=0.95, help='Train/val split ratio')
    args = parser.parse_args()
    
    print("=" * 60)
    print("📚 EXTERNAL BRAIN - Dataset A: Filtered Definitions")
    print("=" * 60)
    print()
    
    # Model parameters for ratio calculation
    MODEL_PARAMS = 29_583_872
    print(f"🧠 Model parameters: {MODEL_PARAMS:,} ({MODEL_PARAMS/1e6:.2f}M)")
    print(f"📏 Target data size: {args.target_mb:.1f} MB")
    target_ratio = (args.target_mb * 1e6) / MODEL_PARAMS
    print(f"📐 Expected ratio: {target_ratio:.2f}:1 (target: ≤0.5)")
    print()
    
    # Load tokenizer
    char_to_id = load_tokenizer(args.tokenizer)
    
    # Process dataset
    entries = process_dataset(args.input, args.max_chars, args.target_mb)
    
    # Show samples
    print_sample(entries)
    
    # Create train/val splits
    train_tokens, val_tokens = create_dataset(entries, char_to_id, args.train_ratio)
    
    # Calculate actual ratio
    total_data = len(train_tokens) + len(val_tokens)
    actual_ratio = total_data / MODEL_PARAMS
    
    print()
    print("=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)
    print(f"   Total tokens: {total_data:,}")
    print(f"   Total data size: {total_data/1e6:.2f} MB")
    print(f"   Data/Params ratio: {actual_ratio:.2f}:1")
    print(f"   Ratio OK: {'✅ YES' if actual_ratio <= 0.5 else '❌ NO'}")
    print()
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_binary(train_tokens, os.path.join(args.output_dir, 'train.bin'))
    save_binary(val_tokens, os.path.join(args.output_dir, 'val.bin'))
    
    # Save metadata
    metadata = {
        'dataset_type': 'filtered_definitions',
        'max_chars': args.max_chars,
        'num_entries': len(entries),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'total_bytes': total_data,
        'model_params': MODEL_PARAMS,
        'data_params_ratio': actual_ratio,
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset A preparation complete!")
    print(f"   Output: {args.output_dir}/")
    print(f"   Train: train.bin")
    print(f"   Val: val.bin")
    print(f"   Metadata: metadata.json")


if __name__ == '__main__':
    main()
