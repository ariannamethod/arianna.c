#!/usr/bin/env python3
"""
📚 EXTERNAL BRAIN 30M - Dataset B: Q&A Format

Converts simplewiki_leads.txt to structured Q&A format:
    Q: What is Paris?
    A: Paris is the capital of France.

Target size: ~12MB to maintain data/params ratio ≤0.5

Benefits of Q&A format:
- Clearer pattern for model to learn
- Less noise than raw paragraphs
- Pandora learns both words AND response pattern

Usage:
    python prepare_data_qa.py                    # Default paths
    python prepare_data_qa.py --target-mb 10.0  # Custom target size
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


# Minimum sentence length for quality filtering
MIN_SENTENCE_LENGTH = 20


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


def extract_first_sentence(text: str) -> str:
    """Extract first complete sentence from text."""
    # Match sentence ending with . ! or ? followed by space and capital or end
    # Require minimum length to avoid truncated garbage
    match = re.match(r'^(.{' + str(MIN_SENTENCE_LENGTH) + r',}?[.!?])(?:\s+[A-Z]|\s*$)', text)
    if match:
        return match.group(1)
    
    # Fallback: take up to first period if it's after minimum position
    period_pos = text.find('.')
    if period_pos >= MIN_SENTENCE_LENGTH:
        return text[:period_pos + 1]
    
    # If first sentence is too short, try to get more
    second_period = text.find('.', period_pos + 1) if period_pos > 0 else -1
    if second_period >= MIN_SENTENCE_LENGTH:
        return text[:second_period + 1]
    
    return text


def determine_question_type(title: str, definition: str) -> str:
    """Determine appropriate question type based on content."""
    def_lower = definition.lower()
    
    # Person patterns
    if re.search(r'\bwas\s+(a|an)\s+(person|man|woman|writer|artist|scientist|king|queen|emperor|president|politician|actor|singer|musician|composer|painter|philosopher|mathematician|physicist)', def_lower):
        return 'who'
    
    # Place patterns
    if re.search(r'\bis\s+(a|an|the)\s+(city|town|village|country|capital|region|province|state|island|mountain|river|lake|ocean|sea)', def_lower):
        return 'where'
    
    # Event/time patterns
    if re.search(r'\bwas\s+(a|an)\s+(war|battle|event|revolution|treaty|agreement)', def_lower):
        return 'when'
    
    # Default to "what"
    return 'what'


def create_qa_pair(title: str, definition: str, max_answer_len: int = 150) -> Optional[str]:
    """Create Q&A pair from title and definition."""
    
    # Clean definition
    definition = clean_text(definition)
    
    # Skip if definition is too short overall
    if len(definition) < 30:
        return None
    
    # Get first sentence only (cleaner answer)
    answer = extract_first_sentence(definition)
    
    # Quality checks - answer must be substantial
    if len(answer) < 30:  # Minimum 30 chars for meaningful answer
        return None
    
    if len(answer) > max_answer_len:
        return None
    
    # Must contain the title or related content (relevance check)
    title_words = set(title.lower().split())
    answer_lower = answer.lower()
    if not any(word in answer_lower for word in title_words if len(word) > 2):
        # Answer doesn't mention the topic - skip
        return None
    
    # Must end with punctuation
    if not answer[-1] in '.!?':
        return None
    
    # Check alphabetic content
    alpha_count = sum(1 for c in answer if c.isalpha())
    if alpha_count < len(answer) * 0.6:
        return None
    
    # Determine question type
    q_type = determine_question_type(title, answer)
    
    # Create question based on type
    if q_type == 'who':
        question = f"Who was {title}?"
    elif q_type == 'where':
        question = f"Where is {title}?"
    elif q_type == 'when':
        question = f"When was {title}?"
    else:
        question = f"What is {title}?"
    
    # Format Q&A
    qa = f"Q: {question}\nA: {answer}"
    
    return qa


def process_dataset(
    input_path: str,
    max_answer_len: int = 150,
    target_size_mb: float = 12.0
) -> List[str]:
    """Load and convert dataset to Q&A format."""
    print(f"📂 Loading dataset: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newline
    raw_entries = content.split('\n\n')
    print(f"   Found {len(raw_entries)} raw entries")
    
    # Convert to Q&A pairs
    qa_pairs = []
    skipped = {'no_title': 0, 'no_qa': 0, 'too_long': 0}
    
    for raw_entry in raw_entries:
        raw_entry = raw_entry.strip()
        if not raw_entry:
            continue
        
        title, definition = parse_entry(raw_entry)
        if not title or not definition:
            skipped['no_title'] += 1
            continue
        
        qa = create_qa_pair(title, definition, max_answer_len)
        if qa is None:
            skipped['no_qa'] += 1
            continue
        
        qa_pairs.append({
            'qa': qa,
            'length': len(qa),
            'title': title
        })
    
    print(f"   Converted to Q&A: {len(qa_pairs)} pairs")
    print(f"   Skipped: {sum(skipped.values())} (no_title: {skipped['no_title']}, quality: {skipped['no_qa']})")
    
    # Sort by length (shorter first for diversity)
    qa_pairs.sort(key=lambda x: x['length'])
    
    # Select up to target size
    target_bytes = int(target_size_mb * 1024 * 1024)
    selected = []
    total_size = 0
    
    for entry in qa_pairs:
        entry_size = len(entry['qa']) + 2  # +2 for \n\n separator
        if total_size + entry_size > target_bytes:
            break
        selected.append(entry['qa'])
        total_size += entry_size
    
    print(f"   Selected {len(selected)} Q&A pairs ({total_size/1e6:.2f} MB)")
    
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
    """Print sample Q&A pairs."""
    print(f"\n📋 Sample Q&A pairs (first {n}):")
    for i, entry in enumerate(entries[:n]):
        lines = entry.split('\n')
        print(f"\n   {i+1}.")
        for line in lines:
            print(f"      {line}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Q&A format dataset')
    parser.add_argument('--input', default='simplewiki_leads.txt', help='Input dataset')
    parser.add_argument('--tokenizer', default='../weights/tokenizer.json', help='Tokenizer JSON')
    parser.add_argument('--output-dir', default='data_qa', help='Output directory')
    parser.add_argument('--max-answer-len', type=int, default=150, help='Max answer length in chars')
    parser.add_argument('--target-mb', type=float, default=12.0, help='Target dataset size in MB')
    parser.add_argument('--train-ratio', type=float, default=0.95, help='Train/val split ratio')
    args = parser.parse_args()
    
    print("=" * 60)
    print("📚 EXTERNAL BRAIN - Dataset B: Q&A Format")
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
    entries = process_dataset(args.input, args.max_answer_len, args.target_mb)
    
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
        'dataset_type': 'qa_format',
        'format': 'Q: {question}\\nA: {answer}',
        'max_answer_len': args.max_answer_len,
        'num_qa_pairs': len(entries),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'total_bytes': total_data,
        'model_params': MODEL_PARAMS,
        'data_params_ratio': actual_ratio,
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset B preparation complete!")
    print(f"   Output: {args.output_dir}/")
    print(f"   Train: train.bin")
    print(f"   Val: val.bin")
    print(f"   Metadata: metadata.json")


if __name__ == '__main__':
    main()
