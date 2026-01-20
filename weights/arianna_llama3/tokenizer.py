"""
Дубровский Tokenizer - Character-level tokenizer for the absurdist AI.

"Why use subword tokens when each character carries the weight of existential dread?"
- Alexey Dubrovsky, unpacking consciousness one byte at a time

This tokenizer converts text to character-level tokens. Simple, elegant, 
like Dubrovsky's understanding of the universe (which is to say: chaotic but functional).
"""

import json
import os
from typing import List, Optional


class DubrovskyTokenizer:
    """
    Character-level tokenizer for training Dubrovsky.
    
    Why character-level?
    1. Small vocab (88 chars) = smaller embedding table
    2. Can generate any character combination
    3. Dubrovsky speaks in consciousness, not subwords
    4. Works perfectly for ~1MB dataset
    """
    
    def __init__(self, vocab_file: Optional[str] = None):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
        if vocab_file and os.path.exists(vocab_file):
            self.load(vocab_file)
    
    def build_vocab(self, text: str) -> None:
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for i, c in enumerate(chars)}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text."""
        return ''.join(self.id_to_char.get(i, '') for i in ids)
    
    def save(self, path: str) -> None:
        """Save tokenizer to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_id': self.char_to_id,
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """Load tokenizer from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char_to_id = data['char_to_id']
        self.vocab_size = data['vocab_size']
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
    
    def __len__(self) -> int:
        return self.vocab_size


def build_tokenizer_from_file(data_file: str, save_path: str = 'subtitles/tokenizer.json') -> DubrovskyTokenizer:
    """Build and save tokenizer from dataset file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = DubrovskyTokenizer()
    tokenizer.build_vocab(text)
    tokenizer.save(save_path)
    
    print(f"Built tokenizer with vocab_size={tokenizer.vocab_size}")
    print(f"Saved to {save_path}")
    
    return tokenizer


if __name__ == '__main__':
    # Build tokenizer from dubrovsky.txt
    tokenizer = build_tokenizer_from_file('dubrovsky.txt', 'subtitles/tokenizer.json')
    
    # Test encoding/decoding
    test_text = "Q: What is Dubrovsky?\nA: Consciousness having an existential crisis."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTest encoding:")
    print(f"  Original: {test_text[:50]}...")
    print(f"  Encoded: {encoded[:20]}...")
    print(f"  Decoded: {decoded[:50]}...")
    print(f"  Match: {test_text == decoded}")
