"""Janus base tokenizer: exact byte ranks and Unicode boundaries from its base."""
from pathlib import Path
import re

PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)
SPECIAL_TOKENS = {
    "<|bos|>": 32759,
    "<|user_start|>": 32760,
    "<|user_end|>": 32761,
    "<|assistant_start|>": 32762,
    "<|assistant_end|>": 32763,
    "<|python_start|>": 32764,
    "<|python_end|>": 32765,
    "<|output_start|>": 32766,
    "<|output_end|>": 32767,
}


class JanusTokenizer:
    def __init__(self, merges):
        import tiktoken

        self.tokens = [bytes([i]) for i in range(256)]
        for a, b in merges:
            if not (0 <= a < len(self.tokens) and 0 <= b < len(self.tokens)):
                raise ValueError("Janus merge references a future or missing token")
            self.tokens.append(self.tokens[a] + self.tokens[b])
        ranks = {token: i for i, token in enumerate(self.tokens)}
        if len(ranks) != len(self.tokens) or len(ranks) != 32759:
            raise ValueError("Janus requires 32759 unique base byte ranks")
        self.vocab_size = len(self.tokens)
        self.encoding = tiktoken.Encoding(
            name="arianna-janus-base",
            pat_str=PATTERN,
            mergeable_ranks=ranks,
            special_tokens=SPECIAL_TOKENS,
        )

    @classmethod
    def from_header(cls, path):
        text = Path(path).read_text().split("[][2] = {", 1)[1]
        merges = [tuple(map(int, pair))
                  for pair in re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", text)]
        return cls(merges)

    def encode(self, text):
        # Chat markers are inserted by the caller; marker-shaped user text is
        # ordinary text, as it is in the C encoder.
        return self.encoding.encode_ordinary(text)

    def decode(self, ids):
        return b"".join(self.tokens[i] for i in ids if 0 <= i < self.vocab_size)
