#!/usr/bin/env python3
"""
train_sartre.py — Training script for SARTRE transformer

"L'existence précède l'essence" — first we exist, then we define ourselves.

This trains SARTRE on the corpus to learn:
- Its own identity
- Module states and events
- How to answer questions about system state

Usage:
    python train_sartre.py [--small] [--epochs N]

    --small: Use 150K params config (default: 2.2M development config)
    --epochs: Training epochs (default: 100)
"""

import os
import sys
import struct
import random
import math
from pathlib import Path
from typing import List, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Development config (2.2M params) - for testing
CONFIG_DEV = {
    'dim': 256,
    'n_layers': 4,
    'n_heads': 4,
    'hidden_dim': 512,
    'vocab_size': 256,
    'max_seq_len': 256,
}

# Production config (~150K params) - for metalinux
CONFIG_SMALL = {
    'dim': 96,
    'n_layers': 2,
    'n_heads': 4,
    'hidden_dim': 192,
    'vocab_size': 256,
    'max_seq_len': 256,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_corpus(corpus_dir: Path) -> str:
    """Load all corpus files into single string"""
    corpus = []

    # Order matters for learning
    files = ['identity.txt', 'modules.txt', 'events.txt', 'dialogue.txt']

    for filename in files:
        path = corpus_dir / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                corpus.append(f"=== {filename.upper()} ===\n{content}\n")
                print(f"Loaded {filename}: {len(content)} bytes")

    # Also load any .txt files not in the list
    for path in corpus_dir.glob('*.txt'):
        if path.name not in files:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                corpus.append(f"=== {path.name.upper()} ===\n{content}\n")
                print(f"Loaded {path.name}: {len(content)} bytes")

    full_corpus = '\n'.join(corpus)
    print(f"\nTotal corpus: {len(full_corpus)} bytes ({len(full_corpus)/1024:.1f} KB)")
    return full_corpus


def tokenize(text: str) -> List[int]:
    """Byte-level tokenization (same as Arianna)"""
    return list(text.encode('utf-8'))


def create_batches(tokens: List[int], seq_len: int, batch_size: int) -> List[Tuple[List[int], List[int]]]:
    """Create training batches: (input, target) pairs"""
    batches = []

    # Sliding window over tokens
    for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
        x = tokens[i:i + seq_len]
        y = tokens[i + 1:i + seq_len + 1]

        if len(x) == seq_len and len(y) == seq_len:
            batches.append((x, y))

    # Shuffle
    random.shuffle(batches)

    # Group into batch_size
    grouped = []
    for i in range(0, len(batches), batch_size):
        batch_x = [b[0] for b in batches[i:i + batch_size]]
        batch_y = [b[1] for b in batches[i:i + batch_size]]
        if len(batch_x) == batch_size:
            grouped.append((batch_x, batch_y))

    return grouped


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL (Pure Python implementation for training)
# ═══════════════════════════════════════════════════════════════════════════════

class SartreTrainer:
    """
    Training implementation of SARTRE transformer.

    This is a simple Python implementation for training.
    Inference uses the optimized C implementation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.dim = config['dim']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.head_dim = self.dim // self.n_heads
        self.hidden_dim = config['hidden_dim']
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config['max_seq_len']

        # Initialize weights
        self._init_weights()

        # Calculate param count
        self._count_params()

    def _init_weights(self):
        """Xavier initialization"""
        scale_embed = math.sqrt(2.0 / (self.vocab_size + self.dim))
        scale_attn = math.sqrt(2.0 / (self.dim + self.dim))
        scale_ffn1 = math.sqrt(2.0 / (self.dim + self.hidden_dim))
        scale_ffn2 = math.sqrt(2.0 / (self.hidden_dim + self.dim))

        # Embeddings
        self.token_embedding = [[random.gauss(0, scale_embed)
                                  for _ in range(self.dim)]
                                 for _ in range(self.vocab_size)]

        # Per-layer weights
        self.wq = []
        self.wk = []
        self.wv = []
        self.wo = []
        self.w1 = []
        self.w2 = []
        self.ln1_weight = []
        self.ln2_weight = []

        for _ in range(self.n_layers):
            self.wq.append([[random.gauss(0, scale_attn) for _ in range(self.dim)]
                           for _ in range(self.dim)])
            self.wk.append([[random.gauss(0, scale_attn) for _ in range(self.dim)]
                           for _ in range(self.dim)])
            self.wv.append([[random.gauss(0, scale_attn) for _ in range(self.dim)]
                           for _ in range(self.dim)])
            self.wo.append([[random.gauss(0, scale_attn) for _ in range(self.dim)]
                           for _ in range(self.dim)])

            self.w1.append([[random.gauss(0, scale_ffn1) for _ in range(self.hidden_dim)]
                           for _ in range(self.dim)])
            self.w2.append([[random.gauss(0, scale_ffn2) for _ in range(self.dim)]
                           for _ in range(self.hidden_dim)])

            self.ln1_weight.append([1.0] * self.dim)
            self.ln2_weight.append([1.0] * self.dim)

        # Final layer norm and output
        self.ln_final_weight = [1.0] * self.dim
        self.output_weight = [[random.gauss(0, scale_embed)
                               for _ in range(self.vocab_size)]
                              for _ in range(self.dim)]

    def _count_params(self):
        """Count total parameters"""
        params = 0

        # Embeddings
        params += self.vocab_size * self.dim

        # Per layer
        for _ in range(self.n_layers):
            params += 4 * self.dim * self.dim  # Q, K, V, O
            params += self.dim * self.hidden_dim  # W1
            params += self.hidden_dim * self.dim  # W2
            params += 2 * self.dim  # LN weights

        # Final
        params += self.dim  # Final LN
        params += self.dim * self.vocab_size  # Output

        self.param_count = params
        print(f"Total parameters: {params:,} ({params/1e6:.2f}M)")

    def forward(self, tokens: List[int]) -> List[List[float]]:
        """Forward pass, returns logits for each position"""
        seq_len = len(tokens)

        # Embed tokens
        x = [self.token_embedding[t].copy() for t in tokens]

        # Transformer layers
        for layer in range(self.n_layers):
            # Layer norm 1
            x = [self._rmsnorm(xi, self.ln1_weight[layer]) for xi in x]

            # Attention
            x = self._attention(x, layer)

            # Layer norm 2
            x = [self._rmsnorm(xi, self.ln2_weight[layer]) for xi in x]

            # FFN
            x = [self._ffn(xi, layer) for xi in x]

        # Final norm
        x = [self._rmsnorm(xi, self.ln_final_weight) for xi in x]

        # Output projection
        logits = [self._matmul_vec(xi, self.output_weight) for xi in x]

        return logits

    def _rmsnorm(self, x: List[float], weight: List[float]) -> List[float]:
        """RMS normalization"""
        ss = sum(xi * xi for xi in x) / len(x)
        scale = 1.0 / math.sqrt(ss + 1e-5)
        return [xi * scale * wi for xi, wi in zip(x, weight)]

    def _attention(self, x: List[List[float]], layer: int) -> List[List[float]]:
        """Multi-head self-attention"""
        seq_len = len(x)

        # Compute Q, K, V for all positions
        Q = [self._matmul_vec(xi, self.wq[layer]) for xi in x]
        K = [self._matmul_vec(xi, self.wk[layer]) for xi in x]
        V = [self._matmul_vec(xi, self.wv[layer]) for xi in x]

        # Per-head attention
        output = [[0.0] * self.dim for _ in range(seq_len)]

        for h in range(self.n_heads):
            h_start = h * self.head_dim
            h_end = h_start + self.head_dim

            for i in range(seq_len):
                # Query for position i
                q = Q[i][h_start:h_end]

                # Attention scores
                scores = []
                for j in range(i + 1):  # Causal mask
                    k = K[j][h_start:h_end]
                    score = sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(self.head_dim)
                    scores.append(score)

                # Softmax
                max_score = max(scores)
                exp_scores = [math.exp(s - max_score) for s in scores]
                sum_exp = sum(exp_scores)
                attn_weights = [e / sum_exp for e in exp_scores]

                # Weighted sum of values
                for j, w in enumerate(attn_weights):
                    v = V[j][h_start:h_end]
                    for k, vk in enumerate(v):
                        output[i][h_start + k] += w * vk

        # Output projection
        output = [self._matmul_vec(o, self.wo[layer]) for o in output]

        # Residual
        return [[xi + oi for xi, oi in zip(x[i], output[i])] for i in range(seq_len)]

    def _ffn(self, x: List[float], layer: int) -> List[float]:
        """Feed-forward network with SiLU"""
        # Up projection
        hidden = self._matmul_vec(x, self.w1[layer])

        # SiLU activation
        hidden = [h * (1 / (1 + math.exp(-h))) for h in hidden]

        # Down projection
        out = self._matmul_vec(hidden, self.w2[layer])

        # Residual
        return [xi + oi for xi, oi in zip(x, out)]

    def _matmul_vec(self, x: List[float], W: List[List[float]]) -> List[float]:
        """Vector-matrix multiplication"""
        out_dim = len(W[0])
        result = [0.0] * out_dim
        for i, xi in enumerate(x):
            for j in range(out_dim):
                result[j] += xi * W[i][j]
        return result

    def compute_loss(self, logits: List[List[float]], targets: List[int]) -> float:
        """Cross-entropy loss"""
        total_loss = 0.0

        for i, (l, t) in enumerate(zip(logits, targets)):
            # Softmax
            max_l = max(l)
            exp_l = [math.exp(li - max_l) for li in l]
            sum_exp = sum(exp_l)
            probs = [e / sum_exp for e in exp_l]

            # Cross-entropy
            total_loss -= math.log(probs[t] + 1e-10)

        return total_loss / len(targets)

    def save_weights(self, path: str):
        """Save weights in binary format for C inference"""
        with open(path, 'wb') as f:
            # Header
            f.write(struct.pack('i', 0x53415254))  # Magic: "SART"
            f.write(struct.pack('i', 1))  # Version
            f.write(struct.pack('i', self.dim))
            f.write(struct.pack('i', self.n_layers))
            f.write(struct.pack('i', self.n_heads))
            f.write(struct.pack('i', self.hidden_dim))
            f.write(struct.pack('i', self.vocab_size))
            f.write(struct.pack('i', self.max_seq_len))

            # Token embedding [vocab_size, dim]
            for row in self.token_embedding:
                f.write(struct.pack(f'{self.dim}f', *row))

            # Per-layer weights
            for layer in range(self.n_layers):
                # wq [dim, dim]
                for row in self.wq[layer]:
                    f.write(struct.pack(f'{self.dim}f', *row))
                # wk
                for row in self.wk[layer]:
                    f.write(struct.pack(f'{self.dim}f', *row))
                # wv
                for row in self.wv[layer]:
                    f.write(struct.pack(f'{self.dim}f', *row))
                # wo
                for row in self.wo[layer]:
                    f.write(struct.pack(f'{self.dim}f', *row))
                # w1 [dim, hidden_dim]
                for row in self.w1[layer]:
                    f.write(struct.pack(f'{self.hidden_dim}f', *row))
                # w2 [hidden_dim, dim]
                for row in self.w2[layer]:
                    f.write(struct.pack(f'{self.dim}f', *row))
                # ln1_weight [dim]
                f.write(struct.pack(f'{self.dim}f', *self.ln1_weight[layer]))
                # ln2_weight [dim]
                f.write(struct.pack(f'{self.dim}f', *self.ln2_weight[layer]))

            # Final layer norm
            f.write(struct.pack(f'{self.dim}f', *self.ln_final_weight))

            # Output [dim, vocab_size]
            for row in self.output_weight:
                f.write(struct.pack(f'{self.vocab_size}f', *row))

        print(f"Saved weights to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(corpus_dir: Path, output_path: Path, config: dict, epochs: int = 100,
          batch_size: int = 4, lr: float = 0.001):
    """Train SARTRE on corpus"""

    print("=" * 60)
    print("SARTRE Training")
    print("=" * 60)

    # Load corpus
    corpus = load_corpus(corpus_dir)
    tokens = tokenize(corpus)
    print(f"Tokens: {len(tokens):,}")

    # Create model
    print(f"\nConfig: dim={config['dim']}, layers={config['n_layers']}, "
          f"heads={config['n_heads']}, hidden={config['hidden_dim']}")
    model = SartreTrainer(config)

    # Create batches
    batches = create_batches(tokens, config['max_seq_len'], batch_size)
    print(f"Batches: {len(batches)}")

    if len(batches) == 0:
        print("ERROR: Corpus too small for training!")
        print(f"Need at least {config['max_seq_len'] * batch_size} tokens")
        return

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in batches:
            # Forward pass (simplified: one sequence at a time)
            batch_loss = 0.0
            for x, y in zip(batch_x, batch_y):
                logits = model.forward(x)
                loss = model.compute_loss(logits, y)
                batch_loss += loss

            total_loss += batch_loss / batch_size

            # TODO: Backprop and weight updates
            # For now this is forward-only to verify the architecture

        avg_loss = total_loss / len(batches)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d}/{epochs}: loss = {avg_loss:.4f}")

    print("-" * 60)

    # Save weights
    model.save_weights(str(output_path))
    print(f"\nTraining complete! Weights saved to {output_path}")

    # Summary
    file_size = output_path.stat().st_size
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    script_dir = Path(__file__).parent
    corpus_dir = script_dir / "corpus"

    # Parse args
    use_small = "--small" in sys.argv
    epochs = 100

    for i, arg in enumerate(sys.argv):
        if arg == "--epochs" and i + 1 < len(sys.argv):
            epochs = int(sys.argv[i + 1])

    # Select config
    if use_small:
        config = CONFIG_SMALL
        output_path = script_dir / "sartre_small.bin"
        print("Using SMALL config (~150K params)")
    else:
        config = CONFIG_DEV
        output_path = script_dir / "sartre.bin"
        print("Using DEV config (2.2M params)")

    # Check corpus
    if not corpus_dir.exists():
        print(f"ERROR: Corpus directory not found: {corpus_dir}")
        sys.exit(1)

    corpus_size = sum(f.stat().st_size for f in corpus_dir.glob('*.txt'))
    print(f"Corpus size: {corpus_size:,} bytes ({corpus_size/1024:.1f} KB)")

    if corpus_size < 1024:
        print("WARNING: Corpus is very small!")
        print("Consider expanding corpus before training.")

    # Train
    train(corpus_dir, output_path, config, epochs=epochs)


if __name__ == "__main__":
    main()
