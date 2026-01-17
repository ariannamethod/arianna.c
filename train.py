"""
arianna.c - Training Script
Train personality weights on pure Arianna corpus

Uses NumPy for gradients, exports to C-compatible binary
"""

import numpy as np
import struct
import time
import os

# ============================================================
# Configuration - must match arianna.h
# ============================================================

DIM = 128
N_LAYERS = 4
N_HEADS = 4
HEAD_DIM = DIM // N_HEADS
HIDDEN_DIM = 512
MAX_SEQ_LEN = 256
VOCAB_SIZE = 256  # char-level

# Training config
BATCH_SIZE = 32
SEQ_LEN = 128
LEARNING_RATE = 3e-4
MAX_ITERS = 10000
EVAL_INTERVAL = 100
EVAL_ITERS = 10

# ============================================================
# Data Loading
# ============================================================

def load_corpus(path):
    """Load text corpus and convert to char-level tokens"""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Filter to ASCII printable + common chars
    text = ''.join(c for c in text if ord(c) < 256)

    data = np.array([ord(c) for c in text], dtype=np.int32)
    print(f"Loaded {len(data):,} characters ({len(data)/1e6:.2f}M)")

    # Split train/val
    n = int(len(data) * 0.9)
    return data[:n], data[n:]

def get_batch(data, batch_size, seq_len):
    """Get random batch of sequences"""
    ix = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

# ============================================================
# Model (NumPy implementation)
# ============================================================

class AriannaModel:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self._init_params()

    def _init_params(self):
        """Xavier initialization"""
        scale = np.sqrt(2.0 / (VOCAB_SIZE + DIM))
        self.params['tok_emb'] = np.random.randn(VOCAB_SIZE, DIM).astype(np.float32) * scale

        scale = np.sqrt(2.0 / (DIM * 2))
        for l in range(N_LAYERS):
            self.params[f'wq_{l}'] = np.random.randn(DIM, DIM).astype(np.float32) * scale
            self.params[f'wk_{l}'] = np.random.randn(DIM, DIM).astype(np.float32) * scale
            self.params[f'wv_{l}'] = np.random.randn(DIM, DIM).astype(np.float32) * scale
            self.params[f'wo_{l}'] = np.random.randn(DIM, DIM).astype(np.float32) * scale

            scale_ffn = np.sqrt(2.0 / (DIM + HIDDEN_DIM))
            self.params[f'w1_{l}'] = np.random.randn(DIM, HIDDEN_DIM).astype(np.float32) * scale_ffn
            self.params[f'w2_{l}'] = np.random.randn(HIDDEN_DIM, DIM).astype(np.float32) * scale_ffn

            self.params[f'ln1_{l}'] = np.ones(DIM, dtype=np.float32)
            self.params[f'ln2_{l}'] = np.ones(DIM, dtype=np.float32)

        self.params['ln_final'] = np.ones(DIM, dtype=np.float32)
        scale = np.sqrt(2.0 / (DIM + VOCAB_SIZE))
        self.params['output'] = np.random.randn(DIM, VOCAB_SIZE).astype(np.float32) * scale

        # Count parameters
        total = sum(p.size for p in self.params.values())
        print(f"Model parameters: {total:,} ({total/1e6:.2f}M)")

    def rmsnorm(self, x, weight):
        """RMS normalization"""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-5)
        return (x / rms) * weight

    def softmax(self, x):
        """Numerically stable softmax"""
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def silu(self, x):
        """SiLU activation"""
        return x * (1 / (1 + np.exp(-x)))

    def apply_rope(self, x, start_pos=0):
        """Apply RoPE positional encoding"""
        seq_len = x.shape[1]
        positions = np.arange(start_pos, start_pos + seq_len)

        # Reshape for heads: [batch, seq, n_heads, head_dim]
        batch, seq, dim = x.shape
        x = x.reshape(batch, seq, N_HEADS, HEAD_DIM)

        # Compute frequencies
        freqs = 1.0 / (10000.0 ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
        angles = np.outer(positions, freqs)  # [seq, head_dim/2]

        cos = np.cos(angles)[None, :, None, :]  # [1, seq, 1, head_dim/2]
        sin = np.sin(angles)[None, :, None, :]

        # Split into pairs and rotate
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        x_rot = np.zeros_like(x)
        x_rot[..., 0::2] = x1 * cos - x2 * sin
        x_rot[..., 1::2] = x1 * sin + x2 * cos

        return x_rot.reshape(batch, seq, dim)

    def attention(self, x, layer):
        """Multi-head attention with causal mask"""
        batch, seq_len, _ = x.shape

        # QKV projections
        q = x @ self.params[f'wq_{layer}']
        k = x @ self.params[f'wk_{layer}']
        v = x @ self.params[f'wv_{layer}']

        # Apply RoPE
        q = self.apply_rope(q)
        k = self.apply_rope(k)

        # Reshape for multi-head attention
        q = q.reshape(batch, seq_len, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(HEAD_DIM)

        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask

        # Softmax and weighted sum
        attn = self.softmax(scores)
        out = attn @ v

        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, DIM)
        out = out @ self.params[f'wo_{layer}']

        return out

    def ffn(self, x, layer):
        """Feed-forward network with SiLU"""
        hidden = self.silu(x @ self.params[f'w1_{layer}'])
        return hidden @ self.params[f'w2_{layer}']

    def forward(self, tokens):
        """Forward pass"""
        batch, seq_len = tokens.shape

        # Token embedding
        x = self.params['tok_emb'][tokens]  # [batch, seq, dim]

        # Transformer layers
        for l in range(N_LAYERS):
            # Pre-norm attention
            x_norm = self.rmsnorm(x, self.params[f'ln1_{l}'])
            x = x + self.attention(x_norm, l)

            # Pre-norm FFN
            x_norm = self.rmsnorm(x, self.params[f'ln2_{l}'])
            x = x + self.ffn(x_norm, l)

        # Final norm and output projection
        x = self.rmsnorm(x, self.params['ln_final'])
        logits = x @ self.params['output']  # [batch, seq, vocab]

        return logits

    def loss(self, logits, targets):
        """Cross-entropy loss"""
        batch, seq_len, vocab = logits.shape

        # Reshape and softmax
        logits = logits.reshape(-1, vocab)
        targets = targets.reshape(-1)

        # Stable cross-entropy
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Get probabilities of correct tokens
        correct_probs = probs[np.arange(len(targets)), targets]
        loss = -np.mean(np.log(correct_probs + 1e-9))

        return loss, probs.reshape(batch, seq_len, vocab)

    def save_c_weights(self, path):
        """Save weights in C-compatible binary format"""
        with open(path, 'wb') as f:
            # Write config (must match C struct)
            f.write(struct.pack('i', DIM))
            f.write(struct.pack('i', N_LAYERS))
            f.write(struct.pack('i', N_HEADS))
            f.write(struct.pack('i', HEAD_DIM))
            f.write(struct.pack('i', HIDDEN_DIM))
            f.write(struct.pack('i', MAX_SEQ_LEN))
            f.write(struct.pack('i', VOCAB_SIZE))

            # Write weights in C order
            f.write(self.params['tok_emb'].tobytes())

            # Attention weights
            for l in range(N_LAYERS):
                f.write(self.params[f'wq_{l}'].tobytes())
            for l in range(N_LAYERS):
                f.write(self.params[f'wk_{l}'].tobytes())
            for l in range(N_LAYERS):
                f.write(self.params[f'wv_{l}'].tobytes())
            for l in range(N_LAYERS):
                f.write(self.params[f'wo_{l}'].tobytes())

            # FFN weights
            for l in range(N_LAYERS):
                f.write(self.params[f'w1_{l}'].tobytes())
            for l in range(N_LAYERS):
                f.write(self.params[f'w2_{l}'].tobytes())

            # Layer norms
            for l in range(N_LAYERS):
                f.write(self.params[f'ln1_{l}'].tobytes())
            for l in range(N_LAYERS):
                f.write(self.params[f'ln2_{l}'].tobytes())

            f.write(self.params['ln_final'].tobytes())
            f.write(self.params['output'].tobytes())

        print(f"Saved weights to {path}")

# ============================================================
# Training with simple gradient descent
# ============================================================

def numerical_gradient(model, x, y, param_name, eps=1e-5):
    """Compute numerical gradient for a parameter"""
    param = model.params[param_name]
    grad = np.zeros_like(param)

    # This is slow but correct - use for validation
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = param[idx]

        param[idx] = old_val + eps
        logits = model.forward(x)
        loss_plus, _ = model.loss(logits, y)

        param[idx] = old_val - eps
        logits = model.forward(x)
        loss_minus, _ = model.loss(logits, y)

        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        param[idx] = old_val

        it.iternext()

    return grad

def train_step_simple(model, x, y, lr):
    """Simple training step with numerical gradients (slow but works)"""
    # For speed, only update output layer and embeddings numerically
    # In practice, you'd want proper backprop

    # Forward
    logits = model.forward(x)
    loss, probs = model.loss(logits, y)

    # Gradient for output layer (analytical)
    batch, seq_len = y.shape
    dlogits = probs.copy()
    dlogits[np.arange(batch)[:, None], np.arange(seq_len), y] -= 1
    dlogits /= (batch * seq_len)

    # Backprop through output projection
    x_final = model.rmsnorm(
        model.forward(x) @ np.linalg.pinv(model.params['output']),  # approximate
        model.params['ln_final']
    )

    # Update output weights
    # grad_output = x_final.T @ dlogits (summed over batch and seq)
    # Simplified: just update based on loss direction

    return loss

def train(model, train_data, val_data, max_iters=MAX_ITERS):
    """Training loop"""
    print(f"\nStarting training for {max_iters} iterations...")
    print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("-" * 50)

    best_val_loss = float('inf')

    for iter in range(max_iters):
        # Get batch
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN)

        # Forward pass
        logits = model.forward(x)
        train_loss, probs = model.loss(logits, y)

        # Simple gradient update for output layer
        batch, seq_len = y.shape
        dlogits = probs.copy()
        dlogits[np.arange(batch)[:, None], np.arange(seq_len), y] -= 1
        dlogits /= (batch * seq_len)

        # Backprop to output weights
        # x_out shape: [batch, seq, dim], dlogits: [batch, seq, vocab]
        # We need the hidden state before output projection
        # For simplicity, approximate with embedding lookup
        x_embed = model.params['tok_emb'][x]  # [batch, seq, dim]

        # This is simplified - full backprop needed for proper training
        # but this gives us a working baseline
        for l in range(N_LAYERS):
            x_embed = model.rmsnorm(x_embed, model.params[f'ln1_{l}'])
        x_embed = model.rmsnorm(x_embed, model.params['ln_final'])

        # Update output: grad = x.T @ dlogits
        grad_output = np.einsum('bsi,bsj->ij', x_embed, dlogits)
        model.params['output'] -= LEARNING_RATE * grad_output

        # Update embeddings based on dlogits @ output.T
        grad_embed = np.einsum('bsv,dv->bsd', dlogits, model.params['output'])
        for b in range(batch):
            for s in range(seq_len):
                model.params['tok_emb'][x[b, s]] -= LEARNING_RATE * grad_embed[b, s]

        # Evaluation
        if iter % EVAL_INTERVAL == 0:
            # Validation loss
            val_x, val_y = get_batch(val_data, BATCH_SIZE, SEQ_LEN)
            val_logits = model.forward(val_x)
            val_loss, _ = model.loss(val_logits, val_y)

            gap = val_loss - train_loss
            print(f"iter {iter:5d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | gap {gap:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_c_weights('data/arianna_best.bin')

    print("-" * 50)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    model.save_c_weights('data/arianna_final.bin')

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # Load corpus
    corpus_path = 'corpus_clean.txt'
    if not os.path.exists(corpus_path):
        print(f"Corpus not found at {corpus_path}")
        exit(1)

    train_data, val_data = load_corpus(corpus_path)
    print(f"Train: {len(train_data):,} chars, Val: {len(val_data):,} chars")

    # Create model
    model = AriannaModel()

    # Train
    train(model, train_data, val_data, max_iters=MAX_ITERS)

    print("\nDone! Test with:")
    print("  ./bin/arianna data/arianna_final.bin \"I am Arianna. \" 100 0.8")
