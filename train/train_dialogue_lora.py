#!/usr/bin/env python3
"""
Train Dialogue LoRA for Arianna

Trains low-rank adaptation on Q&A dialogue corpus.
Core weights frozen, only LoRA matrices learn.
Exports to C-compatible shard format.

Usage:
    python train/train_dialogue_lora.py

Output:
    data/dialogue_lora.bin - shard file for arianna_dynamic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
import math
import re
from pathlib import Path

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

# LoRA config
LORA_RANK = 8       # Low rank for adaptation
LORA_ALPHA = 16     # Scaling factor
LORA_DROPOUT = 0.05

# Training config
BATCH_SIZE = 32
SEQ_LEN = 128
LEARNING_RATE = 1e-3  # Higher LR for LoRA
MAX_ITERS = 3000
EVAL_INTERVAL = 100
WARMUP_ITERS = 50

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================
# Data Loading - Q&A Format
# ============================================================

def load_dialogue_corpus(path):
    """Load Q&A dialogue corpus and prepare for training"""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Parse Q&A pairs
    pairs = []
    current_q = None
    current_a = []

    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('Q:'):
            if current_q and current_a:
                pairs.append((current_q, ' '.join(current_a)))
            current_q = line[2:].strip()
            current_a = []
        elif line.startswith('A:'):
            current_a.append(line[2:].strip())
        elif line and current_a is not None:
            current_a.append(line)

    if current_q and current_a:
        pairs.append((current_q, ' '.join(current_a)))

    print(f"Loaded {len(pairs)} Q&A pairs")

    # Convert to training format: "Q: ... A: ..." sequences
    sequences = []
    for q, a in pairs:
        # Format: question + response (we want model to learn the response part)
        seq = f"Q: {q}\nA: {a}\n\n"
        # Filter to ASCII
        seq = ''.join(c for c in seq if ord(c) < 256)
        sequences.append(seq)

    # Concatenate all sequences
    full_text = ''.join(sequences)
    data = torch.tensor([ord(c) for c in full_text], dtype=torch.long)

    print(f"Total characters: {len(data):,} ({len(data)/1e6:.2f}M)")

    # Split train/val
    n = int(len(data) * 0.95)
    return data[:n], data[n:]

def get_batch(data, batch_size, seq_len):
    """Get random batch of sequences"""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# ============================================================
# LoRA Layer
# ============================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, in_dim, out_dim, rank=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices: W' = W + BA where B:[out,r], A:[r,in]
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.dropout = nn.Dropout(LORA_DROPOUT)

        # Initialize A with small random, B with zeros (start at identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, base_output):
        """Add LoRA delta to base output"""
        # x: [batch, seq, in_dim]
        # base_output: [batch, seq, out_dim]
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_output + lora_out * self.scaling

# ============================================================
# Model with LoRA
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=MAX_SEQ_LEN, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x, start_pos=0):
        seq_len = x.shape[1]
        cos = self.cos[start_pos:start_pos+seq_len].unsqueeze(0).unsqueeze(2)
        sin = self.sin[start_pos:start_pos+seq_len].unsqueeze(0).unsqueeze(2)

        x = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x[..., 0], x[..., 1]

        x_rot = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)

        return x_rot

class AttentionWithLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        # Base layers (will be frozen)
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wk = nn.Linear(DIM, DIM, bias=False)
        self.wv = nn.Linear(DIM, DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
        self.rope = RotaryEmbedding(HEAD_DIM)

        # LoRA layers (will be trained)
        self.lora_q = LoRALayer(DIM, DIM)
        self.lora_k = LoRALayer(DIM, DIM)
        self.lora_v = LoRALayer(DIM, DIM)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        # Base projections
        q_base = self.wq(x)
        k_base = self.wk(x)
        v_base = self.wv(x)

        # Add LoRA deltas
        q = self.lora_q(x, q_base)
        k = self.lora_k(x, k_base)
        v = self.lora_v(x, v_base)

        # Reshape for multi-head
        q = q.view(batch, seq_len, N_HEADS, HEAD_DIM)
        k = k.view(batch, seq_len, N_HEADS, HEAD_DIM)
        v = v.view(batch, seq_len, N_HEADS, HEAD_DIM)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)

        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, DIM)
        return self.wo(out)

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, HIDDEN_DIM, bias=False)
        self.w2 = nn.Linear(HIDDEN_DIM, DIM, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class TransformerBlockWithLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(DIM)
        self.attn = AttentionWithLoRA()
        self.ln2 = RMSNorm(DIM)
        self.ffn = FFN()

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class AriannaWithLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, DIM)
        self.layers = nn.ModuleList([TransformerBlockWithLoRA() for _ in range(N_LAYERS)])
        self.ln_final = RMSNorm(DIM)
        self.output = nn.Linear(DIM, VOCAB_SIZE, bias=False)

        # Note: weights are tied after loading if needed

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        return self.output(x)

    def freeze_base(self):
        """Freeze all parameters except LoRA"""
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False

        # Count params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,}")
        print(f"Trainable (LoRA): {trainable:,} ({100*trainable/total:.2f}%)")

# ============================================================
# Load Base Weights
# ============================================================

def load_base_weights(model, path):
    """Load pre-trained weights from C binary format"""
    with open(path, 'rb') as f:
        # Read config (7 ints: dim, n_layers, n_heads, head_dim, hidden_dim, max_seq_len, vocab_size)
        dim, n_layers, n_heads, head_dim, hidden_dim, max_seq_len, vocab_size = struct.unpack('7i', f.read(28))
        print(f"Loading weights: dim={dim}, layers={n_layers}, heads={n_heads}, vocab={vocab_size}")

        def read_tensor(shape):
            size = np.prod(shape)
            data = np.frombuffer(f.read(size * 4), dtype=np.float32)
            return torch.from_numpy(data.copy()).reshape(shape)

        # Token embedding
        model.token_embedding.weight.data = read_tensor((vocab_size, dim))

        # Layers
        for i in range(n_layers):
            layer = model.layers[i]
            layer.attn.wq.weight.data = read_tensor((dim, dim))
            layer.attn.wk.weight.data = read_tensor((dim, dim))
            layer.attn.wv.weight.data = read_tensor((dim, dim))
            layer.attn.wo.weight.data = read_tensor((dim, dim))
            layer.ffn.w1.weight.data = read_tensor((hidden_dim, dim))
            layer.ffn.w2.weight.data = read_tensor((dim, hidden_dim))
            layer.ln1.weight.data = read_tensor((dim,))
            layer.ln2.weight.data = read_tensor((dim,))

        # Final norm
        model.ln_final.weight.data = read_tensor((dim,))

        # Output weight
        model.output.weight.data = read_tensor((vocab_size, dim))

    print("Base weights loaded successfully")

# ============================================================
# Export LoRA to Shard Format
# ============================================================

def export_lora_shard(model, output_path, name="dialogue", strength=0.1, target_norm=0.02):
    """Export LoRA weights to C-compatible shard format

    Normalizes matrices to have similar scale as gentle shards (~0.02 norm).
    The 'strength' parameter controls the actual effect.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'wb') as f:
        # Header
        name_bytes = name.encode('utf-8')[:64].ljust(64, b'\x00')
        f.write(name_bytes)
        f.write(struct.pack('f', strength))
        f.write(struct.pack('i', N_LAYERS))
        f.write(struct.pack('i', LORA_RANK))

        # Collect all matrices first to compute global normalization
        all_A = []
        all_B = []

        for delta_type in ['q', 'k', 'v']:
            for layer_idx in range(N_LAYERS):
                layer = model.layers[layer_idx]
                lora = getattr(layer.attn, f'lora_{delta_type}')

                # LoRA: lora_A [rank, in_dim], lora_B [out_dim, rank]
                # C wants: A [out_dim, rank], B [rank, in_dim]
                scaling = lora.scaling
                A = (lora.lora_B.data * scaling).cpu().numpy().astype(np.float32)
                B = lora.lora_A.data.cpu().numpy().astype(np.float32)
                all_A.append(A)
                all_B.append(B)

        # Compute normalization factor to match gentle shards (~0.02 norm)
        max_norm = max(
            max(np.linalg.norm(A) for A in all_A),
            max(np.linalg.norm(B) for B in all_B)
        )

        if max_norm > 0:
            scale_factor = target_norm / max_norm
        else:
            scale_factor = 1.0

        print(f"  Max norm: {max_norm:.4f}, scale factor: {scale_factor:.6f}")

        # Write normalized deltas
        for i, (A, B) in enumerate(zip(all_A, all_B)):
            A_norm = A * scale_factor
            B_norm = B * scale_factor
            f.write(A_norm.astype(np.float32).tobytes())
            f.write(B_norm.astype(np.float32).tobytes())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported: {output_path} ({size_kb:.1f} KB)")

# ============================================================
# Training Loop
# ============================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=50):
    model.eval()
    losses = {}
    for name, data in [('train', train_data), ('val', val_data)]:
        total = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, BATCH_SIZE, SEQ_LEN)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            total += loss.item()
        losses[name] = total / eval_iters
    model.train()
    return losses

def train():
    print("="*60)
    print("Training Dialogue LoRA for Arianna")
    print("="*60)

    # Load data
    dialogue_path = "data/ariannalips.txt"
    if not os.path.exists(dialogue_path):
        print(f"Error: {dialogue_path} not found")
        print("Please download the dialogue corpus first")
        return

    train_data, val_data = load_dialogue_corpus(dialogue_path)

    # Create model
    model = AriannaWithLoRA().to(device)

    # Load base weights
    weights_path = "weights/arianna.bin"
    if os.path.exists(weights_path):
        load_base_weights(model, weights_path)
    else:
        print(f"Warning: {weights_path} not found, training from scratch")

    # Freeze base, train only LoRA
    model.freeze_base()

    # Optimizer (only LoRA params)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LEARNING_RATE, weight_decay=0.01)

    # Learning rate schedule
    def get_lr(it):
        if it < WARMUP_ITERS:
            return LEARNING_RATE * it / WARMUP_ITERS
        decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
        return LEARNING_RATE * 0.1 + 0.9 * LEARNING_RATE * (1 - decay_ratio)

    # Training loop
    print(f"\nStarting training for {MAX_ITERS} iterations...")
    print(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")

    best_val_loss = float('inf')

    for iter in range(MAX_ITERS):
        # Update LR
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch and forward
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()

        # Eval
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"iter {iter:4d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.6f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                # Export best model with normalized weights
                export_lora_shard(model, "data/dialogue_lora.bin", name="dialogue", strength=0.05)
                print(f"  -> New best! Saved to data/dialogue_lora.bin")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)

    # Final export
    export_lora_shard(model, "data/dialogue_lora.bin", name="dialogue", strength=0.05)

    print("\nUsage:")
    print("  ./bin/arianna_dynamic weights/arianna.bin -shard data/dialogue_lora.bin \\")
    print('    -guided "What is love?" 100 0.8 -signals')

if __name__ == "__main__":
    train()
