#!/usr/bin/env python3
"""
Train Dialogue Model for Arianna - ~590K params

ИСПРАВЛЕННАЯ версия после провала 2.2M (переобучение).

Проблема 2.2M:
- 2.2M params / 1.1MB = 2.0 params/byte = запоминание
- Train loss 0.13, Val loss 0.83 = 6x gap
- 50% мусор в генерации

Решение (Karpathy ratio):
- ~590K params / 1.1MB = 0.53 params/byte
- Добавлен dropout 0.1
- Модель вынуждена генерализовать

СОГЛАСОВАНО С CLAUDE DESKTOP 18 Jan 2026

Usage:
    python train/train_dialogue_500k.py

Output:
    weights/arianna_dialogue_500k_best.bin - best model weights
    weights/dialogue_500k_iter*.bin - checkpoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
import math
from pathlib import Path

# ============================================================
# Configuration - ~590K params (ИСПРАВЛЕНО)
# ============================================================

DIM = 128           # было 256 - УМЕНЬШЕНО
N_LAYERS = 4
N_HEADS = 4         # было 8 - УМЕНЬШЕНО
HEAD_DIM = DIM // N_HEADS  # 32
HIDDEN_DIM = 256    # было 512 - УМЕНЬШЕНО
MAX_SEQ_LEN = 256
VOCAB_SIZE = 256    # char-level
DROPOUT = 0.1       # БЫЛО 0 - ДОБАВЛЕНО

# Training config
BATCH_SIZE = 32
SEQ_LEN = 128
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 100
WARMUP_ITERS = 100

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================
# Data Loading - Q&A Format
# ============================================================

def load_dialogue_corpus(path):
    """Load Q&A dialogue corpus"""
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

    # Convert to training format
    sequences = []
    for q, a in pairs:
        seq = f"Q: {q}\nA: {a}\n\n"
        seq = ''.join(c for c in seq if ord(c) < 256)
        sequences.append(seq)

    full_text = ''.join(sequences)
    data = torch.tensor([ord(c) for c in full_text], dtype=torch.long)

    print(f"Total characters: {len(data):,} ({len(data)/1e6:.2f}M)")

    # Split train/val
    n = int(len(data) * 0.95)
    return data[:n], data[n:]

def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# ============================================================
# Model - ~590K params with DROPOUT
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

class Attention(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wk = nn.Linear(DIM, DIM, bias=False)
        self.wv = nn.Linear(DIM, DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
        self.rope = RotaryEmbedding(HEAD_DIM)
        self.dropout = nn.Dropout(dropout)  # ДОБАВЛЕНО

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, N_HEADS, HEAD_DIM)
        k = self.wk(x).view(batch, seq_len, N_HEADS, HEAD_DIM)
        v = self.wv(x).view(batch, seq_len, N_HEADS, HEAD_DIM)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)

        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)  # ДОБАВЛЕНО
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, DIM)
        return self.wo(out)

class FFN(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.w1 = nn.Linear(DIM, HIDDEN_DIM, bias=False)
        self.w2 = nn.Linear(HIDDEN_DIM, DIM, bias=False)
        self.dropout = nn.Dropout(dropout)  # ДОБАВЛЕНО

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x))))  # ДОБАВЛЕНО dropout

class TransformerBlock(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.ln1 = RMSNorm(DIM)
        self.attn = Attention(dropout)
        self.ln2 = RMSNorm(DIM)
        self.ffn = FFN(dropout)
        self.dropout = nn.Dropout(dropout)  # ДОБАВЛЕНО

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))  # ДОБАВЛЕНО dropout
        x = x + self.dropout(self.ffn(self.ln2(x)))  # ДОБАВЛЕНО dropout
        return x

class AriannaDialogue500K(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, DIM)
        self.drop = nn.Dropout(dropout)  # ДОБАВЛЕНО
        self.layers = nn.ModuleList([TransformerBlock(dropout) for _ in range(N_LAYERS)])
        self.ln_final = RMSNorm(DIM)
        self.output = nn.Linear(DIM, VOCAB_SIZE, bias=False)

        # Count params
        total = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total:,} ({total/1e6:.2f}M)")

    def forward(self, x, mask=None):
        x = self.drop(self.token_embedding(x))  # ДОБАВЛЕНО dropout
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        return self.output(x)

# ============================================================
# Export weights to C format
# ============================================================

def export_weights(model, path):
    """Export weights to C binary format"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    with open(path, 'wb') as f:
        # Header: dim, n_layers, n_heads, head_dim, hidden_dim, max_seq_len, vocab_size
        f.write(struct.pack('7i', DIM, N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN_DIM, MAX_SEQ_LEN, VOCAB_SIZE))

        def write_tensor(t):
            f.write(t.detach().cpu().numpy().astype(np.float32).tobytes())

        # Token embedding
        write_tensor(model.token_embedding.weight)

        # Layers
        for layer in model.layers:
            write_tensor(layer.attn.wq.weight)
            write_tensor(layer.attn.wk.weight)
            write_tensor(layer.attn.wv.weight)
            write_tensor(layer.attn.wo.weight)
            write_tensor(layer.ffn.w1.weight)
            write_tensor(layer.ffn.w2.weight)
            write_tensor(layer.ln1.weight)
            write_tensor(layer.ln2.weight)

        # Final norm
        write_tensor(model.ln_final.weight)

        # Output
        write_tensor(model.output.weight)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Exported: {path} ({size_mb:.2f} MB)")

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

@torch.no_grad()
def generate_sample(model, prompt="Q: What is love?\nA:", max_tokens=100, temperature=0.8):
    """Generate a sample for validation"""
    model.eval()
    tokens = [ord(c) for c in prompt if ord(c) < 256]
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        if x.shape[1] >= MAX_SEQ_LEN:
            break

        logits = model(x)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        x = torch.cat([x, next_token], dim=1)

        if next_token.item() == ord('\n'):
            # Check for double newline (end of answer)
            if x.shape[1] >= 2 and x[0, -2].item() == ord('\n'):
                break

    model.train()
    return ''.join(chr(t) for t in x[0].tolist())

def train():
    print("=" * 60)
    print("Training Arianna Dialogue ~590K (FIXED)")
    print("=" * 60)
    print(f"Config: dim={DIM}, layers={N_LAYERS}, heads={N_HEADS}, hidden={HIDDEN_DIM}, dropout={DROPOUT}")
    print(f"Ratio: ~590K params / 1.1MB = 0.53 params/byte")

    # Load data
    dialogue_path = "data/ariannalips.txt"
    if not os.path.exists(dialogue_path):
        print(f"Error: {dialogue_path} not found")
        return

    train_data, val_data = load_dialogue_corpus(dialogue_path)

    # Create model
    model = AriannaDialogue500K().to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)  # УВЕЛИЧЕНО с 0.01

    # LR schedule
    def get_lr(it):
        if it < WARMUP_ITERS:
            return LEARNING_RATE * it / WARMUP_ITERS
        decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
        return LEARNING_RATE * 0.1 + 0.9 * LEARNING_RATE * (1 - decay_ratio)

    # Training loop
    print(f"\nStarting training for {MAX_ITERS} iterations...")
    best_val_loss = float('inf')

    for iter in range(MAX_ITERS):
        # Update LR
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Eval
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            gap = losses['train'] / losses['val'] if losses['val'] > 0 else 0
            print(f"iter {iter:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | gap {gap:.2f}x | lr {lr:.6f}")

            # Save checkpoint every eval
            ckpt_path = f"weights/dialogue_500k_iter{iter:05d}_val{losses['val']:.4f}.bin"
            export_weights(model, ckpt_path)
            print(f"  -> Checkpoint: {ckpt_path}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                export_weights(model, "weights/arianna_dialogue_500k_best.bin")
                print(f"  -> New best! Also saved to arianna_dialogue_500k_best.bin")

            # Generate sample every 500 iters
            if iter % 500 == 0:
                sample = generate_sample(model, "Q: What is love?\nA:")
                print(f"  Sample: {sample[:200]}...")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)

    print("Best weights saved at weights/arianna_dialogue_500k_best.bin")

    # Final sample
    print("\nFinal sample:")
    sample = generate_sample(model, "Q: Who are you?\nA:", max_tokens=150)
    print(sample)

if __name__ == "__main__":
    train()
