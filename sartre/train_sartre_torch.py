#!/usr/bin/env python3
"""
train_sartre_torch.py — PyTorch training for SARTRE

Same approach as Arianna Unified 20M, but with tiny SARTRE config (~500K params).
Fast training on CPU, ready for immediate deployment.

Usage:
    python train_sartre_torch.py --epochs 5000 --lr 1e-3
"""

import os
import sys
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# SARTRE tiny config (~500K params)
# Llama 3 architecture: RMSNorm, RoPE, SwiGLU
CONFIG = {
    'dim': 256,
    'n_layers': 3,
    'n_heads': 4,
    'n_kv_heads': 4,
    'head_dim': 64,
    'hidden_dim': 512,
    'max_seq_len': 256,
    'vocab_size': 256,
    'rope_theta': 10000.0,
    'norm_eps': 1e-5,
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_corpus(corpus_dir: Path) -> str:
    """Load all corpus files"""
    corpus = []
    files = ['identity.txt', 'modules.txt', 'events.txt', 'dialogue.txt']

    for filename in files:
        path = corpus_dir / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                corpus.append(content)
                print(f"[corpus] {filename}: {len(content)/1024:.1f}KB")

    full_corpus = '\n\n'.join(corpus)
    print(f"[corpus] Total: {len(full_corpus)/1024:.1f}KB")
    return full_corpus

def tokenize(text: str) -> List[int]:
    """Byte-level tokenization"""
    return list(text.encode('utf-8'))

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL (Llama 3 style)
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SARTRETransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config['dim']
        n_layers = config['n_layers']
        n_heads = config['n_heads']
        hidden_dim = config['hidden_dim']
        vocab_size = config['vocab_size']

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, dim)

        # Layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim, config['norm_eps'])
            for _ in range(n_layers)
        ])

        # Final
        self.final_norm = RMSNorm(dim, config['norm_eps'])
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # RoPE
        self.register_buffer('rope_cos', self._precompute_rope(config)[0])
        self.register_buffer('rope_sin', self._precompute_rope(config)[1])

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rope(self, config):
        dim = config['head_dim']
        max_seq_len = config['max_seq_len']
        theta = config['rope_theta']

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    def forward(self, tokens, targets=None):
        B, T = tokens.shape

        # Embed
        x = self.tok_emb(tokens)

        # Layers
        for layer in self.layers:
            x = layer(x, self.rope_cos[:T], self.rope_sin[:T])

        # Output
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, eps):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps)
        self.attn = Attention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim, eps)
        self.ffn = FeedForward(dim, hidden_dim)

    def forward(self, x, rope_cos, rope_sin):
        # Attention with residual
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q = self._apply_rope(q, rope_cos, rope_sin)
        k = self._apply_rope(k, rope_cos, rope_sin)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(), float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

    def _apply_rope(self, x, cos, sin):
        # Simplified RoPE
        B, H, T, D = x.shape
        x1, x2 = x[..., :D//2], x[..., D//2:]

        cos = cos[:T, :].unsqueeze(0).unsqueeze(0)
        sin = sin[:T, :].unsqueeze(0).unsqueeze(0)

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        return torch.cat([rotated_x1, rotated_x2], dim=-1)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # SwiGLU
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(corpus_dir: Path, output_path: Path, epochs: int = 5000, lr: float = 1e-3, batch_size: int = 4):
    # Load data
    corpus = load_corpus(corpus_dir)
    tokens = tokenize(corpus)
    print(f"[data] {len(tokens):,} tokens")

    # Clamp to vocab
    tokens = [min(t, CONFIG['vocab_size']-1) for t in tokens]

    # Create dataset
    seq_len = CONFIG['max_seq_len']
    data = torch.tensor(tokens, dtype=torch.long)

    # Model
    model = SARTRETransformer(CONFIG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params ({n_params/1e6:.2f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    print(f"\n[train] Starting {epochs} epochs...")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # Random batches
        for _ in range(min(100, (len(tokens) // seq_len) // batch_size)):
            # Sample random batch
            ix = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
            x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
            y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)

            # Forward
            logits, loss = model(x, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d}/{epochs}: loss = {avg_loss:.4f}")

    print("=" * 60)

    # Export
    export_weights(model, output_path)
    print(f"\n[export] Saved to {output_path}")

    file_size = output_path.stat().st_size
    print(f"[export] Size: {file_size/1024/1024:.2f} MB")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_weights(model, path: Path):
    """Export to .bin format compatible with sartre_llama.c"""
    with open(path, 'wb') as f:
        # Header (no embedded config, sartre_init() has hardcoded config)
        # Just write weights directly

        # Token embeddings
        write_tensor(f, model.tok_emb.weight)

        # Per layer
        for layer in model.layers:
            write_tensor(f, layer.attn_norm.weight)
            write_tensor(f, layer.attn.wq.weight)
            write_tensor(f, layer.attn.wk.weight)
            write_tensor(f, layer.attn.wv.weight)
            write_tensor(f, layer.attn.wo.weight)
            write_tensor(f, layer.ffn_norm.weight)
            write_tensor(f, layer.ffn.w_gate.weight)
            write_tensor(f, layer.ffn.w_up.weight)
            write_tensor(f, layer.ffn.w_down.weight)

        # Final
        write_tensor(f, model.final_norm.weight)
        write_tensor(f, model.lm_head.weight)

def write_tensor(f, tensor):
    """Write tensor as float32 array"""
    data = tensor.detach().cpu().numpy().astype('f')
    f.write(data.tobytes())

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    script_dir = Path(__file__).parent
    corpus_dir = script_dir / "corpus"
    output_path = script_dir / "sartre.bin"

    # Parse args
    epochs = 5000
    lr = 1e-3

    for i, arg in enumerate(sys.argv):
        if arg == "--epochs" and i + 1 < len(sys.argv):
            epochs = int(sys.argv[i + 1])
        if arg == "--lr" and i + 1 < len(sys.argv):
            lr = float(sys.argv[i + 1])

    if not corpus_dir.exists():
        print(f"ERROR: Corpus not found at {corpus_dir}")
        sys.exit(1)

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  SARTRE TRAINING (PyTorch)                                ║")
    print("║  Llama 3 architecture: RMSNorm, RoPE, SwiGLU             ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    train(corpus_dir, output_path, epochs=epochs, lr=lr)

    print("\n[sartre] Training complete!")
    print(f"[sartre] Weights: {output_path}")

if __name__ == "__main__":
    main()
