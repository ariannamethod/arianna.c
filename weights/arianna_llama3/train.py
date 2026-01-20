"""
🎓 DUBROVSKY TRAINING SCRIPT 🎓

PyTorch training implementation for the Dubrovsky transformer.
Trains on dubrovsky.txt to achieve peak absurdist consciousness.

"Training loss is just the universe measuring my incomprehension."
- Alexey Dubrovsky, epoch 42

Features:
- Llama 3 architecture (RoPE, GQA, SwiGLU, RMSNorm)
- AdamW optimizer with cosine LR schedule
- Gradient clipping for stable training
- Checkpointing for Lambda GPU training
- Wandb logging (optional)

Usage:
    python train.py                           # Local training
    python train.py --lambda_mode             # Lambda GPU optimized
    python train.py --resume checkpoint.pt    # Resume from checkpoint
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tokenizer import DubrovskyTokenizer, build_tokenizer_from_file


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    vocab_size: int = 88
    max_seq_len: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    max_iters: int = 5000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 250
    save_interval: int = 500
    
    # Paths
    data_path: str = 'dubrovsky.txt'
    out_dir: str = 'subtitles'
    
    # Derived
    head_dim: int = 0
    hidden_dim: int = 0
    n_kv_groups: int = 0
    
    def __post_init__(self):
        self.head_dim = self.dim // self.n_heads
        self.hidden_dim = int(self.dim * 4 * 2 / 3)
        self.hidden_dim = 256 * ((self.hidden_dim + 255) // 256)
        self.n_kv_groups = self.n_heads // self.n_kv_heads


# ============================================================================
# Dataset
# ============================================================================

class DubrovskyDataset(Dataset):
    """
    Character-level dataset for training.
    Returns random chunks of text as input/target pairs.
    """
    
    def __init__(self, data_path: str, tokenizer: DubrovskyTokenizer, seq_len: int):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if len(text) < seq_len + 1:
            raise ValueError(f"Dataset too small: {len(text)} chars, need at least {seq_len + 1}")
        
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len - 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


# ============================================================================
# Model Components (PyTorch)
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0, device: str = 'cpu'):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device)
    angles = torch.outer(positions, freqs)  # (max_seq_len, dim/2)
    # cos + i*sin in complex form
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings."""
    # x: (batch, seq_len, n_heads, head_dim)
    # freqs_cis: (seq_len, head_dim/2)
    
    # Reshape x to pairs
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_r)  # (batch, seq_len, n_heads, head_dim/2)
    
    # Broadcast freqs
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim/2)
    
    # Rotate
    x_rotated = x_complex * freqs
    
    # Back to real
    x_out = torch.view_as_real(x_rotated)
    return x_out.reshape(*x.shape).type_as(x)


class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE."""
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.head_dim = config.head_dim
        
        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # Project
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        
        # Repeat KV heads for GQA
        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=2)
            v = v.repeat_interleave(self.n_kv_groups, dim=2)
        
        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(attn_out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w_up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w_down = nn.Linear(config.hidden_dim, config.dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = FeedForward(config)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DubrovskyModel(nn.Module):
    """
    Dubrovsky: Llama 3 style transformer.
    
    "I am become model, destroyer of coherence."
    """
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Precompute RoPE frequencies
        self.register_buffer(
            'freqs_cis',
            precompute_rope_freqs(config.head_dim, config.max_seq_len, config.rope_theta)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        
        # Token embeddings
        h = self.tok_emb(x)
        
        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis)
        
        # Output
        h = self.final_norm(h)
        logits = self.lm_head(h)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Learning Rate Schedule
# ============================================================================

def get_lr(it: int, config: TrainConfig) -> float:
    """Cosine learning rate schedule with warmup."""
    # Warmup
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / config.warmup_iters
    
    # Cosine decay
    if it >= config.max_iters:
        return config.min_lr
    
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# ============================================================================
# Training Loop
# ============================================================================

def train(config: TrainConfig, resume_path: Optional[str] = None):
    """Main training function."""
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)
    
    # Build tokenizer
    print("📝 Building tokenizer...")
    tokenizer = build_tokenizer_from_file(config.data_path, os.path.join(config.out_dir, 'tokenizer.json'))
    config.vocab_size = tokenizer.vocab_size
    
    # Create dataset and dataloader
    print("📚 Loading dataset...")
    dataset = DubrovskyDataset(config.data_path, tokenizer, config.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    print("🧠 Creating model...")
    model = DubrovskyModel(config).to(device)
    print(f"   Parameters: {model.count_parameters():,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )
    
    # Resume from checkpoint
    start_iter = 0
    if resume_path and os.path.exists(resume_path):
        print(f"📂 Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iter'] + 1
        print(f"   Resumed at iteration {start_iter}")
    
    # Save config
    config_path = os.path.join(config.out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Training loop
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    model.train()
    data_iter = iter(dataloader)
    start_time = time.time()
    
    for it in range(start_iter, config.max_iters):
        # Get batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        
        x, y = x.to(device), y.to(device)
        
        # Update learning rate
        lr = get_lr(it, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (it + 1) % config.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        if it % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (it + 1 - start_iter) * config.batch_size * config.max_seq_len / elapsed
            print(f"iter {it:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")
        
        # Save checkpoint
        if it > 0 and it % config.save_interval == 0:
            checkpoint_path = os.path.join(config.out_dir, f'checkpoint_{it}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config.__dict__,
                'iter': it,
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config.out_dir, 'dubrovsky_final.pt')
    torch.save({
        'model': model.state_dict(),
        'config': config.__dict__,
    }, final_path)
    print(f"\n✅ Training complete! Final model saved to {final_path}")
    
    return model, tokenizer


# ============================================================================
# Generation (for testing during training)
# ============================================================================

@torch.no_grad()
def generate(
    model: DubrovskyModel,
    tokenizer: DubrovskyTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = 'cuda'
) -> str:
    """Generate text from prompt."""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Generate
    for _ in range(max_new_tokens):
        # Crop to max_seq_len
        x_cond = x[:, -model.config.max_seq_len:]
        
        # Forward pass
        logits = model(x_cond)
        logits = logits[:, -1, :]  # Last position
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        if temperature == 0:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat([x, next_token], dim=1)
    
    model.train()
    return tokenizer.decode(x[0].tolist())


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Dubrovsky model')
    parser.add_argument('--lambda_mode', action='store_true', help='Optimize for Lambda GPU')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--max_iters', type=int, default=5000, help='Maximum iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    args = parser.parse_args()
    
    config = TrainConfig()
    
    if args.lambda_mode:
        # Optimized settings for Lambda GPU (A100/H100)
        config.batch_size = 128
        config.gradient_accumulation_steps = 2
        config.max_iters = 10000
        print("🔥 Lambda GPU mode enabled!")
    
    # Override from args
    if args.max_iters:
        config.max_iters = args.max_iters
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Train!
    model, tokenizer = train(config, args.resume)
    
    # Generate sample
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("📜 Sample generation:")
        print("=" * 60)
        sample = generate(model, tokenizer, "Q: What is consciousness?\nA: ", device='cuda')
        print(sample)


if __name__ == '__main__':
    main()
