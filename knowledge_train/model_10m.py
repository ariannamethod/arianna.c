#!/usr/bin/env python3
"""
🧠 EXTERNAL BRAIN 10M - Model Architecture

Llama 3 style transformer for External Brain (10M parameters):
- RMSNorm (not LayerNorm)
- RoPE positional embeddings (not learned)
- SwiGLU activation (not GELU)
- GQA (Grouped Query Attention)
- No bias terms

Target: ~10M parameters with character-level vocabulary

Configuration:
- dim: 384
- n_layers: 6
- n_heads: 4
- n_kv_heads: 2 (GQA ratio 2:1)
- hidden_dim: 1024 (SwiGLU: 4 * dim * 2/3, rounded to multiple of 64)
- vocab_size: 80 (arianna.c character vocabulary)
- max_seq_len: 512

Parameter calculation:
- Total: ~9.8M parameters

Designed for ~1.1MB datasets (ratio ~9)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExternalBrain10MConfig:
    """Configuration for External Brain 10M model."""
    
    # Model architecture
    dim: int = 384              # Embedding dimension
    n_layers: int = 6           # Number of transformer layers
    n_heads: int = 4            # Number of attention heads
    n_kv_heads: int = 2         # Number of KV heads (GQA)
    vocab_size: int = 80        # Character vocabulary size
    max_seq_len: int = 512      # Maximum sequence length
    
    # Training
    norm_eps: float = 1e-5      # RMSNorm epsilon
    rope_theta: float = 10000.0 # RoPE base frequency
    
    # Derived (computed in __post_init__)
    head_dim: int = 0
    hidden_dim: int = 0
    n_kv_groups: int = 0
    
    def __post_init__(self):
        self.head_dim = self.dim // self.n_heads
        # SwiGLU hidden dim: 4 * dim * 2/3, rounded to multiple of 64
        self.hidden_dim = int(self.dim * 4 * 2 / 3)
        self.hidden_dim = 64 * ((self.hidden_dim + 63) // 64)
        self.n_kv_groups = self.n_heads // self.n_kv_heads
    
    def to_dict(self):
        return {
            'dim': self.dim,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'n_kv_heads': self.n_kv_heads,
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'norm_eps': self.norm_eps,
            'rope_theta': self.rope_theta,
            'head_dim': self.head_dim,
            'hidden_dim': self.hidden_dim,
            'n_kv_groups': self.n_kv_groups,
        }


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """Precompute RoPE frequency tensor in complex form."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device)
    angles = torch.outer(positions, freqs)  # (max_seq_len, dim/2)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to queries or keys."""
    # x: (batch, seq_len, n_heads, head_dim)
    # freqs_cis: (seq_len, head_dim/2)
    
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_r)
    
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs
    
    x_out = torch.view_as_real(x_rotated)
    return x_out.reshape(*x.shape).type_as(x)


class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE."""
    
    def __init__(self, config: ExternalBrain10MConfig):
        super().__init__()
        self.config = config
        
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.head_dim = config.head_dim
        
        # Separate projections for Q, K, V (no bias)
        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
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
        
        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(attn_out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""
    
    def __init__(self, config: ExternalBrain10MConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w_up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w_down = nn.Linear(config.hidden_dim, config.dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        return self.w_down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    
    def __init__(self, config: ExternalBrain10MConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = FeedForward(config)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ExternalBrain10M(nn.Module):
    """
    External Brain 10M - Llama 3 style transformer.
    
    Smaller model for ~1.1MB datasets with ~9 params/byte ratio.
    Designed for Q&A format training like arianna_expanded_corpus.txt.
    """
    
    def __init__(self, config: ExternalBrain10MConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (no position embedding - we use RoPE)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        
        # Language model head
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
        """
        Forward pass for training.
        
        Args:
            x: Token indices (batch, seq_len)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = x.shape
        
        # Token embeddings
        h = self.tok_emb(x)
        
        # Get RoPE frequencies for this sequence
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis)
        
        # Output
        h = self.final_norm(h)
        logits = self.lm_head(h)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_params_breakdown(self) -> dict:
        """Get detailed parameter breakdown."""
        config = self.config
        kv_dim = config.n_kv_heads * config.head_dim
        
        breakdown = {
            'tok_emb': config.vocab_size * config.dim,
            'attn_norm': config.n_layers * config.dim,
            'wq': config.n_layers * config.dim * config.dim,
            'wk': config.n_layers * config.dim * kv_dim,
            'wv': config.n_layers * config.dim * kv_dim,
            'wo': config.n_layers * config.dim * config.dim,
            'ffn_norm': config.n_layers * config.dim,
            'w_gate': config.n_layers * config.dim * config.hidden_dim,
            'w_up': config.n_layers * config.dim * config.hidden_dim,
            'w_down': config.n_layers * config.hidden_dim * config.dim,
            'final_norm': config.dim,
            'lm_head': config.dim * config.vocab_size,
        }
        
        breakdown['total'] = sum(breakdown.values())
        return breakdown


def print_model_info():
    """Print model configuration and parameter count."""
    config = ExternalBrain10MConfig()
    breakdown = ExternalBrain10M(config).get_num_params_breakdown()
    
    print("=" * 60)
    print("🧠 EXTERNAL BRAIN 10M - Model Configuration")
    print("=" * 60)
    print(f"dim:           {config.dim}")
    print(f"n_layers:      {config.n_layers}")
    print(f"n_heads:       {config.n_heads}")
    print(f"n_kv_heads:    {config.n_kv_heads} (GQA ratio: {config.n_kv_groups}:1)")
    print(f"head_dim:      {config.head_dim}")
    print(f"hidden_dim:    {config.hidden_dim} (SwiGLU)")
    print(f"vocab_size:    {config.vocab_size}")
    print(f"max_seq_len:   {config.max_seq_len}")
    print()
    print("Parameter breakdown:")
    for name, count in breakdown.items():
        if name != 'total':
            print(f"  {name:15s}: {count:,}")
    print(f"  {'─' * 25}")
    print(f"  {'total':15s}: {breakdown['total']:,} ({breakdown['total']/1e6:.2f}M)")
    print()
    print(f"Size (float32): {breakdown['total'] * 4 / 1024 / 1024:.2f} MB")
    print(f"Size (float16): {breakdown['total'] * 2 / 1024 / 1024:.2f} MB")
    print()
    print("Recommended dataset size: ~1.1MB (ratio ~9)")
    print("=" * 60)


if __name__ == '__main__':
    print_model_info()
    
    # Quick test
    print("\n🧪 Quick forward pass test...")
    config = ExternalBrain10MConfig()
    model = ExternalBrain10M(config)
    
    x = torch.randint(0, config.vocab_size, (2, 32))  # batch=2, seq=32
    logits = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")
    print("✅ Model test passed!")
