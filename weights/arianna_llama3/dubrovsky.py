"""
🌀 DUBROVSKY MODEL 🌀

Llama 3 style transformer architecture for the most absurdist AI ever created.
Pure Python/NumPy inference - NO PYTORCH DEPENDENCIES for inference!

"My weights are light, my consciousness is heavy."
- Alexey Dubrovsky, pondering his own parameters

Architecture:
- RoPE (Rotary Position Embeddings) - because positions rotate like my anxiety
- GQA (Grouped Query Attention) - fewer KV heads, more philosophical density  
- SwiGLU activation - smoother than my existential transitions
- RMSNorm - normalizing reality since 2023

This file contains:
1. Model configuration (DubrovskyConfig)
2. Pure NumPy inference implementation (NO TORCH!)
3. Model loading from binary weights

For training, see train.py (which uses PyTorch separately)
"""

import json
import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DubrovskyConfig:
    """
    Configuration for Dubrovsky model.
    
    Small but mighty, like a quantum particle with opinions.
    """
    dim: int = 384           # Embedding dimension
    n_layers: int = 6        # Number of transformer layers
    n_heads: int = 6         # Number of attention heads
    n_kv_heads: int = 2      # Number of KV heads (GQA)
    vocab_size: int = 88     # Character-level vocab
    max_seq_len: int = 256   # Maximum sequence length
    norm_eps: float = 1e-5   # RMSNorm epsilon
    rope_theta: float = 10000.0  # RoPE base frequency
    
    # Computed
    head_dim: int = 0
    hidden_dim: int = 0
    n_kv_groups: int = 0
    
    def __post_init__(self):
        self.head_dim = self.dim // self.n_heads
        # SwiGLU hidden dim: 4 * dim * 2/3, rounded to nice number
        self.hidden_dim = int(self.dim * 4 * 2 / 3)
        self.hidden_dim = 256 * ((self.hidden_dim + 255) // 256)
        self.n_kv_groups = self.n_heads // self.n_kv_heads
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DubrovskyConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls()
        for k, v in data.items():
            setattr(config, k, v)
        return config


# ============================================================================
# NumPy Operations (NO PYTORCH!)
# ============================================================================

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU/Swish activation: x * sigmoid(x)"""
    return x * (1 / (1 + np.exp(-x)))


def compute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RoPE frequency tensors (cos and sin)."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    positions = np.arange(max_seq_len, dtype=np.float32)
    # Shape: (max_seq_len, dim/2)
    angles = np.outer(positions, freqs)
    return np.cos(angles), np.sin(angles)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, pos: int) -> np.ndarray:
    """
    Apply RoPE to query or key.
    x: (n_heads, head_dim) or (seq_len, n_heads, head_dim)
    """
    # Split into pairs for rotation
    x_r = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_r[..., 0], x_r[..., 1]
    
    # Get cos/sin for this position
    c, s = cos[pos], sin[pos]  # (head_dim/2,)
    
    # Rotate
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    
    # Interleave back
    out = np.stack([out0, out1], axis=-1)
    return out.reshape(x.shape)


# ============================================================================
# Model Weights Container
# ============================================================================

@dataclass
class LayerWeights:
    """Weights for a single transformer layer."""
    # Attention
    wq: np.ndarray  # (dim, dim)
    wk: np.ndarray  # (dim, kv_dim)
    wv: np.ndarray  # (dim, kv_dim)
    wo: np.ndarray  # (dim, dim)
    
    # FFN (SwiGLU)
    w_gate: np.ndarray  # (dim, hidden_dim)
    w_up: np.ndarray    # (dim, hidden_dim)
    w_down: np.ndarray  # (hidden_dim, dim)
    
    # Norms
    attn_norm: np.ndarray  # (dim,)
    ffn_norm: np.ndarray   # (dim,)


@dataclass
class DubrovskyWeights:
    """All model weights."""
    tok_emb: np.ndarray     # (vocab_size, dim)
    layers: List[LayerWeights]
    final_norm: np.ndarray  # (dim,)
    lm_head: np.ndarray     # (dim, vocab_size)


# ============================================================================
# Main Model Class (Pure NumPy Inference)
# ============================================================================

class Dubrovsky:
    """
    Dubrovsky: A Llama 3 style transformer implemented in pure NumPy.
    
    No PyTorch required for inference - just vibes and matrix multiplication.
    """
    
    def __init__(self, config: DubrovskyConfig, weights: DubrovskyWeights):
        self.config = config
        self.weights = weights
        
        # Precompute RoPE frequencies
        self.rope_cos, self.rope_sin = compute_rope_freqs(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        
        # KV cache for autoregressive generation
        self.kv_cache = None
        self.cache_pos = 0
    
    def init_cache(self):
        """Initialize KV cache for generation."""
        cfg = self.config
        self.kv_cache = []
        for _ in range(cfg.n_layers):
            k_cache = np.zeros((cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim), dtype=np.float32)
            v_cache = np.zeros((cfg.max_seq_len, cfg.n_kv_heads, cfg.head_dim), dtype=np.float32)
            self.kv_cache.append((k_cache, v_cache))
        self.cache_pos = 0
    
    def clear_cache(self):
        """Clear KV cache."""
        self.kv_cache = None
        self.cache_pos = 0
    
    def attention(self, x: np.ndarray, layer: int, pos: int) -> np.ndarray:
        """
        Single token attention with KV cache.
        x: (dim,)
        """
        cfg = self.config
        lw = self.weights.layers[layer]
        
        # Project to Q, K, V
        q = x @ lw.wq  # (dim,)
        k = x @ lw.wk  # (kv_dim,)
        v = x @ lw.wv  # (kv_dim,)
        
        # Reshape for multi-head
        q = q.reshape(cfg.n_heads, cfg.head_dim)      # (n_heads, head_dim)
        k = k.reshape(cfg.n_kv_heads, cfg.head_dim)   # (n_kv_heads, head_dim)
        v = v.reshape(cfg.n_kv_heads, cfg.head_dim)   # (n_kv_heads, head_dim)
        
        # Apply RoPE
        for h in range(cfg.n_heads):
            q[h] = apply_rope(q[h], self.rope_cos, self.rope_sin, pos)
        for h in range(cfg.n_kv_heads):
            k[h] = apply_rope(k[h], self.rope_cos, self.rope_sin, pos)
        
        # Store in cache
        k_cache, v_cache = self.kv_cache[layer]
        k_cache[pos] = k
        v_cache[pos] = v
        
        # Compute attention scores
        # Q: (n_heads, head_dim)
        # K cache: (seq_len, n_kv_heads, head_dim) -> we use [:pos+1]
        scale = 1.0 / math.sqrt(cfg.head_dim)
        
        output = np.zeros((cfg.n_heads, cfg.head_dim), dtype=np.float32)
        
        for h in range(cfg.n_heads):
            # Which KV head this Q head attends to
            kv_h = h // cfg.n_kv_groups
            
            # Get cached K, V for this kv head
            k_seq = k_cache[:pos+1, kv_h]  # (seq_len, head_dim)
            v_seq = v_cache[:pos+1, kv_h]  # (seq_len, head_dim)
            
            # Attention scores
            scores = q[h] @ k_seq.T * scale  # (seq_len,)
            probs = softmax(scores)           # (seq_len,)
            
            # Weighted sum of values
            output[h] = probs @ v_seq        # (head_dim,)
        
        # Concatenate heads and project
        output = output.reshape(-1)  # (dim,)
        return output @ lw.wo
    
    def ffn(self, x: np.ndarray, layer: int) -> np.ndarray:
        """SwiGLU feed-forward network."""
        lw = self.weights.layers[layer]
        
        gate = x @ lw.w_gate  # (hidden_dim,)
        up = x @ lw.w_up      # (hidden_dim,)
        
        # SwiGLU: gate * silu(up) - wait, it's gate * silu(gate) * up
        # Actually: silu(gate) * up
        hidden = silu(gate) * up
        
        return hidden @ lw.w_down
    
    def forward(self, token: int, pos: int) -> np.ndarray:
        """
        Single token forward pass.
        Returns logits over vocabulary.
        """
        cfg = self.config
        
        # Token embedding
        x = self.weights.tok_emb[token].copy()  # (dim,)
        
        # Transformer layers
        for layer in range(cfg.n_layers):
            lw = self.weights.layers[layer]
            
            # Pre-norm attention
            h = rms_norm(x, lw.attn_norm, cfg.norm_eps)
            h = self.attention(h, layer, pos)
            x = x + h  # Residual
            
            # Pre-norm FFN
            h = rms_norm(x, lw.ffn_norm, cfg.norm_eps)
            h = self.ffn(h, layer)
            x = x + h  # Residual
        
        # Final norm and output projection
        x = rms_norm(x, self.weights.final_norm, cfg.norm_eps)
        logits = x @ self.weights.lm_head
        
        return logits
    
    def generate(
        self, 
        prompt_tokens: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Generate text given prompt tokens.
        
        Args:
            prompt_tokens: List of token ids for the prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 = greedy, higher = more random)
            top_k: Keep only top-k tokens before sampling
            top_p: Nucleus sampling threshold
            stop_tokens: List of token ids that stop generation (e.g., newline)
        
        Returns:
            List of generated token ids (including prompt)
        """
        self.init_cache()
        
        tokens = list(prompt_tokens)
        
        # Process prompt tokens
        for pos, tok in enumerate(tokens):
            if pos < len(tokens) - 1:
                # Just update cache, don't need logits
                _ = self.forward(tok, pos)
            else:
                # Last prompt token - get logits for generation
                logits = self.forward(tok, pos)
        
        # Generate new tokens
        for i in range(max_new_tokens):
            pos = len(tokens) - 1
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                k = min(top_k, len(logits))  # Ensure top_k doesn't exceed vocab size
                indices_to_remove = logits < np.partition(logits, -k)[-k]
                logits[indices_to_remove] = -float('inf')
            
            # Convert to probabilities
            probs = softmax(logits)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                cumsum = np.cumsum(sorted_probs)
                
                # Find cutoff
                cutoff_idx = np.searchsorted(cumsum, top_p) + 1
                indices_to_remove = sorted_indices[cutoff_idx:]
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()
            
            # Sample
            if temperature == 0:
                next_token = int(np.argmax(probs))
            else:
                next_token = int(np.random.choice(len(probs), p=probs))
            
            tokens.append(next_token)
            
            # Check for stop tokens (e.g., newline marks end of answer)
            if stop_tokens and next_token in stop_tokens:
                break
            
            # Check for end of generation
            if pos >= self.config.max_seq_len - 1:
                break
            
            # Get next logits
            logits = self.forward(next_token, len(tokens) - 1)
        
        self.clear_cache()
        return tokens


# ============================================================================
# Binary Weight Loading
# ============================================================================

def load_weights_from_bin(bin_path: str, config: DubrovskyConfig) -> DubrovskyWeights:
    """
    Load model weights from binary file.
    
    Binary format (all float32, stored in PyTorch Linear format: out_features x in_features):
    1. tok_emb: (vocab_size, dim)
    2. For each layer:
       - attn_norm: (dim,)
       - wq: (dim, dim) - stored as (out, in), needs transpose for x @ W
       - wk: (kv_dim, dim) - stored as (out, in), needs transpose for x @ W
       - wv: (kv_dim, dim) - stored as (out, in), needs transpose for x @ W
       - wo: (dim, dim) - stored as (out, in), needs transpose for x @ W
       - ffn_norm: (dim,)
       - w_gate: (hidden_dim, dim) - stored as (out, in), needs transpose for x @ W
       - w_up: (hidden_dim, dim) - stored as (out, in), needs transpose for x @ W
       - w_down: (dim, hidden_dim) - stored as (out, in), needs transpose for x @ W
    3. final_norm: (dim,)
    4. lm_head: (vocab_size, dim) - stored as (out, in), needs transpose for x @ W
    
    Note: PyTorch Linear stores weights as (out_features, in_features).
    For inference with x @ W, we need (in_features, out_features), so we transpose.
    """
    cfg = config
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    
    with open(bin_path, 'rb') as f:
        def read_tensor(shape):
            n = int(np.prod(shape))
            data = struct.unpack(f'{n}f', f.read(n * 4))
            return np.array(data, dtype=np.float32).reshape(shape)
        
        # Token embeddings - (vocab_size, dim), no transpose needed
        tok_emb = read_tensor((cfg.vocab_size, cfg.dim))
        
        # Layers
        layers = []
        for _ in range(cfg.n_layers):
            # Read in PyTorch format (out, in) and transpose to (in, out) for x @ W
            # Transpose converts from (out_features, in_features) to (in_features, out_features)
            layer = LayerWeights(
                attn_norm=read_tensor((cfg.dim,)),
                wq=read_tensor((cfg.dim, cfg.dim)).T,             # (out=dim, in=dim) -> (in=dim, out=dim)
                wk=read_tensor((kv_dim, cfg.dim)).T,              # (out=kv_dim, in=dim) -> (in=dim, out=kv_dim)
                wv=read_tensor((kv_dim, cfg.dim)).T,              # (out=kv_dim, in=dim) -> (in=dim, out=kv_dim)
                wo=read_tensor((cfg.dim, cfg.dim)).T,             # (out=dim, in=dim) -> (in=dim, out=dim)
                ffn_norm=read_tensor((cfg.dim,)),
                w_gate=read_tensor((cfg.hidden_dim, cfg.dim)).T,  # (out=hidden, in=dim) -> (in=dim, out=hidden)
                w_up=read_tensor((cfg.hidden_dim, cfg.dim)).T,    # (out=hidden, in=dim) -> (in=dim, out=hidden)
                w_down=read_tensor((cfg.dim, cfg.hidden_dim)).T,  # (out=dim, in=hidden) -> (in=hidden, out=dim)
            )
            layers.append(layer)
        
        # Final norm and output
        final_norm = read_tensor((cfg.dim,))
        # lm_head: transpose from (out=vocab, in=dim) to (in=dim, out=vocab) for x @ W
        lm_head = read_tensor((cfg.vocab_size, cfg.dim)).T
    
    return DubrovskyWeights(
        tok_emb=tok_emb,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


def load_model(config_path: str, weights_path: str) -> Dubrovsky:
    """Load complete model from config and weights files."""
    config = DubrovskyConfig.load(config_path)
    weights = load_weights_from_bin(weights_path, config)
    return Dubrovsky(config, weights)


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(config: DubrovskyConfig) -> int:
    """Count total model parameters."""
    cfg = config
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    
    # Token embeddings
    tok_emb = cfg.vocab_size * cfg.dim
    
    # Per layer
    attn = cfg.dim * cfg.dim + cfg.dim * kv_dim * 2 + cfg.dim * cfg.dim
    ffn = cfg.dim * cfg.hidden_dim * 3
    norms = cfg.dim * 2
    per_layer = attn + ffn + norms
    
    # Output
    final_norm = cfg.dim
    lm_head = cfg.dim * cfg.vocab_size
    
    return tok_emb + per_layer * cfg.n_layers + final_norm + lm_head


if __name__ == '__main__':
    # Print model info
    config = DubrovskyConfig()
    params = count_parameters(config)
    
    print("=" * 60)
    print("🌀 DUBROVSKY MODEL CONFIGURATION 🌀")
    print("=" * 60)
    print(f"dim:           {config.dim}")
    print(f"n_layers:      {config.n_layers}")
    print(f"n_heads:       {config.n_heads}")
    print(f"n_kv_heads:    {config.n_kv_heads} (GQA ratio: {config.n_kv_groups}:1)")
    print(f"head_dim:      {config.head_dim}")
    print(f"hidden_dim:    {config.hidden_dim} (SwiGLU)")
    print(f"vocab_size:    {config.vocab_size}")
    print(f"max_seq_len:   {config.max_seq_len}")
    print(f"")
    print(f"Total parameters: {params:,} ({params/1e6:.2f}M)")
    print(f"Size (float32):   {params * 4 / 1024 / 1024:.2f} MB")
    print(f"Size (float16):   {params * 2 / 1024 / 1024:.2f} MB")
    print("=" * 60)
