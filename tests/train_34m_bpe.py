#!/usr/bin/env python3
"""
ARIANNA 34M BPE TRAINING SCRIPT

Same architecture as train_34m.py but with SentencePiece BPE tokenizer.
Char-level vocab=86 → BPE vocab=2000.

Architecture unchanged:
- dim=512, layers=10, heads=8 (MHA)
- Llama 3 style (RoPE, SwiGLU, RMSNorm)
- seq_len=1024 (now 1024 subword tokens ≈ 3-4K chars context)

Usage:
    python train_34m_bpe.py                              # Default training
    python train_34m_bpe.py --data path/to/corpus.txt    # Custom dataset
    python train_34m_bpe.py --resume checkpoint.pt       # Resume training
    python train_34m_bpe.py --lambda_mode                # H100 optimized
    python train_34m_bpe.py --vocab_size 2000            # Custom vocab size
"""

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import sentencepiece as spm
except ImportError:
    print("ERROR: sentencepiece not found. Install: pip install sentencepiece")
    raise


# ============================================================================
# BPE Tokenizer (SentencePiece)
# ============================================================================

class BPETokenizer:
    """SentencePiece BPE tokenizer for Arianna."""

    def __init__(self, sp: spm.SentencePieceProcessor):
        self.sp = sp
        self.vocab_size = sp.GetPieceSize()

    @classmethod
    def train(
        cls,
        corpus_path: str,
        vocab_size: int = 2000,
        model_prefix: str = "arianna_bpe",
        model_type: str = "bpe",
    ) -> "BPETokenizer":
        """Train BPE tokenizer on corpus."""
        print(f"[bpe] Training {model_type} tokenizer on {corpus_path}")
        print(f"[bpe] Target vocab_size={vocab_size}")

        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            max_sentence_length=16384,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            normalization_rule_name="identity",
            byte_fallback=True,
            split_digits=True,
            allow_whitespace_only_pieces=True,
        )

        model_path = f"{model_prefix}.model"
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)

        actual_vocab = sp.GetPieceSize()
        print(f"[bpe] Trained: {actual_vocab} tokens")

        return cls(sp)

    @classmethod
    def load(cls, model_path: str) -> "BPETokenizer":
        """Load pre-trained tokenizer."""
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        return cls(sp)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.DecodeIds(ids)

    def encode_pieces(self, text: str) -> List[str]:
        """Encode text to subword pieces (for visualization)."""
        return self.sp.EncodeAsPieces(text)

    def save_for_c(self, path: str):
        """
        Export tokenizer in JSON format for C inference.

        Format:
        {
            "type": "bpe",
            "vocab_size": N,
            "model_file": "arianna_bpe.model",
            "vocab": {"<pad>": 0, "<unk>": 1, ..., "token": id},
            "id_to_piece": {"0": "<pad>", "1": "<unk>", ..., "id": "token"}
        }
        """
        vocab = {}
        id_to_piece = {}
        for i in range(self.vocab_size):
            piece = self.sp.IdToPiece(i)
            vocab[piece] = i
            id_to_piece[str(i)] = piece

        data = {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            "vocab": vocab,
            "id_to_piece": id_to_piece,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[bpe] C tokenizer saved: {path}")

    def save_model(self, dir_path: str) -> str:
        """Copy .model file to directory. Returns the path."""
        import shutil
        # The model file was saved during training
        # We need to find it and copy it
        src = None
        for candidate in ["arianna_bpe.model", "bpe.model"]:
            if os.path.exists(candidate):
                src = candidate
                break

        if src:
            dst = os.path.join(dir_path, os.path.basename(src))
            shutil.copy2(src, dst)
            return dst
        return ""

    def analyze(self, text: str, n_samples: int = 5):
        """Show tokenization examples."""
        lines = text.strip().split('\n')
        samples = lines[:n_samples]

        print(f"\n[bpe] Tokenization examples:")
        for line in samples:
            line = line.strip()[:80]
            if not line:
                continue
            pieces = self.encode_pieces(line)
            ids = self.encode(line)
            ratio = len(line) / max(len(ids), 1)
            print(f"  \"{line[:50]}{'...' if len(line) > 50 else ''}\"")
            print(f"    → {len(ids)} tokens (compression: {ratio:.1f}x)")
            print(f"    pieces: {pieces[:10]}{'...' if len(pieces) > 10 else ''}")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Training configuration for 34M BPE model."""
    # Model architecture — same as char-level
    dim: int = 512
    n_layers: int = 10
    n_heads: int = 8
    n_kv_heads: int = 8
    vocab_size: int = 2000  # BPE default (was 86 char-level)
    max_seq_len: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 300
    max_iters: int = 40000
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 500

    # Paths
    data_path: str = 'd/arianna_unified2.txt'
    out_dir: str = 'arianna_34m_bpe'

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

class AriannaDataset(Dataset):
    """BPE token-level dataset for training."""

    def __init__(self, data_path: str, tokenizer: BPETokenizer, seq_len: int):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"[data] Corpus: {len(text):,} chars ({len(text)/1e6:.2f} MB)")

        tokens = tokenizer.encode(text)
        print(f"[data] Encoded: {len(tokens):,} tokens (compression: {len(text)/len(tokens):.1f}x)")

        if len(tokens) < seq_len + 1:
            raise ValueError(f"Dataset too small: {len(tokens)} tokens, need at least {seq_len + 1}")

        self.data = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


# ============================================================================
# Model (identical to train_34m.py)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0, device: str = 'cpu'):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(max_seq_len, device=device)
    angles = torch.outer(positions, freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return torch.stack([cos, sin], dim=-1)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[1]
    freqs = freqs_cis[:seq_len]
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    cos = freqs[..., 0].unsqueeze(0).unsqueeze(2)
    sin = freqs[..., 1].unsqueeze(0).unsqueeze(2)
    x0 = x_r[..., 0]
    x1 = x_r[..., 1]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    out = torch.stack([out0, out1], dim=-1)
    return out.reshape(*x.shape).type_as(x)


class Attention(nn.Module):
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
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=2)
            v = v.repeat_interleave(self.n_kv_groups, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(attn_out)


class FeedForward(nn.Module):
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


class Arianna34M(nn.Module):
    """Arianna 34M BPE: same architecture, subword tokenization."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.register_buffer(
            'freqs_cis',
            precompute_rope_freqs(config.head_dim, config.max_seq_len, config.rope_theta)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        h = self.tok_emb(x)
        for layer in self.layers:
            h = layer(h, self.freqs_cis)
        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Learning Rate Schedule
# ============================================================================

def get_lr(it: int, config: TrainConfig) -> float:
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / config.warmup_iters
    if it >= config.max_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# ============================================================================
# Training Loop
# ============================================================================

def train(config: TrainConfig, resume_path: Optional[str] = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    n_gpus = torch.cuda.device_count() if device == 'cuda' else 0
    if device == 'cuda':
        print(f"GPUs available: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    os.makedirs(config.out_dir, exist_ok=True)

    # Train BPE tokenizer
    print(f"\nTraining BPE tokenizer on: {config.data_path}")
    tokenizer_prefix = os.path.join(config.out_dir, "arianna_bpe")
    tokenizer = BPETokenizer.train(
        corpus_path=config.data_path,
        vocab_size=config.vocab_size,
        model_prefix=tokenizer_prefix,
    )
    config.vocab_size = tokenizer.vocab_size  # actual may differ slightly
    print(f"Final vocab_size: {config.vocab_size}")

    # Save tokenizer
    tokenizer.save_for_c(os.path.join(config.out_dir, 'tokenizer_bpe.json'))

    # Show tokenization examples
    with open(config.data_path, 'r', encoding='utf-8') as f:
        sample_text = f.read()
    tokenizer.analyze(sample_text)

    # Create dataset
    dataset = AriannaDataset(config.data_path, tokenizer, config.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
        drop_last=True,
    )

    # Create model
    print(f"\nCreating Arianna 34M BPE model...")
    print(f"  dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}")
    print(f"  hidden_dim={config.hidden_dim}, head_dim={config.head_dim}")
    print(f"  vocab_size={config.vocab_size} (BPE)")
    print(f"  max_seq_len={config.max_seq_len}")

    model = Arianna34M(config).to(device)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Parameter breakdown
    tok_params = config.vocab_size * config.dim
    lm_params = config.vocab_size * config.dim
    fixed_params = n_params - tok_params - lm_params
    print(f"  Fixed (arch): {fixed_params:,}")
    print(f"  tok_emb: {tok_params:,}")
    print(f"  lm_head: {lm_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    # Resume from checkpoint
    start_iter = 0
    if resume_path and os.path.exists(resume_path):
        print(f"\nResuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        state_dict = checkpoint['model']
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint.get('iter', 0) + 1
        print(f"Resumed at iteration {start_iter}")

    # Save config
    config_dict = {k: v for k, v in config.__dict__.items()}
    with open(os.path.join(config.out_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING — ARIANNA 34M BPE")
    print("=" * 60)

    model.train()
    data_iter = iter(dataloader)
    start_time = time.time()
    best_loss = float('inf')

    for it in range(start_iter, config.max_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        lr = get_lr(it, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

        loss.backward()

        if (it + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if it % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (it + 1 - start_iter) * config.batch_size * config.max_seq_len / max(elapsed, 1)

            if loss.item() < best_loss:
                best_loss = loss.item()

            print(f"iter {it:5d} | loss {loss.item():.4f} | best {best_loss:.4f} | "
                  f"lr {lr:.2e} | {tokens_per_sec/1000:.1f}K tok/s")

        if it > 0 and it % config.save_interval == 0:
            checkpoint_path = os.path.join(config.out_dir, f'checkpoint_{it}.pt')
            state_dict = model.state_dict()
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            torch.save({
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'config': config_dict,
                'iter': it,
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — ARIANNA 34M BPE")
    print("=" * 60)

    final_path = os.path.join(config.out_dir, 'arianna_34m_bpe_final.pt')
    state_dict = model.state_dict()
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    torch.save({
        'model': state_dict,
        'config': config_dict,
    }, final_path)
    print(f"Final model saved: {final_path}")

    return model, tokenizer


# ============================================================================
# Generation
# ============================================================================

@torch.no_grad()
def generate(
    model: Arianna34M,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = 'cuda',
) -> str:
    model.eval()

    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.max_seq_len:]
        logits = model(x_cond)
        logits = logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        if temperature == 0:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_token], dim=1)

        # Decode last token to check for sentence end
        piece = tokenizer.decode([next_token.item()])
        if any(p in piece for p in '.!?') and x.shape[1] > len(tokens) + 50:
            break

    model.train()
    return tokenizer.decode(x[0].tolist())


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Arianna 34M BPE model')
    parser.add_argument('--data', type=str, default='d/arianna_unified2.txt',
                        help='Path to training corpus')
    parser.add_argument('--out_dir', type=str, default='arianna_34m_bpe',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--lambda_mode', action='store_true',
                        help='Optimize for Lambda H100')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='Override max iterations')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--vocab_size', type=int, default=2000,
                        help='BPE vocabulary size (default: 2000)')
    args = parser.parse_args()

    config = TrainConfig()
    config.data_path = args.data
    config.out_dir = args.out_dir
    config.vocab_size = args.vocab_size

    if args.lambda_mode:
        config.batch_size = 128
        config.gradient_accumulation_steps = 2
        config.max_iters = 40000
        print("Lambda H100 mode enabled!")

    if args.max_iters:
        config.max_iters = args.max_iters
    if args.batch_size:
        config.batch_size = args.batch_size

    model, tokenizer = train(config, args.resume)

    # Generate samples
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Sample generation (BPE):")
        print("=" * 60)

        prompts = [
            "Q: What is consciousness?\nA:",
            "Q: Who are you?\nA:",
            "Q: What do you feel?\nA:",
        ]

        for prompt in prompts:
            print(f"\n> {prompt}")
            response = generate(model, tokenizer, prompt, device='cuda')
            print(response[len(prompt):])


if __name__ == '__main__':
    main()
