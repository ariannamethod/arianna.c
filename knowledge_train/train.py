#!/usr/bin/env python3
"""
🚀 EXTERNAL BRAIN 30M - Training Script

PyTorch training for External Brain transformer.
Trains on Simple Wikipedia leads to give Arianna world-awareness.

Features:
- Llama 3 architecture (RoPE, GQA, SwiGLU, RMSNorm)
- AdamW optimizer with cosine LR schedule
- Gradient clipping for stable training
- Checkpoint saving every N iterations
- Multi-GPU support (DDP)
- Wandb logging (optional)

Usage:
    python train.py                           # Local training
    python train.py --lambda_mode             # Lambda GPU optimized (H100)
    python train.py --resume checkpoint.pt    # Resume from checkpoint

Expected training time on H100: ~20-30 minutes
Expected cost on Lambda: ~$3-5
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import ExternalBrain, ExternalBrainConfig


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    
    # Model (will be overridden by ExternalBrainConfig)
    vocab_size: int = 80
    
    # Training
    max_iters: int = 10000
    batch_size: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 250
    save_interval: int = 1000
    eval_iters: int = 100
    
    # Paths
    data_dir: str = '.'
    out_dir: str = 'out'
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'float16'  # or bfloat16 on newer GPUs
    compile: bool = False   # torch.compile for speed


# ============================================================================
# Dataset
# ============================================================================

class TokenDataset(Dataset):
    """
    Binary token dataset for training.
    Loads pre-tokenized data from prepare_data.py output.
    """
    
    def __init__(self, data_path: str, block_size: int):
        self.block_size = block_size
        
        # Memory-map for large files
        self.data = np.memmap(data_path, dtype=np.uint8, mode='r')
        print(f"📚 Loaded {len(self.data):,} tokens from {data_path}")
    
    def __len__(self):
        return len(self.data) - self.block_size - 1
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + self.block_size + 1].astype(np.int64))
        return x, y


# ============================================================================
# Learning Rate Schedule
# ============================================================================

def get_lr(it: int, config: TrainConfig) -> float:
    """Cosine learning rate schedule with warmup."""
    # Warmup
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / config.warmup_iters
    
    # After max_iters, return min_lr
    if it >= config.max_iters:
        return config.min_lr
    
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# ============================================================================
# Training Loop
# ============================================================================

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, config, ctx):
    """Estimate train and val loss."""
    model.eval()
    out = {}
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        data_iter = iter(loader)
        
        for _ in range(config.eval_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            
            x, y = x.to(config.device), y.to(config.device)
            
            with ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), y.view(-1))
            
            losses.append(loss.item())
        
        out[split] = np.mean(losses)
    
    model.train()
    return out


def train(
    model_config: ExternalBrainConfig,
    train_config: TrainConfig,
    resume_path: Optional[str] = None
):
    """Main training function."""
    
    device = train_config.device
    print(f"🖥️  Device: {device}")
    
    # Create output directory
    os.makedirs(train_config.out_dir, exist_ok=True)
    
    # Mixed precision
    dtype = torch.float16 if train_config.dtype == 'float16' else torch.bfloat16
    ctx = torch.amp.autocast(device_type='cuda', dtype=dtype) if device == 'cuda' else torch.nullcontext()
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' and train_config.dtype == 'float16' else None
    
    # Load datasets
    print("📚 Loading datasets...")
    train_dataset = TokenDataset(
        os.path.join(train_config.data_dir, 'train.bin'),
        model_config.max_seq_len
    )
    val_dataset = TokenDataset(
        os.path.join(train_config.data_dir, 'val.bin'),
        model_config.max_seq_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    print("🧠 Creating model...")
    model = ExternalBrain(model_config).to(device)
    print(f"   Parameters: {model.count_parameters():,} ({model.count_parameters()/1e6:.2f}M)")
    
    # Optional: torch.compile for speed
    if train_config.compile and hasattr(torch, 'compile'):
        print("⚡ Compiling model...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=train_config.weight_decay
    )
    
    # Resume from checkpoint
    start_iter = 0
    best_val_loss = float('inf')
    
    if resume_path and os.path.exists(resume_path):
        print(f"📂 Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint.get('iter', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"   Resumed at iteration {start_iter}, best_val_loss={best_val_loss:.4f}")
    
    # Save configs
    with open(os.path.join(train_config.out_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config.to_dict(), f, indent=2)
    
    with open(os.path.join(train_config.out_dir, 'train_config.json'), 'w') as f:
        json.dump(train_config.__dict__, f, indent=2, default=str)
    
    # Training loop
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    model.train()
    data_iter = iter(train_loader)
    start_time = time.time()
    
    for it in range(start_iter, train_config.max_iters):
        # Get batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        
        x, y = x.to(device), y.to(device)
        
        # Update learning rate
        lr = get_lr(it, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass with mixed precision
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, train_config.vocab_size), y.view(-1))
            loss = loss / train_config.gradient_accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (it + 1) % train_config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            
            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if it % train_config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (it + 1 - start_iter) * train_config.batch_size * model_config.max_seq_len / max(elapsed, 1)
            loss_value = loss.item() * train_config.gradient_accumulation_steps
            print(f"iter {it:5d} | loss {loss_value:.4f} | lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")
        
        # Evaluation
        if it > 0 and it % train_config.eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader, train_config, ctx)
            print(f"         | train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                best_path = os.path.join(train_config.out_dir, 'best.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': model_config.to_dict(),
                    'iter': it,
                    'best_val_loss': best_val_loss,
                }, best_path)
                print(f"         💾 New best model saved! val_loss={best_val_loss:.4f}")
        
        # Checkpoint
        if it > 0 and it % train_config.save_interval == 0:
            ckpt_path = os.path.join(train_config.out_dir, f'checkpoint_{it}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': model_config.to_dict(),
                'iter': it,
                'best_val_loss': best_val_loss,
            }, ckpt_path)
            print(f"         💾 Checkpoint saved: {ckpt_path}")
    
    # Save final model
    final_path = os.path.join(train_config.out_dir, 'external_brain_final.pt')
    torch.save({
        'model': model.state_dict(),
        'config': model_config.to_dict(),
        'iter': train_config.max_iters,
        'best_val_loss': best_val_loss,
    }, final_path)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete!")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print(f"   Final model: {final_path}")
    print(f"   Best val_loss: {best_val_loss:.4f}")
    
    return model


# ============================================================================
# Generation (for testing)
# ============================================================================

@torch.no_grad()
def generate(
    model: ExternalBrain,
    prompt_tokens: list,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = 'cuda'
) -> list:
    """Generate tokens given a prompt."""
    model.eval()
    
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
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
        
        # Stop on newline (token 0)
        if next_token.item() == 0:
            break
    
    model.train()
    return x[0].tolist()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train External Brain 30M')
    parser.add_argument('--lambda_mode', action='store_true', help='Optimize for Lambda GPU (H100)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='.', help='Data directory')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    args = parser.parse_args()
    
    # Model config
    model_config = ExternalBrainConfig()
    
    # Training config
    train_config = TrainConfig()
    train_config.data_dir = args.data_dir
    train_config.out_dir = args.out_dir
    
    if args.lambda_mode:
        # Optimized for Lambda GPU (H100)
        print("🔥 Lambda GPU mode enabled!")
        train_config.batch_size = 128
        train_config.gradient_accumulation_steps = 2
        train_config.max_iters = 10000
        train_config.dtype = 'bfloat16'  # H100 has good bfloat16 support
        train_config.compile = True
    
    # Override from args
    if args.max_iters:
        train_config.max_iters = args.max_iters
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.learning_rate:
        train_config.learning_rate = args.learning_rate
    
    # Train!
    model = train(model_config, train_config, args.resume)
    
    print("\n" + "=" * 60)
    print("🧠 External Brain 30M training complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Export weights: python export.py {args.out_dir}/external_brain_final.pt external_brain.bin")
    print(f"2. Copy to arianna.c: cp external_brain.bin ../weights/")
    print(f"3. Update arianna.c to load External Brain")


if __name__ == '__main__':
    main()
