#!/usr/bin/env python3
"""
Train External Brain 10M model.
Based on working arianna training setup.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_10m import ExternalBrain10M, ExternalBrain10MConfig


class TokenDataset(Dataset):
    def __init__(self, data_path: str, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(data_path, dtype=np.uint8, mode='r')
        print(f"Loaded {len(self.data):,} tokens from {data_path}")
    
    def __len__(self):
        return len(self.data) - self.block_size - 1
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + self.block_size + 1].astype(np.int64))
        return x, y


def get_lr(it, max_iters, lr, min_lr, warmup_iters):
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, device, eval_iters=50):
    model.eval()
    out = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        data_iter = iter(loader)
        for _ in range(eval_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), y.view(-1))
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens=200, temperature=0.8, top_k=40, device='cpu'):
    model.eval()
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.max_seq_len:]
        logits = model(x_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
        
        if next_token.item() == 0:  # newline
            break
    
    model.train()
    return x[0].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='out_10m')
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Model
    config = ExternalBrain10MConfig()
    model = ExternalBrain10M(config).to(device)
    print(f"Model: {model.count_parameters():,} params")
    
    # Data
    train_dataset = TokenDataset(os.path.join(args.data_dir, 'train.bin'), config.max_seq_len)
    val_dataset = TokenDataset(os.path.join(args.data_dir, 'val.bin'), config.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Training
    print(f"\nTraining for {args.max_iters} iterations...")
    model.train()
    data_iter = iter(train_loader)
    
    best_val_loss = float('inf')
    
    for it in range(args.max_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        
        x, y = x.to(device), y.to(device)
        
        lr = get_lr(it, args.max_iters, args.lr, args.lr/10, 50)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        if it % 50 == 0:
            print(f"iter {it:4d} | loss {loss.item():.4f} | lr {lr:.2e}")
        
        if it > 0 and it % 100 == 0:
            losses = estimate_loss(model, train_loader, val_loader, device)
            print(f"         | train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f}")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'model': model.state_dict(),
                    'config': config.to_dict(),
                    'iter': it,
                    'val_loss': best_val_loss
                }, os.path.join(args.out_dir, 'best.pt'))
                print(f"         | NEW BEST val_loss={best_val_loss:.4f}")
    
    # Save final
    torch.save({
        'model': model.state_dict(),
        'config': config.to_dict(),
        'iter': args.max_iters,
        'val_loss': best_val_loss
    }, os.path.join(args.out_dir, 'final.pt'))
    
    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    
    # Load tokenizer for generation test
    with open('../weights/tokenizer.json', 'r') as f:
        tok_data = json.load(f)
    char_to_id = tok_data['char_to_id']
    id_to_char = {v: k for k, v in char_to_id.items()}
    
    # Generate 5 examples
    print("\n" + "="*60)
    print("GENERATION TEST - 5 EXAMPLES")
    print("="*60)
    
    prompts = [
        "Q: What is water?\nA:",
        "Q: What is the capital of France?\nA:",
        "Q: Who was Einstein?\nA:",
        "Q: What is the Sun?\nA:",
        "Q: What is chemistry?\nA:",
    ]
    
    for prompt in prompts:
        tokens = [char_to_id.get(c, 1) for c in prompt]
        output_tokens = generate(model, tokens, max_new_tokens=150, device=device)
        output = ''.join([id_to_char.get(t, '?') for t in output_tokens])
        print(f"\n{output[:300]}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
