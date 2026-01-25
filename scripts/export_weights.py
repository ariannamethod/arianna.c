#!/usr/bin/env python3
"""
Universal export script for Arianna models (20M/34M) to .bin format.
Supports both train_20m.py and train_34m.py checkpoints.
"""

import struct
import torch
import sys
import os

# Add paths for model imports
sys.path.insert(0, 'train')
sys.path.insert(0, 'tests')


def detect_model_type(checkpoint):
    """Detect if this is 20M or 34M based on checkpoint structure."""
    state = checkpoint.get('model', checkpoint)

    # Check embedding dimension
    if 'tok_emb.weight' in state:
        vocab, dim = state['tok_emb.weight'].shape
    else:
        raise ValueError("Cannot find tok_emb.weight in checkpoint")

    # Count layers
    n_layers = 0
    while f'layers.{n_layers}.attn.wq.weight' in state:
        n_layers += 1

    # Detect based on dim
    if dim == 448:
        return '20m', dim, n_layers
    elif dim == 512:
        return '34m', dim, n_layers
    else:
        return 'unknown', dim, n_layers


def export_checkpoint(checkpoint_path, output_path, data_path=None):
    """Export PyTorch checkpoint to .bin format."""

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_type, dim, n_layers = detect_model_type(checkpoint)
    print(f"Detected: {model_type} (dim={dim}, layers={n_layers})")

    # Get model state dict
    if 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint

    # Infer config from weights
    vocab_size, _ = state['tok_emb.weight'].shape
    n_heads = 8  # Both models use 8 heads
    n_kv_heads = 8  # Both use full attention
    head_dim = dim // n_heads

    # Hidden dim from w_gate
    hidden_dim, _ = state['layers.0.ffn.w_gate.weight'].shape

    max_seq_len = 1024  # Extended context
    rope_theta = 10000.0
    norm_eps = 1e-5
    n_kv_groups = n_heads // n_kv_heads

    print(f"\nConfig:")
    print(f"  dim: {dim}")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads}")
    print(f"  n_kv_heads: {n_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  vocab_size: {vocab_size}")

    # Export
    print(f"\nExporting to: {output_path}")

    with open(output_path, 'wb') as f:
        # Magic: "naka" = 0x616B616E
        f.write(struct.pack('<I', 0x616B616E))

        # Config
        f.write(struct.pack('<i', dim))
        f.write(struct.pack('<i', n_layers))
        f.write(struct.pack('<i', n_heads))
        f.write(struct.pack('<i', n_kv_heads))
        f.write(struct.pack('<i', head_dim))
        f.write(struct.pack('<i', hidden_dim))
        f.write(struct.pack('<i', max_seq_len))
        f.write(struct.pack('<i', vocab_size))
        f.write(struct.pack('<i', n_kv_groups))
        f.write(struct.pack('<f', rope_theta))
        f.write(struct.pack('<f', norm_eps))

        def write_tensor(name, key):
            tensor = state[key]
            data = tensor.detach().float().numpy().flatten()
            f.write(data.tobytes())
            print(f"  {name}: {tensor.shape} -> {len(data)} floats")

        # Token embeddings
        write_tensor("tok_emb", "tok_emb.weight")

        # Per-layer weights
        for i in range(n_layers):
            print(f"  Layer {i}:")
            write_tensor("    attn_norm", f"layers.{i}.attn_norm.weight")
            write_tensor("    wq", f"layers.{i}.attn.wq.weight")
            write_tensor("    wk", f"layers.{i}.attn.wk.weight")
            write_tensor("    wv", f"layers.{i}.attn.wv.weight")
            write_tensor("    wo", f"layers.{i}.attn.wo.weight")
            write_tensor("    ffn_norm", f"layers.{i}.ffn_norm.weight")
            write_tensor("    w_gate", f"layers.{i}.ffn.w_gate.weight")
            write_tensor("    w_up", f"layers.{i}.ffn.w_up.weight")
            write_tensor("    w_down", f"layers.{i}.ffn.w_down.weight")

        # Final norm and output
        write_tensor("final_norm", "final_norm.weight")
        write_tensor("lm_head", "lm_head.weight")

    size = os.path.getsize(output_path)
    print(f"\nDone! Exported {size:,} bytes ({size/1024/1024:.2f} MB)")

    # Save tokenizer if we can find it
    if 'tokenizer' in checkpoint:
        tok = checkpoint['tokenizer']
        tokenizer_path = output_path.replace('.bin', '_tokenizer.json')
        import json
        with open(tokenizer_path, 'w') as tf:
            json.dump({
                'char_to_id': tok.get('char_to_idx', tok),
                'id_to_char': {str(k): v for k, v in tok.get('idx_to_char', {}).items()},
                'vocab_size': vocab_size
            }, tf, indent=2)
        print(f"Tokenizer saved: {tokenizer_path}")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_weights.py <checkpoint.pt> [output.bin]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = checkpoint_path.replace('.pt', '.bin')

    export_checkpoint(checkpoint_path, output_path)
