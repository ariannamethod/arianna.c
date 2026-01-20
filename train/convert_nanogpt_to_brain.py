#!/usr/bin/env python3
"""
Convert nanoGPT checkpoint (.pt) to external_brain.c format (.bin)

nanoGPT uses GPT-2 architecture, same as external_brain.c
Just need to reorder weights and add header.

Usage:
    python convert_nanogpt_to_brain.py <checkpoint.pt> <output.bin>
"""

import torch
import struct
import numpy as np
import sys
import os

def convert_nanogpt_to_brain(checkpoint_path, output_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove '_orig_mod.' prefix if present (from torch.compile)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Extract dimensions
    wte = state_dict['transformer.wte.weight']
    vocab_size, dim = wte.shape

    wpe = state_dict['transformer.wpe.weight']
    max_seq = wpe.shape[0]

    # Count layers
    n_layers = 0
    while f'transformer.h.{n_layers}.ln_1.weight' in state_dict:
        n_layers += 1

    # Get n_head from config or infer
    config = checkpoint.get('config', {})
    model_args = checkpoint.get('model_args', {})
    n_heads = model_args.get('n_head', config.get('n_head', 6))

    # hidden_dim = 4 * dim (GPT-2 standard)
    c_fc_weight = state_dict['transformer.h.0.mlp.c_fc.weight']
    hidden_dim = c_fc_weight.shape[0]

    print(f"Model config:")
    print(f"  dim: {dim}")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  max_seq: {max_seq}")
    print(f"  hidden_dim: {hidden_dim}")

    # Check if model has bias
    has_bias = f'transformer.h.0.ln_1.bias' in state_dict
    print(f"  has_bias: {has_bias}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'wb') as f:
        # Header: dim, n_layers, n_heads, vocab_size, max_seq, hidden_dim, use_fp16
        use_fp16 = 0  # float32
        f.write(struct.pack('7i', dim, n_layers, n_heads, vocab_size, max_seq, hidden_dim, use_fp16))

        def write_tensor(name, t, transpose=False):
            if transpose:
                t = t.T
            data = t.detach().cpu().float().numpy()
            f.write(data.tobytes())
            print(f"  {name}: {t.shape}")

        def write_zeros(name, shape):
            """Write zeros for missing bias"""
            size = np.prod(shape)
            data = np.zeros(size, dtype=np.float32)
            f.write(data.tobytes())
            print(f"  {name}: {shape} (zeros)")

        def get_or_zeros(key, shape):
            if key in state_dict:
                return state_dict[key]
            else:
                return torch.zeros(shape)

        print("Writing weights:")

        # Token embedding: [vocab_size, dim]
        write_tensor('wte', state_dict['transformer.wte.weight'])

        # Position embedding: [max_seq, dim]
        write_tensor('wpe', state_dict['transformer.wpe.weight'])

        # Layers
        for i in range(n_layers):
            prefix = f'transformer.h.{i}'

            # Layer norm 1
            write_tensor(f'ln1_w[{i}]', state_dict[f'{prefix}.ln_1.weight'])
            write_tensor(f'ln1_b[{i}]', get_or_zeros(f'{prefix}.ln_1.bias', (dim,)))

            # Attention c_attn: [dim, 3*dim] -> need transpose for C
            # PyTorch Linear: [out_features, in_features]
            # C matmul expects: [in_features, out_features]
            write_tensor(f'c_attn_w[{i}]', state_dict[f'{prefix}.attn.c_attn.weight'], transpose=True)
            write_tensor(f'c_attn_b[{i}]', get_or_zeros(f'{prefix}.attn.c_attn.bias', (3 * dim,)))

            # Attention c_proj: [dim, dim]
            write_tensor(f'c_proj_w[{i}]', state_dict[f'{prefix}.attn.c_proj.weight'], transpose=True)
            write_tensor(f'c_proj_b[{i}]', get_or_zeros(f'{prefix}.attn.c_proj.bias', (dim,)))

            # Layer norm 2
            write_tensor(f'ln2_w[{i}]', state_dict[f'{prefix}.ln_2.weight'])
            write_tensor(f'ln2_b[{i}]', get_or_zeros(f'{prefix}.ln_2.bias', (dim,)))

            # MLP c_fc: [dim, hidden_dim]
            write_tensor(f'c_fc_w[{i}]', state_dict[f'{prefix}.mlp.c_fc.weight'], transpose=True)
            write_tensor(f'c_fc_b[{i}]', get_or_zeros(f'{prefix}.mlp.c_fc.bias', (hidden_dim,)))

            # MLP c_proj: [hidden_dim, dim]
            write_tensor(f'c_proj2_w[{i}]', state_dict[f'{prefix}.mlp.c_proj.weight'], transpose=True)
            write_tensor(f'c_proj2_b[{i}]', get_or_zeros(f'{prefix}.mlp.c_proj.bias', (dim,)))

        # Final layer norm
        write_tensor('ln_f_w', state_dict['transformer.ln_f.weight'])
        write_tensor('ln_f_b', get_or_zeros('transformer.ln_f.bias', (dim,)))

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nExported: {output_path} ({size_mb:.2f} MB)")
    print(f"\nNOTE: This is char-level model (vocab={vocab_size})")
    print(f"      external_brain.c needs char tokenizer, not BPE!")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_nanogpt_to_brain.py <checkpoint.pt> <output.bin>")
        sys.exit(1)

    convert_nanogpt_to_brain(sys.argv[1], sys.argv[2])
