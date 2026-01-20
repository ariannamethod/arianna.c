#!/usr/bin/env python3
"""
Export nanoGPT checkpoint (.pt) to binary format (.bin) for C inference.

Usage:
    python export_nanogpt_to_bin.py <checkpoint.pt> <output.bin>
"""

import torch
import struct
import numpy as np
import sys
import os

def export_nanogpt(checkpoint_path, output_path):
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
    vocab_size, n_embd = wte.shape

    wpe = state_dict['transformer.wpe.weight']
    block_size = wpe.shape[0]

    # Count layers
    n_layer = 0
    while f'transformer.h.{n_layer}.ln_1.weight' in state_dict:
        n_layer += 1

    # Get n_head from config or infer
    config = checkpoint.get('config', {})
    model_args = checkpoint.get('model_args', {})
    n_head = model_args.get('n_head', config.get('n_head', 6))

    print(f"Model config:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  n_embd: {n_embd}")
    print(f"  n_layer: {n_layer}")
    print(f"  n_head: {n_head}")
    print(f"  block_size: {block_size}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'wb') as f:
        # Header: vocab_size, n_embd, n_layer, n_head, block_size
        f.write(struct.pack('5i', vocab_size, n_embd, n_layer, n_head, block_size))

        def write_tensor(name, t):
            data = t.detach().cpu().float().numpy()
            f.write(data.tobytes())
            print(f"  {name}: {t.shape}")

        def write_if_exists(name, key):
            if key in state_dict:
                write_tensor(name, state_dict[key])
                return True
            return False

        print("Writing weights:")

        # Token embedding
        write_tensor('wte', state_dict['transformer.wte.weight'])

        # Position embedding
        write_tensor('wpe', state_dict['transformer.wpe.weight'])

        # Layers
        for i in range(n_layer):
            prefix = f'transformer.h.{i}'

            # Layer norm 1
            write_tensor(f'ln_1.w[{i}]', state_dict[f'{prefix}.ln_1.weight'])
            write_if_exists(f'ln_1.b[{i}]', f'{prefix}.ln_1.bias')

            # Attention (c_attn is combined QKV)
            write_tensor(f'c_attn.w[{i}]', state_dict[f'{prefix}.attn.c_attn.weight'])
            write_if_exists(f'c_attn.b[{i}]', f'{prefix}.attn.c_attn.bias')
            write_tensor(f'c_proj.w[{i}]', state_dict[f'{prefix}.attn.c_proj.weight'])
            write_if_exists(f'c_proj.b[{i}]', f'{prefix}.attn.c_proj.bias')

            # Layer norm 2
            write_tensor(f'ln_2.w[{i}]', state_dict[f'{prefix}.ln_2.weight'])
            write_if_exists(f'ln_2.b[{i}]', f'{prefix}.ln_2.bias')

            # MLP
            write_tensor(f'c_fc.w[{i}]', state_dict[f'{prefix}.mlp.c_fc.weight'])
            write_if_exists(f'c_fc.b[{i}]', f'{prefix}.mlp.c_fc.bias')
            write_tensor(f'c_proj_mlp.w[{i}]', state_dict[f'{prefix}.mlp.c_proj.weight'])
            write_if_exists(f'c_proj_mlp.b[{i}]', f'{prefix}.mlp.c_proj.bias')

        # Final layer norm
        write_tensor('ln_f.w', state_dict['transformer.ln_f.weight'])
        write_if_exists('ln_f.b', 'transformer.ln_f.bias')

        # lm_head (output projection)
        if 'lm_head.weight' in state_dict:
            write_tensor('lm_head', state_dict['lm_head.weight'])
        else:
            print("  lm_head: tied with wte")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nExported: {output_path} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python export_nanogpt_to_bin.py <checkpoint.pt> <output.bin>")
        sys.exit(1)

    export_nanogpt(sys.argv[1], sys.argv[2])
