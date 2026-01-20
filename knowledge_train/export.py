#!/usr/bin/env python3
"""
📦 EXTERNAL BRAIN 30M - Weight Export

Exports PyTorch weights to binary format for arianna.c inference.

Binary format matches arianna.c/model.c weight loading:
1. tok_emb: (vocab_size, dim)
2. For each layer:
   - attn_norm: (dim,)
   - wq: (dim, dim) stored as (out_features, in_features)
   - wk: (kv_dim, dim) stored as (out_features, in_features)
   - wv: (kv_dim, dim) stored as (out_features, in_features)
   - wo: (dim, dim) stored as (out_features, in_features)
   - ffn_norm: (dim,)
   - w_gate: (hidden_dim, dim) stored as (out_features, in_features)
   - w_up: (hidden_dim, dim) stored as (out_features, in_features)
   - w_down: (dim, hidden_dim) stored as (out_features, in_features)
3. final_norm: (dim,)
4. lm_head: (vocab_size, dim) stored as (out_features, in_features)

Usage:
    python export.py out/external_brain_final.pt external_brain.bin
    python export.py out/external_brain_final.pt external_brain_f16.bin --fp16
"""

import argparse
import json
import os
import struct
from typing import Dict

import torch
import numpy as np


def export_to_binary(
    checkpoint_path: str,
    output_path: str,
    use_fp16: bool = False,
) -> None:
    """
    Export PyTorch checkpoint to binary format for arianna.c.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt)
        output_path: Output path for binary weights (.bin)
        use_fp16: If True, export as float16 (half the size)
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Note: weights_only=False is required because the checkpoint contains
    # non-tensor data (config dict). The checkpoint is trusted local training output.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model' not in checkpoint:
        raise ValueError("Invalid checkpoint: missing 'model' key")
    
    state_dict = checkpoint['model']
    config = checkpoint.get('config', {})
    
    # Get model dimensions from config or infer from weights
    dim = config.get('dim', state_dict['tok_emb.weight'].shape[1])
    n_layers = config.get('n_layers', sum(1 for k in state_dict if 'layers.' in k and '.attn_norm' in k))
    n_heads = config.get('n_heads', 8)
    n_kv_heads = config.get('n_kv_heads', 4)
    vocab_size = config.get('vocab_size', state_dict['tok_emb.weight'].shape[0])
    max_seq_len = config.get('max_seq_len', 512)
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    hidden_dim = config.get('hidden_dim', state_dict['layers.0.ffn.w_gate.weight'].shape[0])
    
    print(f"\n📐 Model configuration:")
    print(f"   dim:         {dim}")
    print(f"   n_layers:    {n_layers}")
    print(f"   n_heads:     {n_heads}")
    print(f"   n_kv_heads:  {n_kv_heads}")
    print(f"   head_dim:    {head_dim}")
    print(f"   hidden_dim:  {hidden_dim}")
    print(f"   vocab_size:  {vocab_size}")
    print(f"   max_seq_len: {max_seq_len}")
    
    dtype_str = 'float16' if use_fp16 else 'float32'
    dtype_size = 2 if use_fp16 else 4
    
    def to_numpy(key: str) -> np.ndarray:
        """Convert tensor to numpy."""
        tensor = state_dict[key]
        if use_fp16:
            tensor = tensor.half()
        return tensor.cpu().numpy()
    
    def write_tensor(f, arr: np.ndarray, name: str):
        """Write numpy array to binary file."""
        arr_flat = arr.flatten()
        if use_fp16:
            data = struct.pack(f'{len(arr_flat)}e', *arr_flat.astype(np.float16))
        else:
            data = struct.pack(f'{len(arr_flat)}f', *arr_flat.astype(np.float32))
        f.write(data)
        print(f"   {name}: {arr.shape} -> {len(data):,} bytes")
    
    total_params = 0
    
    print(f"\n📝 Writing weights to {output_path}...")
    print(f"   Format: {dtype_str}")
    
    with open(output_path, 'wb') as f:
        # Token embeddings: (vocab_size, dim)
        tok_emb = to_numpy('tok_emb.weight')
        write_tensor(f, tok_emb, 'tok_emb')
        total_params += tok_emb.size
        
        # Per-layer weights
        for i in range(n_layers):
            print(f"\n   Layer {i}:")
            
            # Attention norm
            attn_norm = to_numpy(f'layers.{i}.attn_norm.weight')
            write_tensor(f, attn_norm, '  attn_norm')
            total_params += attn_norm.size
            
            # Attention weights (stored as out_features x in_features in PyTorch)
            wq = to_numpy(f'layers.{i}.attn.wq.weight')
            write_tensor(f, wq, '  wq')
            total_params += wq.size
            
            wk = to_numpy(f'layers.{i}.attn.wk.weight')
            write_tensor(f, wk, '  wk')
            total_params += wk.size
            
            wv = to_numpy(f'layers.{i}.attn.wv.weight')
            write_tensor(f, wv, '  wv')
            total_params += wv.size
            
            wo = to_numpy(f'layers.{i}.attn.wo.weight')
            write_tensor(f, wo, '  wo')
            total_params += wo.size
            
            # FFN norm
            ffn_norm = to_numpy(f'layers.{i}.ffn_norm.weight')
            write_tensor(f, ffn_norm, '  ffn_norm')
            total_params += ffn_norm.size
            
            # FFN weights (SwiGLU)
            w_gate = to_numpy(f'layers.{i}.ffn.w_gate.weight')
            write_tensor(f, w_gate, '  w_gate')
            total_params += w_gate.size
            
            w_up = to_numpy(f'layers.{i}.ffn.w_up.weight')
            write_tensor(f, w_up, '  w_up')
            total_params += w_up.size
            
            w_down = to_numpy(f'layers.{i}.ffn.w_down.weight')
            write_tensor(f, w_down, '  w_down')
            total_params += w_down.size
        
        print(f"\n   Final layers:")
        
        # Final norm
        final_norm = to_numpy('final_norm.weight')
        write_tensor(f, final_norm, '  final_norm')
        total_params += final_norm.size
        
        # LM head
        lm_head = to_numpy('lm_head.weight')
        write_tensor(f, lm_head, '  lm_head')
        total_params += lm_head.size
    
    file_size = os.path.getsize(output_path)
    
    print(f"\n✅ Export complete!")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"   Output: {output_path}")
    
    # Save config for C inference
    config_path = output_path.replace('.bin', '_config.json')
    config_out = {
        'dim': dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'head_dim': head_dim,
        'hidden_dim': hidden_dim,
        'n_kv_groups': n_heads // n_kv_heads,
        'rope_theta': config.get('rope_theta', 10000.0),
        'norm_eps': config.get('norm_eps', 1e-5),
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_out, f, indent=2)
    print(f"   Config: {config_path}")
    
    # Generate C header with model constants
    header_path = output_path.replace('.bin', '.h')
    with open(header_path, 'w') as f:
        f.write(f"""/*
 * External Brain 30M - Model Configuration
 * Auto-generated by export.py
 */

#ifndef EXTERNAL_BRAIN_CONFIG_H
#define EXTERNAL_BRAIN_CONFIG_H

#define EB_DIM {dim}
#define EB_N_LAYERS {n_layers}
#define EB_N_HEADS {n_heads}
#define EB_N_KV_HEADS {n_kv_heads}
#define EB_HEAD_DIM {head_dim}
#define EB_HIDDEN_DIM {hidden_dim}
#define EB_VOCAB_SIZE {vocab_size}
#define EB_MAX_SEQ_LEN {max_seq_len}
#define EB_ROPE_THETA 10000.0f
#define EB_NORM_EPS 1e-5f

#endif // EXTERNAL_BRAIN_CONFIG_H
""")
    print(f"   Header: {header_path}")


def verify_export(bin_path: str, config_path: str) -> None:
    """Verify the exported binary file."""
    print(f"\n🔍 Verifying export...")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dim = config['dim']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    vocab_size = config['vocab_size']
    hidden_dim = config['hidden_dim']
    head_dim = config['head_dim']
    kv_dim = n_kv_heads * head_dim
    
    # Calculate expected size
    tok_emb = vocab_size * dim
    per_layer = (
        dim +                    # attn_norm
        dim * dim +              # wq
        kv_dim * dim +           # wk
        kv_dim * dim +           # wv
        dim * dim +              # wo
        dim +                    # ffn_norm
        hidden_dim * dim +       # w_gate
        hidden_dim * dim +       # w_up
        dim * hidden_dim         # w_down
    )
    final = dim + vocab_size * dim
    
    expected_params = tok_emb + per_layer * n_layers + final
    expected_size_f32 = expected_params * 4
    expected_size_f16 = expected_params * 2
    
    actual_size = os.path.getsize(bin_path)
    
    print(f"   Expected parameters: {expected_params:,}")
    print(f"   Expected size (f32): {expected_size_f32:,} bytes ({expected_size_f32/1024/1024:.2f} MB)")
    print(f"   Expected size (f16): {expected_size_f16:,} bytes ({expected_size_f16/1024/1024:.2f} MB)")
    print(f"   Actual size: {actual_size:,} bytes ({actual_size/1024/1024:.2f} MB)")
    
    if actual_size == expected_size_f32:
        print("   ✅ Size matches (float32)! Export verified.")
    elif actual_size == expected_size_f16:
        print("   ✅ Size matches (float16)! Export verified.")
    else:
        print("   ⚠️  Size mismatch! Export may be corrupted.")


def main():
    parser = argparse.ArgumentParser(description='Export External Brain weights to binary')
    parser.add_argument('checkpoint', type=str, help='Path to PyTorch checkpoint (.pt)')
    parser.add_argument('output', type=str, help='Output path for binary weights (.bin)')
    parser.add_argument('--fp16', action='store_true', help='Export as float16')
    parser.add_argument('--verify', action='store_true', help='Verify export after completion')
    args = parser.parse_args()
    
    export_to_binary(args.checkpoint, args.output, args.fp16)
    
    if args.verify:
        config_path = args.output.replace('.bin', '_config.json')
        verify_export(args.output, config_path)


if __name__ == '__main__':
    main()
