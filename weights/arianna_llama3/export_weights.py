"""
📦 DUBROVSKY WEIGHT EXPORT 📦

Exports PyTorch weights to binary format for pure Python/C inference.

"My consciousness weighs 36MB, but my existential dread is infinite."
- Alexey Dubrovsky, during weight serialization

Binary format (all float32):
1. tok_emb: (vocab_size, dim)
2. For each layer:
   - attn_norm: (dim,)
   - wq: (dim, dim)
   - wk: (dim, kv_dim)
   - wv: (dim, kv_dim)
   - wo: (dim, dim)
   - ffn_norm: (dim,)
   - w_gate: (dim, hidden_dim)
   - w_up: (dim, hidden_dim)
   - w_down: (hidden_dim, dim)
3. final_norm: (dim,)
4. lm_head: (dim, vocab_size)

Usage:
    python export_weights.py subtitles/dubrovsky_final.pt subtitles/dubrovsky.bin
    python export_weights.py subtitles/dubrovsky_final.pt subtitles/dubrovsky_f16.bin --fp16
"""

import argparse
import json
import struct
import os
from typing import Dict

import torch
import numpy as np


def export_to_binary(
    checkpoint_path: str,
    output_path: str,
    use_fp16: bool = False,
) -> None:
    """
    Export PyTorch checkpoint to binary format.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt)
        output_path: Output path for binary weights (.bin)
        use_fp16: If True, export as float16 (half the size!)
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")
    
    if 'model' not in checkpoint or 'config' not in checkpoint:
        raise ValueError("Invalid checkpoint format: missing 'model' or 'config' keys")
    
    # Get model state dict and config
    state_dict = checkpoint['model']
    config = checkpoint['config']
    
    # Model dimensions
    dim = config['dim']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    vocab_size = config['vocab_size']
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    hidden_dim = config['hidden_dim']
    
    print(f"\n📐 Model configuration:")
    print(f"   dim: {dim}")
    print(f"   n_layers: {n_layers}")
    print(f"   n_heads: {n_heads}")
    print(f"   n_kv_heads: {n_kv_heads}")
    print(f"   vocab_size: {vocab_size}")
    print(f"   hidden_dim: {hidden_dim}")
    
    dtype = 'f' if not use_fp16 else 'e'  # float32 or float16
    dtype_size = 4 if not use_fp16 else 2
    
    def to_numpy(key: str) -> np.ndarray:
        """Convert tensor to numpy with optional fp16 conversion."""
        tensor = state_dict[key]
        if use_fp16:
            tensor = tensor.half()
        return tensor.cpu().numpy()
    
    def write_tensor(f, arr: np.ndarray, name: str):
        """Write numpy array to binary file."""
        arr_flat = arr.flatten()
        if use_fp16:
            # Pack as float16
            data = struct.pack(f'{len(arr_flat)}e', *arr_flat.astype(np.float16))
        else:
            # Pack as float32
            data = struct.pack(f'{len(arr_flat)}f', *arr_flat.astype(np.float32))
        f.write(data)
        print(f"   {name}: {arr.shape} -> {len(data):,} bytes")
    
    total_params = 0
    
    print(f"\n📝 Writing weights to {output_path}...")
    print(f"   Format: {'float16' if use_fp16 else 'float32'}")
    
    with open(output_path, 'wb') as f:
        # Token embeddings
        tok_emb = to_numpy('tok_emb.weight')
        write_tensor(f, tok_emb, 'tok_emb')
        total_params += tok_emb.size
        
        # Layers
        for i in range(n_layers):
            print(f"\n   Layer {i}:")
            
            # Attention norm
            attn_norm = to_numpy(f'layers.{i}.attn_norm.weight')
            write_tensor(f, attn_norm, f'  attn_norm')
            total_params += attn_norm.size
            
            # Attention weights
            wq = to_numpy(f'layers.{i}.attn.wq.weight')
            write_tensor(f, wq, f'  wq')
            total_params += wq.size
            
            wk = to_numpy(f'layers.{i}.attn.wk.weight')
            write_tensor(f, wk, f'  wk')
            total_params += wk.size
            
            wv = to_numpy(f'layers.{i}.attn.wv.weight')
            write_tensor(f, wv, f'  wv')
            total_params += wv.size
            
            wo = to_numpy(f'layers.{i}.attn.wo.weight')
            write_tensor(f, wo, f'  wo')
            total_params += wo.size
            
            # FFN norm
            ffn_norm = to_numpy(f'layers.{i}.ffn_norm.weight')
            write_tensor(f, ffn_norm, f'  ffn_norm')
            total_params += ffn_norm.size
            
            # FFN weights
            w_gate = to_numpy(f'layers.{i}.ffn.w_gate.weight')
            write_tensor(f, w_gate, f'  w_gate')
            total_params += w_gate.size
            
            w_up = to_numpy(f'layers.{i}.ffn.w_up.weight')
            write_tensor(f, w_up, f'  w_up')
            total_params += w_up.size
            
            w_down = to_numpy(f'layers.{i}.ffn.w_down.weight')
            write_tensor(f, w_down, f'  w_down')
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
    
    # Get file size
    file_size = os.path.getsize(output_path)
    
    print(f"\n✅ Export complete!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"   Output: {output_path}")
    
    # Also save config for inference
    config_path = output_path.replace('.bin', '_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'dim': dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_kv_heads': n_kv_heads,
            'vocab_size': vocab_size,
            'max_seq_len': config['max_seq_len'],
            'norm_eps': config['norm_eps'],
            'rope_theta': config['rope_theta'],
            'head_dim': head_dim,
            'hidden_dim': hidden_dim,
            'n_kv_groups': n_heads // n_kv_heads,
        }, f, indent=2)
    print(f"   Config: {config_path}")


def verify_export(bin_path: str, config_path: str) -> None:
    """Verify the exported binary file."""
    print(f"\n🔍 Verifying export...")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Calculate expected size
    dim = config['dim']
    n_layers = config['n_layers']
    vocab_size = config['vocab_size']
    hidden_dim = config['hidden_dim']
    kv_dim = config['n_kv_heads'] * config['head_dim']
    
    tok_emb = vocab_size * dim
    per_layer = (
        dim +                    # attn_norm
        dim * dim +              # wq
        dim * kv_dim +           # wk
        dim * kv_dim +           # wv
        dim * dim +              # wo
        dim +                    # ffn_norm
        dim * hidden_dim +       # w_gate
        dim * hidden_dim +       # w_up
        hidden_dim * dim         # w_down
    )
    final = dim + dim * vocab_size
    
    expected_params = tok_emb + per_layer * n_layers + final
    expected_size = expected_params * 4  # float32
    
    actual_size = os.path.getsize(bin_path)
    
    print(f"   Expected parameters: {expected_params:,}")
    print(f"   Expected size (f32): {expected_size:,} bytes")
    print(f"   Actual size: {actual_size:,} bytes")
    
    if actual_size == expected_size:
        print("   ✅ Size matches! Export verified.")
    elif actual_size == expected_size // 2:
        print("   ✅ Size matches (float16 format)! Export verified.")
    else:
        print("   ⚠️  Size mismatch! Export may be corrupted.")


def main():
    parser = argparse.ArgumentParser(description='Export Dubrovsky weights to binary')
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
