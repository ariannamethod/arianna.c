#!/usr/bin/env python3
"""
Convert GPT-2 safetensors to binary format for C inference.

Layout:
  Header: dim, n_layers, n_heads, vocab_size, n_positions, hidden_dim (6 ints)
  wte: [vocab_size, dim]
  wpe: [n_positions, dim]
  For each layer:
    ln_1_weight, ln_1_bias: [dim]
    c_attn_weight: [dim, 3*dim] (Q,K,V combined)
    c_attn_bias: [3*dim]
    c_proj_weight: [dim, dim]
    c_proj_bias: [dim]
    ln_2_weight, ln_2_bias: [dim]
    c_fc_weight: [dim, hidden_dim]
    c_fc_bias: [hidden_dim]
    c_proj2_weight: [hidden_dim, dim]
    c_proj2_bias: [dim]
  ln_f_weight, ln_f_bias: [dim]
"""

import numpy as np
from safetensors import safe_open
import struct
import sys

def convert_gpt2(input_path, output_path):
    print(f"Loading {input_path}...")

    with safe_open(input_path, framework='pt') as f:
        # Get config from tensor shapes
        wte = f.get_tensor('transformer.wte.weight').float().numpy()
        wpe = f.get_tensor('transformer.wpe.weight').float().numpy()

        vocab_size, dim = wte.shape
        n_positions = wpe.shape[0]

        # Count layers
        n_layers = 0
        while f'transformer.h.{n_layers}.ln_1.weight' in f.keys():
            n_layers += 1

        # Get hidden dim from FFN
        c_fc = f.get_tensor('transformer.h.0.mlp.c_fc.weight').float().numpy()
        hidden_dim = c_fc.shape[1]

        # Infer n_heads from c_attn shape
        c_attn = f.get_tensor('transformer.h.0.attn.c_attn.weight').float().numpy()
        n_heads = dim // 64  # Assuming head_dim = 64

        print(f"Config: dim={dim}, layers={n_layers}, heads={n_heads}, vocab={vocab_size}, pos={n_positions}, hidden={hidden_dim}")

        with open(output_path, 'wb') as out:
            # Write header (with dtype flag: 0=float32, 1=float16)
            use_fp16 = True  # Use FP16 for smaller file
            dtype_flag = 1 if use_fp16 else 0
            header = struct.pack('iiiiiii', dim, n_layers, n_heads, vocab_size, n_positions, hidden_dim, dtype_flag)
            out.write(header)

            dtype = np.float16 if use_fp16 else np.float32

            # Write embeddings
            print(f"Writing wte: {wte.shape}")
            out.write(wte.astype(dtype).tobytes())

            print(f"Writing wpe: {wpe.shape}")
            out.write(wpe.astype(dtype).tobytes())

            # Write layers
            for l in range(n_layers):
                prefix = f'transformer.h.{l}'

                # LayerNorm 1
                ln1_w = f.get_tensor(f'{prefix}.ln_1.weight').float().numpy()
                ln1_b = f.get_tensor(f'{prefix}.ln_1.bias').float().numpy()
                out.write(ln1_w.astype(dtype).tobytes())
                out.write(ln1_b.astype(dtype).tobytes())

                # Attention
                c_attn_w = f.get_tensor(f'{prefix}.attn.c_attn.weight').float().numpy()
                c_attn_b = f.get_tensor(f'{prefix}.attn.c_attn.bias').float().numpy()
                c_proj_w = f.get_tensor(f'{prefix}.attn.c_proj.weight').float().numpy()
                c_proj_b = f.get_tensor(f'{prefix}.attn.c_proj.bias').float().numpy()

                out.write(c_attn_w.astype(dtype).tobytes())
                out.write(c_attn_b.astype(dtype).tobytes())
                out.write(c_proj_w.astype(dtype).tobytes())
                out.write(c_proj_b.astype(dtype).tobytes())

                # LayerNorm 2
                ln2_w = f.get_tensor(f'{prefix}.ln_2.weight').float().numpy()
                ln2_b = f.get_tensor(f'{prefix}.ln_2.bias').float().numpy()
                out.write(ln2_w.astype(dtype).tobytes())
                out.write(ln2_b.astype(dtype).tobytes())

                # FFN
                c_fc_w = f.get_tensor(f'{prefix}.mlp.c_fc.weight').float().numpy()
                c_fc_b = f.get_tensor(f'{prefix}.mlp.c_fc.bias').float().numpy()
                c_proj2_w = f.get_tensor(f'{prefix}.mlp.c_proj.weight').float().numpy()
                c_proj2_b = f.get_tensor(f'{prefix}.mlp.c_proj.bias').float().numpy()

                out.write(c_fc_w.astype(dtype).tobytes())
                out.write(c_fc_b.astype(dtype).tobytes())
                out.write(c_proj2_w.astype(dtype).tobytes())
                out.write(c_proj2_b.astype(dtype).tobytes())

                print(f"  Layer {l}: written")

            # Final LayerNorm
            ln_f_w = f.get_tensor('transformer.ln_f.weight').float().numpy()
            ln_f_b = f.get_tensor('transformer.ln_f.bias').float().numpy()
            out.write(ln_f_w.astype(dtype).tobytes())
            out.write(ln_f_b.astype(dtype).tobytes())

            print(f"Written to {output_path}")

            # Calculate and print size
            import os
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"File size: {size_mb:.1f} MB")

if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'weights/gpt2_30m/model.safetensors'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'weights/gpt2_30m/gpt2_30m.bin'
    convert_gpt2(input_path, output_path)
