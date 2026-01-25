#!/usr/bin/env python3
"""
Export PyTorch 34M model to .bin format for arianna.c inference.

Format: naka magic + config + weights (float32)
"""

import struct
import torch
import sys
sys.path.insert(0, 'tests')
from train_34m import Arianna34M, TrainConfig, CharTokenizer


def export_to_bin(checkpoint_path: str, output_path: str, data_path: str = 'd/arianna_unified2.txt'):
    """Export PyTorch checkpoint to .bin format."""

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load tokenizer to get vocab size
    print(f"Loading tokenizer from: {data_path}")
    with open(data_path, 'r') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)

    # Create config
    config = TrainConfig()
    config.vocab_size = tokenizer.vocab_size

    print(f"\nModel config:")
    print(f"  dim: {config.dim}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_kv_heads: {config.n_kv_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  vocab_size: {config.vocab_size}")

    # Load model
    model = Arianna34M(config)
    model.load_state_dict(checkpoint['model'])

    # Export
    print(f"\nExporting to: {output_path}")

    with open(output_path, 'wb') as f:
        # Magic: "naka" = 0x616B616E
        f.write(struct.pack('<I', 0x616B616E))

        # Config
        f.write(struct.pack('<i', config.dim))
        f.write(struct.pack('<i', config.n_layers))
        f.write(struct.pack('<i', config.n_heads))
        f.write(struct.pack('<i', config.n_kv_heads))
        f.write(struct.pack('<i', config.head_dim))
        f.write(struct.pack('<i', config.hidden_dim))
        f.write(struct.pack('<i', config.max_seq_len))
        f.write(struct.pack('<i', config.vocab_size))
        n_kv_groups = config.n_heads // config.n_kv_heads
        f.write(struct.pack('<i', n_kv_groups))
        f.write(struct.pack('<f', config.rope_theta))
        f.write(struct.pack('<f', config.norm_eps))

        def write_tensor(name, tensor):
            data = tensor.detach().float().numpy().flatten()
            f.write(data.tobytes())
            print(f"  {name}: {tensor.shape} -> {len(data)} floats")

        # Token embeddings
        write_tensor("tok_emb", model.tok_emb.weight)

        # Per-layer weights
        for i, layer in enumerate(model.layers):
            print(f"  Layer {i}:")
            write_tensor(f"    attn_norm", layer.attn_norm.weight)
            write_tensor(f"    wq", layer.attn.wq.weight)
            write_tensor(f"    wk", layer.attn.wk.weight)
            write_tensor(f"    wv", layer.attn.wv.weight)
            write_tensor(f"    wo", layer.attn.wo.weight)
            write_tensor(f"    ffn_norm", layer.ffn_norm.weight)
            write_tensor(f"    w_gate", layer.ffn.w_gate.weight)
            write_tensor(f"    w_up", layer.ffn.w_up.weight)
            write_tensor(f"    w_down", layer.ffn.w_down.weight)

        # Final norm and output
        write_tensor("final_norm", model.final_norm.weight)
        write_tensor("lm_head", model.lm_head.weight)

    import os
    size = os.path.getsize(output_path)
    print(f"\nDone! Exported {size:,} bytes ({size/1024/1024:.2f} MB)")

    # Also save tokenizer (use char_to_id for C compatibility)
    tokenizer_path = output_path.replace('.bin', '_tokenizer.json')
    import json
    with open(tokenizer_path, 'w') as f:
        json.dump({
            'char_to_id': tokenizer.char_to_idx,
            'id_to_char': {str(k): v for k, v in tokenizer.idx_to_char.items()},
            'vocab_size': tokenizer.vocab_size
        }, f, indent=2)
    print(f"Tokenizer saved: {tokenizer_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='weights/arianna_34m_final.pt')
    parser.add_argument('--output', default='weights/arianna_34m.bin')
    parser.add_argument('--data', default='d/arianna_unified2.txt')
    args = parser.parse_args()

    export_to_bin(args.checkpoint, args.output, args.data)
