#!/usr/bin/env python3
"""
Export PyTorch 34M BPE model to .bin format for arianna.c inference.

Format: naka magic + config + weights (float32)
Same binary layout as char-level export — only vocab_size differs.

Also exports:
- SentencePiece .model file (for C BPE tokenizer)
- tokenizer_bpe.json (vocab mapping)

Usage:
    python scripts/export_34m_bpe_to_bin.py
    python scripts/export_34m_bpe_to_bin.py --checkpoint arianna_34m_bpe/arianna_34m_bpe_final.pt
    python scripts/export_34m_bpe_to_bin.py --output weights/arianna_34m_bpe.bin
"""

import struct
import os
import sys
import shutil
import json
import argparse

import torch

sys.path.insert(0, 'tests')
from train_34m_bpe import Arianna34M, TrainConfig, BPETokenizer


def export_to_bin(
    checkpoint_path: str,
    output_path: str,
    tokenizer_model_path: str,
):
    """Export BPE PyTorch checkpoint to .bin format."""

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get config from checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        config = TrainConfig()
        for k, v in saved_config.items():
            if hasattr(config, k):
                setattr(config, k, v)
        config.__post_init__()
    else:
        config = TrainConfig()

    # Load tokenizer to verify vocab size
    print(f"Loading BPE tokenizer: {tokenizer_model_path}")
    tokenizer = BPETokenizer.load(tokenizer_model_path)
    assert tokenizer.vocab_size == config.vocab_size, (
        f"Tokenizer vocab ({tokenizer.vocab_size}) != config vocab ({config.vocab_size})"
    )

    print(f"\nModel config:")
    print(f"  dim: {config.dim}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_kv_heads: {config.n_kv_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  vocab_size: {config.vocab_size} (BPE)")

    # Load model
    model = Arianna34M(config)
    state_dict = checkpoint['model']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Export .bin
    print(f"\nExporting to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'wb') as f:
        # Magic: "naka" = 0x616B616E
        f.write(struct.pack('<I', 0x616B616E))

        # Config (same layout as char-level)
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

    size = os.path.getsize(output_path)
    print(f"\nDone! Exported {size:,} bytes ({size/1024/1024:.2f} MB)")

    # Expected size calculation
    expected = 4 + 11 * 4 + n_params * 4  # magic + config + weights
    print(f"Expected: {expected:,} bytes")
    if size == expected:
        print("SIZE MATCH OK")
    else:
        print(f"WARNING: size mismatch! delta = {size - expected}")

    # Copy SentencePiece .model next to .bin
    out_dir = os.path.dirname(output_path) or '.'
    sp_dest = os.path.join(out_dir, 'arianna_bpe.model')
    shutil.copy2(tokenizer_model_path, sp_dest)
    print(f"SentencePiece model copied: {sp_dest}")

    # Export vocab JSON for C tokenizer
    tokenizer.save_for_c(os.path.join(out_dir, 'tokenizer_bpe.json'))

    print(f"\nAll artifacts:")
    print(f"  {output_path}  ({size/1024/1024:.1f} MB)")
    print(f"  {sp_dest}")
    print(f"  {os.path.join(out_dir, 'tokenizer_bpe.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export Arianna 34M BPE to .bin')
    parser.add_argument('--checkpoint',
                        default='arianna_34m_bpe/arianna_34m_bpe_final.pt',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output',
                        default='weights/arianna_34m_bpe.bin',
                        help='Output .bin path')
    parser.add_argument('--tokenizer',
                        default='arianna_34m_bpe/arianna_bpe.model',
                        help='Path to SentencePiece .model file')
    args = parser.parse_args()

    export_to_bin(args.checkpoint, args.output, args.tokenizer)
