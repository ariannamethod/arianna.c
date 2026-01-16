#!/usr/bin/env python3
"""
Merge LoRA shard into base weights

Takes dialogue_lora.bin (or any shard) and merges it into base weights,
producing a new weights file that has the LoRA "baked in".

This means you can use the merged weights WITHOUT loading the shard at runtime!

Usage:
    python train/merge_lora.py weights/arianna.bin data/dialogue_lora.bin weights/arianna_dialogue.bin

The merged weights can be used with basic arianna inference:
    ./bin/arianna weights/arianna_dialogue.bin "She finds that " 100 0.8
"""

import struct
import numpy as np
import sys
import os

def read_weights(path):
    """Read arianna weights file

    Layout (from model.c):
    - header: 7 ints
    - tok_emb: vocab_size * dim
    - wq: n_layers * dim * dim (all layers together)
    - wk: n_layers * dim * dim
    - wv: n_layers * dim * dim
    - wo: n_layers * dim * dim
    - w1: n_layers * dim * hidden_dim
    - w2: n_layers * hidden_dim * dim
    - ln1_weight: n_layers * dim
    - ln2_weight: n_layers * dim
    - ln_final_weight: dim
    - output_weight: dim * vocab_size
    """
    with open(path, 'rb') as f:
        # Header: 7 ints
        header = struct.unpack('7i', f.read(28))
        dim, n_layers, n_heads, head_dim, hidden_dim, max_seq_len, vocab_size = header

        print(f"Base weights: dim={dim}, layers={n_layers}, heads={n_heads}, vocab={vocab_size}")

        weights = {}

        # Token embeddings
        weights['tok_emb'] = np.frombuffer(f.read(vocab_size * dim * 4), dtype=np.float32).reshape(vocab_size, dim)

        # Attention weights (all layers together) - copy() to make writable
        weights['wq'] = np.frombuffer(f.read(n_layers * dim * dim * 4), dtype=np.float32).reshape(n_layers, dim, dim).copy()
        weights['wk'] = np.frombuffer(f.read(n_layers * dim * dim * 4), dtype=np.float32).reshape(n_layers, dim, dim).copy()
        weights['wv'] = np.frombuffer(f.read(n_layers * dim * dim * 4), dtype=np.float32).reshape(n_layers, dim, dim).copy()
        weights['wo'] = np.frombuffer(f.read(n_layers * dim * dim * 4), dtype=np.float32).reshape(n_layers, dim, dim).copy()

        # FFN weights
        weights['w1'] = np.frombuffer(f.read(n_layers * dim * hidden_dim * 4), dtype=np.float32).reshape(n_layers, hidden_dim, dim)
        weights['w2'] = np.frombuffer(f.read(n_layers * hidden_dim * dim * 4), dtype=np.float32).reshape(n_layers, dim, hidden_dim)

        # Layer norms
        weights['ln1'] = np.frombuffer(f.read(n_layers * dim * 4), dtype=np.float32).reshape(n_layers, dim)
        weights['ln2'] = np.frombuffer(f.read(n_layers * dim * 4), dtype=np.float32).reshape(n_layers, dim)

        # Final norm and output
        weights['ln_final'] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
        weights['output'] = np.frombuffer(f.read(dim * vocab_size * 4), dtype=np.float32).reshape(vocab_size, dim)

        return header, weights

def read_shard(path, dim, n_layers):
    """Read LoRA shard file"""
    with open(path, 'rb') as f:
        name = f.read(64).decode('utf-8', errors='ignore').strip('\x00')
        strength, = struct.unpack('f', f.read(4))
        shard_layers, = struct.unpack('i', f.read(4))
        rank, = struct.unpack('i', f.read(4))

        print(f"Shard: name={name}, strength={strength}, layers={shard_layers}, rank={rank}")

        deltas = {'q': [], 'k': [], 'v': []}

        # Read Q deltas for all layers
        for l in range(n_layers):
            A = np.frombuffer(f.read(dim * rank * 4), dtype=np.float32).reshape(dim, rank)
            B = np.frombuffer(f.read(rank * dim * 4), dtype=np.float32).reshape(rank, dim)
            deltas['q'].append((A, B))

        # Read K deltas
        for l in range(n_layers):
            A = np.frombuffer(f.read(dim * rank * 4), dtype=np.float32).reshape(dim, rank)
            B = np.frombuffer(f.read(rank * dim * 4), dtype=np.float32).reshape(rank, dim)
            deltas['k'].append((A, B))

        # Read V deltas
        for l in range(n_layers):
            A = np.frombuffer(f.read(dim * rank * 4), dtype=np.float32).reshape(dim, rank)
            B = np.frombuffer(f.read(rank * dim * 4), dtype=np.float32).reshape(rank, dim)
            deltas['v'].append((A, B))

        return name, strength, deltas

def write_weights(path, header, weights):
    """Write weights file in same layout as model.c expects"""
    dim, n_layers, n_heads, head_dim, hidden_dim, max_seq_len, vocab_size = header

    with open(path, 'wb') as f:
        # Header
        f.write(struct.pack('7i', *header))

        # Token embeddings
        f.write(weights['tok_emb'].astype(np.float32).tobytes())

        # Attention weights (all layers together)
        f.write(weights['wq'].astype(np.float32).tobytes())
        f.write(weights['wk'].astype(np.float32).tobytes())
        f.write(weights['wv'].astype(np.float32).tobytes())
        f.write(weights['wo'].astype(np.float32).tobytes())

        # FFN weights
        f.write(weights['w1'].astype(np.float32).tobytes())
        f.write(weights['w2'].astype(np.float32).tobytes())

        # Layer norms
        f.write(weights['ln1'].astype(np.float32).tobytes())
        f.write(weights['ln2'].astype(np.float32).tobytes())

        # Final norm and output
        f.write(weights['ln_final'].astype(np.float32).tobytes())
        f.write(weights['output'].astype(np.float32).tobytes())

def merge_lora(base_path, shard_path, output_path, scale=1.0):
    """Merge LoRA shard into base weights"""

    # Read base weights
    header, weights = read_weights(base_path)
    dim, n_layers, n_heads, head_dim, hidden_dim, max_seq_len, vocab_size = header

    # Read shard
    name, strength, deltas = read_shard(shard_path, dim, n_layers)

    # Effective scale
    effective_scale = strength * scale
    print(f"Merging with effective scale: {effective_scale}")

    # Merge deltas into weights
    # weights['wq'] has shape [n_layers, dim, dim]
    for l in range(n_layers):
        # Q: W_new = W + scale * A @ B
        A_q, B_q = deltas['q'][l]
        delta_q = A_q @ B_q  # [dim, dim]
        weights['wq'][l] = weights['wq'][l] + effective_scale * delta_q

        # K
        A_k, B_k = deltas['k'][l]
        delta_k = A_k @ B_k
        weights['wk'][l] = weights['wk'][l] + effective_scale * delta_k

        # V
        A_v, B_v = deltas['v'][l]
        delta_v = A_v @ B_v
        weights['wv'][l] = weights['wv'][l] + effective_scale * delta_v

        # Report change magnitude
        q_change = np.linalg.norm(delta_q) * effective_scale
        print(f"  Layer {l}: delta norm (Q/K/V) = {q_change:.6f} / {np.linalg.norm(delta_k)*effective_scale:.6f} / {np.linalg.norm(delta_v)*effective_scale:.6f}")

    # Write merged weights
    write_weights(output_path, header, weights)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nMerged weights saved to: {output_path} ({size_mb:.2f} MB)")
    print(f"\nUsage:")
    print(f"  ./bin/arianna_dynamic {output_path} \"She finds that \" 100 0.8")
    print(f"  (No need for -shard flag anymore!)")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python merge_lora.py <base_weights> <shard> <output>")
        print("       python merge_lora.py <base_weights> <shard> <output> <scale>")
        print("")
        print("Example:")
        print("  python train/merge_lora.py weights/arianna.bin data/dialogue_lora.bin weights/arianna_dialogue.bin")
        sys.exit(1)

    base_path = sys.argv[1]
    shard_path = sys.argv[2]
    output_path = sys.argv[3]
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    merge_lora(base_path, shard_path, output_path, scale)
