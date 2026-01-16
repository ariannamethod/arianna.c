#!/usr/bin/env python3
"""
Create a test experience shard for arianna_dynamic.c
"""

import numpy as np
import struct
import sys

# Model config (must match arianna.h)
DIM = 128
N_LAYERS = 4
DELTA_RANK = 8

def create_shard(name, output_path, strength=0.1, noise_scale=0.01):
    """Create a shard with small random deltas."""

    with open(output_path, 'wb') as f:
        # Header
        name_bytes = name.encode('utf-8')[:64].ljust(64, b'\x00')
        f.write(name_bytes)
        f.write(struct.pack('f', strength))  # strength
        f.write(struct.pack('i', N_LAYERS))  # n_layers
        f.write(struct.pack('i', DELTA_RANK))  # rank

        # Q deltas for each layer
        for layer in range(N_LAYERS):
            # A: [out_dim, rank] = [DIM, DELTA_RANK]
            A = np.random.randn(DIM, DELTA_RANK).astype(np.float32) * noise_scale
            f.write(A.tobytes())
            # B: [rank, in_dim] = [DELTA_RANK, DIM]
            B = np.random.randn(DELTA_RANK, DIM).astype(np.float32) * noise_scale
            f.write(B.tobytes())

        # K deltas
        for layer in range(N_LAYERS):
            A = np.random.randn(DIM, DELTA_RANK).astype(np.float32) * noise_scale
            f.write(A.tobytes())
            B = np.random.randn(DELTA_RANK, DIM).astype(np.float32) * noise_scale
            f.write(B.tobytes())

        # V deltas
        for layer in range(N_LAYERS):
            A = np.random.randn(DIM, DELTA_RANK).astype(np.float32) * noise_scale
            f.write(A.tobytes())
            B = np.random.randn(DELTA_RANK, DIM).astype(np.float32) * noise_scale
            f.write(B.tobytes())

    # Report size
    size = 64 + 4 + 4 + 4  # header
    size += N_LAYERS * 3 * (DIM * DELTA_RANK + DELTA_RANK * DIM) * 4  # deltas
    print(f"Created shard: {output_path}")
    print(f"  Name: {name}")
    print(f"  Strength: {strength}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Rank: {DELTA_RANK}")
    print(f"  Size: {size/1024:.1f} KB")

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_test_shard.py <output.bin> [name] [strength] [noise_scale]")
        print("\nExample:")
        print("  python create_test_shard.py data/test_shard.bin warmth 0.2 0.01")
        return

    output_path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "test_shard"
    strength = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    noise_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01

    np.random.seed(42)  # Reproducible
    create_shard(name, output_path, strength, noise_scale)

if __name__ == "__main__":
    main()
