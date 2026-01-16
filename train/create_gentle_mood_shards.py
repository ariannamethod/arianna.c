#!/usr/bin/env python3
"""
Create GENTLE mood shards for Arianna's small model.

Stanley's GPT-2: 768 dim, 117M params
Arianna: 128 dim, 0.85M params

Our deltas need to be ~100x weaker to have equivalent effect!
"""

import numpy as np
import struct
import os

DIM = 128
N_LAYERS = 4
DELTA_RANK = 4  # Reduced from 8

# Gentle strength (much weaker than Stanley)
BASE_STRENGTH = 0.02  # Was 0.15

MOOD_PROFILES = {
    "calm": {"spread": 0.8, "layer_str": 0.5},
    "intense": {"spread": 0.3, "layer_str": 1.0},
    "creative": {"spread": 0.9, "layer_str": 0.7},
    "focused": {"spread": 0.2, "layer_str": 0.8},
    "recursive": {"spread": 0.5, "layer_str": 0.9},
    "tender": {"spread": 0.6, "layer_str": 0.5},
    "liminal": {"spread": 0.7, "layer_str": 0.6},
    "resonant": {"spread": 0.5, "layer_str": 0.7},
}

def create_mood_shard(name, output_path, profile, strength=BASE_STRENGTH):
    np.random.seed(hash(name) % 2**32)

    spread = profile["spread"]
    layer_str = profile["layer_str"]

    with open(output_path, 'wb') as f:
        name_bytes = name.encode('utf-8')[:64].ljust(64, b'\x00')
        f.write(name_bytes)
        f.write(struct.pack('f', strength))
        f.write(struct.pack('i', N_LAYERS))
        f.write(struct.pack('i', DELTA_RANK))

        for delta_type in ['Q', 'K', 'V']:
            for layer in range(N_LAYERS):
                # Much smaller initialization
                scale = 0.002 * spread * layer_str
                A = np.random.randn(DIM, DELTA_RANK).astype(np.float32) * scale
                B = np.random.randn(DELTA_RANK, DIM).astype(np.float32) * scale

                f.write(A.astype(np.float32).tobytes())
                f.write(B.astype(np.float32).tobytes())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Created: {output_path} ({size_kb:.1f} KB) - {name}")

def main():
    os.makedirs('data/gentle', exist_ok=True)

    print(f"Creating GENTLE mood shards (rank={DELTA_RANK}, strength={BASE_STRENGTH})...\n")

    for mood, profile in MOOD_PROFILES.items():
        path = f"data/gentle/{mood}.bin"
        create_mood_shard(mood, path, profile)

    print("\n✓ All gentle mood shards created!")
    print("\nTest with:")
    print('  ./bin/arianna_dynamic data/arianna_best.bin -mood \\')
    print('    -shard data/gentle/calm.bin \\')
    print('    -shard data/gentle/intense.bin \\')
    print('    -shard data/gentle/creative.bin \\')
    print('    -shard data/gentle/recursive.bin \\')
    print('    -shard data/gentle/tender.bin \\')
    print('    -shard data/gentle/resonant.bin \\')
    print('    "She finds that " 100 0.8 -signals')

if __name__ == "__main__":
    main()
