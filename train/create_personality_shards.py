#!/usr/bin/env python3
"""
Create personality-biased shards to demonstrate how deltas affect speech.

Each shard biases attention in a specific direction:
- tension_shard: increases attention to conflict/uncertainty patterns
- warmth_shard: increases attention to emotional/intimate patterns
- focus_shard: narrows attention, more deterministic
- chaos_shard: spreads attention, more creative/chaotic
"""

import numpy as np
import struct
import os

# Model config
DIM = 128
N_LAYERS = 4
DELTA_RANK = 8

def create_biased_shard(name, output_path, strength, bias_type):
    """Create a shard with specific attention bias."""

    np.random.seed(hash(name) % 2**32)

    with open(output_path, 'wb') as f:
        # Header
        name_bytes = name.encode('utf-8')[:64].ljust(64, b'\x00')
        f.write(name_bytes)
        f.write(struct.pack('f', strength))
        f.write(struct.pack('i', N_LAYERS))
        f.write(struct.pack('i', DELTA_RANK))

        for delta_type in ['Q', 'K', 'V']:
            for layer in range(N_LAYERS):
                # Base random
                A = np.random.randn(DIM, DELTA_RANK).astype(np.float32) * 0.01
                B = np.random.randn(DELTA_RANK, DIM).astype(np.float32) * 0.01

                # Apply bias based on type
                if bias_type == 'tension':
                    # Tension: amplify later layers (deeper processing)
                    # Q gets boosted - asking more questions
                    if delta_type == 'Q':
                        A *= (1.0 + layer * 0.3)
                    # Sharper, more pointed attention
                    B[:, :DIM//4] *= 2.0

                elif bias_type == 'warmth':
                    # Warmth: boost V (value) to enrich output
                    # Spread attention more evenly
                    if delta_type == 'V':
                        A *= 1.5
                        B *= 1.2
                    # Earlier layers matter more (surface warmth)
                    if layer < 2:
                        A *= 1.3

                elif bias_type == 'focus':
                    # Focus: sharpen K (keys) for precise matching
                    if delta_type == 'K':
                        # Concentrate on specific dimensions
                        mask = np.zeros_like(B)
                        mask[:, ::4] = 1.0  # Every 4th dimension
                        B = B * mask * 3.0
                    # Later layers for deep focus
                    if layer >= 2:
                        A *= 1.5

                elif bias_type == 'chaos':
                    # Chaos: add noise to Q, spread K
                    if delta_type == 'Q':
                        A += np.random.randn(*A.shape).astype(np.float32) * 0.02
                    if delta_type == 'K':
                        B += np.random.randn(*B.shape).astype(np.float32) * 0.02
                    # All layers equally chaotic
                    A *= 1.2

                elif bias_type == 'poetic':
                    # Poetic: rhythm in attention (periodic patterns)
                    freq = 2 * np.pi * np.arange(DIM) / DIM
                    wave = np.sin(freq * (layer + 1)).astype(np.float32)
                    if delta_type == 'V':
                        B *= (1.0 + 0.3 * wave.reshape(1, -1))
                    A *= 1.1

                f.write(A.astype(np.float32).tobytes())
                f.write(B.astype(np.float32).tobytes())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Created: {output_path} ({size_kb:.1f} KB) - {bias_type}")

def main():
    os.makedirs('data/shards', exist_ok=True)

    shards = [
        ('tension', 'data/shards/tension.bin', 0.15, 'tension'),
        ('warmth', 'data/shards/warmth.bin', 0.15, 'warmth'),
        ('focus', 'data/shards/focus.bin', 0.12, 'focus'),
        ('chaos', 'data/shards/chaos.bin', 0.10, 'chaos'),
        ('poetic', 'data/shards/poetic.bin', 0.12, 'poetic'),
    ]

    print("Creating personality shards...\n")
    for name, path, strength, bias in shards:
        create_biased_shard(name, path, strength, bias)

    print("\n✓ All shards created!")
    print("\nTest with:")
    print('  ./bin/arianna_dynamic data/arianna_best.bin "She finds that " 100 0.8')
    print('  ./bin/arianna_dynamic data/arianna_best.bin -shard data/shards/tension.bin "She finds that " 100 0.8')
    print('  ./bin/arianna_dynamic data/arianna_best.bin -shard data/shards/warmth.bin "She finds that " 100 0.8')

if __name__ == "__main__":
    main()
