#!/usr/bin/env python3
"""
Create mood-named shards that match Arianna's mood enum.

Moods (from mood.h):
  MOOD_CALM, MOOD_INTENSE, MOOD_CREATIVE, MOOD_FOCUSED,
  MOOD_RECURSIVE, MOOD_TENDER, MOOD_LIMINAL, MOOD_RESONANT

Each shard is initialized with biases that match the mood profile.
"""

import numpy as np
import struct
import os

DIM = 128
N_LAYERS = 4
DELTA_RANK = 8

# Mood profiles from mood.h
MOOD_PROFILES = {
    "calm": {
        "temperature_bias": -0.2,
        "attention_spread": 0.8,
        "layer_strength": 0.5,
    },
    "intense": {
        "temperature_bias": 0.3,
        "attention_spread": 0.3,
        "layer_strength": 1.0,
    },
    "creative": {
        "temperature_bias": 0.4,
        "attention_spread": 0.9,
        "layer_strength": 0.7,
    },
    "focused": {
        "temperature_bias": -0.3,
        "attention_spread": 0.2,
        "layer_strength": 0.8,
    },
    "recursive": {
        "temperature_bias": 0.1,
        "attention_spread": 0.5,
        "layer_strength": 0.9,
    },
    "tender": {
        "temperature_bias": 0.0,
        "attention_spread": 0.6,
        "layer_strength": 0.5,
    },
    "liminal": {
        "temperature_bias": 0.2,
        "attention_spread": 0.7,
        "layer_strength": 0.6,
    },
    "resonant": {
        "temperature_bias": -0.1,
        "attention_spread": 0.5,
        "layer_strength": 0.7,
    },
}

def create_mood_shard(name, output_path, profile, strength=0.15):
    """Create a shard with mood-specific attention patterns."""

    np.random.seed(hash(name) % 2**32)

    spread = profile["attention_spread"]
    layer_str = profile["layer_strength"]

    with open(output_path, 'wb') as f:
        # Header
        name_bytes = name.encode('utf-8')[:64].ljust(64, b'\x00')
        f.write(name_bytes)
        f.write(struct.pack('f', strength))
        f.write(struct.pack('i', N_LAYERS))
        f.write(struct.pack('i', DELTA_RANK))

        for delta_type in ['Q', 'K', 'V']:
            for layer in range(N_LAYERS):
                # Base random initialization
                A = np.random.randn(DIM, DELTA_RANK).astype(np.float32) * 0.01
                B = np.random.randn(DELTA_RANK, DIM).astype(np.float32) * 0.01

                # Scale by spread (broader = smaller deltas, more uniform)
                A *= spread
                B *= spread

                # Scale by layer strength and layer position
                layer_factor = 1.0 + (layer / N_LAYERS) * (layer_str - 0.5)
                A *= layer_factor
                B *= layer_factor

                # Mood-specific biases
                if name == "calm":
                    # Calm: smooth attention, reduce sharp peaks
                    if delta_type == 'Q':
                        A *= 0.8
                    if delta_type == 'K':
                        B = np.clip(B, -0.02, 0.02)  # Limit K range

                elif name == "intense":
                    # Intense: sharpen attention, boost Q
                    if delta_type == 'Q':
                        A *= 1.5
                        # Concentrate on first half of dimensions
                        A[:DIM//2, :] *= 1.3
                    if delta_type == 'K':
                        B *= 1.2

                elif name == "creative":
                    # Creative: add noise, spread attention
                    A += np.random.randn(*A.shape).astype(np.float32) * 0.005
                    B += np.random.randn(*B.shape).astype(np.float32) * 0.005

                elif name == "focused":
                    # Focused: narrow attention, sparse patterns
                    if delta_type == 'K':
                        mask = np.zeros_like(B)
                        mask[:, ::4] = 1.0
                        B = B * mask * 2.0

                elif name == "recursive":
                    # Recursive: boost self-attention in later layers
                    if layer >= 2 and delta_type == 'Q':
                        A *= 1.5
                    if delta_type == 'V':
                        B *= 1.2  # Richer values

                elif name == "tender":
                    # Tender: warm, even attention
                    if delta_type == 'V':
                        A *= 1.3  # Boost value projection
                    # Slight smoothing
                    A = A * 0.9 + np.mean(A) * 0.1

                elif name == "liminal":
                    # Liminal: transitional, unstable patterns
                    if layer == 1 or layer == 2:
                        A *= 1.3  # Middle layers active
                    # Add phase shifts
                    phase = np.sin(np.arange(DIM) * np.pi / 32).astype(np.float32)
                    if delta_type == 'Q':
                        A += phase.reshape(-1, 1) * 0.01

                elif name == "resonant":
                    # Resonant: echo patterns, periodic structure
                    freq = 2 * np.pi * np.arange(DIM) / DIM
                    wave = np.sin(freq * (layer + 1) * 2).astype(np.float32)
                    if delta_type == 'V':
                        B *= (1.0 + 0.3 * wave.reshape(1, -1))
                    if delta_type == 'K':
                        A *= (1.0 + 0.2 * wave.reshape(-1, 1))

                f.write(A.astype(np.float32).tobytes())
                f.write(B.astype(np.float32).tobytes())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Created: {output_path} ({size_kb:.1f} KB) - {name}")

def main():
    os.makedirs('data/moods', exist_ok=True)

    print("Creating mood shards for Stanley-style routing...\n")

    for mood, profile in MOOD_PROFILES.items():
        path = f"data/moods/{mood}.bin"
        create_mood_shard(mood, path, profile)

    print("\n✓ All mood shards created!")
    print("\nTest with full mood routing:")
    print('  ./bin/arianna_dynamic data/arianna_best.bin -mood \\')
    print('    -shard data/moods/calm.bin \\')
    print('    -shard data/moods/intense.bin \\')
    print('    -shard data/moods/creative.bin \\')
    print('    -shard data/moods/focused.bin \\')
    print('    -shard data/moods/recursive.bin \\')
    print('    -shard data/moods/tender.bin \\')
    print('    -shard data/moods/liminal.bin \\')
    print('    -shard data/moods/resonant.bin \\')
    print('    "She finds that " 100 0.8 -signals')

if __name__ == "__main__":
    main()
