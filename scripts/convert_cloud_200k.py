#!/usr/bin/env python3
"""
Convert Cloud 200K weights from NPZ to BIN format.

Architecture: 100→128→64→32→1 per chamber (4 layers)
- 6 chambers × ~23K = ~138K params
- 1 observer × ~41K params
- Total: ~180K ≈ 200K params
"""

import numpy as np
import struct
import os
import sys

def convert_chamber(npz_path, bin_path):
    """Convert single chamber NPZ to BIN."""
    data = np.load(npz_path)

    # Expected keys for 100→128→64→32→1
    expected = [
        ('W1', (100, 128)), ('b1', (128,)),
        ('W2', (128, 64)), ('b2', (64,)),
        ('W3', (64, 32)), ('b3', (32,)),
        ('W4', (32, 1)), ('b4', (1,))
    ]

    # Check what's actually in the file
    print(f"  Keys in {os.path.basename(npz_path)}: {list(data.keys())}")
    for key in data.keys():
        print(f"    {key}: {data[key].shape}")

    with open(bin_path, 'wb') as f:
        total_floats = 0
        for key, expected_shape in expected:
            if key in data:
                arr = data[key].astype(np.float32).flatten()
                f.write(arr.tobytes())
                total_floats += len(arr)
                print(f"    Wrote {key}: {len(arr)} floats")
            else:
                # Try alternative key names
                alt_keys = {
                    'W1': ['fc1.weight', 'fc1_weight'],
                    'b1': ['fc1.bias', 'fc1_bias'],
                    'W2': ['fc2.weight', 'fc2_weight'],
                    'b2': ['fc2.bias', 'fc2_bias'],
                    'W3': ['fc3.weight', 'fc3_weight'],
                    'b3': ['fc3.bias', 'fc3_bias'],
                    'W4': ['fc4.weight', 'fc4_weight'],
                    'b4': ['fc4.bias', 'fc4_bias'],
                }
                found = False
                for alt in alt_keys.get(key, []):
                    if alt in data:
                        arr = data[alt].astype(np.float32).flatten()
                        f.write(arr.tobytes())
                        total_floats += len(arr)
                        print(f"    Wrote {key} (from {alt}): {len(arr)} floats")
                        found = True
                        break
                if not found:
                    print(f"    WARNING: {key} not found!")

        print(f"  Total: {total_floats} floats = {total_floats * 4} bytes")

    return total_floats

def convert_observer(npz_path, bin_path):
    """Convert observer NPZ to BIN."""
    data = np.load(npz_path)

    print(f"  Keys in {os.path.basename(npz_path)}: {list(data.keys())}")
    for key in data.keys():
        print(f"    {key}: {data[key].shape}")

    with open(bin_path, 'wb') as f:
        total_floats = 0
        for key in sorted(data.keys()):
            arr = data[key].astype(np.float32).flatten()
            f.write(arr.tobytes())
            total_floats += len(arr)
            print(f"    Wrote {key}: {len(arr)} floats")

        print(f"  Total: {total_floats} floats = {total_floats * 4} bytes")

    return total_floats

def main():
    src_dir = "/tmp/haze_check/cloud/models"
    dst_dir = "/Users/ataeff/Downloads/arianna.c/weights/cloud"

    os.makedirs(dst_dir, exist_ok=True)

    chambers = ['fear', 'love', 'rage', 'void']
    total_params = 0

    print("Converting Cloud 200K weights...")
    print()

    for chamber in chambers:
        npz_path = os.path.join(src_dir, f"chamber_{chamber}.npz")
        bin_path = os.path.join(dst_dir, f"chamber_{chamber}.bin")

        if os.path.exists(npz_path):
            print(f"Chamber {chamber.upper()}:")
            params = convert_chamber(npz_path, bin_path)
            total_params += params
        else:
            print(f"WARNING: {npz_path} not found!")

    # Observer
    obs_npz = os.path.join(src_dir, "observer.npz")
    obs_bin = os.path.join(dst_dir, "observer.bin")
    if os.path.exists(obs_npz):
        print(f"\nObserver:")
        params = convert_observer(obs_npz, obs_bin)
        total_params += params

    print(f"\n{'='*50}")
    print(f"Total params: {total_params} ({total_params/1000:.1f}K)")
    print(f"Expected: ~180K for Cloud 200K")

    if total_params < 50000:
        print("\nWARNING: Weights are for smaller architecture!")
        print("The NPZ files may contain 3-layer weights, not 4-layer.")

if __name__ == "__main__":
    main()
