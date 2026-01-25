#!/usr/bin/env python3
"""
Convert float32 weights to float16 format.
Changes magic from 0x616B616E ("naka") to 0x36316B6E ("nk16").
"""

import struct
import numpy as np
import os
import sys

# Magic numbers
MAGIC_F32 = 0x616B616E  # "naka"
MAGIC_F16 = 0x36316B6E  # "nk16"

def convert_f32_to_f16(input_path, output_path):
    """Convert float32 weights file to float16."""

    with open(input_path, 'rb') as f:
        # Read magic
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != MAGIC_F32:
            print(f"Error: expected float32 magic (0x{MAGIC_F32:08x}), got 0x{magic:08x}")
            return False

        # Read config (44 bytes: 9 ints + 2 floats)
        config_data = f.read(44)

        # Read all weights as float32
        weights_f32 = np.frombuffer(f.read(), dtype=np.float32)

    # Convert to float16
    weights_f16 = weights_f32.astype(np.float16)

    # Write output
    with open(output_path, 'wb') as f:
        # Write float16 magic
        f.write(struct.pack('<I', MAGIC_F16))
        # Config unchanged
        f.write(config_data)
        # Weights as float16
        f.write(weights_f16.tobytes())

    # Report
    in_size = os.path.getsize(input_path)
    out_size = os.path.getsize(output_path)
    print(f"Converted: {input_path} -> {output_path}")
    print(f"  Float32: {in_size:,} bytes ({in_size/1024/1024:.2f} MB)")
    print(f"  Float16: {out_size:,} bytes ({out_size/1024/1024:.2f} MB)")
    print(f"  Reduction: {100*(1 - out_size/in_size):.1f}%")
    print(f"  Weights: {len(weights_f32):,}")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_to_f16.py <input.bin> [output.bin]")
        print("  If output not specified, replaces .bin with _f16.bin")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = input_path.replace('.bin', '_f16.bin')

    if not convert_f32_to_f16(input_path, output_path):
        sys.exit(1)
