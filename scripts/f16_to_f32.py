#!/usr/bin/env python3
"""
Convert float16 weights back to float32 for C inference.
Run this before compiling/running if you only have f16 weights.
"""

import struct
import numpy as np
import os
import sys

MAGIC_F32 = 0x616B616E  # "naka"
MAGIC_F16 = 0x36316B6E  # "nk16"

def convert_f16_to_f32(input_path, output_path):
    """Convert float16 weights file to float32."""

    with open(input_path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != MAGIC_F16:
            if magic == MAGIC_F32:
                print(f"Already float32: {input_path}")
                return True
            print(f"Error: unknown magic 0x{magic:08x}")
            return False

        config_data = f.read(44)
        weights_f16 = np.frombuffer(f.read(), dtype=np.float16)

    # Convert to float32
    weights_f32 = weights_f16.astype(np.float32)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', MAGIC_F32))
        f.write(config_data)
        f.write(weights_f32.tobytes())

    in_size = os.path.getsize(input_path)
    out_size = os.path.getsize(output_path)
    print(f"Converted: {input_path} -> {output_path}")
    print(f"  Float16: {in_size/1024/1024:.2f} MB -> Float32: {out_size/1024/1024:.2f} MB")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python f16_to_f32.py <input_f16.bin> <output_f32.bin>")
        sys.exit(1)

    if not convert_f16_to_f32(sys.argv[1], sys.argv[2]):
        sys.exit(1)
