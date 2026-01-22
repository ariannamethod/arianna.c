#!/usr/bin/env python3
"""
Convert Cloud .npz weights to binary format for Go.
Each file: raw float32 arrays, no headers.
"""

import numpy as np
from pathlib import Path
import struct

def convert_npz_to_bin(npz_path: Path, bin_dir: Path):
    """Convert single .npz to binary files."""
    data = np.load(npz_path)
    name = npz_path.stem  # e.g., "chamber_fear"

    out_path = bin_dir / f"{name}.bin"

    with open(out_path, 'wb') as f:
        # Write number of arrays
        keys = list(data.keys())
        f.write(struct.pack('<I', len(keys)))

        for key in keys:
            arr = data[key].astype(np.float32).flatten()
            # Write: name_len, name, shape_len, shape, data
            name_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            shape = data[key].shape
            f.write(struct.pack('<I', len(shape)))
            for dim in shape:
                f.write(struct.pack('<I', dim))

            f.write(struct.pack('<I', arr.size))
            f.write(arr.tobytes())

    print(f"  {npz_path.name} -> {out_path.name} ({out_path.stat().st_size:,} bytes)")

def main():
    haze_models = Path("/Users/ataeff/Downloads/haze_temp/cloud/models")
    out_dir = Path("/Users/ataeff/Downloads/arianna.c/weights/cloud")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Converting Cloud weights to binary format for Go:")
    print()

    # Convert chamber weights
    for chamber in ["fear", "love", "rage", "void"]:
        npz_path = haze_models / f"chamber_{chamber}.npz"
        if npz_path.exists():
            convert_npz_to_bin(npz_path, out_dir)

    # Convert observer
    observer_path = haze_models / "observer.npz"
    if observer_path.exists():
        convert_npz_to_bin(observer_path, out_dir)

    # Create empty flow/complex (will be random init in Go)
    print()
    print("Note: chamber_flow and chamber_complex not in haze/models")
    print("      Will be random initialized in Go")

    print()
    print(f"Output directory: {out_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
