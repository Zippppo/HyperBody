#!/usr/bin/env python3
import os
from pathlib import Path
from collections import defaultdict

def check_consistency():
    voxel_dir = Path("Dataset/voxel_data")
    extra_dir = Path("extra_data")

    # Get all npz files
    voxel_files = {f.name: f.stat().st_size for f in voxel_dir.glob("*.npz")}
    extra_files = {f.name: f.stat().st_size for f in extra_dir.glob("*.npz")}

    # Find duplicates and unique files
    duplicates = set(voxel_files.keys()) & set(extra_files.keys())
    voxel_only = set(voxel_files.keys()) - set(extra_files.keys())
    extra_only = set(extra_files.keys()) - set(voxel_files.keys())

    # Print results
    print(f"Dataset/voxel_data: {len(voxel_files)} files")
    print(f"extra_data: {len(extra_files)} files")
    print(f"\nDuplicate files: {len(duplicates)}")

    if duplicates:
        print("\nDuplicate files found:")
        size_mismatch = []
        for fname in sorted(duplicates):
            v_size = voxel_files[fname]
            e_size = extra_files[fname]
            if v_size != e_size:
                size_mismatch.append((fname, v_size, e_size))
                print(f"  {fname} - SIZE MISMATCH: voxel={v_size}, extra={e_size}")
            else:
                print(f"  {fname} - size match: {v_size}")

        if size_mismatch:
            print(f"\n⚠️  {len(size_mismatch)} files have size mismatches!")

    print(f"\nUnique to Dataset/voxel_data: {len(voxel_only)}")
    print(f"Unique to extra_data: {len(extra_only)}")

    # Show file number ranges
    def get_file_num(fname):
        return int(fname.split('_')[1].split('.')[0])

    if voxel_files:
        v_nums = [get_file_num(f) for f in voxel_files.keys()]
        print(f"\nDataset/voxel_data range: {min(v_nums):08d} - {max(v_nums):08d}")

    if extra_files:
        e_nums = [get_file_num(f) for f in extra_files.keys()]
        print(f"extra_data range: {min(e_nums):08d} - {max(e_nums):08d}")

if __name__ == "__main__":
    check_consistency()
