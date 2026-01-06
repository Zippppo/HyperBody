#!/usr/bin/env python3
"""GPU Stress Test - Keep GPUs at 100% utilization"""

import argparse
import signal
import sys
import multiprocessing as mp
from typing import List

import torch

stop_flag = mp.Event()


def stress_gpu(device_id: int, matrix_size: int = 8192):
    """Run continuous matrix multiplication on specified GPU."""
    device = torch.device(f'cuda:{device_id}')
    print(f"[GPU {device_id}] Starting stress test...")

    try:
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)

        while not stop_flag.is_set():
            torch.mm(a, b)

    except Exception as e:
        print(f"[GPU {device_id}] Error: {e}")
    finally:
        print(f"[GPU {device_id}] Stopped")


def parse_gpu_ids(gpu_arg: str, max_gpus: int) -> List[int]:
    """Parse GPU IDs from argument string."""
    if gpu_arg == 'all':
        return list(range(max_gpus))

    ids = []
    for part in gpu_arg.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))

    return [i for i in ids if 0 <= i < max_gpus]


def main():
    parser = argparse.ArgumentParser(description='GPU Stress Test')
    parser.add_argument('-g', '--gpus', default='all',
                        help='GPU IDs: "all", "0,1,2", "0-3", or "0,2-3"')
    parser.add_argument('-s', '--size', type=int, default=8192,
                        help='Matrix size (default: 8192)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPUs")

    gpu_ids = parse_gpu_ids(args.gpus, gpu_count)
    if not gpu_ids:
        print("Error: No valid GPU IDs")
        sys.exit(1)

    print(f"Target GPUs: {gpu_ids}")
    print("Press Ctrl+C to stop\n")

    # Setup signal handler
    def signal_handler(sig, frame):
        print("\nStopping...")
        stop_flag.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start processes
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=stress_gpu, args=(gpu_id, args.size))
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join()

    print("All GPUs stopped")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
