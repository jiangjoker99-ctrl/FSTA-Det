"""Run the self-contained checks without requiring pytest."""

import argparse
import runpy
import subprocess
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ddp', action='store_true', help='Include the two-rank CPU check')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    count = 0
    for file in sorted((root / 'tests').glob('test_*.py')):
        namespace = runpy.run_path(str(file))
        tests = sorted(
            (name, function) for name, function in namespace.items()
            if name.startswith('test_') and callable(function))
        for name, function in tests:
            function()
            count += 1
            print(f'PASS {file.name}::{name}', flush=True)
    if args.ddp:
        subprocess.run([sys.executable, str(root / 'tools/check_ddp.py')],
                       cwd=root, check=True)
    print(f'{count} tests passed')


if __name__ == '__main__':
    main()
