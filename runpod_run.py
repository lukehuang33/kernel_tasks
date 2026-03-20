"""Run a 2048x2048 PyTorch matmul remotely on Runpod B200.

Usage:
    pip install runpod-flash
    flash login
    python runpod_run.py

This mirrors the local-entrypoint style used by the Modal examples in this
workspace, but dispatches the computation to Runpod Flash.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from matmul_worker import DEFAULT_DIM, DEFAULT_DTYPE, TARGET_GPU, run_matmul

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2048x2048 PyTorch matmul on Runpod B200."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help=f"Square matrix dimension to multiply (default: {DEFAULT_DIM}).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default=DEFAULT_DTYPE,
        help=f"PyTorch dtype to use (default: {DEFAULT_DTYPE}).",
    )
    return parser.parse_args()


async def main() -> None:
    # Flash uses the current working directory as the project root when it
    # generates the local dev server. Force it to this file's directory so a
    # parent folder name like "runpod-matmul-test" does not become part of the
    # Python import path.
    os.chdir(PROJECT_DIR)

    args = parse_args()
    print(
        f"Dispatching torch.matmul for {args.dim}x{args.dim} matrices "
        f"on Runpod {TARGET_GPU.name}..."
    )
    result = await run_matmul(dim=args.dim, dtype=args.dtype)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
