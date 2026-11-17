#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

PRECISIONS = ("fp32", "fp16")
THREAD_COUNTS = (1, 2, 4)
RATES = (1.0, 2.0, 3.0)


def config_basename(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("must be a non-empty name without path separators")
    return value


def run_grid(basename: str) -> int:
    repository_root = Path(__file__).resolve().parents[2]
    runner = repository_root / "scripts" / "eval" / "run_kitti_sequences.py"

    for precision in PRECISIONS:
        for n_threads in THREAD_COUNTS:
            for rate in RATES:
                config = f"{basename}_{precision}_t{n_threads}_r{rate}"
                command = [
                    "uv",
                    "run",
                    str(runner),
                    "--precision",
                    precision,
                    "--n-threads",
                    str(n_threads),
                    "--rate",
                    str(rate),
                    "--config",
                    config,
                ]
                print(f"\n==> Running configuration {config}", flush=True)
                result = subprocess.run(command, cwd=repository_root, check=False)
                if result.returncode != 0:
                    print(f"Error: configuration {config} failed (exit code {result.returncode})")
                    return result.returncode

    print(f"Completed {len(PRECISIONS) * len(THREAD_COUNTS) * len(RATES)} configurations.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all KITTI sequences for every detector design-axis combination.")
    parser.add_argument(
        "--config-basename",
        required=True,
        type=config_basename,
        help="Base name used to build a unique --config label for each run",
    )
    args = parser.parse_args()
    return run_grid(args.config_basename)


if __name__ == "__main__":
    raise SystemExit(main())
