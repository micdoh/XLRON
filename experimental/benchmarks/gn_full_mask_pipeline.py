"""Benchmark new kernels first, then run matched heuristics from frozen sources."""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimization-root", type=Path, required=True)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--gpu-memory-fraction", type=float)
    p.add_argument("--benchmark-output", default="runs/masking-v8-validated")
    p.add_argument("--heuristic-output", default="runs/heuristics-validated")
    args = p.parse_args()
    optimization = args.optimization_root.resolve()
    training = args.training_root.resolve()
    env = dict(os.environ, PYTHONPATH=str(optimization))
    benchmark = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "experimental.benchmarks.gn_full_mask_followup",
        "--gpu",
        str(args.gpu),
        "--predicate-packing",
        "--state-input",
        "dense-state.npz",
        "--output",
        args.benchmark_output,
    ]
    sharing = (
        ["--gpu-memory-fraction", str(args.gpu_memory_fraction)]
        if args.gpu_memory_fraction is not None
        else []
    )
    benchmark += sharing
    code = subprocess.call(benchmark, cwd=optimization, env=env)
    if code == 0:
        subprocess.call(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "experimental.benchmarks.gn_log_full_mask_report",
                "--directory",
                args.benchmark_output,
                "--name",
                "full-mask-predicate-packing-v8",
            ],
            cwd=optimization,
            env=env,
        )
    # Heuristics use the previously validated kernel version, independent of
    # whether the experimental rewrite improves performance or passes its gate.
    raise SystemExit(
        subprocess.call(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "experimental.benchmarks.gn_full_mask_campaign",
                "--gpu",
                str(args.gpu),
                "--kind",
                "heuristics",
                "--output",
                args.heuristic_output,
                "--initial-state",
                str(training / "dense-state.npz"),
                "--resume-model",
                str(args.checkpoint),
            ]
            + sharing,
            cwd=training,
            env=dict(os.environ, PYTHONPATH=str(training)),
        )
    )


if __name__ == "__main__":
    main()
