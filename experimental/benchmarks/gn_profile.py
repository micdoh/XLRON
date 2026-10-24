"""Capture one warmed-up GN kernel with Nsight Compute's CUDA profiler API range.

Uses gn_scaling's arguments. Set CUDART_PATH to the local CUDA runtime library.
Example (use a free physical GPU):
  CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  CUDART_PATH=/usr/local/cuda/lib64/libcudart.so.12 \
    ncu --profile-from-start off --launch-count 40 --section SpeedOfLight \
    --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
    --section LaunchStats --section Occupancy --cache-control none --clock-control none \
    --export /tmp/gn-profile \
    python -m experimental.benchmarks.gn_profile --slots 420 --envs 64 \
    --independent-streams --warmup-steps 128 --kernels mask --iterations 1

Collect counter diagnostics with this script; use unprofiled gn_scaling runs for
throughput numbers. The launch limit may cover only part of the captured range.
"""

import ctypes
import argparse
from functools import partial
import os
from pathlib import Path
import sys

import jax

from experimental.benchmarks import gn_scaling
from xlron.environments.gn_model import isrs_gn_model
from xlron.environments import env_funcs


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--step-xpm-backend", choices=["jax", "pallas", "pallas_grouped"], default="jax")
    parser.add_argument("--path-tile", type=int, nargs=2, metavar=("SLOTS", "WARPS"))
    parser.add_argument("--candidate-snr-backend", choices=["jax", "pallas"], default="jax")
    parser.add_argument("--xpm-tile", type=int, nargs=3)
    parser.add_argument("--xpm-backend", default="jax")
    parser.add_argument("--path-snr-backend", default="jax")
    args, remaining = parser.parse_known_args()
    if args.path_tile:
        from xlron.kernels import gn_path_snr

        gn_path_snr.path_snr_pallas = partial(
            gn_path_snr.path_snr_pallas,
            block_slots=args.path_tile[0], num_warps=args.path_tile[1],
        )

    if args.candidate_snr_backend != "jax":
        env_funcs.get_candidate_snr_evidence = partial(
            env_funcs.get_candidate_snr_evidence, backend=args.candidate_snr_backend
        )
    if args.step_xpm_backend != "jax":
        isrs_gn_model.get_snr = partial(isrs_gn_model.get_snr, xpm_backend=args.step_xpm_backend)
    isrs_gn_model.get_snr_fused = partial(
        isrs_gn_model.get_snr_fused,
        xpm_backend=args.xpm_backend,
        xpm_tile=tuple(args.xpm_tile) if args.xpm_tile else None,
    )
    env_funcs.get_lightpath_snr = partial(
        env_funcs.get_lightpath_snr, nsr_backend=args.path_snr_backend
    )
    sys.argv = [sys.argv[0], *remaining]
    runtime = ctypes.CDLL(os.environ.get("CUDART_PATH", "libcudart.so.12"))
    original_measure = gn_scaling.measure

    def profile(fn, arg, iterations):
        metrics, result = original_measure(fn, arg, iterations)
        executable = jax.jit(fn).lower(arg).compile()
        if output := os.environ.get("GN_PROFILE_HLO"):
            Path(output).write_text(executable.as_text())
        jax.block_until_ready(executable(arg))
        status = runtime.cudaProfilerStart()
        if status:
            raise RuntimeError(f"cudaProfilerStart failed with status {status}")
        try:
            result = jax.block_until_ready(executable(arg))
        finally:
            status = runtime.cudaProfilerStop()
            if status:
                raise RuntimeError(f"cudaProfilerStop failed with status {status}")
        return metrics, result

    gn_scaling.measure = profile
    gn_scaling.main()


if __name__ == "__main__":
    main()
