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
import os
from pathlib import Path

import jax

from experimental.benchmarks import gn_scaling


def main():
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
