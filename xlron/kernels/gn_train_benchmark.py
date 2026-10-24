"""Run real PPO with an opt-in experimental XPM implementation.

Example: python -m xlron.kernels.gn_train_benchmark --xpm-backend pallas_compact \
    --env_type=rmsa_gn_model --link_resources=1376 --NUM_ENVS=16 ...

The override precedes importing the training entrypoint. The ordinary CLI and
production defaults are unchanged; this wrapper is for controlled comparisons.
"""

import argparse
import ctypes
import os
import runpy
import sys
from contextlib import contextmanager
from functools import partial

from xlron.train.compilation import configure_jax_runtime


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--pool-before-projection", action="store_true")
    parser.add_argument(
        "--step-xpm-backend", choices=["jax", "pallas", "pallas_grouped"], default="jax"
    )
    parser.add_argument("--path-tile", type=int, nargs=2, metavar=("SLOTS", "WARPS"))
    parser.add_argument("--candidate-snr-backend", choices=["jax", "pallas"], default="jax")
    parser.add_argument("--xpm-tile", type=int, nargs=3, metavar=("ROWS", "COLUMNS", "WARPS"))
    parser.add_argument(
        "--profile-cuda-increment",
        type=int,
        default=0,
        help="Capture this execution increment with the CUDA profiler API (0 disables)",
    )
    parser.add_argument(
        "--path-snr-backend",
        choices=["jax", "pallas", "pallas_fused", "pallas_precomputed"],
        default="jax",
    )
    parser.add_argument(
        "--xpm-backend",
        choices=[
            "jax",
            "pallas_compact",
            "pallas_sparse",
            "pallas_reuse",
            "pallas_skip",
            "pallas_inline",
            "pallas_grouped",
            "mosaic",
        ],
        required=True,
    )
    args, remaining = parser.parse_known_args()
    configure_jax_runtime(remaining)
    from xlron.environments import env_funcs
    from xlron.environments.gn_model import isrs_gn_model
    from xlron.train import train_utils

    if args.pool_before_projection:
        train_utils.ActorCriticTransformer = partial(
            train_utils.ActorCriticTransformer, pool_before_projection=True
        )

    original_setup_wandb = train_utils.setup_wandb

    def setup_wandb(*setup_args, **setup_kwargs):
        original_setup_wandb(*setup_args, **setup_kwargs)
        train_utils.wandb.config.update(
            {
                "kernel_xpm_backend": args.xpm_backend,
                "kernel_step_xpm_backend": args.step_xpm_backend,
                "kernel_candidate_snr_backend": args.candidate_snr_backend,
                "kernel_path_snr_backend": args.path_snr_backend,
                "kernel_xpm_tile": args.xpm_tile,
                "kernel_path_tile": args.path_tile,
                "pool_before_projection": args.pool_before_projection,
                "experiment_source_fingerprint": os.environ.get("XLRON_SOURCE_FINGERPRINT"),
            }
        )

    train_utils.setup_wandb = setup_wandb

    if args.path_tile:
        from xlron.kernels import gn_path_snr

        gn_path_snr.path_snr_pallas = partial(
            gn_path_snr.path_snr_pallas,
            block_slots=args.path_tile[0],
            num_warps=args.path_tile[1],
        )

    if args.candidate_snr_backend != "jax":
        env_funcs.get_candidate_snr_evidence = partial(
            env_funcs.get_candidate_snr_evidence, backend=args.candidate_snr_backend
        )

    if args.step_xpm_backend != "jax":
        isrs_gn_model.get_snr = partial(isrs_gn_model.get_snr, xpm_backend=args.step_xpm_backend)

    if args.profile_cuda_increment:
        from xlron.environments.wrappers import Profiler

        runtime = ctypes.CDLL(os.environ.get("CUDART_PATH", "libcudart.so.12"))
        original_section = Profiler.section
        execution_count = 0

        @contextmanager
        def profiled_section(self, tag, frames=None):
            nonlocal execution_count
            if tag == "EXECUTION":
                execution_count += 1
            capture = tag == "EXECUTION" and execution_count == args.profile_cuda_increment
            if capture and (status := runtime.cudaProfilerStart()):
                raise RuntimeError(f"cudaProfilerStart failed with status {status}")
            try:
                with original_section(self, tag, frames=frames):
                    yield
            finally:
                if capture and (status := runtime.cudaProfilerStop()):
                    raise RuntimeError(f"cudaProfilerStop failed with status {status}")

        Profiler.section = profiled_section

    isrs_gn_model.get_snr_fused = partial(
        isrs_gn_model.get_snr_fused,
        xpm_backend=args.xpm_backend,
        xpm_tile=tuple(args.xpm_tile) if args.xpm_tile else None,
    )
    env_funcs.get_lightpath_snr = partial(
        env_funcs.get_lightpath_snr, nsr_backend=args.path_snr_backend
    )
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module("xlron.train.train", run_name="__main__")


if __name__ == "__main__":
    main()
