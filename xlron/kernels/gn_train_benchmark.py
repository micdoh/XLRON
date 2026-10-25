"""Run real PPO with an opt-in experimental XPM implementation.

Example: python -m xlron.kernels.gn_train_benchmark --xpm-backend pallas_compact \
    --env_type=rmsa_gn_model --link_resources=1376 --NUM_ENVS=16 ...

The override precedes importing the training entrypoint. The ordinary CLI and
production defaults are unchanged; this wrapper is for controlled comparisons.
"""

import argparse
import ctypes
import hashlib
import os
import runpy
import sys
from contextlib import contextmanager
from functools import partial

from xlron.train.compilation import configure_jax_runtime


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--initial-model", help="Load these weights while saving to --MODEL_PATH")
    parser.add_argument("--upgrade-modulation-head", action="store_true")
    parser.add_argument(
        "--initial-env-state", help="Validated admitted physical-state NPZ bootstrap"
    )
    parser.add_argument("--bootstrap-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--per-env-output", help="Record per-environment blocking and anomaly states"
    )
    parser.add_argument("--pool-before-projection", action="store_true")
    parser.add_argument("--new-lightpath-prefilter", action="store_true")
    parser.add_argument("--matmul-precision", choices=("default", "high", "highest"))
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
    if args.matmul_precision:
        import jax

        jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    from xlron.environments import env_funcs
    from xlron.environments.gn_model import isrs_gn_model
    from xlron.train import train_utils

    if args.new_lightpath_prefilter:
        from xlron.environments.gn_model import rmsa_gn_model

        env_funcs.mask_slots_rmsa_gn_model = partial(
            env_funcs.mask_slots_rmsa_gn_model, prefilter_new=True
        )
        rmsa_gn_model.mask_slots_rmsa_gn_model = env_funcs.mask_slots_rmsa_gn_model
        # Heuristic warmup imports this function directly rather than calling
        # env.action_mask, so update that alias as well.
        train_utils.mask_slots_rmsa_gn_model = env_funcs.mask_slots_rmsa_gn_model

    if args.initial_model:
        original_load_model = train_utils.load_model

        def load_initial_model(config, key):
            initial_config = config.copy()
            initial_config.MODEL_PATH = args.initial_model
            if args.upgrade_modulation_head:
                initial_config.transformer_modulation_features = False
            model = original_load_model(initial_config, key)
            if args.upgrade_modulation_head:
                import jax

                from xlron.models.modulation_actor import ModulationAwareActorCritic

                if not config.transformer_modulation_features:
                    raise ValueError(
                        "Modulation migration requires transformer_modulation_features"
                    )
                model = ModulationAwareActorCritic(model, jax.random.fold_in(key, 20261025))
            return model

        train_utils.load_model = load_initial_model

    if args.initial_env_state:
        from pathlib import Path

        from xlron.kernels.gn_initial_state import restore_physical_state

        original_data_setup = train_utils.experiment_data_setup

        def initial_state_setup(config, *setup_args, **setup_kwargs):
            if config.ENV_WARMUP_STEPS:
                raise ValueError(
                    "Physical bootstrap uses ENV_WARMUP_STEPS=0; log provenance separately"
                )
            runner, env, params = original_data_setup(config, *setup_args, **setup_kwargs)
            runner = restore_physical_state(runner, params, config, args.initial_env_state)
            print(f"Restored admitted physical bootstrap: {args.initial_env_state}", flush=True)
            if args.bootstrap_warmup_steps:
                import jax

                warmup_config = config.copy()
                warmup_config.ENV_WARMUP_STEPS = args.bootstrap_warmup_steps
                key = jax.random.PRNGKey(config.SEED + 20261025)
                keys = jax.random.split(key, config.NUM_ENVS) if config.NUM_ENVS > 1 else key
                inputs = (keys, runner[1], runner[2])
                warmup = train_utils.get_warmup_fn(inputs, env, params, runner[0], warmup_config)
                if config.NUM_ENVS > 1:
                    warmup = jax.vmap(warmup, in_axes=(0, None))
                outer, observation = jax.block_until_ready(jax.jit(warmup)(inputs, runner[0]))
                outer = train_utils.reset_warmup_metric_counters(outer)
                if config.USE_GNN or config.USE_TRANSFORMER:
                    observation = (outer.env_state, params)
                runner = (runner[0], outer, observation, runner[3], runner[4])
                print(
                    f"Discarded {args.bootstrap_warmup_steps} bootstrap requests per replica",
                    flush=True,
                )
            return runner, env, params

        train_utils.experiment_data_setup = initial_state_setup
        state_digest = hashlib.sha256(Path(args.initial_env_state).read_bytes()).hexdigest()
    else:
        state_digest = None

    if args.per_env_output:
        from xlron.kernels.gn_campaign_diagnostics import install_diagnostics

        install_diagnostics(train_utils, args.per_env_output)

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
                "kernel_new_lightpath_prefilter": args.new_lightpath_prefilter,
                "matmul_precision": args.matmul_precision,
                "pool_before_projection": args.pool_before_projection,
                "experiment_source_fingerprint": os.environ.get("XLRON_SOURCE_FINGERPRINT"),
                "initial_checkpoint": args.initial_model,
                "gpu_sharing": os.environ.get("XLRON_GPU_SHARING"),
                "jax_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
                "initial_physical_state": args.initial_env_state,
                "initial_physical_state_sha256": state_digest,
                "bootstrap_warmup_steps": args.bootstrap_warmup_steps,
                "bootstrap_note": "Common admitted 3000-request heuristic states; independent future traffic; initial transient is not a steady-state evaluation"
                if args.initial_env_state
                else None,
                "upgraded_modulation_head": args.upgrade_modulation_head,
                "optimizer_restored": False if args.initial_model else None,
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
