"""Run the standard environment benchmark with an explicit experimental XPM backend.

Example: python -m xlron.kernels.gn_environment_benchmark --xpm-backend pallas_compact \
    --slots 1376 --envs 16 --warmup-steps 128 --independent-streams \
    --kernels mask rollout --output /tmp/gn-pallas.json

The override happens before JIT tracing and applies to warmup, masks and steps.
No production default or environment configuration is changed by this wrapper.
"""

import argparse
import json
import os
import runpy
import sys
from functools import partial
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--no-full-format-pruning", action="store_true")
    parser.add_argument("--new-lightpath-prefilter", action="store_true")
    parser.add_argument("--pack-old-predicates", action="store_true")
    parser.add_argument("--reuse-screened-power", action="store_true")
    parser.add_argument("--narrow-new-snr", action="store_true")
    parser.add_argument("--reuse-unaffected-snr", action="store_true")
    parser.add_argument("--compact-snr-updates", action="store_true")
    parser.add_argument("--screen-batch-size", type=int)
    parser.add_argument(
        "--step-xpm-backend", choices=["jax", "pallas", "pallas_grouped"], default="jax"
    )
    parser.add_argument("--path-tile", type=int, nargs=2, metavar=("SLOTS", "WARPS"))
    parser.add_argument("--candidate-snr-backend", choices=["jax", "pallas"], default="jax")
    parser.add_argument("--xpm-tile", type=int, nargs=3, metavar=("ROWS", "COLUMNS", "WARPS"))
    parser.add_argument(
        "--path-snr-backend",
        choices=["jax", "pallas", "pallas_fused", "pallas_precomputed"],
        default="jax",
    )
    parser.add_argument(
        "--reference-script", type=Path, help="Use an archived workload script unchanged"
    )
    parser.add_argument(
        "--compiler-profile",
        choices=["default", "library"],
        default="default",
        help="library bypasses pathological GPU GEMM autotuning; use the same profile for controls",
    )
    parser.add_argument(
        "--xpm-backend",
        choices=[
            "jax",
            "jax_dense",
            "jax_compact",
            "pallas",
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
    if args.compiler_profile == "library":
        flags = os.environ.get("XLA_FLAGS", "")
        for option in ("--xla_gpu_enable_triton_gemm=false", "--xla_gpu_autotune_level=0"):
            if option not in flags.split():
                flags += " " + option
        os.environ["XLA_FLAGS"] = flags.strip()
    # Parsing and compiler flags must precede imports whose default arguments
    # allocate JAX arrays and initialize the CUDA backend.
    from experimental.benchmarks import gn_scaling
    from xlron.environments import env_funcs
    from xlron.environments.gn_model import isrs_gn_model

    if (
        args.no_full_format_pruning
        or args.new_lightpath_prefilter
        or args.pack_old_predicates
        or args.reuse_screened_power
        or args.narrow_new_snr
        or args.reuse_unaffected_snr
        or args.compact_snr_updates
        or args.screen_batch_size is not None
    ):
        from xlron.environments.gn_model import rmsa_gn_model

        env_funcs.mask_slots_rmsa_gn_model = partial(
            env_funcs.mask_slots_rmsa_gn_model,
            prune_formats=not args.no_full_format_pruning,
            prefilter_new=args.new_lightpath_prefilter,
            pack_old=args.pack_old_predicates,
            reuse_screened_power=args.reuse_screened_power,
            narrow_new_snr=args.narrow_new_snr,
            reuse_unaffected_snr=args.reuse_unaffected_snr,
            compact_snr_updates=args.compact_snr_updates,
            screen_batch_size=args.screen_batch_size,
        )
        rmsa_gn_model.mask_slots_rmsa_gn_model = env_funcs.mask_slots_rmsa_gn_model

    if args.path_tile:
        from xlron.kernels import gn_path_snr

        gn_path_snr.path_snr_pallas = partial(
            gn_path_snr.path_snr_pallas,
            block_slots=args.path_tile[0],
            num_warps=args.path_tile[1],
        )

    if args.candidate_snr_backend != "jax":
        env_funcs.get_candidate_baseline_snr = partial(
            env_funcs.get_candidate_baseline_snr, backend=args.candidate_snr_backend
        )
        env_funcs.get_candidate_snr_evidence = partial(
            env_funcs.get_candidate_snr_evidence, backend=args.candidate_snr_backend
        )

    if args.xpm_backend != "jax":
        isrs_gn_model.get_snr_fused = partial(
            isrs_gn_model.get_snr_fused,
            xpm_backend=args.xpm_backend,
            xpm_tile=tuple(args.xpm_tile) if args.xpm_tile else None,
        )
    if args.step_xpm_backend != "jax":
        isrs_gn_model.get_snr = partial(isrs_gn_model.get_snr, xpm_backend=args.step_xpm_backend)
    if args.path_snr_backend != "jax":
        env_funcs.get_lightpath_snr = partial(
            env_funcs.get_lightpath_snr, nsr_backend=args.path_snr_backend
        )
    sys.argv = [sys.argv[0], *remaining]
    if args.reference_script:
        runpy.run_path(str(args.reference_script), run_name="__main__")
    else:
        gn_scaling.main()
    if "--output" in remaining:
        path = Path(remaining[remaining.index("--output") + 1])
        report = json.loads(path.read_text())
        report["xpm_backend"] = args.xpm_backend
        report["step_xpm_backend"] = args.step_xpm_backend
        report["candidate_snr_backend"] = args.candidate_snr_backend
        report["xpm_tile"] = args.xpm_tile
        report["path_tile"] = args.path_tile
        report["path_snr_backend"] = args.path_snr_backend
        report["compiler_profile"] = args.compiler_profile
        report["full_format_pruning"] = not args.no_full_format_pruning
        report["new_lightpath_prefilter"] = args.new_lightpath_prefilter
        report["pack_old_predicates"] = args.pack_old_predicates
        report["reuse_screened_power"] = args.reuse_screened_power
        report["narrow_new_snr"] = args.narrow_new_snr
        report["reuse_unaffected_snr"] = args.reuse_unaffected_snr
        report["compact_snr_updates"] = args.compact_snr_updates
        report["screen_batch_size"] = args.screen_batch_size
        report["xla_flags"] = os.environ.get("XLA_FLAGS", "")
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
