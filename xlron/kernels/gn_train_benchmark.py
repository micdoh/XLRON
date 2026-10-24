"""Run real PPO with an opt-in experimental XPM implementation.

Example: python -m xlron.kernels.gn_train_benchmark --xpm-backend pallas_compact \
    --env_type=rmsa_gn_model --link_resources=1376 --NUM_ENVS=16 ...

The override precedes importing the training entrypoint. The ordinary CLI and
production defaults are unchanged; this wrapper is for controlled comparisons.
"""

import argparse
import runpy
import sys
from functools import partial

from xlron.train.compilation import configure_jax_runtime


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
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

    isrs_gn_model.get_snr_fused = partial(isrs_gn_model.get_snr_fused, xpm_backend=args.xpm_backend)
    env_funcs.get_lightpath_snr = partial(
        env_funcs.get_lightpath_snr, nsr_backend=args.path_snr_backend
    )
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module("xlron.train.train", run_name="__main__")


if __name__ == "__main__":
    main()
