"""Run the standard environment benchmark with an explicit experimental XPM backend.

Example: python -m xlron.kernels.gn_environment_benchmark --xpm-backend pallas_compact \
    --slots 1376 --envs 16 --warmup-steps 128 --independent-streams \
    --kernels mask rollout --output /tmp/gn-pallas.json

The override happens before JIT tracing and applies to warmup, masks and steps.
No production default or environment configuration is changed by this wrapper.
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

from experimental.benchmarks import gn_scaling
from xlron.environments.gn_model import isrs_gn_model


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--xpm-backend",
        choices=["jax", "jax_dense", "jax_compact", "pallas", "pallas_compact"],
        required=True,
    )
    args, remaining = parser.parse_known_args()
    isrs_gn_model.get_snr_fused = partial(isrs_gn_model.get_snr_fused, xpm_backend=args.xpm_backend)
    sys.argv = [sys.argv[0], *remaining]
    gn_scaling.main()
    if "--output" in remaining:
        path = Path(remaining[remaining.index("--output") + 1])
        report = json.loads(path.read_text())
        report["xpm_backend"] = args.xpm_backend
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
