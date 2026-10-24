"""Autotune the experimental XPM kernel against the equivalent dense JAX formula.

Example: CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m xlron.kernels.gn_xpm_benchmark --channels 1376 --links 64 \
  --output /tmp/xpm-1376.json

This is a coefficient microbenchmark; a kernel speedup is not a rollout speedup.
"""

import argparse
import itertools
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experimental.benchmarks.gn_scaling import measure
from xlron.kernels.gn_xpm import xpm_compact_jax, xpm_pallas, xpm_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=420)
    parser.add_argument("--links", type=int, default=64)
    parser.add_argument("--occupancy", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--compact", action="store_true", help="Skip zero interferers in Pallas")
    parser.add_argument("--rows", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--columns", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = args.channels
    rng = np.random.default_rng(42)
    power = rng.uniform(1e-5, 3e-4, (args.links, n)).astype(np.float32)
    occupied = rng.uniform(size=power.shape) < args.occupancy
    power *= occupied
    bandwidth = rng.choice([12.5e9, 25e9, 50e9], power.shape).astype(np.float32) * occupied
    frequency = np.broadcast_to(np.linspace(-8e12, 8e12, n, dtype=np.float32), power.shape)
    inputs = tuple(jnp.asarray(x) for x in (power, frequency, bandwidth))
    physics = {
        "a": 4.6e-5,
        "a_bar": 4.6e-5,
        "gamma": 1.2e-3,
        "cr": 2.8e-17,
        "beta2": -2.1e-26,
        "beta3": 1.4e-40,
        "beta4": 0.0,
    }
    reference_fn = jax.vmap(partial(xpm_reference, **physics))
    reference, expected = measure(lambda x: reference_fn(*x), inputs, args.iterations)
    expected = np.asarray(expected)
    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "jax": jax.__version__,
        "device": jax.devices()[0].device_kind,
        "reference": reference,
        "variants": [],
    }
    pure_fn = jax.vmap(partial(xpm_compact_jax, **physics))
    pure, actual = measure(lambda x: pure_fn(*x), inputs, args.iterations)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-4)
    pure["speedup"] = reference["median_ms"] / pure["median_ms"]
    report["compact_jax"] = pure
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for rows, columns, warps in itertools.product(args.rows, args.columns, args.warps):
        config = {"block_i": rows, "block_k": columns, "num_warps": warps}
        print(f"Benchmarking {config}", flush=True)
        entry = {"config": config}
        try:
            fn = jax.vmap(partial(xpm_pallas, **physics, **config, compact=args.compact))
            metrics, actual = measure(lambda x, fn=fn: fn(*x), inputs, args.iterations)
            actual = np.asarray(actual)
            np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-4)
            entry.update(metrics)
            entry["speedup"] = reference["median_ms"] / metrics["median_ms"]
            entry["max_relative_error"] = float(
                np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-10))
            )
        except (AssertionError, ValueError, RuntimeError, TypeError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["variants"].append(entry)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
    valid = [v for v in report["variants"] if "speedup" in v]
    if valid:
        report["best"] = max(valid, key=lambda v: v["speedup"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report.get("best", {"error": "no valid kernel"})), flush=True)


if __name__ == "__main__":
    main()
