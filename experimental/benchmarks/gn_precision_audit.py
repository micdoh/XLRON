"""Audit the current GN coefficient numerics against an independent float64 equation.

This is not physical propagation validation. Production formulas and default
arithmetic are unchanged. The rescaled experimental control applies exact binary
frequency units before very small SI dispersion coefficients enter float32.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from xlron.environments.gn_model.isrs_gn_model import get_snr_fused
from xlron.kernels.gn_xpm import xpm_compact_jax, xpm_pallas, xpm_reference
from xlron.kernels.gn_xpm_numerics import scaled_xpm_inputs


def numpy_xpm64(power, frequency, bandwidth, **physics):
    """Independent float64 row-wise evaluation of the implemented XPM equation.

    No shared JAX helper or compiled kernel is used. This checks numerical
    implementation accuracy, not validation of the physical approximation.
    """
    p, f, b = [np.asarray(x, dtype=np.float64) for x in (power, frequency, bandwidth)]
    a, abar = physics["a"], physics["a_bar"]
    ap = a + abar
    result = np.zeros_like(p)
    tilt = (ap - f * np.sum(p) * physics["cr"]) ** 2
    for i in range(len(p)):
        phase = (
            2
            * np.pi**2
            * (f[i] - f)
            * (
                physics["beta2"]
                + np.pi * physics["beta3"] * (f[i] + f)
                + 2 / 3 * np.pi**2 * physics["beta4"] * (f[i] ** 2 + f[i] * f + f**2)
            )
        )
        denominator = b * phase * abar * (2 * a + abar)
        valid = denominator != 0
        terms = (tilt - a**2) / a * np.arctan(phase * b[i] / a)
        terms += (ap**2 - tilt) / ap * np.arctan(phase * b[i] / ap)
        weights = 32 / 27 * (p / max(p[i], 1e-12)) ** 2 * physics["gamma"] ** 2
        result[i] = np.sum(weights[valid] * terms[valid] / denominator[valid])
    return result


def compare(actual, reference):
    a, r = np.asarray(actual), np.asarray(reference)
    return {
        "max_relative_error": float(np.max(np.abs(a - r) / np.maximum(np.abs(r), 1e-12))),
        "max_absolute_error": float(np.max(np.abs(a - r))),
    }


def audit():
    n = 257
    rng = np.random.default_rng(7)
    power = rng.uniform(1e-5, 3e-4, n).astype(np.float32)
    bandwidth = rng.choice([12.5e9, 25e9, 50e9], n).astype(np.float32)
    frequency = np.linspace(-8e12, 8e12, n, dtype=np.float32)
    power[::5] = bandwidth[::5] = 0
    frequency[2] = frequency[1]
    physics = {
        "a": 4.6e-5,
        "a_bar": 4.6e-5,
        "gamma": 1.2e-3,
        "cr": 2.8e-17,
        "beta2": -2.1e-26,
        "beta3": 1.4e-40,
        "beta4": -2e-55,
    }
    args = tuple(jnp.asarray(x) for x in (power, frequency, bandwidth))
    reference = numpy_xpm64(*args, **physics)
    f_scaled, coefficients = scaled_xpm_inputs(args[1], physics)
    normalized = args[0], f_scaled, args[2]
    report = {
        "scope": "Implementation/numerical audit, not validation against physical propagation",
        "jax": jax.__version__,
        "device": jax.devices()[0].device_kind,
        "physics": physics,
        "channels": n,
        "xpm": {},
        "full_snr": {},
    }
    implementations = {"dense_jax": xpm_reference, "packed_jax": xpm_compact_jax}
    if jax.default_backend() == "gpu":
        implementations["triton"] = partial(xpm_pallas, compact=True)
    for name, fn in implementations.items():
        original = jax.jit(lambda *x, fn=fn: fn(*x, **physics))(*args)
        scaled = jax.jit(lambda *x, fn=fn: fn(*x, **coefficients))(*normalized)
        report["xpm"][name] = {
            "original": compare(original, reference),
            "binary_scaled": compare(scaled, reference),
        }
        assert report["xpm"][name]["binary_scaled"]["max_relative_error"] < 5e-6

    common = {
        "num_spans": 3,
        "span_length": 80_000.0,
        "num_channels": n,
        "ref_lambda": 1550e-9,
        "attenuation": 4.6e-5,
        "attenuation_bar": 4.6e-5,
        "nonlinear_coeff": 1.2e-3,
        "raman_gain_slope": 2.8e-17,
        "dispersion_coeff": 17e-6,
        "dispersion_slope": 67.0,
    }
    for coherent in [False, True]:
        inputs32 = (*args, jnp.full((n,), 5.0), jnp.full((n,), 30.0))

        def evaluate(p, f, bw, nf, trx, coherent=coherent):
            return get_snr_fused(
                p,
                f,
                bw,
                **common,
                coherent=coherent,
                amplifier_noise_figure=nf,
                transceiver_snr=trx,
                xpm_backend="jax_dense",
            )

        actual = np.asarray(jax.jit(evaluate)(*inputs32))
        with jax.enable_x64(True):
            inputs64 = tuple(jnp.asarray(np.asarray(x), dtype=jnp.float64) for x in inputs32)
            reference64 = np.asarray(jax.jit(evaluate)(*inputs64))
        occupied = power > 0
        difference = 10 * np.log10(actual[occupied] / reference64[occupied])
        report["full_snr"][f"coherent_{coherent}"] = {
            "reference": "Same implemented equations evaluated in float64; no equation corrections",
            "max_absolute_db_difference": float(np.max(np.abs(difference))),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
