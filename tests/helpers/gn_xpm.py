"""Deterministic XPM physics, inputs and float64 references shared by kernel tests."""

import jax.numpy as jnp
import numpy as np

PARAMS = {
    "a": 4.6e-5,
    "a_bar": 4.6e-5,
    "gamma": 1.2e-3,
    "cr": 2.8e-17,
    "beta2": -2.1e-26,
    "beta3": 1.4e-40,
    "beta4": 0.0,
}


def inputs(n, seed=7):
    rng = np.random.default_rng(seed)
    power = rng.uniform(1e-5, 3e-4, n).astype(np.float32)
    bandwidth = rng.choice([12.5e9, 25e9, 50e9], n).astype(np.float32)
    frequency = np.linspace(-8e12, 8e12, n, dtype=np.float32)
    power[::5] = 0
    bandwidth[::5] = 0
    if n > 3:
        frequency[2] = frequency[1]  # Same-centre/self terms must be excluded.
    return tuple(jnp.asarray(x) for x in (power, frequency, bandwidth))


def numpy_xpm64(power, frequency, bandwidth, **physics):
    """Independent float64 row-wise evaluation of the XPM equation, Ref. [1, Eq. (11)].

    No shared JAX helper or compiled kernel is used. This checks numerical
    implementation accuracy, not the physical approximation itself.
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


def scaled_xpm_inputs(frequency, physics):
    """Binary-rescale frequencies so float32 keeps beta3/beta4 (see issue #64).

    SI beta4 (~1e-55) is below the float32 range and CPU execution flushes
    subnormal beta3 (~1e-40). A power-of-two frequency unit leaves the XPM output
    unchanged while keeping these coefficients representable.
    """
    scale = 2.0**40
    coefficients = dict(physics)
    coefficients["beta2"] *= scale
    coefficients["beta3"] *= scale**2
    coefficients["beta4"] *= scale**3
    coefficients["cr"] *= scale
    return frequency / scale, coefficients
