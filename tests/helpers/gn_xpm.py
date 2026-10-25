"""Deterministic XPM physics and channel inputs shared by kernel tests."""

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
