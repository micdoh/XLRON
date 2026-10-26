"""Accuracy and batching checks for gn_xpm.py (interpret mode on CPU, native on GPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.gn_xpm import PARAMS, inputs, numpy_xpm64, scaled_xpm_inputs
from xlron.environments.gn_model.isrs_gn_model import _xpm_compact, get_snr_fused, xpm_pairs
from xlron.kernels.gn_xpm import xpm_pallas


def xpm_reference(power, frequency, bandwidth, **physics):
    """Dense ordinary-JAX XPM coefficient (the get_snr_fused formula)."""
    pairs = xpm_pairs(
        power[:, None],
        frequency[:, None],
        bandwidth[:, None],
        power[None, :],
        frequency[None, :],
        bandwidth[None, :],
        jnp.sum(power),
        **physics,
    )
    return jnp.sum(pairs, axis=1)


@pytest.mark.parametrize("n", [1, 7, 31, 129])
@pytest.mark.parametrize("persistent_rows", [None, 1, 3])
def test_matches_dense_with_padding_and_empty_slots(n, persistent_rows):
    args = list(inputs(n))
    if n > 1:
        args[0] = args[0].at[1].set(0)  # Positive-bandwidth rows survive even at zero power.
    expected = xpm_reference(*args, **PARAMS)
    actual = jax.jit(
        lambda *x: xpm_pallas(*x, **PARAMS, block_i=8, block_k=16, persistent_rows=persistent_rows)
    )(*args)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)


def test_repeated_runs_preserve_every_output_and_interferer():
    # Runs cross tile boundaries; equal centres with unequal power/bandwidth stay
    # distinct. Includes empty gaps, zero-power live rows and singleton tails.
    p, f, b = (jnp.repeat(x, 5) for x in inputs(13))
    p = p.at[6].multiply(2).at[16].set(0)
    b = b.at[7].multiply(2).at[20:35].set(0)
    actual = jax.jit(lambda *x: xpm_pallas(*x, **PARAMS, block_i=8, block_k=8))(p, f, b)
    np.testing.assert_allclose(actual, xpm_reference(p, f, b, **PARAMS), rtol=3e-5, atol=1e-5)


def test_nested_vmap_and_jit_with_different_live_counts():
    args = [jnp.stack([inputs(13, seed)[i] for seed in range(3)]) for i in range(3)]
    # Different live counts, including an empty link, exercise dynamic loop bounds.
    args[0] = args[0].at[0].set(0).at[1, :7].set(0)
    expected = jax.vmap(lambda *x: xpm_reference(*x, **PARAMS))(*args)
    actual = jax.jit(jax.vmap(lambda *x: xpm_pallas(*x, **PARAMS)))(*args)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)
    compact = jax.jit(jax.vmap(lambda *x: _xpm_compact(*x, **PARAMS)))(*args)
    np.testing.assert_allclose(compact, expected, rtol=3e-5, atol=1e-5)


@pytest.mark.parametrize("n", [33, 257])
def test_rescaled_float32_matches_float64_reference(n):
    """With binary frequency rescaling float32 keeps beta3/beta4 (issue #64)."""
    p, f, b = inputs(n)
    physics = {**PARAMS, "beta4": -2e-55}
    f_scaled, scaled = scaled_xpm_inputs(f, physics)
    actual = jax.jit(lambda *x: xpm_pallas(*x, **scaled))(p, f_scaled, b)
    np.testing.assert_allclose(actual, numpy_xpm64(p, f, b, **physics), rtol=5e-6, atol=1e-5)


@pytest.mark.parametrize("n", [33, 257])
@pytest.mark.xfail(strict=True, reason="Issue #64: SI float32 beta4 underflow; CPU loses beta3")
def test_si_float32_known_dispersion_underflow(n):
    args = inputs(n)
    physics = {**PARAMS, "beta4": -2e-55}
    actual = jax.jit(lambda *x: xpm_reference(*x, **physics))(*args)
    np.testing.assert_allclose(actual, numpy_xpm64(*args, **physics), rtol=5e-6, atol=1e-5)


@pytest.mark.parametrize("coherent", [False, True])
def test_full_snr_pallas_backend_matches_jax_including_empty_links(coherent):
    args = [jnp.stack([inputs(140, seed)[i] for seed in range(3)]) for i in range(3)]
    args[0] = args[0].at[0].set(0).at[1, :80].set(0)
    physics = {
        "num_spans": 3,
        "span_length": 80_000.0,
        "num_channels": 140,
        "ref_lambda": 1550e-9,
        "attenuation": 4.6e-5,
        "attenuation_bar": 4.6e-5,
        "nonlinear_coeff": 1.2e-3,
        "raman_gain_slope": 2.8e-17,
        "dispersion_coeff": 17e-6,
        "dispersion_slope": 0.067e3,
        "amplifier_noise_figure": jnp.full((140,), 5.0),
        "transceiver_snr": jnp.full((140,), 30.0),
        "coherent": coherent,
    }
    expected = jax.jit(jax.vmap(lambda *x: get_snr_fused(*x, **physics)))(*args)
    actual = jax.jit(jax.vmap(lambda *x: get_snr_fused(*x, **physics, xpm_backend="pallas")))(
        *args
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(actual)[0], -1e5)
