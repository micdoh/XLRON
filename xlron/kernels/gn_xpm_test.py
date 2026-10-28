"""Accuracy and batching checks for gn_xpm.py (interpret mode on CPU, native on GPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.gn_xpm import PARAMS, inputs, kernel_physics, numpy_xpm64
from xlron.environments.gn_model.isrs_gn_model import (
    _xpm_compact,
    get_snr_fused,
    xpm_after_insertion,
    xpm_pairs,
)
from xlron.kernels.gn_xpm import xpm_pallas

# PARAMS holds the SI truth for numpy_xpm64; the kernels take scaled dispersion.
KERNEL = kernel_physics()


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
    expected = xpm_reference(*args, **KERNEL)
    actual = jax.jit(
        lambda *x: xpm_pallas(*x, **KERNEL, block_i=8, block_k=16, persistent_rows=persistent_rows)
    )(*args)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)


def test_repeated_runs_preserve_every_output_and_interferer():
    # Runs cross tile boundaries; equal centres with unequal power/bandwidth stay
    # distinct. Includes empty gaps, zero-power live rows and singleton tails.
    p, f, b = (jnp.repeat(x, 5) for x in inputs(13))
    p = p.at[6].multiply(2).at[16].set(0)
    b = b.at[7].multiply(2).at[20:35].set(0)
    actual = jax.jit(lambda *x: xpm_pallas(*x, **KERNEL, block_i=8, block_k=8))(p, f, b)
    np.testing.assert_allclose(actual, xpm_reference(p, f, b, **KERNEL), rtol=3e-5, atol=1e-5)


def test_nested_vmap_and_jit_with_different_live_counts():
    args = [jnp.stack([inputs(13, seed)[i] for seed in range(3)]) for i in range(3)]
    # Different live counts, including an empty link, exercise dynamic loop bounds.
    args[0] = args[0].at[0].set(0).at[1, :7].set(0)
    expected = jax.vmap(lambda *x: xpm_reference(*x, **KERNEL))(*args)
    actual = jax.jit(jax.vmap(lambda *x: xpm_pallas(*x, **KERNEL)))(*args)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)
    compact = jax.jit(jax.vmap(lambda *x: _xpm_compact(*x, **KERNEL)))(*args)
    np.testing.assert_allclose(compact, expected, rtol=3e-5, atol=1e-5)


@pytest.mark.parametrize("n", [33, 257])
def test_float32_keeps_higher_order_dispersion(n):
    """Scaled beta3/beta4 keep the kernel on the float64 answer (issue #64).

    SI beta4 is below the float32 range and CPU arithmetic flushes subnormal
    beta3 to zero, so the unscaled coefficients lose both contributions here.
    """
    p, f, b = inputs(n)
    physics = {**PARAMS, "beta4": -2e-55}
    actual = jax.jit(lambda *x: xpm_pallas(*x, **kernel_physics(physics)))(p, f, b)
    np.testing.assert_allclose(actual, numpy_xpm64(p, f, b, **physics), rtol=5e-6, atol=1e-5)


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
    actual = jax.jit(jax.vmap(lambda *x: get_snr_fused(*x, **physics, xpm_backend="pallas")))(*args)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(actual)[0], -1e5)


class TestPolynomialXpm:
    @staticmethod
    @pytest.mark.parametrize("scale", [0.5, 1.0, 3.0])
    def test_coefficients_give_the_xpm_at_any_total_power(scale):
        p, f, b = (jnp.repeat(x, 3) for x in inputs(41))
        s0, s1, s2 = jax.jit(lambda *x: xpm_pallas(*x, **KERNEL, polynomial=True))(p, f, b)
        # At the arrays' own total power: the direct kernel. Elsewhere: the float64
        # reference with that total (scaled powers change only the tilt's total).
        total = jnp.sum(p)
        direct = xpm_pallas(p, f, b, **KERNEL)
        np.testing.assert_allclose(s0 + total * s1 + total**2 * s2, direct, rtol=3e-5, atol=1e-6)
        expected = numpy_xpm64(p, f, b, **PARAMS, total=float(total) * scale)
        moved = s0 + scale * total * s1 + (scale * total) ** 2 * s2
        np.testing.assert_allclose(moved, expected, rtol=3e-5, atol=1e-6)

    @staticmethod
    @pytest.mark.parametrize("start, width", [(20, 3), (0, 1), (60, 4)])
    def test_insertion_matches_a_full_re_solve(start, width):
        p, f, b = (np.array(x) for x in inputs(64))
        p[start : start + width] = b[start : start + width] = 0  # free spectrum
        before = tuple(jnp.asarray(x) for x in (p, f, b))
        coefficients = xpm_pallas(*before, **KERNEL, polynomial=True)
        inside = (np.arange(64) >= start) & (np.arange(64) < start + width)
        centre = f[start : start + width].mean()
        after = (
            jnp.where(inside, 2e-4, before[0]),
            jnp.where(inside, centre, before[1]),
            jnp.where(inside, 12.5e9, before[2]),
        )
        inserted = jax.jit(
            lambda c, *x: xpm_after_insertion(
                c, *x, jnp.asarray(inside), 2e-4, centre, 12.5e9, width, **KERNEL
            )
        )(coefficients, *after)
        np.testing.assert_allclose(inserted, xpm_pallas(*after, **KERNEL), rtol=3e-5, atol=1e-6)
        np.testing.assert_allclose(inserted, numpy_xpm64(*after, **PARAMS), rtol=3e-5, atol=1e-6)
