"""Numerical and batching checks for the experimental tiled GN kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experimental.benchmarks.gn_precision_audit import numpy_xpm64
from xlron.environments.gn_model.isrs_gn_model import get_snr_fused
from xlron.kernels.gn_xpm import xpm_compact_jax, xpm_pallas, xpm_reference
from xlron.kernels.gn_xpm_numerics import scaled_xpm_inputs

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


@pytest.mark.parametrize("n", [33, 257])
@pytest.mark.xfail(
    strict=True, reason="Issue #64: SI float32 beta4 underflow; CPU also loses beta3"
)
def test_legacy_float32_known_dispersion_underflow(n):
    args = inputs(n)
    physics = {**PARAMS, "beta4": -2e-55}
    expected = numpy_xpm64(*args, **physics)
    for fn in [xpm_reference, xpm_compact_jax]:
        actual = jax.jit(lambda *x, fn=fn: fn(*x, **physics))(*args)
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-5)


@pytest.mark.parametrize("n", [33, 257])
def test_scaled_float32_matches_independent_float64_reference(n):
    p, f, b = inputs(n)
    physics = {**PARAMS, "beta4": -2e-55}
    expected = numpy_xpm64(p, f, b, **physics)
    f_scaled, scaled = scaled_xpm_inputs(f, physics)
    for fn in [xpm_reference, xpm_compact_jax]:
        actual = jax.jit(lambda *x, fn=fn: fn(*x, **scaled))(p, f_scaled, b)
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-5)


@pytest.mark.parametrize("n", [1, 7, 31, 129])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("compact_rows", [False, True])
def test_interpret_matches_dense_with_padding_and_empty_slots(n, compact, compact_rows):
    args = list(inputs(n))
    if n > 1:
        args[0] = args[0].at[1].set(0)  # Positive-bandwidth rows must survive even at zero power.
    expected = xpm_reference(*args, **PARAMS)
    actual = jax.jit(
        lambda *x: xpm_pallas(
            *x, **PARAMS, compact=compact, compact_rows=compact_rows, interpret=True
        )
    )(*args)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)


def test_nested_vmap_and_jit():
    args = [jnp.stack([inputs(13, seed)[i] for seed in range(3)]) for i in range(3)]
    # Different live counts, including an empty link, exercise dynamic loop bounds.
    args[0] = args[0].at[0].set(0).at[1, :7].set(0)
    fn = jax.jit(jax.vmap(lambda *x: xpm_pallas(*x, **PARAMS, compact=True, interpret=True)))
    expected = jax.vmap(lambda *x: xpm_reference(*x, **PARAMS))(*args)
    np.testing.assert_allclose(fn(*args), expected, rtol=3e-5, atol=1e-5)
    pure = jax.jit(jax.vmap(lambda *x: xpm_compact_jax(*x, **PARAMS)))(*args)
    np.testing.assert_allclose(pure, expected, rtol=3e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["interpret", "cuda"])
@pytest.mark.parametrize(
    "mode",
    ["reuse_rows", "skip_empty_tiles", "skip_empty_columns", "group_interferers", "group_reuse"],
)
def test_repeated_rows_preserve_every_output_and_interferer(backend, mode):
    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    # Runs cross tile boundaries; equal centres with unequal power/bandwidth must
    # remain distinct. Include empty gaps, zero-power live rows and singleton tails.
    p, f, b = (jnp.repeat(x, 5) for x in inputs(13))
    p = p.at[6].multiply(2).at[16].set(0)
    b = b.at[7].multiply(2).at[20:35].set(0)
    args = (p, f, b)
    options = (
        {"group_interferers": True, "reuse_rows": True} if mode == "group_reuse" else {mode: True}
    )
    actual = jax.jit(
        lambda *x: xpm_pallas(
            *x,
            **PARAMS,
            compact=mode != "skip_empty_columns",
            **options,
            interpret=backend == "interpret",
        )
    )(*args)
    expected = xpm_reference(*args, **PARAMS)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires a CUDA GPU")
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("compact_rows", [False, True])
def test_cuda_kernel_matches_dense(compact, compact_rows):
    args = inputs(1376)
    actual = jax.jit(
        lambda *x: xpm_pallas(*x, **PARAMS, compact=compact, compact_rows=compact_rows)
    )(*args)
    expected = xpm_reference(*args, **PARAMS)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-4)
    expected64 = numpy_xpm64(*args, **PARAMS)
    np.testing.assert_allclose(actual, expected64, rtol=5e-6, atol=1e-5)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires a CUDA GPU")
@pytest.mark.parametrize("n", [7, 140, 1376])
def test_mosaic_kernel_matches_independent_float64_reference(n):
    from xlron.kernels.gn_xpm_mosaic import xpm_mosaic

    p, f, b = inputs(n)
    physics = {**PARAMS, "beta4": -2e-55}
    f_scaled, scaled = scaled_xpm_inputs(f, physics)
    actual = jax.jit(lambda *x: xpm_mosaic(*x, **scaled))(p, f_scaled, b)
    np.testing.assert_allclose(actual, numpy_xpm64(p, f, b, **physics), rtol=5e-6, atol=1e-5)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires a CUDA GPU")
def test_scaled_triton_matches_independent_float64_reference():
    p, f, b = inputs(1376)
    physics = {**PARAMS, "beta4": -2e-55}
    f_scaled, scaled = scaled_xpm_inputs(f, physics)
    actual = jax.jit(lambda *x: xpm_pallas(*x, **scaled, compact=True))(p, f_scaled, b)
    np.testing.assert_allclose(actual, numpy_xpm64(p, f, b, **physics), rtol=5e-6, atol=1e-5)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires a CUDA GPU")
@pytest.mark.parametrize("coherent", [False, True])
def test_cuda_full_snr_matches_dense_including_empty_links(coherent):
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
    for backend in [
        "jax_compact",
        "pallas_compact",
        "pallas_sparse",
        "pallas_reuse",
        "pallas_skip",
        "pallas_inline",
        "pallas_grouped",
        "mosaic",
    ]:
        actual = jax.jit(
            jax.vmap(lambda *x, backend=backend: get_snr_fused(*x, **physics, xpm_backend=backend))
        )(*args)
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-5)
        np.testing.assert_array_equal(np.asarray(actual)[0], -1e5)
        occupied = np.asarray(expected) > 0
        db_difference = 10 * np.log10(np.asarray(actual)[occupied] / np.asarray(expected)[occupied])
        assert np.max(np.abs(db_difference)) < 1e-4
