"""Generic step-model XPM must retain its distinct existing broadcast convention."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments.gn_model import isrs_gn_model as gn
from xlron.kernels import gn_xpm_legacy
from xlron.kernels.gn_xpm_numerics import scaled_xpm_inputs
from tests.helpers.gn_xpm import PARAMS, inputs


def reference(power, frequency, bandwidth, physics):
    f = frequency
    phi = (
        2
        * jnp.pi**2
        * (f[:, None] - f[None, :])
        * (
            physics["beta2"]
            + jnp.pi * physics["beta3"] * (f[:, None] + f[None, :])
            + 2
            / 3
            * jnp.pi**2
            * physics["beta4"]
            * (f[:, None] ** 2 + f[:, None] * f[None, :] + f[None, :] ** 2)
        )
    )
    tilt = (physics["a"] + physics["a_bar"] - f * jnp.sum(power) * physics["cr"]) ** 2
    return gn._xpm(
        power,
        power.T,
        phi,
        tilt[None, :],
        bandwidth,
        bandwidth.T,
        physics["a"],
        physics["a_bar"],
        physics["gamma"],
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("n", [1, 17, 140])
def test_selected_outputs_retain_every_interferer(native, n):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    p, f, b = (jnp.repeat(x, 3) for x in inputs(n))
    if n > 1:
        p = p.at[1].set(0).at[3].set(1e-4)
        b = b.at[3].set(0)
    indices = jnp.array([len(p) - 1, 0, len(p) - 1, len(p) // 2])
    expected = jax.jit(lambda p, f, b: reference(p, f, b, PARAMS)[indices])(p, f, b)
    actual = jax.jit(
        lambda p, f, b: gn_xpm_legacy.xpm_selected_pallas(
            p,
            f,
            b,
            indices,
            **PARAMS,
            interpret=not native,
        )
    )(p, f, b)
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-5)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("programs", [1, 4, 16])
def test_persistent_row_scheduling_preserves_every_output(native, programs):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    p, f, b = (jnp.repeat(x, 3) for x in inputs(140))
    p = p.at[4:8].set(0)
    b = b.at[17].set(0)

    def run(p, f, b, persistent):
        return gn_xpm_legacy.xpm_legacy_pallas(
            p,
            f,
            b,
            **PARAMS,
            group_interferers=True,
            persistent_rows=persistent,
            interpret=not native,
        )

    expected = jax.jit(lambda p, f, b: run(p, f, b, None))(p, f, b)
    actual = jax.jit(lambda p, f, b: run(p, f, b, programs))(p, f, b)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reuse_rows", [False, True])
@pytest.mark.parametrize("group_interferers", [False, True])
@pytest.mark.parametrize("n", [1, 17, 140])
def test_generic_coefficient_matches_current_broadcasting(native, reuse_rows, group_interferers, n):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    p, f, b = (jnp.repeat(x, 3) for x in inputs(n))
    if n > 1:
        p = p.at[1].set(0).at[3].set(1e-4)
        b = b.at[3].set(0)  # Retain this power in the Raman sum.
    expected = jax.jit(lambda p, f, b: reference(p, f, b, PARAMS))(p, f, b)
    actual = jax.jit(
        lambda p, f, b: gn_xpm_legacy.xpm_legacy_pallas(
            p,
            f,
            b,
            **PARAMS,
            reuse_rows=reuse_rows,
            group_interferers=group_interferers,
            interpret=not native,
        )
    )(p, f, b)
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-5)
    if n > 1:
        assert np.any(np.asarray(expected)[np.asarray(b) == 0] != 0)


@pytest.mark.parametrize("native", [False, True])
def test_generic_scaled_coefficient_matches_independent_float64(native):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    p, f, b = inputs(33)
    params = {**PARAMS, "beta4": -2e-55}
    power, frequency, bandwidth = (np.asarray(x, np.float64) for x in (p, f, b))
    expected = np.zeros(len(power))
    a, ab, gamma, cr = (params[key] for key in ("a", "a_bar", "gamma", "cr"))
    for i, fi in enumerate(frequency):
        for k, fk in enumerate(frequency):
            phi = (
                2
                * np.pi**2
                * (fi - fk)
                * (
                    params["beta2"]
                    + np.pi * params["beta3"] * (fi + fk)
                    + 2 / 3 * np.pi**2 * params["beta4"] * (fi**2 + fi * fk + fk**2)
                )
            )
            denom = bandwidth[k] * phi * ab * (2 * a + ab)
            if denom:
                tilt = (a + ab - fk * np.sum(power) * cr) ** 2
                expected[i] += (
                    32
                    / 27
                    * (power[k] / max(power[k], 1e-12)) ** 2
                    * gamma**2
                    / denom
                    * (
                        (tilt - a**2) / a * np.arctan(phi * bandwidth[k] / a)
                        + ((a + ab) ** 2 - tilt)
                        / (a + ab)
                        * np.arctan(phi * bandwidth[k] / (a + ab))
                    )
                )
    scaled_f, scaled = scaled_xpm_inputs(f, params)
    actual = jax.jit(
        lambda p, f, b: gn_xpm_legacy.xpm_legacy_pallas(p, f, b, **scaled, interpret=not native)
    )(p, scaled_f, b)
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-5)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("coherent", [False, True])
@pytest.mark.parametrize("subchannels", [1, 3])
@pytest.mark.parametrize("backend", ["pallas", "pallas_grouped"])
def test_generic_full_snr_and_diagnostics_match(
    native, coherent, subchannels, backend, monkeypatch
):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    if not native:
        monkeypatch.setattr(
            gn_xpm_legacy,
            "xpm_legacy_pallas",
            partial(gn_xpm_legacy.xpm_legacy_pallas, interpret=True),
        )
    p, f, b = inputs(140)
    p = p.at[49:56].set(1e-4)
    f = f.at[49:56].set(f[49])
    b = b.at[49:56].set(50e9)
    config = {
        "num_channels": 140,
        "num_spans": 3,
        "length": jnp.array([80000.0, 80000.0, 80000.0, 0.0]),
        "uniform_spans": True,
        "attenuation_i": jnp.asarray(PARAMS["a"]),
        "attenuation_bar_i": jnp.asarray(PARAMS["a_bar"]),
        "nonlinear_coeff": jnp.asarray(PARAMS["gamma"]),
        "raman_gain_slope_i": jnp.asarray(PARAMS["cr"]),
        "dispersion_coeff": jnp.asarray(17e-6),
        "dispersion_slope": jnp.asarray(0.067e3),
        "coherent": coherent,
        "num_subchannels": subchannels,
        "span_lumped_loss_db": 1.7,
        "amplifier_noise_figure": jnp.linspace(4, 7, 140),
        "transceiver_snr": jnp.full(140, 30.0),
    }
    expected = jax.jit(
        lambda p, f, b: gn.get_snr(ch_power_w_i=p, ch_centre_i=f, ch_bandwidth_i=b, **config)
    )(p, f, b)
    actual = jax.jit(
        lambda p, f, b: gn.get_snr(
            ch_power_w_i=p, ch_centre_i=f, ch_bandwidth_i=b, xpm_backend=backend, **config
        )
    )(p, f, b)
    for result, control in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(result, control, rtol=5e-6, atol=1e-5)
