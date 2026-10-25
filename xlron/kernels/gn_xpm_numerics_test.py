"""Accuracy checks for gn_xpm_numerics.py."""

import jax
import numpy as np
import pytest
from experimental.benchmarks.gn_precision_audit import numpy_xpm64
from tests.helpers.gn_xpm import PARAMS, inputs
from xlron.kernels.gn_xpm_numerics import scaled_xpm_inputs
from xlron.kernels.gn_xpm import xpm_reference, xpm_compact_jax, xpm_pallas


@pytest.mark.parametrize("n", [33, 257])
def test_scaled_float32_matches_independent_float64_reference(n):
    p, f, b = inputs(n)
    physics = {**PARAMS, "beta4": -2e-55}
    expected = numpy_xpm64(p, f, b, **physics)
    f_scaled, scaled = scaled_xpm_inputs(f, physics)
    for fn in [xpm_reference, xpm_compact_jax]:
        actual = jax.jit(lambda *x, fn=fn: fn(*x, **scaled))(p, f_scaled, b)
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=1e-5)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires a CUDA GPU")
def test_scaled_triton_matches_independent_float64_reference():
    p, f, b = inputs(1376)
    physics = {**PARAMS, "beta4": -2e-55}
    f_scaled, scaled = scaled_xpm_inputs(f, physics)
    actual = jax.jit(lambda *x: xpm_pallas(*x, **scaled, compact=True))(p, f_scaled, b)
    np.testing.assert_allclose(actual, numpy_xpm64(p, f, b, **physics), rtol=5e-6, atol=1e-5)
