"""Accuracy checks for gn_xpm_mosaic.py."""

import jax
import numpy as np
import pytest
from experimental.benchmarks.gn_precision_audit import numpy_xpm64
from tests.helpers.gn_xpm import PARAMS, inputs
from xlron.kernels.gn_xpm_numerics import scaled_xpm_inputs


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires a CUDA GPU")
@pytest.mark.parametrize("n", [7, 140, 1376])
def test_mosaic_kernel_matches_independent_float64_reference(n):
    from xlron.kernels.gn_xpm_mosaic import xpm_mosaic

    p, f, b = inputs(n)
    physics = {**PARAMS, "beta4": -2e-55}
    f_scaled, scaled = scaled_xpm_inputs(f, physics)
    actual = jax.jit(lambda *x: xpm_mosaic(*x, **scaled))(p, f_scaled, b)
    np.testing.assert_allclose(actual, numpy_xpm64(p, f, b, **physics), rtol=5e-6, atol=1e-5)
