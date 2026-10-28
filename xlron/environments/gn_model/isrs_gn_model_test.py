"""Unit tests for isrs_gn_model.py helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments.gn_model import isrs_gn_model as gn


class TestAseWithIsrsGainAt:
    @staticmethod
    @pytest.mark.parametrize("lumped", [None, 1.5])
    def test_matches_the_full_spectrum_ase_at_the_selected_channels(lumped):
        rng = np.random.default_rng(0)
        n = 97
        power = jnp.asarray(rng.uniform(0, 3e-4, n) * (rng.random(n) > 0.3), jnp.float32)
        centre = jnp.asarray(np.linspace(-6e12, 6e12, n), jnp.float32)
        bandwidth = jnp.full((n,), 12.5e9, jnp.float32)
        nf = jnp.asarray(rng.uniform(4, 6, n), jnp.float32)
        args = (4.6e-5, 80e3, 1577.5e-9)
        full = gn._calculate_ase_with_isrs_gain(
            nf, args[0], args[1], args[2], centre, bandwidth, power, 2.8e-17, lumped
        )
        at = jnp.array([3, 40, 41, 96])
        selected = gn._ase_with_isrs_gain_at(
            nf[at], args[0], args[1], args[2], centre, power, 2.8e-17, lumped,
            centre[at], bandwidth[at], power[at],
        )  # fmt: skip
        np.testing.assert_allclose(selected, full[at], rtol=1e-6)
