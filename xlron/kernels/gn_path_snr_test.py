"""Every stored hop/slot, empty paths, padding and full ROADM/TRX SNR parity."""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments.dataclasses import GNModelEnvParams
from xlron.environments.env_funcs import get_lightpath_snr
from xlron.environments.make_env import make
from xlron.kernels import gn_path_snr


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("slots", [1, 17, 140])
def test_path_nsr_matches_independent_hop_sum(native, slots):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(13)
    links, paths, hops = 8, 19, 6
    table = np.full((paths, hops), links, np.int32)
    for row in table[:-1]:
        count = rng.integers(1, hops + 1)
        row[:count] = np.sort(rng.choice(links, count, replace=False))
    snr = rng.uniform(1, 100, (2, links, slots)).astype(np.float32)
    index = rng.integers(-1, paths - 1, snr.shape, dtype=np.int32)
    # Zero on-path SNR must give infinity; padded zero-SNR links must be ignored.
    snr[0, -1] = 0
    expected = np.zeros_like(snr)
    with np.errstate(divide="ignore"):
        for batch, link, slot in np.ndindex(snr.shape):
            for hop in table[index[batch, link, slot]]:
                if hop < links:
                    expected[batch, link, slot] += 1 / snr[batch, hop, slot]
    actual = jax.jit(
        jax.vmap(
            lambda s, p: gn_path_snr.path_nsr_pallas(s, p, jnp.asarray(table), interpret=not native)
        )
    )(jnp.asarray(snr), jnp.asarray(index))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("backend", ["pallas", "pallas_fused", "pallas_precomputed"])
def test_full_path_snr_with_roadm_and_transceivers(native, backend, monkeypatch):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    if not native:
        monkeypatch.setattr(
            gn_path_snr, "path_nsr_pallas", partial(gn_path_snr.path_nsr_pallas, interpret=True)
        )
        monkeypatch.setattr(
            gn_path_snr, "path_snr_pallas", partial(gn_path_snr.path_snr_pallas, interpret=True)
        )
    env, params = make(
        {
            "env_type": "rmsa_gn_model",
            "topology_name": "5node_directed",
            "link_resources": 17,
            "k": 3,
            "values_bw": [100],
            "slot_size": 12.5,
            "guardband": 0,
        },
        log_wrapper=False,
    )
    _, state = env.reset(jax.random.PRNGKey(0), params)
    params = cast(GNModelEnvParams, params)
    rng = np.random.default_rng(7)
    shape = state.link_snr_array.shape
    state = state.replace(
        link_snr_array=jnp.asarray(rng.uniform(1, 100, shape).astype(np.float32)),
        path_index_array=jnp.asarray(
            rng.integers(-1, params.path_link_indices.val.shape[0] - 1, shape, dtype=np.int32)
        ),
        channel_power_array=jnp.full(shape, 1e-4),
        channel_centre_bw_array=jnp.full(shape, 12.5),
        channel_centre_freq_array=jnp.broadcast_to(jnp.linspace(-1000, 1000, shape[1]), shape),
    )
    expected = jax.jit(lambda s: get_lightpath_snr(s, params))(state)
    actual = jax.jit(lambda s: get_lightpath_snr(s, params, nsr_backend=backend))(state)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-5)
