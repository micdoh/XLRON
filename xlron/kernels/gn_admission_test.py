"""Native kernels retain full admission checks on an occupied small network."""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments import env_funcs
from xlron.environments.dataclasses import HashableArrayWrapper, RMSAGNModelEnvParams
from xlron.environments.gn_model import isrs_gn_model
from xlron.environments.make_env import make


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires CUDA")
@pytest.mark.parametrize("backend", ["pallas_inline", "pallas_reuse", "pallas_grouped"])
def test_native_mask_matches_jax_with_snr_margin_power_and_existing_path_checks(
    backend, monkeypatch
):
    env, params = make(
        {
            "env_type": "rmsa_gn_model",
            "topology_name": "5node_directed",
            "k": 2,
            "link_resources": 24,
            "slot_size": 25,
            "guardband": 0,
            "values_bw": [100],
            "incremental_loading": True,
            "max_requests": 100,
            "include_no_op": False,
            "power_per_channel": -10,
        },
        log_wrapper=False,
    )
    params = cast(RMSAGNModelEnvParams, params)
    _, state = env.reset(jax.random.PRNGKey(3), params)
    for step in range(4):
        mask, _, mods = env.action_mask(state, params)
        assert np.any(np.asarray(mask))
        state = state.replace(link_slot_mask=mask, mod_format_mask=mods)
        _, state, *_ = env.step(
            jax.random.PRNGKey(step), state, jnp.array([jnp.argmax(mask), 0]), params
        )
    assert np.any(np.asarray(state.channel_power_array) > 0)
    mods = params.modulations_array.val
    rejecting_thresholds = mods.at[:, 2].set(1000)
    existing_mod = int(
        np.asarray(state.modulation_format_index_array)[np.asarray(state.path_index_array) >= 0][0]
    )
    existing_thresholds = mods.at[:, 2].set(-1000).at[existing_mod, 2].set(1000)
    variants = [
        params,
        params.replace(modulations_array=HashableArrayWrapper(rejecting_thresholds)),
        params.replace(snr_margin=1000),
        params.replace(max_power_per_fibre=0.0),
        params.replace(modulations_array=HashableArrayWrapper(existing_thresholds)),
    ]
    # Evaluate the control before installing any custom-call override.
    expected = [tuple(np.asarray(x) for x in env.action_mask(state, p)) for p in variants]
    assert np.any(expected[0][0])
    for index in (1, 2, 3):
        assert not np.any(expected[index][0])
    assert bool(env_funcs.check_snr_sufficient(state, variants[-1]))

    monkeypatch.setattr(
        isrs_gn_model, "get_snr_fused", partial(isrs_gn_model.get_snr_fused, xpm_backend=backend)
    )
    monkeypatch.setattr(
        env_funcs,
        "get_lightpath_snr",
        partial(env_funcs.get_lightpath_snr, nsr_backend="pallas_precomputed"),
    )
    # Jitted callers otherwise retain the ordinary-JAX implementations traced above.
    jax.clear_caches()
    try:
        executable = jax.jit(lambda s: env.action_mask(s, variants[0])).lower(state).compile()
        hlo = executable.as_text()
        assert "gn_xpm_tiled" in hlo
        assert "gn_path_snr_fused" in hlo
        for p, control in zip(variants, expected, strict=True):
            actual = env.action_mask(state, p)
            for result, reference in zip(actual, control, strict=True):
                np.testing.assert_array_equal(result, reference)
    finally:
        # Leave no compiled functions that capture monkeypatched globals.
        jax.clear_caches()
