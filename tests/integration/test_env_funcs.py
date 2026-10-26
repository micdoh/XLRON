"""gn_kernel_backend=pallas keeps every admission check on an occupied small network.

Kernels run natively on a GPU and in the Pallas interpreter elsewhere.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments import env_funcs
from xlron.environments.dataclasses import HashableArrayWrapper, RMSAGNModelEnvParams
from xlron.environments.make_env import make

ON_GPU = jax.default_backend() == "gpu"


def _make(slots, mode, values_bw=(100,), incremental=True):
    env, params = make(
        {
            "env_type": "rmsa_gn_model",
            "topology_name": "5node_directed",
            "k": 2,
            "link_resources": slots,
            "slot_size": 25,
            "guardband": 0,
            "values_bw": list(values_bw),
            "incremental_loading": incremental,
            "max_requests": 100,
            "include_no_op": False,
            "power_per_channel": -10,
            "gn_mask_mode": mode,
        },
        log_wrapper=False,
    )
    return env, cast(RMSAGNModelEnvParams, params)


@pytest.mark.parametrize("slots,mode", [(24, "ff_lf"), (24, "all_slots"), (140, "all_slots")])
def test_pallas_mask_matches_jax_with_snr_margin_power_and_existing_path_checks(slots, mode):
    env, params = _make(slots, mode)
    _, state = env.reset(jax.random.PRNGKey(3), params)
    for step in range(4):
        mask, _, mods = env.action_mask(state, params)
        assert np.any(np.asarray(mask))
        state = state.replace(link_slot_mask=mask, mod_format_mask=mods)
        _, state, *_ = env.step(jax.random.PRNGKey(step), state, jnp.argmax(mask), params)
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
    expected = [tuple(np.asarray(x) for x in env.action_mask(state, p)) for p in variants]
    assert np.any(expected[0][0])
    for index in (1, 2, 3):
        assert not np.any(expected[index][0])
    assert bool(env_funcs.check_snr_sufficient(state, variants[-1]))

    kernel = params.replace(gn_kernel_backend="pallas")
    if ON_GPU:
        hlo = jax.jit(lambda s: env.action_mask(s, kernel)).lower(state).compile().as_text()
        assert "gn_xpm" in hlo
        assert ("gn_path_snr_evidence" if mode == "all_slots" else "gn_path_snr") in hlo
    for p, control in zip(variants, expected, strict=True):
        actual = env.action_mask(state, p.replace(gn_kernel_backend="pallas"))
        for result, reference in zip(actual, control, strict=True):
            np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("slots", [24, 140])
def test_pallas_step_preserves_full_state_and_admissions(slots):
    env, params = _make(slots, "ff_lf", values_bw=(100, 200, 400), incremental=False)
    _, initial = env.reset(jax.random.PRNGKey(17), params)
    kernel = params.replace(gn_kernel_backend="pallas")

    def run(p):
        state, outputs, actions = initial, [], []
        for step in range(6):
            mask, _, mods = env.action_mask(state, p)
            # Reuse a previous slot choice for a subsequent request as well.
            action = actions[-1] if step == 3 else jnp.argmax(mask)
            actions.append(action)
            output = env.step(
                jax.random.PRNGKey(step),
                state.replace(link_slot_mask=mask, mod_format_mask=mods),
                action,
                p,
            )
            outputs.append(jax.tree.map(np.asarray, output))
            state = output[1]
        return outputs

    if ON_GPU:
        hlo = (
            jax.jit(lambda s: env.step(jax.random.PRNGKey(0), s, jnp.int32(0), kernel))
            .lower(initial)
            .compile()
            .as_text()
        )
        assert "gn_xpm" in hlo
    for actual, expected in zip(run(kernel), run(params), strict=True):
        for result, control in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(result, control, rtol=5e-6, atol=2e-5)
