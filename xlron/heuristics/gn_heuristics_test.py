"""Every advertised heuristic must run with GN masks and sentinel path rows."""

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from box import Box

from experimental.benchmarks.gn_cls_tuning_campaign import HEURISTICS
from xlron.environments.dataclasses import HashableArrayWrapper
from xlron.environments.make_env import make
from xlron.heuristics.heuristics import capacity_loss
from xlron.train.train_utils import select_action_eval


@pytest.fixture(scope="module")
def gn_setup():
    env, params = make(
        {
            "env_type": "rmsa_gn_model",
            "topology_name": "5node_directed",
            "k": 2,
            "link_resources": 24,
            "slot_size": 25,
            "guardband": 0,
            "values_bw": [100],
            "power_per_channel": -10,
            "max_requests": 100,
            "include_no_op": False,
        }
    )
    obs, state = env.reset(jax.random.PRNGKey(3), params)
    return env, params, obs, state


def test_inventory_matches_actual_dispatch():
    source = Path("xlron/train/train_utils.py").read_text()
    function = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "select_action_eval"
    )
    actual = {
        node.comparators[0].value
        for node in ast.walk(function)
        if isinstance(node, ast.Compare)
        and "path_heuristic" in ast.unparse(node.left)
        and isinstance(node.comparators[0], ast.Constant)
    }
    assert actual == set(HEURISTICS)


@pytest.mark.parametrize("heuristic", HEURISTICS)
def test_all_heuristics_select_and_step_with_gn(gn_setup, heuristic):
    env, params, obs, state = gn_setup
    config = Box(
        {
            "EVAL_HEURISTIC": True,
            "env_type": "rmsa_gn_model",
            "path_heuristic": heuristic,
            "launch_power_type": "fixed",
        }
    )
    # Use an occupied state as well as an empty one: the GN modulation mask and
    # the sentinel row must remain valid after an accepted placement.
    for i in range(2):
        state, action, _, _ = select_action_eval(
            (jax.random.PRNGKey(i), state, obs), env, params, None, config
        )
        assert action.shape == (2,)
        index = int(action[0])
        assert 0 <= index < params.k_paths * params.link_resources
        assert np.asarray(state.env_state.link_slot_mask)[index] > 0
        obs, state, *_ = env.step(jax.random.PRNGKey(i + 50), state, action, params)
    assert int(state.env_state.accepted_services) == 2


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_mscl_ignores_gn_sentinel_and_preserves_capacity(packed, k):
    from dataclasses import fields
    from unittest.mock import patch

    from xlron.environments.dataclasses import RSAEnvParams
    from xlron.heuristics.heuristics import get_action_mask

    env, params = make(
        {
            "env_type": "rmsa_gn_model",
            "topology_name": "5node_directed",
            "k": k,
            "link_resources": 24,
            "slot_size": 25,
            "guardband": 0,
            "values_bw": [100],
            "power_per_channel": -10,
            "pack_path_bits": packed,
        },
        log_wrapper=False,
    )
    _, state = env.reset(jax.random.PRNGKey(3), params)
    state = state.replace(
        link_slot_array=jnp.asarray(
            np.random.default_rng(7).integers(0, 2, state.link_slot_array.shape),
            dtype=state.link_slot_array.dtype,
        )
    )
    mask = get_action_mask(state, params)
    ordinary = RSAEnvParams(**{f.name: getattr(params, f.name) for f in fields(RSAEnvParams)})
    ordinary = ordinary.replace(
        path_link_array=HashableArrayWrapper(params.path_link_array.val[:-1])
    )
    with patch("xlron.heuristics.heuristics.get_action_mask", return_value=mask):
        expected, _ = capacity_loss(state, ordinary, num_interfering_routes=k)
        actual, _ = capacity_loss(state, params, num_interfering_routes=k)
    np.testing.assert_array_equal(actual, expected)
