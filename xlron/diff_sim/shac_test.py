"""Tests for the SHAC analytic-policy-gradient learner (xlron/diff_sim/shac.py).

The critical property under test: the actor receives a NONZERO analytic gradient
through the differentiable environment when there is blocking pressure, with
ENT_COEF=0 so no gradient reaches the actor through the entropy bonus -- the only
path is reward -> soft collision check -> soft slot mask -> straight-through
soft action -> policy logits. If a refactor of the env soft ops or the
straight-through action ever severs that path, test_actor_gradient_nonzero fails.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.diff_sim.shac import get_shac_learner_fn
from xlron.environments.make_env import process_config
from xlron.train.train_utils import experiment_data_setup


def _make_config(**overrides):
    base = dict(
        env_type="rsa",
        topology_name="5node_directed",
        link_resources=10,
        k=2,
        values_bw="1",
        slot_size=1,
        guardband=0,
        load=150.0,
        mean_service_holding_time=25.0,
        continuous_operation=True,
        ENV_WARMUP_STEPS=50,
        warmup_action_type="heuristic",
        path_heuristic="ksp_ff",
        differentiable=True,
        temperature=5.0,
        SHAC=True,
        ROLLOUT_LENGTH=16,
        NUM_ENVS=2,
        TOTAL_TIMESTEPS=64,
        LR=1e-3,
        GAMMA=0.99,
        GAE_LAMBDA=0.95,
        VF_COEF=0.5,
        ENT_COEF=0.0,
        NUM_LAYERS=2,
        NUM_UNITS=32,
        SEED=0,
    )
    base.update(overrides)
    return process_config(base)


def _setup(config):
    rng = jax.random.PRNGKey(config.SEED)
    runner_state, env, env_params = experiment_data_setup(config, rng)
    learner_fn = get_shac_learner_fn(env, env_params, runner_state, config)
    return runner_state, env, env_params, learner_fn


@pytest.mark.slow
def test_shac_learner_runs_and_losses_finite():
    config = _make_config()
    runner_state, env, env_params, learner_fn = _setup(config)
    out = jax.jit(learner_fn)(runner_state)

    loss_info = out["loss_info"]
    for key, val in loss_info.items():
        assert bool(jnp.all(jnp.isfinite(val))), f"{key} not finite: {val}"
    # Env advanced: per-update metrics have shape (NUM_UPDATES, H, N)
    lengths = out["metrics"]["lengths"]
    assert lengths.shape == (config.NUM_UPDATES, config.ROLLOUT_LENGTH, config.NUM_ENVS)
    # Continuous operation: step counter strictly increases across the rollout
    assert bool(jnp.all(lengths[:, -1, :] > lengths[:, 0, :]))


@pytest.mark.slow
def test_actor_gradient_nonzero_under_blocking():
    """With ENT_COEF=0, actor gradients exist ONLY via the analytic reward path.

    Load is high enough that blocks occur inside the horizon, so the soft
    collision check must transmit d(reward)/d(logits) != 0.
    """
    config = _make_config(load=300.0, ENV_WARMUP_STEPS=200)
    runner_state, env, env_params, learner_fn = _setup(config)
    out = jax.jit(learner_fn)(runner_state)

    # Blocking actually occurred during the run (otherwise the test is vacuous)
    returns = out["metrics"]["returns"]
    assert float(jnp.sum(returns < 0)) > 0, "no blocking events; raise load"

    grad_norm = np.asarray(out["loss_info"]["loss/grad_norm"])
    assert np.all(np.isfinite(grad_norm))
    assert np.any(grad_norm > 1e-8), f"gradient identically zero: {grad_norm}"

    # Parameters actually moved
    p0 = jax.tree_util.tree_leaves(runner_state[0].model_params)
    p1 = jax.tree_util.tree_leaves(out["runner_state"][0].model_params)
    moved = any(bool(jnp.any(a != b)) for a, b in zip(p0, p1) if a is not None)
    assert moved, "model parameters did not change after updates"


@pytest.mark.slow
def test_shac_mode_forward_runs():
    config = _make_config(SHAC_FORWARD="mode", TOTAL_TIMESTEPS=32)
    runner_state, env, env_params, learner_fn = _setup(config)
    out = jax.jit(learner_fn)(runner_state)
    assert bool(jnp.all(jnp.isfinite(out["loss_info"]["loss/total_loss"])))


@pytest.mark.slow
def test_shac_flat_surrogate_runs():
    config = _make_config(SHAC_ACTION_SURROGATE="flat", TOTAL_TIMESTEPS=32)
    runner_state, env, env_params, learner_fn = _setup(config)
    out = jax.jit(learner_fn)(runner_state)
    assert bool(jnp.all(jnp.isfinite(out["loss_info"]["loss/total_loss"])))
    assert bool(jnp.all(jnp.isfinite(out["loss_info"]["loss/grad_norm"])))


@pytest.mark.slow
def test_shac_no_bootstrap_runs():
    config = _make_config(SHAC_VALUE_BOOTSTRAP=False, TOTAL_TIMESTEPS=32)
    runner_state, env, env_params, learner_fn = _setup(config)
    out = jax.jit(learner_fn)(runner_state)
    assert bool(jnp.all(jnp.isfinite(out["loss_info"]["loss/total_loss"])))
    chex.assert_trees_all_close(
        out["loss_info"]["loss/terminal_value_mean"],
        jnp.zeros_like(out["loss_info"]["loss/terminal_value_mean"]),
    )


def test_shac_requires_differentiable():
    config = _make_config(differentiable=False)
    rng = jax.random.PRNGKey(0)
    runner_state, env, env_params = experiment_data_setup(config, rng)
    with pytest.raises(ValueError, match="differentiable"):
        get_shac_learner_fn(env, env_params, runner_state, config)
