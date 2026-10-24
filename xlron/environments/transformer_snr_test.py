"""GN transformer inputs use current, finite per-channel SNR features."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl import flags

import xlron.parameter_flags  # noqa: F401
from xlron.environments.env_funcs import (
    generate_request_rsa,
    get_obs_transformer,
    get_snr_link_array,
)
from xlron.environments.make_env import make, process_config
from xlron.train.train_utils import init_network


def setup(env_type, mode):
    config = {name: flags.FLAGS[name].default for name in flags.FLAGS}
    config.update(
        env_type=env_type,
        topology_name="5node_directed",
        k=2,
        link_resources=8,
        USE_TRANSFORMER=True,
        transformer_obs_type=mode,
        transformer_embedding_size=8,
        transformer_intermediate_size=16,
        transformer_num_heads=2,
        num_wire_features=2,
        launch_power_type="fixed",
        power_per_channel=-10,
        incremental_loading=True,
        values_bw=[100],
        slot_size=25,
        guardband=0,
        TOTAL_TIMESTEPS=4,
        STEPS_PER_INCREMENT=4,
        ROLLOUT_LENGTH=2,
        NUM_ENVS=1,
        ENV_WARMUP_STEPS=0,
    )
    config = process_config(config)
    env, params = make(config, log_wrapper=False)
    _, state = env.reset(jax.random.PRNGKey(3), params)
    config.ACTION_DIM = env.num_actions(params)
    return config, env, params, state


@pytest.mark.parametrize("env_type", ["rsa_gn_model", "rmsa_gn_model"])
@pytest.mark.parametrize("mode", ["snr", "occupancy_snr", "departure_snr"])
def test_snr_feature_values_shapes_and_policy_gradients(env_type, mode):
    config, _, params, state = setup(env_type, mode)
    snr = state.link_snr_array.at[0, :5].set(jnp.array([10.0, 100.0, jnp.nan, jnp.inf, -1.0]))
    power = state.channel_power_array.at[0, :5].set(1e-4)
    # Empty slots must stay zero even when their cache contains a positive value.
    snr = snr.at[0, 5].set(100.0)
    state = state.replace(link_snr_array=snr, channel_power_array=power)
    tokens = get_obs_transformer(state, params)
    start = config.num_wire_features + (params.link_resources if mode != "snr" else 0)
    np.testing.assert_allclose(tokens[0, start : start + 6], [1 / 3, 2 / 3, 0, 1, 0, 0], atol=1e-6)
    request_cols = 6 if mode == "departure_snr" else 5
    expected_width = (
        config.num_wire_features
        + params.link_resources * (1 if mode == "snr" else 2)
        + 2
        + request_cols
    )
    assert tokens.shape == (params.num_links, expected_width)
    assert np.all(np.isfinite(tokens))
    model = init_network(config, jax.random.PRNGKey(4))

    @eqx.filter_value_and_grad
    def loss(network):
        pi, value = network(state, params)
        assert pi.logits.shape == (config.ACTION_DIM,)
        return jnp.mean(pi.logits**2) + jnp.mean(value**2)

    value, gradient = loss(model)
    assert np.isfinite(value)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(gradient))


@pytest.mark.parametrize("env_type", ["rsa_gn_model", "rmsa_gn_model"])
def test_snr_cache_is_refreshed_after_departures(env_type):
    _, env, params, state = setup(env_type, "occupancy_snr")
    mask_output = env.action_mask(state, params)
    state = state.replace(link_slot_mask=mask_output[0], full_link_slot_mask=mask_output[1])
    if env_type == "rmsa_gn_model":
        state = state.replace(mod_format_mask=mask_output[2])
        action = jnp.array([jnp.argmax(mask_output[0]), 0])
    else:
        action = jnp.argmax(mask_output[0])
    _, state, *_ = env.step(jax.random.PRNGKey(5), state, action, params)
    assert np.any(np.asarray(state.channel_power_array) > 0)
    # Force expiration and poison the old cache: the next request must rebuild it.
    state = state.replace(
        link_slot_departure_array=jnp.zeros_like(state.link_slot_departure_array),
        link_snr_array=jnp.full_like(state.link_snr_array, 123),
    )
    params = params.replace(incremental_loading=False)
    next_state = generate_request_rsa(jax.random.PRNGKey(6), state, params)
    assert not np.any(np.asarray(next_state.channel_power_array) > 0)
    np.testing.assert_allclose(next_state.link_snr_array, get_snr_link_array(next_state, params))


def test_snr_observations_reject_non_gn_environments():
    with pytest.raises(AssertionError, match="requires a GN environment"):
        make({"env_type": "rsa", "transformer_obs_type": "snr"})
