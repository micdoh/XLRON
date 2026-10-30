"""Small GN environment and transformer configuration shared by tests."""

import jax
import jax.numpy as jnp
from absl import flags

import xlron.parameter_flags  # noqa: F401 -- register shared environment flags
from xlron.environments import env_funcs
from xlron.environments.make_env import make, process_config


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


def make_gn_env(env_type="rsa_gn_model", **overrides):
    """A 5-node GN environment from the flag defaults with overrides: (env, params, state)."""
    config = {name: flags.FLAGS[name].default for name in flags.FLAGS}
    config.update(
        env_type=env_type,
        topology_name="5node_directed",
        k=2,
        launch_power_type="fixed",
        incremental_loading=True,
        TOTAL_TIMESTEPS=4,
        STEPS_PER_INCREMENT=4,
        ROLLOUT_LENGTH=2,
        NUM_ENVS=1,
        ENV_WARMUP_STEPS=0,
    )
    config.update(overrides)
    env, params = make(process_config(config), log_wrapper=False)
    _, state = env.reset(jax.random.PRNGKey(3), params)
    return env, params, state


def place_channel(env, params, state, slot, datarate):
    """Implement an RSA-GN lightpath for the state's node pair on its first path at slot.

    Goes through the environment's own action processing and writer, so the
    channel arrays follow the environment's convention. Returns (state, info).
    """
    state = state.replace(request_array=state.request_array.at[1].set(datarate))
    info = env.process_action(state, jnp.array([slot, 0.0]), params)
    return env_funcs.implement_action_rsa_gn_model(state, info, params), info
