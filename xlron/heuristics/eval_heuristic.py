import chex
import orbax.checkpoint
import pathlib
import optax
from flax.training import orbax_utils
from typing import NamedTuple, Callable, Dict
from absl import flags
from flax import struct
from flax.training.train_state import TrainState
from gymnax.environments import environment
from xlron.environments.env_funcs import *
from xlron.train.train_utils import *
from xlron.environments.vone import *
from xlron.environments.rsa import *
from xlron.heuristics.heuristics import *


def get_eval_fn(
    env: environment.Environment,
    env_params: EnvParams,
    eval_state: TrainState,
    config: flags.FlagValues,
) -> Callable:


    # COLLECT TRAJECTORIES
    def _env_episode(runner_state, unused):

        def _env_step(runner_state, unused):
            eval_state, env_state, last_obs, rng_step, rng_epoch = runner_state

            rng_step, action_key, step_key = jax.random.split(rng_step, 3)

            # SELECT ACTION
            action_key = jax.random.split(action_key, config.NUM_ENVS)
            select_action_fn = lambda x: select_action_eval(x, env, env_params, eval_state, config)
            select_action_fn = jax.vmap(select_action_fn)
            select_action_state = (action_key, env_state, last_obs)
            env_state, action, _, _ = select_action_fn(select_action_state)

            # STEP ENV
            step_key = jax.random.split(step_key, config.NUM_ENVS)
            step_fn = lambda x, y, z: env.step(x, y, z, env_params)
            step_fn = jax.vmap(step_fn)
            obsv, env_state, reward, done, info = step_fn(step_key, env_state, action)

            obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
            transition = Transition(
                done, action, reward, last_obs, info
            )
            runner_state = (eval_state, env_state, obsv, rng_step, rng_epoch)

            if config.DEBUG:
                jax.debug.print("request {}", env_state.env_state.request_array, ordered=config.ORDERED)
                jax.debug.print("link_slot_array {}", env_state.env_state.link_slot_array, ordered=config.ORDERED)
                if env_params.__class__.__name__ == "RSAGNModelEnvParams":
                    jax.debug.print("link_snr_array {}", env_state.env_state.link_snr_array, ordered=config.ORDERED)
                    jax.debug.print("channel_power_array {}", env_state.env_state.channel_power_array, ordered=config.ORDERED)
                    jax.debug.print("modulation_format_index_array {}", env_state.env_state.modulation_format_index_array, ordered=config.ORDERED)
                    jax.debug.print("channel_centre_bw_array {}", env_state.env_state.channel_centre_bw_array, ordered=config.ORDERED)
                jax.debug.print("link_slot_mask {}", env_state.env_state.link_slot_mask, ordered=config.ORDERED)
                jax.debug.print("action {}", action, ordered=config.ORDERED)
                jax.debug.print("reward {}", reward, ordered=config.ORDERED)

            return runner_state, transition

        runner_state, traj_episode = jax.lax.scan(
            _env_step, runner_state, None, config.max_requests
        )

        metric = traj_episode.info

        return runner_state, metric

    def eval_fn(runner_state):

        NUM_EPISODES = config.TOTAL_TIMESTEPS // config.max_requests // config.NUM_ENVS
        runner_state, metric = jax.lax.scan(
            _env_episode, runner_state, None, NUM_EPISODES
        )
        return {"runner_state": runner_state, "metrics": metric}

    return eval_fn
