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
            eval_state, env_state, last_obs, step_key, rng_epoch = runner_state

            action_key, next_step_key = jax.random.split(step_key)

            # SELECT ACTION
            select_action_state = (action_key, env_state, last_obs)
            env_state, action, _, _ = select_action_eval(select_action_state, env, env_params, eval_state, config)

            # STEP ENV
            obsv, env_state, reward, done, info = env.step(step_key, env_state, action, env_params)

            obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
            transition = Transition(
                done, action, reward, last_obs, info
            )
            runner_state = (eval_state, env_state, obsv, next_step_key, rng_epoch)

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

        # VECTORISE ENV STEP
        _env_step_vmap = jax.vmap(
            _env_step,
            in_axes=((None, 0, 0, 0, None), None),
            out_axes=((None, 0, 0, 0, None), 0)
        ) if config.NUM_ENVS > 1 else _env_step

        rng_step = runner_state[3]
        rng_step, *step_keys = jax.random.split(rng_step, config.NUM_ENVS + 1)
        step_keys = jnp.array(step_keys) if config.NUM_ENVS > 1 else step_keys[0]
        runner_state = runner_state[:3] + (step_keys,) + runner_state[4:]
        runner_state, traj_episode = jax.lax.scan(
            _env_step_vmap, runner_state, None, config.max_requests
        )
        runner_state = runner_state[:3] + (rng_step,) + runner_state[4:]

        metric = traj_episode.info

        return runner_state, metric

    def eval_fn(runner_state):

        NUM_EPISODES = config.STEPS_PER_INCREMENT // config.max_requests // config.NUM_ENVS
        runner_state, metric = jax.lax.scan(
            _env_episode, runner_state, None, NUM_EPISODES
        )
        return {"runner_state": runner_state, "metrics": metric}

    return eval_fn
