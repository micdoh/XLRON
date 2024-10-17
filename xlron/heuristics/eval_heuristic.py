import orbax.checkpoint
import pathlib
import optax
from flax.training import orbax_utils
from typing import NamedTuple, Callable
from absl import flags
from flax import struct
from flax.training.train_state import TrainState
from xlron.environments.env_funcs import *
from xlron.train.train_utils import *
from xlron.environments.vone import *
from xlron.environments.rsa import *
from xlron.heuristics.heuristics import *


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@struct.dataclass
class EvalState:
    apply_fn: Callable
    params: chex.Array


def select_action_eval(config, env_state, env_params, env, eval_state, rng_key, last_obs):
    if config.EVAL_HEURISTIC:
        if config.env_type.lower() == "vone":
            raise NotImplementedError(f"VONE heuristics not yet implemented")

        elif config.env_type.lower() in ["rsa", "rwa", "rmsa", "deeprmsa", "rwa_lightpath_reuse", "rsa_gn_model"]:
            if config.path_heuristic.lower() == "ksp_ff":
                action = jax.vmap(ksp_ff, in_axes=(0, None))(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "ff_ksp":
                action = jax.vmap(ff_ksp, in_axes=(0, None))(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "kmc_ff":
                action = jax.vmap(kmc_ff, in_axes=(0, None))(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "kmf_ff":
                action = jax.vmap(kmf_ff, in_axes=(0, None))(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "ksp_mu":
                action = jax.vmap(ksp_mu, in_axes=(0, None, None, None))(env_state.env_state, env_params, False, True)
            elif config.path_heuristic.lower() == "ksp_mu_nonrel":
                action = jax.vmap(ksp_mu, in_axes=(0, None, None, None))(env_state.env_state, env_params, False, False)
            elif config.path_heuristic.lower() == "ksp_mu_unique":
                action = jax.vmap(ksp_mu, in_axes=(0, None, None, None))(env_state.env_state, env_params, True, True)
            elif config.path_heuristic.lower() == "mu_ksp":
                action = jax.vmap(mu_ksp, in_axes=(0, None, None, None))(env_state.env_state, env_params, False, True)
            elif config.path_heuristic.lower() == "mu_ksp_nonrel":
                action = jax.vmap(mu_ksp, in_axes=(0, None, None, None))(env_state.env_state, env_params, False, False)
            elif config.path_heuristic.lower() == "mu_ksp_unique":
                action = jax.vmap(mu_ksp, in_axes=(0, None, None, None))(env_state.env_state, env_params, True, True)
            elif config.path_heuristic.lower() == "kca_ff":
                action = jax.vmap(kca_ff, in_axes=(0, None))(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "kme_ff":
                action = jax.vmap(kme_ff, in_axes=(0, None))(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "ksp_bf":
                action = jax.vmap(ksp_bf, in_axes=(0, None))(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "bf_ksp":
                action = jax.vmap(bf_ksp, in_axes=(0, None))(env_state.env_state, env_params)
            else:
                raise ValueError(f"Invalid path heuristic {config.path_heuristic}")

        else:
            raise ValueError(f"Invalid environment type {config.env_type}")
    else:
        rng_key = jax.random.split(rng_key, config.NUM_ENVS)
        select_action_state = (rng_key, env_state, last_obs)
        action, _, _ = jax.vmap(select_action, in_axes=(0, None, None, None, None))(select_action_state, env, env_params, eval_state, config)
    return action


def get_warmup_fn(warmup_state, env, params, eval_state, config) -> Tuple[EnvState, chex.Array]:
    """Warmup period for DeepRMSA."""

    def warmup_fn(warmup_state):

        rng, state, last_obs = warmup_state

        def warmup_step(i, val):
            _rng, _state, _params, _last_obs = val
            # SELECT ACTION
            _rng, action_key, step_key = jax.random.split(_rng, 3)
            select_action_state = (_rng, _state, _last_obs)
            action = select_action_eval(config, _state, _params, env, eval_state, action_key, _last_obs)
            # STEP ENV
            rng_step = jax.random.split(step_key, config.NUM_ENVS)
            obsv, _state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                rng_step, _state, action, _params
            )
            obsv = (_state.env_state, params) if config.USE_GNN else tuple([obsv])
            return _rng, _state, _params, obsv

        vals = jax.lax.fori_loop(0, config.ENV_WARMUP_STEPS, warmup_step,
                                 (rng, state, params, last_obs))

        return vals[1], vals[3]

    return warmup_fn


def make_eval(config):

    # INIT ENV
    env, env_params = define_env(config)

    def evaluate(rng):

        # RESET ENV
        rng, warmup_rng, reset_key = jax.random.split(rng, 3)
        reset_key = jax.random.split(reset_key, config.NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])

        # # LOAD MODEL
        if config.EVAL_MODEL:
            network, last_obs = init_network(config, env, env_state, env_params)
            network_params = config.model["model"]["params"]
            apply = network.apply
            print('Evaluating model')
        else:
            network_params = apply = None

        eval_state = EvalState(apply_fn=apply, params=network_params)

        # Recreate DeepRMSA warmup period
        if config.ENV_WARMUP_STEPS:
            warmup_state = (warmup_rng, env_state, obsv)
            warmup_fn = get_warmup_fn(warmup_state, env, env_params, eval_state, config)
            env_state, obsv = warmup_fn(warmup_state)

        # COLLECT TRAJECTORIES
        def _env_episode(runner_state, unused):

            def _env_step(runner_state, unused):

                env_state, last_obs, rng = runner_state
                rng, action_key, step_key = jax.random.split(rng, 3)

                # SELECT ACTION
                action = select_action_eval(config, env_state, env_params, env, eval_state, action_key, last_obs)

                # STEP ENV

                step_key = jax.random.split(step_key, config.NUM_ENVS)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    step_key, env_state, action, env_params
                )
                if config.PROFILE:
                    jax.profiler.save_device_memory_profile("memory_step.prof")
                obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
                transition = Transition(
                    done, action, reward, last_obs, info
                )
                runner_state = (env_state, obsv, rng)

                if config.DEBUG:
                    jax.debug.print("link_slot_array {}", env_state.env_state.link_slot_array, ordered=config.ORDERED)
                    jax.debug.print("link_slot_mask {}", env_state.env_state.link_slot_mask, ordered=config.ORDERED)
                    jax.debug.print("action {}", action, ordered=config.ORDERED)
                    jax.debug.print("reward {}", reward, ordered=config.ORDERED)

                return runner_state, transition

            runner_state, traj_episode = jax.lax.scan(
                _env_step, runner_state, None, config.max_timesteps
            )
            if config.PROFILE:
                jax.profiler.save_device_memory_profile("memory_scan.prof")

            metric = traj_episode.info

            return runner_state, metric

        runner_state = (env_state, obsv, rng)
        NUM_EPISODES = config.TOTAL_TIMESTEPS // config.max_timesteps // config.NUM_ENVS
        runner_state, metric = jax.lax.scan(
            _env_episode, runner_state, None, NUM_EPISODES
        )
        return {"runner_state": runner_state, "metrics": metric}

    return evaluate
