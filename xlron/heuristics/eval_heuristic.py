from typing import NamedTuple
from xlron.environments.env_funcs import *
from xlron.train.ppo import make_train
from xlron.environments.vone import *
from xlron.environments.rsa import *
from xlron.heuristics.heuristics import *


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_eval_heuristic(config):

    config_dict = {k: v.value for k, v in config.__flags.items()}
    if config.env_type.lower() == "vone":
        env, env_params = make_vone_env(config_dict)
    elif config.env_type.lower() in ["rsa", "rmsa", "rwa", "deeprmsa"]:
        env, env_params = make_rsa_env(config_dict)
    else:
        raise ValueError(f"Invalid environment type {config.env_type}")
    env = LogWrapper(env)

    def evaluate(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # COLLECT TRAJECTORIES
        def _env_episode(runner_state, unused):

            def _env_step(runner_state, unused):

                env_state, last_obs, rng = runner_state

                # SELECT ACTION
                if config.env_type.lower() == "vone":
                    raise NotImplementedError(f"VONE heuristics not yet implemented")

                elif config.env_type.lower() in ["rsa", "rwa", "rmsa", "deeprmsa"]:
                    if config.path_heuristic.lower() == "ksp_ff":
                        action = jax.vmap(ksp_ff, in_axes=(0, None))(env_state.env_state, env_params)
                    elif config.path_heuristic.lower() == "ff_ksp":
                        action = jax.vmap(ff_ksp, in_axes=(0, None))(env_state.env_state, env_params)
                    else:
                        raise ValueError(f"Invalid path heuristic {config.path_heuristic}")

                else:
                    raise ValueError(f"Invalid environment type {config.env_type}")

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.NUM_ENVS)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
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

            metric = traj_episode.info

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (env_state, obsv, _rng)
        NUM_EPISODES = config.TOTAL_TIMESTEPS // config.max_timesteps // config.NUM_ENVS
        runner_state, metric = jax.lax.scan(
            _env_episode, runner_state, None, NUM_EPISODES
        )
        return {"runner_state": runner_state, "metrics": metric}

    return evaluate
