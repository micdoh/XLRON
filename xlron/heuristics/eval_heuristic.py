import orbax.checkpoint
import pathlib
import optax
from flax.training import orbax_utils
from typing import NamedTuple
from absl import flags
from flax.training.train_state import TrainState
from xlron.environments.env_funcs import *
from xlron.train.ppo import make_train, define_env, init_network, select_action
from xlron.environments.vone import *
from xlron.environments.rsa import *
from xlron.heuristics.heuristics import *


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_eval(config):

    # INIT ENV
    env, env_params = define_env(config)

    def evaluate(rng):

        # RESET ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])

        # # LOAD MODEL
        if config.EVAL_MODEL:
            # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            # # This isn't working - is it because it's in JIT? or should I use restore_args?
            # # https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#multi-host-multi-process-checkpointing
            # #network, last_obs = init_network(config, env_params, model)
            # #network, last_obs = init_network(rng, config, env, env_state, env_params)
            # network, last_obs = init_network(rng, config, env, env_state, env_params)
            # network_params = network.init(_rng, *init_x)
            # tx = optax.chain(
            #     optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            #     optax.adam(learning_rate=lambda x: x, eps=1e-5),
            # )
            # train_state = TrainState.create(
            #     apply_fn=network.apply,
            #     params=network_params,
            #     tx=tx,
            # )
            # save_data = {"model": train_state, "config": config}
            # restore_args = orbax_utils.restore_args_from_target(save_data)
            # model = orbax_checkpointer.restore(pathlib.Path(config.MODEL_PATH))#, restore_args)
            # network_params = model["model"].params

            network, last_obs = init_network(rng, config, env, env_state, env_params)
            network_params = config.model["model"]["params"]
            print('wow')

        # COLLECT TRAJECTORIES
        def _env_episode(runner_state, unused):

            def _env_step(runner_state, unused):

                env_state, last_obs, rng = runner_state

                # SELECT ACTION
                if config.EVAL_HEURISTIC:
                    if config.env_type.lower() == "vone":
                        raise NotImplementedError(f"VONE heuristics not yet implemented")

                    elif config.env_type.lower() in ["rsa", "rwa", "rmsa", "deeprmsa", "rwa_lightpath_reuse"]:
                        if config.path_heuristic.lower() == "ksp_ff":
                            action = jax.vmap(ksp_ff, in_axes=(0, None))(env_state.env_state, env_params)
                        elif config.path_heuristic.lower() == "ff_ksp":
                            action = jax.vmap(ff_ksp, in_axes=(0, None))(env_state.env_state, env_params)
                        elif config.path_heuristic.lower() == "kmc_ff":
                            action = jax.vmap(kmc_ff, in_axes=(0, None))(env_state.env_state, env_params)
                        elif config.path_heuristic.lower() == "kmf_ff":
                            action = jax.vmap(kmf_ff, in_axes=(0, None))(env_state.env_state, env_params)
                        else:
                            raise ValueError(f"Invalid path heuristic {config.path_heuristic}")

                    else:
                        raise ValueError(f"Invalid environment type {config.env_type}")
                else:
                    action, _, _, rng = select_action(
                        rng, env, env_state, env_params, network, network_params, config, last_obs, deterministic=config.deterministic
                    )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.NUM_ENVS)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
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
