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
            action, _, _ = select_action_fn(select_action_state)

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

    def eval_fn(runner_state):

        NUM_EPISODES = config.TOTAL_TIMESTEPS // config.max_timesteps // config.NUM_ENVS
        runner_state, metric = jax.lax.scan(
            _env_episode, runner_state, None, NUM_EPISODES
        )
        return {"runner_state": runner_state, "metrics": metric}

    return eval_fn


def experiment_data_setup(config: flags.FlagValues, rng: chex.PRNGKey) -> Tuple:

    # INIT ENV
    env, env_params = define_env(config)
    rng, rng_step, rng_epoch, warmup_key, reset_key, network_key = jax.random.split(rng, 6)
    reset_key = jax.random.split(reset_key, config.NUM_ENVS)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
    obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])

    # TRAINING MODE
    if not config.EVAL_HEURISTIC and not config.EVAL_MODEL:

        # INIT NETWORK
        network, init_x = init_network(config, env, env_state, env_params)
        init_x = (jax.tree.map(lambda x: x[0], init_x[0]), init_x[1]) if config.USE_GNN else init_x

        if config.RETRAIN_MODEL:
            network_params = config.model["model"]["params"]
            print('Retraining model')
        else:
            network_params = network.init(network_key, *init_x)

        # INIT LEARNING RATE SCHEDULE AND OPTIMIZER
        lr_schedule = make_lr_schedule(config)
        tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(learning_rate=lr_schedule, eps=config.ADAM_EPS, b1=config.ADAM_BETA1, b2=config.ADAM_BETA2),
        )

        runner_state = TrainState.create(
            apply_fn=network.apply,
            sample_fn=network.sample_action,
            params=network_params,
            tx=tx,
        )

    # EVALUATION MODE
    else:
        # LOAD MODEL
        if config.EVAL_HEURISTIC:
            network_params = apply = sample = None

        elif config.EVAL_MODEL:
            network, last_obs = init_network(config, env, env_state, env_params)
            network_params = config.model["model"]["params"]
            apply = network.apply
            sample = network.sample_action
            print('Evaluating model')

        runner_state = EvalState(apply_fn=apply, sample_fn=sample, params=network_params)

    # Recreate DeepRMSA warmup period
    warmup_key = jax.random.split(warmup_key, config.NUM_ENVS)
    warmup_state = (warmup_key, env_state, obsv)
    warmup_fn = get_warmup_fn(warmup_state, env, env_params, runner_state, config)
    warmup_fn = jax.vmap(warmup_fn)
    env_state, obsv = warmup_fn(warmup_state)

    # Initialise eval state
    init_runner_state = (runner_state, env_state, obsv, rng_step, rng_epoch)

    return init_runner_state, env, env_params
