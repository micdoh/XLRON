from jax.experimental.pallas.ops.tpu.megablox.gmm import partial
import jax
import jax.numpy as jnp
import optax
from absl import flags
from flax.training.train_state import TrainState
from gymnax.environments import environment

from xlron.environments.dataclasses import (
    EnvParams,
)
from xlron.environments.env_funcs import process_path_action
from xlron.environments.diff_utils import *
from xlron.train.train_utils import *


@partial(jax.jit, static_argnames=("params",))
def decombine_actions(actions, params):
    paths, slots = jax.vmap(process_path_action, in_axes=(None, None, 0))(None, params, actions)
    return jnp.vstack((paths, slots)).T


@partial(jax.jit, static_argnames=("params",))
def combine_actions(paths, slots, params):
    def combine(action, params):
        path, slot = action
        return path * params.link_resources + slot

    actions = jax.vmap(combine, in_axes=(0, None))((paths, slots), params)
    return actions


def get_learner_fn(
    env: environment.Environment,
    env_params: EnvParams,
    train_state: TrainState,
    config: flags.FlagValues,
) -> Callable:
    # Create optimizer
    optimizer = optax.adam(config.OPTIMIZATION_LEARNING_RATE)

    def _env_step(_runner_state, action):
        _, env_state, last_obs, rng_step, rng_epoch = _runner_state
        rng_step, action_key, step_key = jax.random.split(rng_step, 3)
        step_key = jax.random.split(step_key, config.NUM_ENVS) if config.NUM_ENVS > 1 else step_key
        step_fn = lambda x, y: env.step(x, y, action, env_params)
        step_fn = jax.vmap(step_fn) if config.NUM_ENVS > 1 else step_fn
        obsv, env_state, reward, done, truncated, info = step_fn(step_key, env_state)
        obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
        _runner_state = (_, env_state, obsv, rng_step, rng_epoch)
        return _runner_state, reward

    def _env_step_interp(_runner_state, action):
        _, env_state, last_obs, rng_step, rng_epoch = _runner_state
        rng_step, action_key, step_key = jax.random.split(rng_step, 3)
        step_key = jax.random.split(step_key, config.NUM_ENVS) if config.NUM_ENVS > 1 else step_key
        step_fn = lambda x, y, z: env.step(x, y, z, env_params)
        step_fn = jax.vmap(step_fn, in_axes=(0, 0, None)) if config.NUM_ENVS > 1 else step_fn
        floor_action = jnp.floor(action)
        ceil_action = jnp.ceil(action)
        # Get interpolation weight based on distance
        weight = action - floor_action  # 0.0 at floor, 1.0 at ceil
        # Evaluate both integer actions (without rounding in the state transition)
        floor_obs, floor_state, floor_reward, floor_done, floor_truncated, floor_info = step_fn(
            step_key, env_state, floor_action
        )
        ceil_obs, ceil_state, ceil_reward, ceil_done, ceil_truncated, ceil_info = step_fn(
            step_key, env_state, ceil_action
        )
        # Interpolate reward
        weighted_reward = (1 - weight) * floor_reward + weight * ceil_reward
        obsv, env_state, reward, done, truncated, info = step_fn(step_key, env_state, action)
        obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
        _runner_state = (_, env_state, obsv, rng_step, rng_epoch)
        return _runner_state, weighted_reward

    def _env_step_gaussian(_runner_state, action):
        _, env_state, last_obs, rng_step, rng_epoch = _runner_state
        rng_step, action_key, step_key = jax.random.split(rng_step, 3)
        step_key = jax.random.split(step_key, config.NUM_ENVS) if config.NUM_ENVS > 1 else step_key
        step_fn = lambda x, y, z: env.step(x, y, z, env_params)
        step_fn = jax.vmap(step_fn, in_axes=(0, 0, None)) if config.NUM_ENVS > 1 else step_fn

        # Sample several nearby actions
        neighbor_range = 2.5
        actions = action - jnp.arange(-neighbor_range, neighbor_range, 0.5)

        # Apply Gaussian weighting
        sigma = getattr(config, "GAUSSIAN_SIGMA", 0.8)
        weights = jnp.exp(-0.5 * ((actions - action) / sigma) ** 2)
        weights = weights / jnp.sum(weights)

        # Evaluate all actions and compute weighted average reward
        rewards = jnp.array([step_fn(step_key, env_state, a)[2] for a in actions])
        weighted_reward = jnp.sum(weights * rewards)

        obsv, env_state, reward, done, truncated, info = step_fn(step_key, env_state, action)
        obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
        _runner_state = (_, env_state, obsv, rng_step, rng_epoch)
        return _runner_state, weighted_reward

    def _env_step_heuristic(_runner_state, unused):
        _train_state, env_state, last_obs, rng_step, rng_epoch = _runner_state
        rng_step, action_key, step_key = jax.random.split(rng_step, 3)
        step_key = jax.random.split(step_key, config.NUM_ENVS) if config.NUM_ENVS > 1 else step_key
        # SELECT ACTION
        action_key = (
            jax.random.split(action_key, config.NUM_ENVS) if config.NUM_ENVS > 1 else action_key
        )
        select_action_fn = lambda x: select_action_eval(x, env, env_params, None, config)
        select_action_fn = jax.vmap(select_action_fn) if config.NUM_ENVS > 1 else select_action_fn
        select_action_state = (action_key, env_state, last_obs)
        env_state, action, _, _ = select_action_fn(select_action_state)
        # STEP ENV
        step_fn = lambda x, y, z: env.step(x, y, z, env_params)
        step_fn = jax.vmap(step_fn) if config.NUM_ENVS > 1 else step_fn
        obsv, env_state, reward, done, truncated, info = step_fn(step_key, env_state, action)
        obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
        _runner_state = (_train_state, env_state, obsv, rng_step, rng_epoch)
        return _runner_state, action

    def _rollout(_runner_state, actions):
        # TODO - it's possible for non-masked actions to be used that don't fail
        # because the slots overflow to the beginning of the array.
        # Masking usually prevents this possibility but the gradient-obtained actions aren't masked.
        step_fn = _env_step_gaussian if getattr(config, "GAUSSIAN_SMOOTHING", False) else _env_step
        out, rewards = jax.lax.scan(step_fn, _runner_state, actions)
        return out, rewards

    def _rollout_eval(_runner_state, actions):
        out, rewards = jax.lax.scan(_env_step, _runner_state, actions)
        return out, rewards

    def _rollout_heuristic(_runner_state):
        out, actions = jax.lax.scan(
            _env_step_heuristic,
            _runner_state,
            jnp.zeros((int(config.max_requests),), dtype=jnp.float32),
        )
        return out, actions

    def create_loss_fn(_runner_state):
        def loss_fn(actions):
            if config.PATH_SLOT_ACTIONS:
                paths, slots = actions[:, 0], actions[:, 1]
                actions = combine_actions(paths, slots, env_params)
            _, rewards = _rollout(_runner_state, actions)
            return jnp.sum(
                rewards
            )  # We might expect loss to be negative but our grads point towards high reward

        return loss_fn

    def create_update_step(loss_fn):
        def _update_step(update_state, _):
            actions, opt_state, best_actions, best_reward = update_state

            # Calculate current loss (negative reward)
            current_reward, grads = jax.value_and_grad(loss_fn)(actions)
            current_reward = current_reward  # Convert back to reward

            # Update optimizer state
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_actions = optax.apply_updates(actions, updates)

            # Combine actions for clipping etc.
            if config.PATH_SLOT_ACTIONS:
                actions = combine_actions(actions[:, 0], actions[:, 1], env_params)
                best_actions = combine_actions(best_actions[:, 0], best_actions[:, 1], env_params)
                new_actions = combine_actions(new_actions[:, 0], new_actions[:, 1], env_params)

            # Clip actions to valid range
            new_actions = jnp.clip(new_actions, 0, env.num_actions(env_params))

            # Update best actions if current reward is better
            best_actions = jnp.where(
                current_reward > best_reward,
                actions,  # Use current actions, not new_actions
                best_actions,
            )
            best_reward = jnp.maximum(current_reward, best_reward)

            jax.debug.print(
                "Reward: {:.1f} Mean grad.: {:.1e} +/- {:.1e}, Mean action: {:.1f}, Max action: {:.1f}",
                current_reward,
                jnp.mean(grads),
                jnp.std(grads),
                jnp.mean(new_actions),
                jnp.max(new_actions),
                ordered=True,
            )

            if config.PATH_SLOT_ACTIONS:
                best_actions = decombine_actions(best_actions, env_params)
                new_actions = decombine_actions(new_actions, env_params)

            return (
                new_actions,
                new_opt_state,
                best_actions,
                best_reward,
            ), current_reward

        return _update_step

    def learner_fn(_runner_state):
        config.EVAL_HEURISTIC = True  # For use in select_action_eval
        _, heur_actions = _rollout_heuristic(_runner_state)
        heur_actions = heur_actions.reshape((int(config.max_requests),)).astype(jnp.float32)
        heur_actions = heur_actions % env.num_actions(env_params)

        # Initialize actions
        if config.INITIALIZE_ACTIONS_HEURISTIC:
            actions = heur_actions
            decombined_actions = decombine_actions(actions, env_params)
        elif config.INITIALIZE_ACTIONS_RANDOM:
            actions = jax.random.uniform(
                jax.random.PRNGKey(0),
                (int(config.max_requests),),
                minval=0,
                maxval=env.num_actions(env_params),
            )
            actions = jnp.floor(actions).astype(jnp.float32)
            actions = actions % env.num_actions(env_params)
            decombined_actions = decombine_actions(actions, env_params)
        elif config.INITIALIZE_ACTIONS_ASCENDING:
            actions = jnp.arange(int(config.max_requests), dtype=jnp.float32)
            actions = actions % env.num_actions(env_params)
            decombined_actions = decombine_actions(actions, env_params)
        elif config.INITIALIZE_ACTIONS_DESCENDING:
            actions = jnp.arange(int(config.max_requests) - 1, -1, -1, dtype=jnp.float32)
            actions = actions % env.num_actions(env_params)
            decombined_actions = decombine_actions(actions, env_params)
        elif config.INITIALIZE_ACTIONS_MAX:
            actions = jnp.full(
                (int(config.max_requests),), env.num_actions(env_params) - 1, dtype=jnp.float32
            )
            decombined_actions = decombine_actions(actions, env_params)
        else:
            actions = jnp.zeros((int(config.max_requests),), dtype=jnp.float32)
            decombined_actions = decombine_actions(actions, env_params)

        if config.PATH_SLOT_ACTIONS:
            jax.debug.print("Using path-slot action decomposition")
            actions = decombined_actions
            heur_actions = decombine_actions(heur_actions, env_params)
        opt_state = optimizer.init(actions)
        jax.debug.print("Initial actions: \n{}", actions, ordered=True)

        # Initialize best_actions and best_reward
        best_actions = actions
        best_reward = jnp.array(-float("inf"))  # Start with worst possible reward

        loss_fn = create_loss_fn(_runner_state)
        # Loss from heuristic actions
        heuristic_loss = loss_fn(heur_actions)
        jax.debug.print("Heuristic reward: {}", heuristic_loss, ordered=True)
        update_step = create_update_step(loss_fn)

        # Optimization loop
        (final_actions, _, best_actions, best_reward), rewards = jax.lax.scan(
            update_step,
            (actions, opt_state, best_actions, best_reward),
            None,
            config.OPTIMIZATION_ITERATIONS,
        )

        if config.PATH_SLOT_ACTIONS:
            final_actions = combine_actions(final_actions[:, 0], final_actions[:, 1], env_params)
            best_actions = combine_actions(best_actions[:, 0], best_actions[:, 1], env_params)
            heur_actions = combine_actions(heur_actions[:, 0], heur_actions[:, 1], env_params)

        # Round to nearest integer actions for final evaluation
        final_actions = jnp.round(final_actions)
        best_actions = jnp.round(best_actions)

        # Run one final evaluation of the best actions to ensure consistent reporting
        final_reward = jnp.sum(_rollout_eval(_runner_state, final_actions)[1])
        best_reward = jnp.sum(_rollout_eval(_runner_state, best_actions)[1])
        # Calculate heuristic reward
        heuristic_reward = jnp.sum(_rollout_eval(_runner_state, heur_actions)[1])

        return {
            "final_actions": final_actions,
            "best_actions": best_actions,
            "heuristic_actions": heur_actions,
            "final_reward": final_reward,
            "best_reward": best_reward,
            "heuristic_reward": heuristic_reward,
            "cumulative_reward": rewards[-1],
            "rewards_per_iteration": rewards,
        }

    return learner_fn
