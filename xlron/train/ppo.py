from typing import Any, Callable, Dict, Tuple

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from box import Box
from gymnax.environments.environment import Environment
from jax import Array

from xlron import dtype_config
from xlron.environments.dataclasses import (
    EnvParams,
    LogEnvState,
    Obsv,
    RSATransition,
    Transition,
    VONETransition,
)
from xlron.environments.env_funcs import process_path_action
from xlron.environments.gn_model.isrs_gn_model import to_dbm
from xlron.environments.wrappers import jit_profiler
from xlron.train.train_utils import TrainState, select_action

RunnerState = Tuple[TrainState, LogEnvState, Obsv, Array, Array]
UpdateState = Tuple[TrainState, Transition, Array, Array, Array, Any, Array]


def compute_trajectory_priority_weights(advantages: Array, alpha: Array) -> Array:
    # Advantages have shape (rollout_length, num_envs)
    advantages = advantages.reshape(-1, 1) if advantages.ndim == 1 else advantages
    priority_weights = jnp.abs(advantages).sum(axis=0)  # Axis 0 is rollout
    return jnp.power(priority_weights + 1e-6, alpha)


def compute_sample_priority_weights(advantages: Array, alpha: Array) -> Array:
    # Advantages have shape (rollout_length, num_envs)
    advantages = advantages.reshape(-1, 1) if advantages.ndim == 1 else advantages
    priority_weights = jnp.abs(advantages)
    return jnp.power(priority_weights + 1e-6, alpha)


def _sample_prioritized_batch(
    batch: Tuple[Transition, Array, Array],
    priority_weights: Array,
    beta: Array,
    rng_key: Array,
    config: Box,
) -> Tuple[Tuple[Transition], Array]:
    batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
    assert batch_size == config.ROLLOUT_LENGTH * config.NUM_ENVS, (
        f"batch size (which comprises {config.NUM_MINIBATCHES} of size {config.MINIBATCH_SIZE}) "
        + f"({batch_size}) must be equal to number of steps ({config.ROLLOUT_LENGTH})"
        + f"* number of envs ({config.NUM_ENVS})  * number of devices ({config.NUM_LEARNERS})"
    )
    shuffle_key, sample_key = jax.random.split(rng_key)

    if config.PRIO_ALPHA == 0.0 or config.PRIO_BETA0 == 1.0:
        # Uniform importance if not using prioritized sampling
        importance_weights = jnp.ones((batch_size,), dtype=dtype_config.LARGE_FLOAT_DTYPE)

    else:
        priority_probs = (priority_weights + 1e-6) / (jnp.sum(priority_weights) + 1e-6)

        if not config.USE_RNN or (config.RHO_CLIP <= 0 or config.C_CLIP <= 0):
            # If not using RNN or VTRACE-style clipping, we can prioritize individual samples
            sampled_indices = jax.random.choice(
                sample_key, batch_size, shape=(batch_size,), p=priority_probs.reshape((batch_size,))
            )

            importance_weights = jnp.power(jnp.take(priority_probs, sampled_indices), -beta)

            batch = jax.tree.map(
                lambda x: jnp.take(x.reshape((-1, *x.shape[2:])), sampled_indices, axis=0).reshape(
                    x.shape
                ),
                batch,
            )

        else:
            # If using RNN we can only prioritize entire trajectories
            sampled_indices = jax.random.choice(
                sample_key, config.NUM_ENVS, shape=(config.NUM_ENVS,), p=priority_probs
            )

            importance_weights = jnp.tile(
                jnp.power(jnp.take(priority_probs, sampled_indices), -beta),
                (config.ROLLOUT_LENGTH, 1),
            )

            # Create a prioritized batch
            batch = jax.tree.map(lambda x: jnp.take(x, sampled_indices, axis=1), batch)

    batch_with_weights = (batch, importance_weights)

    if config.NUM_ENVS > 1:
        batch_with_weights = jax.tree.map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]), batch_with_weights
        )

    # Only shuffle the batch if we're not using an RNN-based policy and not using VTRACE-style clipping
    if config.RHO_CLIP <= 0 or config.C_CLIP <= 0:
        if not config.USE_RNN:
            # Shuffle the batch
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch_with_weights = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), batch_with_weights
            )

    minibatches = jax.tree.map(
        lambda x: jnp.reshape(x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])),
        batch_with_weights,
    )

    return minibatches[0], minibatches[1]


def _env_step(
    runner_state: RunnerState,
    unused: Any,
    env: Environment,
    env_params: EnvParams,
    config: Box,
) -> Tuple[RunnerState, Transition]:
    """Single environment step. Called via scan with closure wrapper."""
    train_state, env_state, last_obs, rng_step, rng_epoch = runner_state

    # Use fold_in to generate unique key for this step, maintains shape
    step_key, action_key = jax.random.split(rng_step)

    select_action_state = (action_key, env_state, last_obs)
    env_state, action, log_prob, value = jit_profiler.call(
        env_params.profile, select_action, select_action_state, env, env_params, train_state, config
    )

    obsv, env_state, reward, terminal, truncated, info = jit_profiler.call(
        env_params.profile, env.step, action_key, env_state, action, env_params
    )
    # Apply reward scaling if configured
    reward = reward * config.REWARD_SCALE

    # PROCESS OBS AND TRANSITION
    obsv = (
        (env_state.env_state, env_params)
        if config.USE_GNN or config.USE_TRANSFORMER
        else tuple([obsv])
    )

    # Create transition based on environment type
    if config.env_type.lower() == "vone":
        transition = VONETransition(
            terminal,
            truncated,
            action,
            value,
            reward,
            log_prob,
            last_obs,
            info,
            env_state.env_state.node_mask_s,
            env_state.env_state.link_slot_mask,
            env_state.env_state.node_mask_d,
            env_state.env_state.valid_mass,
        )
    else:
        transition = RSATransition(
            terminal,
            truncated,
            action,
            value,
            reward,
            log_prob,
            last_obs,
            info,
            env_state.env_state.link_slot_mask,
            env_state.env_state.valid_mass,
        )

    # DEBUG LOGGING FOR OPTICAL NETWORKS
    if config.DEBUG:
        path_action = action[0][0] if config.env_type.lower() == "rsa_gn_model" else action
        path_index, slot_index = process_path_action(env_state.env_state, env_params, path_action)
        path = env_params.path_link_array[path_index]

        def get_path_links(x):
            return jnp.dot(path, x)

        jax.debug.print(
            "state.request_array {}", env_state.env_state.request_array, ordered=config.ORDERED
        )
        jax.debug.print("action {}", action, ordered=config.ORDERED)
        jax.debug.print("log_prob {}", log_prob, ordered=config.ORDERED)
        jax.debug.print("reward {}", reward, ordered=config.ORDERED)
        jax.debug.print(
            "link_slot_array {}",
            get_path_links(env_state.env_state.link_slot_array),
            ordered=config.ORDERED,
        )

        if config.env_type.lower() == "vone":
            jax.debug.print(
                "node_mask_s {}", env_state.env_state.node_mask_s, ordered=config.ORDERED
            )
            jax.debug.print(
                "node_mask_d {}", env_state.env_state.node_mask_d, ordered=config.ORDERED
            )
            jax.debug.print(
                "action_history {}", env_state.env_state.action_history, ordered=config.ORDERED
            )
            jax.debug.print(
                "action_counter {}", env_state.env_state.action_counter, ordered=config.ORDERED
            )
            jax.debug.print(
                "request_array {}", env_state.env_state.request_array, ordered=config.ORDERED
            )
            jax.debug.print(
                "node_capacity_array {}",
                env_state.env_state.node_capacity_array,
                ordered=config.ORDERED,
            )
        elif config.env_type.lower() == "rsa_gn_model":
            jax.debug.print(
                "modulation_format_index_array {}",
                get_path_links(env_state.env_state.modulation_format_index_array),
                ordered=config.ORDERED,
            )
            jax.debug.print(
                "channel_centre_bw_array {}",
                get_path_links(env_state.env_state.channel_centre_bw_array),
                ordered=config.ORDERED,
            )
            jax.debug.print(
                "link_snr_array {}",
                get_path_links(env_state.env_state.link_snr_array),
                ordered=config.ORDERED,
            )
            jax.debug.print(
                "channel_power_array {}",
                get_path_links(env_state.env_state.channel_power_array),
                ordered=config.ORDERED,
            )

    runner_state_out = (train_state, env_state, obsv, step_key, rng_epoch)
    return runner_state_out, transition


def _calculate_puffer_advantage(
    train_state: TrainState,
    traj_batch: Transition,
    last_value: Array,
    importance_ratio: Array,
    config: Box,
) -> Tuple[Array, Array, Array]:
    """
    Calculate Puffer Advantage (generalization of GAE and VTrace).

    Contains nested `_get_advantages` helper for the scan.

    Calculate Puffer Advantage, a generalization of GAE and VTrace.

    Args:
        traj_batch: Trajectory batch containing transitions
        last_val: Value estimate for the last state

    Returns:
        advantages: Computed advantages
        targets: Value targets (advantages + values)
        deltas: TD errors

    Note:
        - When config.RHO_CLIP=inf and config.C_CLIP=inf, this reduces to standard GAE
        - When lambda=1, this reduces to VTrace
        - traj_batch.importance should contain importance sampling ratios
    """
    # Optionally anneal GAE_LAMBDA to higher value to increase horizon
    if config.GAE_LAMBDA is None:
        # Multiply by 3 so that more time spent in high lambda at end of training
        frac = (
            3
            * train_state.step
            / (config.NUM_INCREMENTS * config.NUM_UPDATES * config.LAMBDA_SCHEDULE_MULTIPLIER)
        )
        sech_frac = 1 - 1 / jnp.cosh(frac)
        lambda_delta = config.FINAL_LAMBDA - config.INITIAL_LAMBDA
        current_lambda = config.INITIAL_LAMBDA + (sech_frac * lambda_delta)
    else:
        current_lambda = config.GAE_LAMBDA

    def _get_advantages(
        gae_and_next_value: Tuple[Array, Array],
        transition_and_importance: Tuple[Transition, Array],
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        gae, next_value = gae_and_next_value
        transition, importance = transition_and_importance
        terminal, value, reward = (
            transition.terminal,
            transition.value,
            transition.reward,
        )
        centered_reward = reward - train_state.avg_reward if config.REWARD_CENTERING else reward

        if config.RHO_CLIP <= 0 or config.C_CLIP <= 0:
            # No clipping applied
            rho_t = importance
            c_t = importance
        else:
            # Apply clipping to importance ratios
            rho_t = jnp.minimum(importance, config.RHO_CLIP)
            c_t = jnp.minimum(importance, config.C_CLIP)

        # Modified TD error calculation with importance sampling
        # delta = rho_t * (r_t+1 + gamma * V(s_t+1) * (1 - terminal_t+1) - V(s_t))
        delta = rho_t * (centered_reward + config.GAMMA * next_value * (1 - terminal) - value)

        # Modified GAE accumulation with clipped importance ratios
        # A_t = delta_t + gamma * lambda * c_t * (1 - terminal_t+1) * A_t+1
        gae = delta + config.GAMMA * current_lambda * c_t * (1 - terminal) * gae

        return (gae, value), (gae, delta)

    _, (advantages, deltas) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_value), last_value),
        (traj_batch, importance_ratio),
        reverse=True,
        unroll=True,
    )
    return advantages, advantages + traj_batch.value, deltas


def _env_rollout_advantages(
    runner_state: RunnerState,
    env: Environment,
    env_params: EnvParams,
    config: Box,
) -> Tuple[RunnerState, Transition, Array, Array, Array, Array]:
    """
    Perform environment rollout and compute advantages.

    This consolidates:
    - Environment stepping via scan over _env_step
    - Last value computation
    - Advantage calculation
    - Reward centering updates
    - Priority computation

    Returns:
        runner_state: Updated runner state (with updated train_state if reward centering)
        traj_batch: Trajectory batch from rollout
        adv: Computed advantages
        targets: Value targets
        priorities: Sample priorities for prioritized replay
    """

    # Create a scan-compatible wrapper that captures env, env_params, config
    def _env_step_wrapper(runner_state, unused):
        return _env_step(runner_state, unused, env, env_params, config)

    _env_step_vmap = (
        jax.vmap(
            _env_step_wrapper,
            in_axes=((None, 0, 0, 0, None), None),
            out_axes=((None, 0, 0, 0, None), 0),
        )
        if config.NUM_ENVS > 1
        else _env_step_wrapper
    )

    rng_step = runner_state[3]
    rng_step, *step_keys_list = jax.random.split(rng_step, config.NUM_ENVS + 1)
    step_keys = jnp.array(step_keys_list) if config.NUM_ENVS > 1 else step_keys_list[0]
    # Include paallel step_keys for scan
    runner_state = runner_state[:3] + (step_keys,) + runner_state[4:]
    runner_state, traj_batch = jax.lax.scan(
        _env_step_vmap, runner_state, None, config.ROLLOUT_LENGTH
    )
    # Reinstate rng_step after scan
    runner_state = runner_state[:3] + (rng_step,) + runner_state[4:]

    # CALCULATE ADVANTAGE
    train_state, env_state, last_obs, _, rng_epoch = runner_state
    last_obs = (
        (env_state.env_state, env_params) if config.USE_GNN or config.USE_TRANSFORMER else last_obs
    )
    axes = (0, None) if config.USE_GNN or config.USE_TRANSFORMER else (0,)
    # With Equinox, the model is called directly
    model = eqx.combine(train_state.model_params, train_state.model_static)
    _, last_val = (
        jax.vmap(model, in_axes=axes)(*last_obs) if (config.NUM_ENVS > 1) else model(*last_obs)
    )

    # Compute advantages here so they can be used to prioritize trajectories with high absolute advantage estimates
    initial_importance_ratio = jnp.ones_like(traj_batch.reward)
    adv, targets, deltas = jit_profiler.call(
        config.PROFILE,
        _calculate_puffer_advantage,
        train_state,
        traj_batch,
        last_val,
        initial_importance_ratio,
        config,
    )

    if config.REWARD_CENTERING:
        train_state = train_state.update_step_size()
        # Extract the one-step TD errors (deltas) from your GAE calculation
        updated_avg_reward = train_state.avg_reward + train_state.reward_stepsize * jnp.mean(deltas)
        adjustment = train_state.avg_reward - updated_avg_reward
        targets = targets + adjustment
        # Update avg_reward using eqx.tree_at
        train_state = eqx.tree_at(
            lambda state: state.avg_reward,
            train_state,
            updated_avg_reward,
        )

    # COMPUTE PRIORITIES AND ANNEALED BETA
    priorities = (
        compute_sample_priority_weights(adv, train_state.prio_alpha)
        if (not config.USE_RNN) or (config.RHO_CLIP <= 0 or config.C_CLIP <= 0)
        else compute_trajectory_priority_weights(adv, train_state.prio_alpha)
    )
    # Anneal beta from initial value to 1.0 over course of training
    progress = (
        train_state.prio_alpha * train_state.step / (config.NUM_UPDATES * config.NUM_INCREMENTS)
    )
    annealed_beta = train_state.prio_beta0 + (1.0 - train_state.prio_beta0) * progress
    train_state = eqx.tree_at(
        lambda state: state.prio_beta,
        train_state,
        annealed_beta,
    )
    return runner_state, traj_batch, adv, targets, priorities


@eqx.filter_value_and_grad(has_aux=True)
def _loss_fn(
    model: eqx.Module,
    train_state: TrainState,
    batch_info: Tuple[Transition | RSATransition | VONETransition, Array, Array, Array],
    config: Box,
) -> Tuple[Array, Tuple[Array, ...]]:
    """
    Compute PPO loss (actor + value + entropy).
    """
    traj_batch, adv, targets, importance_weights = batch_info
    # RERUN NETWORK - with Equinox, vmap the model directly
    axes = (0, None) if config.USE_GNN or config.USE_TRANSFORMER else (0,)
    pi, value = jax.vmap(model, in_axes=axes)(*traj_batch.obs)

    # HANDLE DIFFERENT ACTION TYPES FOR OPTICAL NETWORKS
    if config.env_type.lower() == "vone":
        # VONE: source, path, destination actions
        pi_source = distrax.Categorical(logits=pi._logits + (-1e8 * (1 - traj_batch.action_mask_s)))
        pi_path = distrax.Categorical(logits=pi._logits + (-1e8 * (1 - traj_batch.action_mask_p)))
        pi_dest = distrax.Categorical(logits=pi._logits + (-1e8 * (1 - traj_batch.action_mask_d)))
        action_s = traj_batch.action[:, 0]
        action_p = traj_batch.action[:, 1]
        action_d = traj_batch.action[:, 2]
        log_prob_source = pi_source.log_prob(action_s)
        log_prob_path = pi_path.log_prob(action_p)
        log_prob_dest = pi_dest.log_prob(action_d)
        log_prob = log_prob_source + log_prob_path + log_prob_dest
        entropy = pi_source.entropy() + pi_path.entropy() + pi_dest.entropy()

    elif config.env_type.lower() == "rsa_gn_model" and config.launch_power_type == "rl":
        # RSA with power control
        path_actions = traj_batch.action[..., 0]
        power_actions = traj_batch.action[..., 1]
        path_dist, power_dist = pi
        path_log_prob = path_entropy = 0.0

        if config.GNN_OUTPUT_RSA:
            pi_masked = distrax.Categorical(
                logits=path_dist._logits + (-1e8 * (1 - traj_batch.action_mask))
            )
            path_log_prob = pi_masked.log_prob(path_actions)
            path_entropy = pi_masked.entropy()

        path_indices = jax.vmap(process_path_action, in_axes=(0, None, 0))(
            traj_batch.obs[0], config, path_actions
        )[0]
        # Re-scale action from [min_power, max_power] to [0, 1]
        power_actions = jnp.astype(
            (to_dbm(power_actions) - config.min_power) / config.step_power, jnp.int32
        )
        # Repeat the power action along the last axis K-paths time
        power_actions = jnp.tile(power_actions[..., None], (1, config.k_paths))
        power_log_prob = power_dist.log_prob(power_actions)
        # Slice log prob to just take the path index
        power_log_prob = jax.vmap(lambda x, i: jax.lax.dynamic_slice(x, (i,), (1,)))(
            power_log_prob, path_indices
        )
        power_entropy = power_dist.entropy()

        log_prob = path_log_prob + power_log_prob
        entropy = path_entropy + power_entropy

        if config.DEBUG:
            jax.debug.print("targets {}", targets, ordered=config.ORDERED)
            jax.debug.print("path_actions {}", path_actions, ordered=config.ORDERED)
            jax.debug.print("power_actions {}", power_actions, ordered=config.ORDERED)
            jax.debug.print("path_log_prob {}", path_log_prob, ordered=config.ORDERED)
            jax.debug.print("power_log_prob {}", power_log_prob, ordered=config.ORDERED)
            jax.debug.print("path_entropy {}", path_entropy, ordered=config.ORDERED)
            jax.debug.print("power_entropy {}", power_entropy, ordered=config.ORDERED)
            jax.debug.print("power logits {}", power_dist._logits, ordered=config.ORDERED)
            jax.debug.print("log_prob {}", log_prob, ordered=config.ORDERED)
            jax.debug.print("entropy {}", entropy, ordered=config.ORDERED)

    elif config.OFF_POLICY_IAM:
        # Ratio will be policy/masked_policy - also known as off-policy invalid action masking
        log_prob = pi.log_prob(traj_batch.action)
        entropy = pi.entropy()

    else:
        # Standard action masking
        pi_masked = distrax.Categorical(
            logits=pi[0]._logits + (-1e8 * (1 - traj_batch.action_mask))
        )
        log_prob = pi_masked.log_prob(traj_batch.action)
        entropy = pi_masked.entropy()

    log_ratio = log_prob - traj_batch.log_prob
    log_ratio = jnp.clip(log_ratio, -config.LOGR_CLIP, config.LOGR_CLIP)
    ratio = jnp.exp(log_ratio)

    # Recalculate the advantage now that we can clip based on the calculated importance ratio
    if config.RHO_CLIP > 0 and config.C_CLIP > 0:
        minibatch_size = config.MINIBATCH_SIZE
        assert config.ROLLOUT_LENGTH % config.NUM_MINIBATCHES == 0, (
            "ROLLOUT_LENGTH must be integer mutliple of NUM_MINIBATCHES"
        )
        traj_batch, value, ratio = jax.tree.map(
            lambda x: x.reshape(
                (config.ROLLOUT_LENGTH // config.NUM_MINIBATCHES, config.NUM_ENVS) + x.shape[1:]
            ),
            (traj_batch, value, ratio),
        )
        adv, _, _ = jit_profiler.call(
            config.PROFILE,
            _calculate_puffer_advantage,
            train_state,
            traj_batch,
            value[-1],
            ratio,
            config,
        )
        adv, traj_batch, value, ratio = jax.tree.map(
            lambda x: x.reshape((minibatch_size,) + x.shape[2:]),
            (adv, traj_batch, value, ratio),
        )

    # --- Hard gate for "no valid actions" ---------------------------------------
    # gate[t]=1 if there exists at least one or two valid actions at s_t, else 0.
    # (Assumes traj_batch.action_mask is boolean or {0,1}.)
    mask_sum = jnp.sum(traj_batch.action_mask, axis=-1)
    gate_any = (mask_sum > 0).astype(jnp.float32)  # at least 1 valid action
    gate_choice = (mask_sum > 1).astype(jnp.float32)  # at least 2 valid actions

    # --- Soft damping using valid-mass ------------------------------------------
    # valid_mass must be computed from the *unmasked* behavior policy at rollout time:
    # valid_mass[t] = sum_a softmax(logits_unmasked_old)[a] * action_mask[a]
    valid_mass = traj_batch.valid_mass.astype(jnp.float32)  # shape like [N], in [0, 1]
    valid_mass0 = config.VALID_MASS_TARGET  # default 0.05 (tune 0.02–0.1)

    # Linear damping (use sqrt for gentler damping if you prefer)
    damp = jnp.clip(valid_mass / valid_mass0, 0.0, 1.0)
    # damp = jnp.sqrt(jnp.clip(I / I0, 0.0, 1.0))  # optional, gentler

    # Combined per-step weight for actor + entropy (must have at least 2 valid actions)
    w = gate_choice * damp
    w_sum = jnp.maximum(w.sum(), 1e-8) # just for numerical stability

    # --- Advantage normalization (weighted stats) --------------------------------
    # Normalize using only weighted-valid steps so empty-mask / low-valid-mass steps don't skew mean/std.
    adv_mean = (adv * w).sum() / w_sum
    adv_var = ((adv - adv_mean) ** 2 * w).sum() / w_sum
    adv_norm = (adv - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
    adv_norm_clipped = jnp.clip(adv_norm, -config.ADV_CLIP, config.ADV_CLIP)

    # Optional: include importance weights (if unused, set importance_weights = 1)
    adv_weighted = importance_weights * adv_norm_clipped

    # --- PPO clipped surrogate (weighted) ----------------------------------------
    loss_actor1 = ratio * adv_weighted
    loss_actor2 = jnp.clip(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * adv_weighted

    actor_loss = -(jnp.minimum(loss_actor1, loss_actor2) * w).sum() / w_sum

    # --- Value loss (ungated) ----------------------------------------------------
    value_loss = 0.5 * jnp.square(value - targets).mean()

    # --- Entropy loss (PER-STEP, weighted) ---------------------------------------
    # entropy must have same leading shape as w (e.g., [minibatch] or [T*B]).
    ent_coef = train_state.ent_schedule(train_state.step)  # scalar
    entropy_loss = -(ent_coef * entropy * w).sum() / w_sum

    # --- Valid mass loss (encourages current policy to place mass on valid actions) -
    if config.VALID_MASS_LOSS_COEF > 0:
        # Recompute valid mass from *current* logits so gradients flow back
        if config.env_type.lower() == "vone":
            current_probs = jax.nn.softmax(pi._logits, axis=-1)
            current_valid_mass = jnp.sum(current_probs * traj_batch.action_mask_p, axis=-1)
        elif config.env_type.lower() == "rsa_gn_model" and config.launch_power_type == "rl":
            current_probs = jax.nn.softmax(pi[0]._logits, axis=-1)
            current_valid_mass = jnp.sum(current_probs * traj_batch.action_mask, axis=-1)
        elif config.OFF_POLICY_IAM:
            current_probs = jax.nn.softmax(pi._logits, axis=-1)
            current_valid_mass = jnp.sum(current_probs * traj_batch.action_mask, axis=-1)
        else:
            current_probs = jax.nn.softmax(pi[0]._logits, axis=-1)
            current_valid_mass = jnp.sum(current_probs * traj_batch.action_mask, axis=-1)
        # Must have at least one valid action, else 0
        validmass_loss = (jnp.square(1.0 - current_valid_mass) * gate_any).sum() / jnp.maximum(gate_any.sum(), 1.0)
    else:
        validmass_loss = jnp.array(0.0)

    # --- Total loss --------------------------------------------------------------
    vml_coef = train_state.vml_schedule(train_state.step)  # scalar
    total_loss = actor_loss + config.VF_COEF * value_loss + entropy_loss + vml_coef * validmass_loss

    if config.DEBUG or config.DEBUG_LOSS:
        jax.debug.print("log_prob {}", log_prob, ordered=config.ORDERED)
        jax.debug.print("entropy {}", entropy, ordered=config.ORDERED)
        jax.debug.print("ratio {}", ratio, ordered=config.ORDERED)
        jax.debug.print("adv {}", adv, ordered=config.ORDERED)
        jax.debug.print("loss_actor1 {}", loss_actor1, ordered=config.ORDERED)
        jax.debug.print("loss_actor2 {}", loss_actor2, ordered=config.ORDERED)
        jax.debug.print("value_loss {}", value_loss, ordered=config.ORDERED)
        jax.debug.print("actor_loss {}", actor_loss, ordered=config.ORDERED)
        jax.debug.print("entropy {}", entropy, ordered=config.ORDERED)
        jax.debug.print("total_loss {}", total_loss, ordered=config.ORDERED)

    return total_loss, (
        log_prob.mean(),
        ratio.mean(),
        adv_weighted.mean(),
        value_loss,
        actor_loss,
        entropy,
        entropy * ent_coef,
        validmass_loss,
    )


def _update_minibatch(
    train_state: TrainState,
    batch_info: Tuple[Transition, Array, Array, Array],
    config: Box,
) -> Tuple[TrainState, Tuple[Array, ...]]:
    """Update on a single minibatch. Called via scan with closure wrapper."""
    model = eqx.combine(train_state.model_params, train_state.model_static)
    total_loss, grads = jit_profiler.call(
        config.PROFILE,
        _loss_fn,
        model,
        train_state,
        batch_info,
        config,
    )
    train_state = train_state.apply_gradients(grads=grads)
    train_state = eqx.tree_at(
        lambda state: state.step,
        train_state,
        train_state.step
        + int(config.STEP_ON_GRADIENT),  # Increment step by config.STEP_ON_GRADIENT
    )
    if config.DEBUG or config.DEBUG_LOSS:
        grad_norm = optax.global_norm(grads)
        jax.debug.print("gradient_norm {}", grad_norm, ordered=config.ORDERED)
    return train_state, total_loss


def _update_epoch(
    update_state: UpdateState,
    unused: Any,
    config: Box,
) -> Tuple[UpdateState, Tuple[Array, ...]]:
    """Single epoch of minibatch updates. Called via scan with closure wrapper."""
    (train_state, traj_batch, adv, targets, rng_step, rng_epoch, priorities) = update_state
    rng_epoch, perm_key = jax.random.split(rng_epoch, 2)

    batch = (traj_batch, adv, targets)
    minibatches, importance_weights_mb = jit_profiler.call(
        config.PROFILE,
        _sample_prioritized_batch,
        batch,
        priorities,
        train_state.prio_beta,
        perm_key,
        config,
    )
    batch_info = (*minibatches, importance_weights_mb)

    # Scan-compatible wrapper
    def _update_minibatch_wrapper(train_state, batch_info):
        return jit_profiler.call(config.PROFILE, _update_minibatch, train_state, batch_info, config)

    train_state, total_loss = jax.lax.scan(_update_minibatch_wrapper, train_state, batch_info)

    update_state = (
        train_state,
        traj_batch,
        adv,
        targets,
        rng_step,
        rng_epoch,
        priorities,
    )
    return update_state, total_loss


def _update_step(
    runner_state: RunnerState,
    unused: Any,
    env: Environment,
    env_params: EnvParams,
    config: Box,
) -> Tuple[RunnerState, Tuple[Dict[str, Array], Dict[str, Array]]]:
    """
    Single update step: rollout + multiple epochs of updates.

    Composes _env_rollout and _update_epoch.
    """

    runner_state, traj_batch, adv, targets, priorities = _env_rollout_advantages(
        runner_state, env, env_params, config
    )
    (train_state, env_state, last_obs, rng_step, rng_epoch) = runner_state

    update_state = (
        train_state,
        traj_batch,
        adv,
        targets,
        rng_step,
        rng_epoch,
        priorities,
    )

    def _update_epoch_wrapper(update_state, unused):
        return jit_profiler.call(config.PROFILE, _update_epoch, update_state, unused, config)

    update_state, loss_info_arrays = jax.lax.scan(
        _update_epoch_wrapper, update_state, None, config.UPDATE_EPOCHS
    )

    # Note: we increment step just once per update loop (not per gradient update)
    train_state = eqx.tree_at(
        lambda state: state.step,
        update_state[0],
        update_state[0].step + (1 - int(config.STEP_ON_GRADIENT)),
    )

    metric = traj_batch.info
    rng_step = update_state[4]
    rng_epoch = update_state[5]
    runner_state = (train_state, env_state, last_obs, rng_step, rng_epoch)

    loss_info = {
        "loss/total_loss": loss_info_arrays[0].reshape(-1),
        "loss/log_prob": loss_info_arrays[1][0].reshape(-1),
        "loss/ratio": loss_info_arrays[1][1].reshape(-1),
        "loss/gae": loss_info_arrays[1][2].reshape(-1),
        "loss/value_loss": loss_info_arrays[1][3].reshape(-1),
        "loss/value_loss_scaled": loss_info_arrays[1][3].reshape(-1) * config.VF_COEF,
        "loss/actor_loss": loss_info_arrays[1][4].reshape(-1),
        "loss/entropy": loss_info_arrays[1][5].reshape(-1),
        "loss/entropy_loss_scaled": loss_info_arrays[1][6].reshape(-1),
        "loss/validmass_loss": loss_info_arrays[1][7].reshape(-1),
        "loss/validmass_loss_scaled": loss_info_arrays[1][7].reshape(-1)
        * train_state.vml_schedule(train_state.step),
        "prioritization/beta": train_state.prio_beta,
        "prioritization/priority_mean": jnp.mean(priorities),
        "prioritization/priority_std": jnp.std(priorities),
    }

    return runner_state, (metric, loss_info)


def get_learner_fn(
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    config: Box,
) -> Callable:
    def _update_step_wrapper(runner_state, unused):
        return _update_step(runner_state, unused, env, env_params, config)

    def learner_fn(runner_state: RunnerState) -> Dict[str, Any]:
        runner_state, (metric_info, loss_info) = jax.lax.scan(
            _update_step_wrapper, runner_state, None, config.NUM_UPDATES
        )

        return {"runner_state": runner_state, "metrics": metric_info, "loss_info": loss_info}

    return learner_fn
