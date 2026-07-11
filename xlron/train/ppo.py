import math
from typing import Any, Callable, Dict, Tuple, cast

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
    VONETransition,
)
from xlron.environments.env_funcs import process_path_action
from xlron.environments.gn_model.isrs_gn_model import to_dbm
from xlron.environments.wrappers import jit_profiler
from xlron.train.train_utils import (
    LossDiagnostics,
    TrainState,
    cast_model_for_compute,
    select_action,
    steps_per_train_state_unit,
)

RunnerState = Tuple[TrainState, LogEnvState, Obsv, Array, Array]
# (train_state, traj_batch, adv, targets, last_val, rng_step, rng_epoch, priorities)
UpdateState = Tuple[
    TrainState, RSATransition | VONETransition, Array, Array, Array, Array, Any, Array
]


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
    batch: Tuple[RSATransition | VONETransition, Array, Array],
    priority_weights: Array,
    beta: Array,
    rng_key: Array,
    config: Box,
) -> Tuple[Tuple[RSATransition | VONETransition], Array]:
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

        if not config.USE_RNN:
            # Without an RNN we can prioritize individual samples: advantages (including
            # VTrace advantages, which are recomputed over the ordered rollout in
            # _recompute_vtrace_advantages before this function runs) are already attached
            # per-sample, so resampling cannot scramble any temporal computation.
            sampled_indices = jax.random.choice(
                sample_key, batch_size, shape=(batch_size,), p=priority_probs.reshape((batch_size,))
            )

            # Standard PER importance weights (Schaul et al. 2016): w_i = (N * P(i))^-beta,
            # normalized by max_i w_i so weights are <= 1 and match the scale of the uniform
            # path above (which uses weights of exactly 1.0).
            importance_weights = jnp.power(
                batch_size * jnp.take(priority_probs, sampled_indices), -beta
            )
            importance_weights = importance_weights / jnp.maximum(jnp.max(importance_weights), 1e-8)
            importance_weights = importance_weights.astype(dtype_config.LARGE_FLOAT_DTYPE)

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

            # Standard PER importance weights over trajectories: w_i = (N * P(i))^-beta,
            # max-normalized (N = NUM_ENVS trajectories).
            trajectory_weights = jnp.power(
                config.NUM_ENVS * jnp.take(priority_probs, sampled_indices), -beta
            )
            trajectory_weights = trajectory_weights / jnp.maximum(jnp.max(trajectory_weights), 1e-8)
            importance_weights = jnp.tile(
                trajectory_weights.astype(dtype_config.LARGE_FLOAT_DTYPE),
                (config.ROLLOUT_LENGTH, 1),
            )

            # Create a prioritized batch
            batch = jax.tree.map(lambda x: jnp.take(x, sampled_indices, axis=1), batch)

    batch_with_weights = (batch, importance_weights)

    if config.NUM_ENVS > 1:
        batch_with_weights = jax.tree.map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]), batch_with_weights
        )

    # Only shuffle the batch if we're not using an RNN-based policy (which needs whole
    # trajectories). Shuffling is safe under VTrace-style clipping because advantages and
    # targets are recomputed over the temporally-ordered rollout *before* this function
    # runs (see _recompute_vtrace_advantages) and ride along with each sample.
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
) -> Tuple[RunnerState, RSATransition | VONETransition]:
    """Single environment step. Called via scan with closure wrapper."""
    train_state, env_state, last_obs, rng_step, rng_epoch = runner_state

    # Split dedicated keys for action sampling and env stepping (a key must never be
    # consumed by two different random consumers); step_key remains the scan carry.
    step_key, action_key, env_key = jax.random.split(rng_step, 3)

    select_action_state = (action_key, env_state, last_obs)
    env_state, action, log_prob, value = jit_profiler.call(
        env_params.profile, select_action, select_action_state, env, env_params, train_state, config
    )

    # Capture the acting state before env.step: on done steps env.step auto-resets, which
    # would replace the action mask / valid mass actually used by select_action with the
    # initial state's all-ones mask and valid_mass=1.0 in the stored transition.
    acting_state = env_state.env_state

    obsv, env_state, reward, terminal, truncated, info = jit_profiler.call(
        env_params.profile, env.step, env_key, env_state, action, env_params
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
            acting_state.node_mask_s,
            acting_state.link_slot_mask,
            acting_state.node_mask_d,
            acting_state.valid_mass,
            acting_state.link_slot_mask,
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
            acting_state.link_slot_mask,
            acting_state.valid_mass,
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
    traj_batch: RSATransition | VONETransition,
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
        # (denominator in train_state.step units: per gradient step if STEP_ON_GRADIENT,
        # else per update loop)
        frac = (
            3
            * train_state.step
            / (
                config.NUM_INCREMENTS
                * config.NUM_UPDATES
                * steps_per_train_state_unit(config)
                * config.LAMBDA_SCHEDULE_MULTIPLIER
            )
        )
        sech_frac = 1 - 1 / jnp.cosh(frac)
        lambda_delta = config.FINAL_LAMBDA - config.INITIAL_LAMBDA
        current_lambda = config.INITIAL_LAMBDA + (sech_frac * lambda_delta)
    else:
        current_lambda = config.GAE_LAMBDA

    def _get_advantages(
        gae_and_next_value: Tuple[Array, Array],
        transition_and_importance: Tuple[RSATransition | VONETransition, Array],
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        gae, next_value = gae_and_next_value
        transition, importance = transition_and_importance
        terminal, truncated, value, reward = (
            transition.terminal,
            transition.truncated,
            transition.value,
            transition.reward,
        )
        # env.step auto-resets on terminal OR truncated, so next_value at a truncation
        # boundary is V(post-reset state) and credit must not flow across it. Masking the
        # bootstrap with done treats truncation as termination (bootstrap 0 rather than
        # V(pre-reset s_t+1)); the faithful alternative would require exposing the
        # pre-reset final observation from env.step.
        done = jnp.logical_or(terminal, truncated)
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
        # delta = rho_t * (r_t+1 + gamma * V(s_t+1) * (1 - done_t+1) - V(s_t))
        delta = rho_t * (centered_reward + config.GAMMA * next_value * (1 - done) - value)

        # Modified GAE accumulation with clipped importance ratios
        # A_t = delta_t + gamma * lambda * c_t * (1 - done_t+1) * A_t+1
        gae = delta + config.GAMMA * current_lambda * c_t * (1 - done) * gae

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
) -> Tuple[RunnerState, RSATransition | VONETransition, Array, Array, Array]:
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
        last_val: Bootstrap value V(s_{T+1}) of the post-rollout state (behaviour policy)
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
    model = cast_model_for_compute(
        model
    )  # mixed-precision compute (no-op unless COMPUTE_DTYPE set)
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
        # Update avg_reward using eqx.tree_at
        train_state = eqx.tree_at(
            lambda state: state.avg_reward,
            train_state,
            updated_avg_reward,
        )

    # COMPUTE PRIORITIES AND ANNEALED BETA
    # Sample-level priorities unless an RNN policy requires whole trajectories; VTrace-style
    # clipping is compatible with sample-level prioritization because its advantages are
    # recomputed on the ordered rollout before resampling (see _recompute_vtrace_advantages).
    priorities = (
        compute_sample_priority_weights(adv, train_state.prio_alpha)
        if not config.USE_RNN
        else compute_trajectory_priority_weights(adv, train_state.prio_alpha)
    )
    # Anneal beta from initial value to 1.0 over course of training
    # (denominator in train_state.step units; clip so beta never exceeds 1.0)
    progress = train_state.step / (
        config.NUM_UPDATES * config.NUM_INCREMENTS * steps_per_train_state_unit(config)
    )
    progress = jnp.clip(progress, 0.0, 1.0)
    annealed_beta = train_state.prio_beta0 + (1.0 - train_state.prio_beta0) * progress
    train_state = eqx.tree_at(
        lambda state: state.prio_beta,
        train_state,
        annealed_beta,
    )
    runner_state = (train_state, env_state, last_obs, runner_state[3], rng_epoch)
    return runner_state, traj_batch, adv, targets, last_val, priorities


def _policy_log_prob_entropy(
    pi: Any,
    traj_batch: RSATransition | VONETransition,
    config: Box,
    targets: Array | None = None,
) -> Tuple[Array, Array, bool]:
    """Log-prob/entropy of the stored actions under the current policy output ``pi``.

    Shared by ``_loss_fn`` (per-minibatch PPO ratio) and ``_recompute_vtrace_advantages``
    (per-epoch VTrace importance ratios over the ordered rollout). Handles the three
    action-head layouts: VONE (three heads), launch-power RSA (path + power heads) and
    the standard masked categorical.

    Args:
        pi: Batched policy output from vmapping the model over ``traj_batch.obs``
        traj_batch: Batch of transitions (any leading batch shape, flattened to (B, ...))
        config: Training config
        targets: Optional value targets, only used for debug printing

    Returns:
        log_prob: Log probability of the stored actions under the current policy
        entropy: Policy entropy (masked where applicable)
        recenter_clip: Whether the standard off-policy IAM branch recenters the ratio
    """
    recenter_clip = False  # only the standard off-policy IAM branch below recenters the clip
    if config.env_type.lower() == "vone":
        # VONE: source, path, destination actions. Slice the logits into per-head blocks
        # ([source nodes | dest nodes | path-slot actions]) exactly as in select_action,
        # so the PPO ratio compares like-for-like per-head log probs.
        vone_batch = cast(VONETransition, traj_batch)
        num_nodes = vone_batch.action_mask_s.shape[-1]
        path_dim = vone_batch.action_mask_p.shape[-1]
        source_logits = pi._logits[..., :num_nodes]
        dest_logits = pi._logits[..., num_nodes : 2 * num_nodes]
        path_logits = pi._logits[..., 2 * num_nodes : 2 * num_nodes + path_dim]
        pi_source = distrax.Categorical(
            logits=source_logits + (-1e8 * (1 - vone_batch.action_mask_s.astype(jnp.float32)))
        )
        pi_path = distrax.Categorical(
            logits=path_logits + (-1e8 * (1 - vone_batch.action_mask_p.astype(jnp.float32)))
        )
        pi_dest = distrax.Categorical(
            logits=dest_logits + (-1e8 * (1 - vone_batch.action_mask_d.astype(jnp.float32)))
        )
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
                logits=path_dist._logits + (-1e8 * (1 - traj_batch.action_mask.astype(jnp.float32)))
            )
            path_log_prob = pi_masked.log_prob(path_actions)
            path_entropy = pi_masked.entropy()

        # Decode the k-path index from the flat action. process_path_action needs the env
        # state and hashable static env_params, neither of which is available here; only
        # the path index is needed, so mirror its decode directly (ceil matches
        # init_link_slot_mask / aggregate_slots).
        num_slot_actions = math.ceil(config.link_resources / config.aggregate_slots)
        path_indices = (path_actions // num_slot_actions).astype(jnp.int32)
        # Invert the sampling-time mapping (see LaunchPowerActorCriticMLP.sample_action):
        # stored actions are linear-unit powers; recover the raw sample the distribution's
        # log_prob expects (power-level index for discrete, [0, 1] Beta sample otherwise).
        if config.discrete_launch_power:
            power_actions = jnp.astype(
                jnp.round((to_dbm(power_actions) - config.min_power) / config.step_power),
                jnp.int32,
            )
        else:
            power_actions = jnp.clip(
                (to_dbm(power_actions) - config.min_power) / (config.max_power - config.min_power),
                config.EPSILON,
                1.0 - config.EPSILON,
            )
        # Repeat the power action along the last axis K-paths time
        power_actions = jnp.tile(power_actions[..., None], (1, config.k))
        power_log_prob = power_dist.log_prob(power_actions)
        # Select the chosen path's log prob / entropy, keeping shape (B,) to match
        # path_log_prob and the rollout-time stored log_prob (a (B,1)/(B,k) leftover here
        # would silently broadcast the ratio to (B,B) or fail at trace time).
        power_log_prob = jnp.take_along_axis(power_log_prob, path_indices[:, None], axis=1)[:, 0]
        power_entropy = jnp.take_along_axis(power_dist.entropy(), path_indices[:, None], axis=1)[
            :, 0
        ]

        log_prob = path_log_prob + power_log_prob
        entropy = path_entropy + power_entropy

        if config.DEBUG:
            if targets is not None:
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
    else:
        # Standard action masking. pi is a bare batched Categorical here (logits (B, A));
        # pi[0] would be distrax batch-indexing, i.e. sample 0's logits broadcast to the batch.
        pi_masked = distrax.Categorical(
            logits=pi._logits + (-1e8 * (1 - traj_batch.action_mask.astype(jnp.float32)))
        )
        # Ratio will be policy/masked_policy - also known as off-policy invalid action masking
        log_prob = (
            pi.log_prob(traj_batch.action)
            if config.OFF_POLICY_IAM
            else pi_masked.log_prob(traj_batch.action)
        )
        recenter_clip = config.OFF_POLICY_IAM and config.get("IAM_RECENTER_CLIP", False)
        entropy = pi_masked.entropy()  # Always use the masked entropy, as we want to encourage exploration within the _valid_ action space

    return log_prob, entropy, recenter_clip


def _importance_ratio(
    log_prob: Array,
    traj_batch: RSATransition | VONETransition,
    recenter_clip: bool,
    config: Box,
) -> Array:
    """Clipped importance ratio of the current policy vs the stored behaviour log-probs."""
    log_ratio = log_prob - traj_batch.log_prob
    # Off-policy IAM ratio sits at mu_old (~0.5) not 1 at no-update; subtract log(mu_old)
    # to recenter on pi_new/pi_old (~1) so the downstream clipping is symmetric for both
    # advantage signs.
    if recenter_clip:
        log_ratio = log_ratio - jnp.log(traj_batch.valid_mass.astype(jnp.float32) + 1e-8)
    log_ratio = jnp.clip(log_ratio, -config.LOGR_CLIP, config.LOGR_CLIP)
    return jnp.exp(log_ratio)


def _recompute_vtrace_advantages(
    train_state: TrainState,
    traj_batch: RSATransition | VONETransition,
    last_val: Array,
    config: Box,
) -> Tuple[Array, Array]:
    """Recompute VTrace advantages and value targets over the temporally-ordered rollout.

    Runs a fresh forward pass with the current policy to get importance ratios, then the
    reverse-time scan over the ordered (ROLLOUT_LENGTH, NUM_ENVS) rollout with the true
    bootstrap value V(s_{T+1}) captured after the rollout. Called at the start of each
    update epoch, *before* prioritized resampling / minibatch shuffling can permute
    temporal order, mirroring how the standard-GAE path computes advantages once on
    ordered data in _env_rollout_advantages.

    Importance ratios are therefore refreshed once per epoch (tracking policy drift
    across UPDATE_EPOCHS) and frozen across the minibatches within an epoch. The scan
    uses the stored behaviour-policy values (traj_batch.value / last_val), consistent
    with standard VTrace.

    (NOTE: IAM_RECENTER_CLIP also recenters this VTrace importance ratio; whether it
    should be recentered here too is an open question - see PR note. Off by default:
    RHO_CLIP<=0.)
    """
    model = eqx.combine(train_state.model_params, train_state.model_static)
    model = cast_model_for_compute(model)
    # Flatten (ROLLOUT_LENGTH, NUM_ENVS, ...) -> (ROLLOUT_LENGTH * NUM_ENVS, ...) for the
    # batched forward pass (same layout _sample_prioritized_batch flattens to later).
    flat_batch = (
        jax.tree.map(
            lambda x: x.reshape((config.ROLLOUT_LENGTH * config.NUM_ENVS,) + x.shape[2:]),
            traj_batch,
        )
        if config.NUM_ENVS > 1
        else traj_batch
    )
    axes = (0, None) if config.USE_GNN or config.USE_TRANSFORMER else (0,)
    pi, _ = jax.vmap(model, in_axes=axes)(*flat_batch.obs)
    log_prob, _, recenter_clip = _policy_log_prob_entropy(pi, flat_batch, config)
    ratio = _importance_ratio(log_prob, flat_batch, recenter_clip, config)
    # Restore time-major (ROLLOUT_LENGTH, NUM_ENVS) layout for the reverse-time scan
    ratio = ratio.reshape(traj_batch.reward.shape)
    adv, targets, _ = jit_profiler.call(
        config.PROFILE,
        _calculate_puffer_advantage,
        train_state,
        traj_batch,
        last_val,
        ratio,
        config,
    )
    return adv, targets


@eqx.filter_value_and_grad(has_aux=True)
def _loss_fn(
    model: eqx.Module,
    train_state: TrainState,
    batch_info: Tuple[RSATransition | VONETransition, Array, Array, Array],
    config: Box,
) -> Tuple[
    Array,
    Tuple[Array, Array, Array, Array, Array, Array, Array, Array, LossDiagnostics],
]:
    """
    Compute PPO loss (actor + value + entropy).
    """
    traj_batch, adv, targets, importance_weights = batch_info
    # RERUN NETWORK - with Equinox, vmap the model directly.
    # Mixed-precision compute: cast the (float32 master) weights to COMPUTE_DTYPE for the forward.
    # This is inside the differentiated region, so gradients flow back to the float32 master.
    model = cast_model_for_compute(model)
    axes = (0, None) if config.USE_GNN or config.USE_TRANSFORMER else (0,)
    pi, value = jax.vmap(model, in_axes=axes)(*traj_batch.obs)

    # HANDLE DIFFERENT ACTION TYPES FOR OPTICAL NETWORKS
    log_prob, entropy, recenter_clip = _policy_log_prob_entropy(pi, traj_batch, config, targets)
    ratio = _importance_ratio(log_prob, traj_batch, recenter_clip, config)

    # NOTE: under VTrace-style clipping (RHO_CLIP > 0 and C_CLIP > 0) the advantages and
    # value targets in batch_info were recomputed at the start of the epoch over the
    # temporally-ordered rollout (see _recompute_vtrace_advantages), so no in-loss
    # advantage recomputation happens here: the minibatch may be freely resampled or
    # shuffled without breaking the reverse-time scan.

    # --- Per-step weight for actor + entropy losses ------------------------------
    mask_sum = jnp.sum(traj_batch.action_mask, axis=-1)

    # Hard gate: zero weight for steps with fewer than IAM_GATING_MIN_ACTIONS valid actions
    if config.IAM_GATING:
        w = (mask_sum >= config.IAM_GATING_MIN_ACTIONS).astype(jnp.float32)
    else:
        w = jnp.ones_like(mask_sum, dtype=jnp.float32)

    # Soft damping: linearly reduce weight when valid mass < VALID_MASS_TARGET
    valid_mass = traj_batch.valid_mass.astype(jnp.float32)
    if config.IAM_DAMPING:
        damp = jnp.clip(valid_mass / config.VALID_MASS_TARGET, 0.0, 1.0)
        w = w * damp
    w_sum = jnp.maximum(w.sum(), 1e-8)  # just for numerical stability

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

    # Self-imitation: optionally drop the policy-gradient term for negative-advantage steps.
    actor_w = (
        w * (adv_weighted > 0).astype(jnp.float32) if config.get("POSITIVE_ADV_ONLY", False) else w
    )
    # Optional per-state valid-mass weighting (numerator only; still normalised by the unweighted
    # count): restores the congestion-aware mu-scaling that the non-recentered ratio (rho ~ mu)
    # applies implicitly but IAM_RECENTER_CLIP removes.
    actor_num = actor_w * valid_mass if config.get("MU_WEIGHT_ACTOR", False) else actor_w
    # Normalize by the full gated count (w_sum), not the positive-only count, so the gradient
    # scale under POSITIVE_ADV_ONLY matches the emergent non-recentered clip (which averages the
    # zeroed negative-adv steps into the denominator) and LR is comparable across the two modes.
    actor_loss = -(jnp.minimum(loss_actor1, loss_actor2) * actor_num).sum() / w_sum

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
            vone_batch = cast(VONETransition, traj_batch)
            num_nodes = vone_batch.action_mask_s.shape[-1]
            path_dim = vone_batch.action_mask_p.shape[-1]
            path_logits = pi._logits[..., 2 * num_nodes : 2 * num_nodes + path_dim]
            current_probs = jax.nn.softmax(path_logits, axis=-1)
            current_valid_mass = jnp.sum(
                current_probs * vone_batch.action_mask_p.astype(jnp.float32), axis=-1
            )
        elif config.env_type.lower() == "rsa_gn_model" and config.launch_power_type == "rl":
            current_probs = jax.nn.softmax(pi[0]._logits, axis=-1)
            current_valid_mass = jnp.sum(current_probs * traj_batch.action_mask, axis=-1)
        elif config.OFF_POLICY_IAM:
            current_probs = jax.nn.softmax(pi._logits, axis=-1)
            current_valid_mass = jnp.sum(current_probs * traj_batch.action_mask, axis=-1)
        else:
            current_probs = jax.nn.softmax(pi._logits, axis=-1)
            current_valid_mass = jnp.sum(current_probs * traj_batch.action_mask, axis=-1)
        validmass_loss = -jnp.log(current_valid_mass + 1e-8).mean()
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

    # Compute enhanced diagnostics if enabled
    if config.ENHANCED_LOGGING:
        # valid_frac: fraction of transitions with meaningful policy gradient signal
        valid_frac = w.sum() / w.shape[0]
        # clip_frac: how often PPO clipping activates
        clip_frac = (jnp.abs(ratio - 1.0) > config.CLIP_EPS).mean()
        # ratio statistics
        ratio_mean = ratio.mean()
        ratio_std = ratio.std()
        ratio_min = ratio.min()
        ratio_max = ratio.max()
        # valid_mass statistics
        valid_mass_mean = valid_mass.mean()
        valid_mass_std = valid_mass.std()
        valid_mass_min = valid_mass.min()
        valid_mass_max = valid_mass.max()
        # n_valid: number of valid actions per step
        n_valid = traj_batch.action_mask.sum(axis=-1)
        n_valid_mean = n_valid.mean()
        n_valid_min = n_valid.min()
        n_valid_max = n_valid.max()
        # adv_mean_raw, adv_std_raw: advantage statistics before normalization
        adv_mean_raw = adv.mean()
        adv_std_raw = adv.std()
        # gate_frac: fraction of steps with >= 2 valid actions
        gate_choice = (mask_sum > 1).astype(jnp.float32)
        gate_frac = gate_choice.mean()
        # log_prob and entropy weighted by gate_choice (steps with >= 2 valid actions)
        log_prob_choice = (log_prob * gate_choice).sum() / jnp.maximum(gate_choice.sum(), 1.0)
        entropy_choice = (entropy * gate_choice).sum() / jnp.maximum(gate_choice.sum(), 1.0)
        # log_prob minimum
        log_prob_min = log_prob.min()
        # invalid action taken fraction
        N = traj_batch.action.shape[0]
        taken_valid = traj_batch.action_mask[jnp.arange(N), traj_batch.action].astype(jnp.float32)
        invalid_taken_frac = 1.0 - taken_valid.mean()
        taken_valid_min = taken_valid.min()
        # recentered ratio pi_new/pi_old (computed regardless of IAM_RECENTER_CLIP so the two
        # modes are comparable): ~1 when recentering would centre the clip, vs ratio ~ mu_old.
        recenter_ratio = jnp.exp(
            jnp.clip(
                log_prob - traj_batch.log_prob - jnp.log(valid_mass + 1e-8),
                -config.LOGR_CLIP,
                config.LOGR_CLIP,
            )
        )
        recenter_ratio_mean = recenter_ratio.mean()
        recenter_ratio_std = recenter_ratio.std()
        # fraction of weighted negative-advantage steps whose actor term is clipped (no gradient):
        # high in unit mode (ratio floored below 1-eps), low once the clip is recentered.
        neg = (adv_weighted < 0).astype(jnp.float32) * w
        neg_clipped = neg * (loss_actor2 < loss_actor1).astype(jnp.float32)
        neg_adv_clip_frac = neg_clipped.sum() / jnp.maximum(neg.sum(), 1.0)
        # fraction of weighted valid steps with positive advantage = the steps the actor learns
        # from under POSITIVE_ADV_ONLY (and the unclipped side of the surrogate).
        frac_pos_adv = ((adv_weighted > 0).astype(jnp.float32) * w).sum() / w_sum

        diagnostics = LossDiagnostics(
            valid_frac=valid_frac,
            clip_frac=clip_frac,
            ratio_mean=ratio_mean,
            ratio_std=ratio_std,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            valid_mass_mean=valid_mass_mean,
            valid_mass_std=valid_mass_std,
            valid_mass_min=valid_mass_min,
            valid_mass_max=valid_mass_max,
            n_valid_mean=n_valid_mean,
            n_valid_min=n_valid_min,
            n_valid_max=n_valid_max,
            adv_mean_raw=adv_mean_raw,
            adv_std_raw=adv_std_raw,
            gate_frac=gate_frac,
            log_prob_choice=log_prob_choice,
            entropy_choice=entropy_choice,
            log_prob_min=log_prob_min,
            invalid_taken_frac=invalid_taken_frac,
            taken_valid_min=taken_valid_min,
            recenter_ratio_mean=recenter_ratio_mean,
            recenter_ratio_std=recenter_ratio_std,
            neg_adv_clip_frac=neg_adv_clip_frac,
            frac_pos_adv=frac_pos_adv,
        )
    else:
        # Placeholder zeros to keep the scan output structure consistent.
        diagnostics = LossDiagnostics.zeros()

    return total_loss, (
        log_prob.mean(),
        ratio.mean(),
        adv_weighted.mean(),
        value_loss,
        actor_loss,
        entropy,
        entropy * ent_coef,
        validmass_loss,
        diagnostics,
    )


def _update_minibatch(
    train_state: TrainState,
    batch_info: Tuple[RSATransition | VONETransition, Array, Array, Array],
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
    (train_state, traj_batch, adv, targets, last_val, rng_step, rng_epoch, priorities) = (
        update_state
    )
    rng_epoch, perm_key = jax.random.split(rng_epoch, 2)

    # VTrace-style clipping: recompute advantages/targets with fresh (current-policy)
    # importance ratios over the temporally-ordered rollout and the true bootstrap value,
    # BEFORE prioritized resampling / shuffling can permute temporal order. The GAE path
    # (RHO_CLIP <= 0 or C_CLIP <= 0) keeps the rollout-time advantages unchanged. The
    # epoch-local values are kept out of the scan carry (which keeps the rollout-time
    # adv/targets) so the carry structure/dtypes stay stable across epochs.
    if config.RHO_CLIP > 0 and config.C_CLIP > 0:
        adv_epoch, targets_epoch = jit_profiler.call(
            config.PROFILE, _recompute_vtrace_advantages, train_state, traj_batch, last_val, config
        )
    else:
        adv_epoch, targets_epoch = adv, targets

    batch = (traj_batch, adv_epoch, targets_epoch)
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
        last_val,
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

    runner_state, traj_batch, adv, targets, last_val, priorities = _env_rollout_advantages(
        runner_state, env, env_params, config
    )
    (train_state, env_state, last_obs, rng_step, rng_epoch) = runner_state

    update_state = (
        train_state,
        traj_batch,
        adv,
        targets,
        last_val,
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
    rng_step = update_state[5]
    rng_epoch = update_state[6]
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

    # Add reward centering diagnostics if enabled
    if config.REWARD_CENTERING:
        loss_info.update(
            {
                "reward_centering/avg_reward": train_state.avg_reward,
                "reward_centering/value_mean": traj_batch.value.mean(),
            }
        )

    # Add enhanced diagnostics if enabled
    if config.ENHANCED_LOGGING:
        diagnostics: LossDiagnostics = loss_info_arrays[1][8]
        loss_info.update(
            {
                f"diagnostics/{name}": getattr(diagnostics, name).reshape(-1)
                for name in LossDiagnostics._fields
            }
        )

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
