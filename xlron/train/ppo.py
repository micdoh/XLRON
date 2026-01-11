import jax
import optax
import jax.numpy as jnp
import distrax
from typing import Any, Callable, Dict, Tuple
from absl import flags
from flax.training.train_state import TrainState
from gymnax.environments import environment
from jax import Array

from xlron.environments.env_funcs import process_path_action
from xlron.environments.gn_model.isrs_gn_model import to_dbm
from xlron.environments.dataclasses import EnvState, EnvParams, VONETransition, RSATransition
from xlron.train.train_utils import *
from xlron import dtype_config


def compute_trajectory_priority_weights(advantages: Array, alpha: float) -> Array:
    """Compute priority weights for entire trajectories based on cumulative absolute advantages.
    
    Args:
        advantages: Array of shape (rollout_length, num_envs)
        alpha: Priority exponent (0 = uniform, 1 = proportional to priority)
    
    Returns:
        Priority weights of shape (num_envs,)
    """
    advantages = advantages.reshape(-1, 1) if advantages.ndim == 1 else advantages
    priority_weights = jnp.abs(advantages).sum(axis=0)  # Sum over rollout dimension
    return jnp.power(priority_weights + 1e-6, alpha)


def compute_sample_priority_weights(advantages: Array, alpha: float) -> Array:
    """Compute priority weights for individual samples based on absolute advantages.
    
    Args:
        advantages: Array of shape (rollout_length, num_envs)
        alpha: Priority exponent (0 = uniform, 1 = proportional to priority)
    
    Returns:
        Priority weights of same shape as advantages
    """
    advantages = advantages.reshape(-1, 1) if advantages.ndim == 1 else advantages
    priority_weights = jnp.abs(advantages)
    return jnp.power(priority_weights + 1e-6, alpha)


def sample_prioritized_batch(
    batch: Tuple,
    priority_weights: Array,
    beta: float,
    rng_key: Array,
    config: flags.FlagValues,
) -> Tuple[Tuple, Array]:
    """Sample minibatches with prioritization and compute importance sampling weights.
    
    Args:
        batch: Tuple of (traj_batch, advantages, targets)
        priority_weights: Priority weights for sampling
        beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        rng_key: JAX random key
        config: Training configuration
    
    Returns:
        Tuple of (minibatches, importance_weights_per_minibatch)
    """
    batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
    assert batch_size == config.ROLLOUT_LENGTH * config.NUM_ENVS, (
        f"batch size (which comprises {config.NUM_MINIBATCHES} of size {config.MINIBATCH_SIZE}) "
        + f"({batch_size}) must be equal to number of steps ({config.ROLLOUT_LENGTH})"
        + f"* number of envs ({config.NUM_ENVS})"
    )
    shuffle_key, sample_key = jax.random.split(rng_key)

    if config.PRIO_ALPHA == 0.0 or config.PRIO_BETA0 == 1.0:
        # Uniform importance if not using prioritized sampling
        importance_weights = jnp.ones((batch_size,), dtype=jnp.float32)
    else:
        priority_probs = (priority_weights + 1e-6) / (jnp.sum(priority_weights) + 1e-6)

        if not config.USE_RNN:
            # If not using RNN, we can prioritize individual samples
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


def get_learner_fn(
    env: environment.Environment,
    env_params: EnvParams,
    train_state: TrainState,
    config: flags.FlagValues,
) -> Callable:

    # TRAIN LOOP
    def _update_step(runner_state, unused):
        # COLLECT TRAJECTORIES

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, step_key, rng_epoch = runner_state

            action_key, next_step_key = jax.random.split(step_key)

            # SELECT ACTION
            select_action_state = (action_key, env_state, last_obs)
            env_state, action, log_prob, value = select_action(
                select_action_state, env, env_params, train_state, config
            )

            # STEP ENV
            obsv, env_state, reward, done, info = env.step(step_key, env_state, action, env_params)

            # Apply reward scaling if configured
            reward = reward * config.REWARD_SCALE

            # PROCESS OBS AND TRANSITION
            obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
            
            # Create transition based on environment type
            if config.env_type.lower() == "vone":
                transition = VONETransition(
                    done, action, value, reward, log_prob, last_obs, info,
                    env_state.env_state.node_mask_s,
                    env_state.env_state.link_slot_mask,
                    env_state.env_state.node_mask_d
                )
            else:
                transition = RSATransition(
                    done, action, value, reward, log_prob, last_obs, info,
                    env_state.env_state.link_slot_mask
                )
            
            runner_state = (train_state, env_state, obsv, next_step_key, rng_epoch)

            # DEBUG LOGGING FOR OPTICAL NETWORKS
            if config.DEBUG:
                path_action = action[0][0] if config.env_type.lower() == "rsa_gn_model" else action
                path_index, slot_index = process_path_action(
                    env_state.env_state, env_params, path_action
                )
                path = env_params.path_link_array[path_index]
                get_path_links = lambda x: jnp.dot(path, x)
                jax.debug.print("state.request_array {}", env_state.env_state.request_array, ordered=config.ORDERED)
                jax.debug.print("action {}", action, ordered=config.ORDERED)
                jax.debug.print("log_prob {}", log_prob, ordered=config.ORDERED)
                jax.debug.print("reward {}", reward, ordered=config.ORDERED)
                jax.debug.print("link_slot_array {}", get_path_links(env_state.env_state.link_slot_array), ordered=config.ORDERED)
                
                if config.env_type.lower() == "vone":
                    jax.debug.print("node_mask_s {}", env_state.env_state.node_mask_s, ordered=config.ORDERED)
                    jax.debug.print("node_mask_d {}", env_state.env_state.node_mask_d, ordered=config.ORDERED)
                    jax.debug.print("action_history {}", env_state.env_state.action_history, ordered=config.ORDERED)
                    jax.debug.print("action_counter {}", env_state.env_state.action_counter, ordered=config.ORDERED)
                    jax.debug.print("request_array {}", env_state.env_state.request_array, ordered=config.ORDERED)
                    jax.debug.print("node_capacity_array {}", env_state.env_state.node_capacity_array, ordered=config.ORDERED)
                elif config.env_type.lower() == "rsa_gn_model":
                    jax.debug.print("modulation_format_index_array {}", get_path_links(env_state.env_state.modulation_format_index_array), ordered=config.ORDERED)
                    jax.debug.print("channel_centre_bw_array {}", get_path_links(env_state.env_state.channel_centre_bw_array), ordered=config.ORDERED)
                    jax.debug.print("link_snr_array {}", get_path_links(env_state.env_state.link_snr_array), ordered=config.ORDERED)
                    jax.debug.print("channel_power_array {}", get_path_links(env_state.env_state.channel_power_array), ordered=config.ORDERED)
            
            return runner_state, transition

        # VECTORISE ENV STEP
        _env_step_vmap = jax.vmap(
            _env_step, in_axes=((None, 0, 0, 0, None), None), out_axes=((None, 0, 0, 0, None), 0)
        ) if config.NUM_ENVS > 1 else _env_step

        rng_step = runner_state[3]
        rng_step, *step_keys = jax.random.split(rng_step, config.NUM_ENVS + 1)
        step_keys = jnp.array(step_keys) if config.NUM_ENVS > 1 else step_keys[0]
        runner_state = runner_state[:3] + (step_keys,) + runner_state[4:]
        runner_state, traj_batch = jax.lax.scan(
            _env_step_vmap, runner_state, None, config.ROLLOUT_LENGTH
        )
        
        if config.DEBUG:
            jax.debug.print("traj_batch.info {}", traj_batch.info, ordered=config.ORDERED)

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, _, rng_epoch = runner_state
        last_obs = (env_state.env_state, env_params) if config.USE_GNN else last_obs
        axes = (None, 0, None) if config.USE_GNN else (None, 0)
        _, last_val = jax.vmap(train_state.apply_fn, in_axes=axes)(
            train_state.params, *last_obs
        ) if (config.NUM_ENVS > 1) else train_state.apply_fn(train_state.params, *last_obs)

        def _calculate_puffer_advantage(traj_batch, last_val, importance_ratio):
            """
            Calculate Puffer Advantage, a generalization of GAE and VTrace.

            Args:
                traj_batch: Trajectory batch containing transitions
                last_val: Value estimate for the last state
                importance_ratio: Importance sampling ratios

            Returns:
                advantages: Computed advantages
                targets: Value targets (advantages + values)
                deltas: TD errors

            Note:
                - When config.RHO_CLIP<=0 and config.C_CLIP<=0, this reduces to standard GAE
                - When lambda=1, this reduces to VTrace
            """
            # Optionally anneal GAE_LAMBDA to higher value to increase horizon
            if config.GAE_LAMBDA is None:
                # Multiply by 3 so that more time spent in high lambda at end of training
                frac = (
                    3
                    * train_state.step
                    / (
                        config.NUM_INCREMENTS
                        * config.NUM_UPDATES
                        * config.LAMBDA_SCHEDULE_MULTIPLIER
                    )
                )
                sech_frac = 1 - 1 / jnp.cosh(frac)
                lambda_delta = config.FINAL_LAMBDA - config.INITIAL_LAMBDA
                current_lambda = config.INITIAL_LAMBDA + (sech_frac * lambda_delta)
            else:
                current_lambda = config.GAE_LAMBDA

            def _get_advantages(gae_and_next_value, transition_and_importance):
                gae, next_value = gae_and_next_value
                transition, importance = transition_and_importance
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                centered_reward = reward - train_state.avg_reward if config.REWARD_CENTERING else reward

                if config.RHO_CLIP <= 0 or config.C_CLIP <= 0:
                    # No clipping applied - standard GAE
                    rho_t = importance
                    c_t = importance
                else:
                    # Apply clipping to importance ratios for VTrace-style advantage
                    rho_t = jnp.minimum(importance, config.RHO_CLIP)
                    c_t = jnp.minimum(importance, config.C_CLIP)

                # Modified TD error calculation with importance sampling
                delta = rho_t * (centered_reward + config.GAMMA * next_value * (1 - done) - value)

                # Modified GAE accumulation with clipped importance ratios
                gae = delta + config.GAMMA * current_lambda * c_t * (1 - done) * gae

                return (gae, value), (gae, delta)

            _, (advantages, deltas) = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                (traj_batch, importance_ratio),
                reverse=True,
                unroll=True,
            )
            return advantages, advantages + traj_batch.value, deltas

        # Compute advantages with initial uniform importance ratios
        initial_importance_ratio = jnp.ones_like(traj_batch.reward)
        advantages, targets, deltas = _calculate_puffer_advantage(
            traj_batch, last_val, initial_importance_ratio
        )

        # REWARD CENTERING
        if config.REWARD_CENTERING:
            train_state = train_state.update_step_size()
            # Extract the one-step TD errors (deltas) from your GAE calculation
            updated_avg_reward = train_state.avg_reward + train_state.reward_stepsize * jnp.mean(deltas)
            # This makes the estimate robust to initialization
            adjustment = train_state.avg_reward - updated_avg_reward
            targets = targets + adjustment
            train_state = train_state.replace(avg_reward=updated_avg_reward)

        # COMPUTE PRIORITIES AND ANNEALED BETA
        if config.PRIO_ALPHA > 0:
            priorities = (
                compute_sample_priority_weights(advantages, config.PRIO_ALPHA)
                if not config.USE_RNN
                else compute_trajectory_priority_weights(advantages, config.PRIO_ALPHA)
            )
            # Anneal beta from initial value to 1.0 over course of training
            progress = (
                train_state.prio_alpha * train_state.step / (config.NUM_UPDATES * config.NUM_INCREMENTS)
            )
            annealed_beta = config.PRIO_BETA0 + (1.0 - config.PRIO_BETA0) * progress
        else:
            # No prioritization
            priorities = jnp.ones_like(advantages)
            annealed_beta = 1.0

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):

            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets, importance_weights = batch_info

                def _loss_fn(params, traj_batch, gae, targets, importance_weights):
                    # RERUN NETWORK
                    axes = (None, 0, None) if config.USE_GNN else (None, 0)
                    pi, value = jax.vmap(train_state.apply_fn, in_axes=axes)(params, *traj_batch.obs)

                    # HANDLE DIFFERENT ACTION TYPES FOR OPTICAL NETWORKS
                    if config.env_type.lower() == "vone":
                        # VONE: source, path, destination actions
                        pi_source = distrax.Categorical(
                            logits=jnp.where(traj_batch.action_mask_s, pi._logits, -1e8)
                        )
                        pi_path = distrax.Categorical(
                            logits=jnp.where(traj_batch.action_mask_p, pi._logits, -1e8)
                        )
                        pi_dest = distrax.Categorical(
                            logits=jnp.where(traj_batch.action_mask_d, pi._logits, -1e8)
                        )
                        action_s = traj_batch.action[:, 0]
                        action_p = traj_batch.action[:, 1]
                        action_d = traj_batch.action[:, 2]
                        log_prob_source = pi_source.log_prob(action_s)
                        log_prob_path = pi_path.log_prob(action_p)
                        log_prob_dest = pi_dest.log_prob(action_d)
                        log_prob = log_prob_source + log_prob_path + log_prob_dest
                        entropy = pi_source.entropy().mean() + pi_path.entropy().mean() + pi_dest.entropy().mean()

                    elif config.ACTION_MASKING:
                        # Standard action masking
                        pi_masked = distrax.Categorical(
                            logits=jnp.where(traj_batch.action_mask, pi[0]._logits, -1e8)
                        )
                        log_prob = pi_masked.log_prob(traj_batch.action)
                        entropy = pi_masked.entropy().mean()

                    elif config.env_type.lower() == "rsa_gn_model" and config.launch_power_type == "rl":
                        # RSA with power control
                        path_actions = traj_batch.action[..., 0]
                        power_actions = traj_batch.action[..., 1]
                        path_dist, power_dist = pi
                        path_log_prob = path_entropy = 0.0
                        
                        if config.GNN_OUTPUT_RSA:
                            pi_masked = distrax.Categorical(
                                logits=jnp.where(traj_batch.action_mask, path_dist._logits, -1e8)
                            )
                            path_log_prob = pi_masked.log_prob(path_actions)
                            path_entropy = pi_masked.entropy().mean()

                        path_indices = jax.vmap(process_path_action, in_axes=(0, None, 0))(
                            traj_batch.obs[0], env_params, path_actions
                        )[0]
                        # Re-scale action from [min_power, max_power] to [0, 1]
                        power_actions = jnp.astype(
                            (to_dbm(power_actions) - env_params.min_power) / env_params.step_power,
                            jnp.int32
                        )
                        # Repeat the power action along the last axis K-paths time
                        power_actions = jnp.tile(power_actions[..., None], (1, env_params.k_paths))
                        power_log_prob = power_dist.log_prob(power_actions)
                        # Slice log prob to just take the path index
                        power_log_prob = jax.vmap(
                            lambda x, i: jax.lax.dynamic_slice(x, (i,), (1,))
                        )(power_log_prob, path_indices)
                        power_entropy = power_dist.entropy().mean()

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

                    else:
                        # Standard action
                        log_prob = pi.log_prob(traj_batch.action)
                        entropy = pi.entropy().mean()

                    # CALCULATE IMPORTANCE RATIO FOR VTRACE-STYLE RECALCULATION
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    
                    # Recalculate the advantage now if using VTrace-style clipping
                    if config.RHO_CLIP > 0 and config.C_CLIP > 0:
                        minibatch_size = config.MINIBATCH_SIZE
                        assert config.ROLLOUT_LENGTH % config.NUM_MINIBATCHES == 0, (
                            "ROLLOUT_LENGTH must be integer multiple of NUM_MINIBATCHES"
                        )
                        traj_batch_reshaped, value_reshaped, ratio_reshaped = jax.tree.map(
                            lambda x: x.reshape(
                                (config.ROLLOUT_LENGTH // config.NUM_MINIBATCHES, config.NUM_ENVS)
                                + x.shape[1:]
                            ),
                            (traj_batch, value, ratio),
                        )
                        gae, _, _ = _calculate_puffer_advantage(
                            traj_batch_reshaped, value_reshaped[-1], ratio_reshaped
                        )
                        gae, traj_batch, value, ratio = jax.tree.map(
                            lambda x: x.reshape((minibatch_size,) + x.shape[2:]),
                            (gae, traj_batch_reshaped, value_reshaped, ratio_reshaped),
                        )

                    # Apply importance weights to correct for prioritized sampling bias
                    gae_normalized = (gae - gae.mean()) / (gae.std() + 1e-8)
                    adv_corrected = importance_weights * gae_normalized

                    # CALCULATE ACTOR LOSS
                    actor_loss1 = ratio * adv_corrected
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.CLIP_EPS,
                            1.0 + config.CLIP_EPS,
                        )
                        * adv_corrected
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                    actor_loss = actor_loss.mean()

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config.CLIP_EPS, config.CLIP_EPS)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # Calculate current entropy coefficient based on training step
                    ent_coef = train_state.ent_schedule(train_state.step)

                    total_loss = (
                        actor_loss
                        + config.VF_COEF * value_loss
                        - ent_coef * entropy
                    )

                    if config.DEBUG:
                        jax.debug.print("log_prob {}", log_prob, ordered=config.ORDERED)
                        jax.debug.print("entropy {}", entropy, ordered=config.ORDERED)
                        jax.debug.print("ratio {}", ratio, ordered=config.ORDERED)
                        jax.debug.print("gae {}", gae, ordered=config.ORDERED)
                        jax.debug.print("actor_loss1 {}", actor_loss1, ordered=config.ORDERED)
                        jax.debug.print("actor_loss2 {}", actor_loss2, ordered=config.ORDERED)
                        jax.debug.print("value_loss {}", value_loss, ordered=config.ORDERED)
                        jax.debug.print("actor_loss {}", actor_loss, ordered=config.ORDERED)
                        jax.debug.print("entropy {}", entropy, ordered=config.ORDERED)
                        jax.debug.print("total_loss {}", total_loss, ordered=config.ORDERED)

                    return total_loss, (
                        log_prob.mean(),
                        ratio.mean(),
                        gae_normalized.mean(),
                        value_loss,
                        actor_loss,
                        entropy,
                        entropy * ent_coef,
                    )

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets, importance_weights
                )
                train_state = train_state.apply_gradients(grads=grads)
                
                if config.DEBUG:
                    grad_norm = optax.global_norm(grads)
                    jax.debug.print("gradient_norm {}", grad_norm, ordered=config.ORDERED)
                
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng_step, rng_epoch, priorities = update_state
            rng_epoch, perm_key = jax.random.split(rng_epoch, 2)

            batch = (traj_batch, advantages, targets)
            
            # Sample prioritized batch with importance weights
            minibatches, importance_weights_mb = sample_prioritized_batch(
                batch, priorities, annealed_beta, perm_key, config
            )
            
            minibatches_with_weights = (*minibatches, importance_weights_mb)

            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches_with_weights
            )
            
            update_state = (
                train_state,
                traj_batch,
                advantages,
                targets,
                rng_step,
                rng_epoch,
                priorities,
            )
            return update_state, total_loss

        update_state = (
            train_state,
            traj_batch,
            advantages,
            targets,
            rng_step,
            rng_epoch,
            priorities,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.UPDATE_EPOCHS
        )
        train_state = update_state[0]
        metric = traj_batch.info
        rng_step = update_state[4]
        rng_epoch = update_state[5]
        runner_state = (train_state, env_state, last_obs, rng_step, rng_epoch)
        
        loss_info_dict = {
            "loss/total_loss": loss_info[0].reshape(-1),
            "loss/log_prob": loss_info[1][0].reshape(-1),
            "loss/ratio": loss_info[1][1].reshape(-1),
            "loss/gae": loss_info[1][2].reshape(-1),
            "loss/value_loss": loss_info[1][3].reshape(-1),
            "loss/value_loss_scaled": loss_info[1][3].reshape(-1) * config.VF_COEF,
            "loss/actor_loss": loss_info[1][4].reshape(-1),
            "loss/entropy": loss_info[1][5].reshape(-1),
            "loss/entropy_loss_scaled": loss_info[1][6].reshape(-1),
        }
        
        # Add prioritization metrics if using prioritized sampling
        if config.PRIO_ALPHA > 0:
            loss_info_dict.update({
                "prioritization/beta": annealed_beta,
                "prioritization/priority_mean": jnp.mean(priorities),
                "prioritization/priority_std": jnp.std(priorities),
            })

        if config.DEBUG:
            jax.debug.print("metric {}", metric, ordered=config.ORDERED)

        return runner_state, (metric, loss_info_dict)

    def learner_fn(update_state):
        train_state, (metric_info, loss_info) = jax.lax.scan(
            _update_step, update_state, None, config.NUM_UPDATES
        )
        return {"runner_state": train_state, "metrics": metric_info, "loss_info": loss_info}

    return learner_fn