import jax
import optax
import jax.numpy as jnp
from absl import flags
from flax.training.train_state import TrainState
from gymnax.environments import environment
from tensorflow_probability.substrates.jax.distributions.student_t import entropy

from xlron.environments.env_funcs import process_path_action
from xlron.environments.gn_model.isrs_gn_model import to_dbm
from xlron.environments.dataclasses import EnvState, EnvParams, VONETransition, RSATransition
from xlron.train.train_utils import *


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
            train_state, env_state, last_obs, rng_step, rng_epoch = runner_state

            rng_step, action_key, step_key = jax.random.split(rng_step, 3)

            # SELECT ACTION
            action_key = jax.random.split(action_key, config.NUM_ENVS)
            select_action_fn = lambda x: select_action(x, env, env_params, train_state, config)
            select_action_fn = jax.vmap(select_action_fn)
            select_action_state = (action_key, env_state, last_obs)
            env_state, action, log_prob, value = select_action_fn(select_action_state)

            # STEP ENV
            step_key = jax.random.split(step_key, config.NUM_ENVS)
            step_fn = lambda x, y, z: env.step(x, y, z, env_params)
            step_fn = jax.vmap(step_fn)
            obsv, env_state, reward, done, info = step_fn(step_key, env_state, action)

            obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
            transition = VONETransition(
                done, action, value, reward, log_prob, last_obs, info, env_state.env_state.node_mask_s,
                env_state.env_state.link_slot_mask,
                env_state.env_state.node_mask_d
            ) if config.env_type.lower() == "vone" else RSATransition(
                done, action, value, reward, log_prob, last_obs, info, env_state.env_state.link_slot_mask
            )
            runner_state = (train_state, env_state, obsv, rng_step, rng_epoch)

            if config.DEBUG:
                path_action = action[0][0] if config.env_type.lower() == "rsa_gn_model" else action
                path_index, slot_index = process_path_action(env_state.env_state, env_params, path_action)
                path = env_params.path_link_array[path_index]
                get_path_links = lambda x: jnp.dot(path, x)
                jax.debug.print("state.request_array {}", env_state.env_state.request_array, ordered=config.ORDERED)
                jax.debug.print("action {}", action, ordered=config.ORDERED)
                jax.debug.print("log_prob {}", log_prob, ordered=config.ORDERED)
                jax.debug.print("reward {}", reward, ordered=config.ORDERED)
                jax.debug.print("link_slot_array {}", get_path_links(env_state.env_state.link_slot_array), ordered=config.ORDERED)
                #jax.debug.print("link_slot_mask {}", env_state.env_state.link_slot_mask, ordered=config.ORDERED)
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

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config.ROLLOUT_LENGTH
        )
        if config.DEBUG:
            jax.debug.print("traj_batch.info {}", traj_batch.info, ordered=config.ORDERED)

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, rng_step, rng_epoch = runner_state
        last_obs = (env_state.env_state, env_params) if config.USE_GNN else last_obs
        axes = (None, 0, None) if config.USE_GNN else (None, 0)
        _, last_val = jax.vmap(train_state.apply_fn, in_axes=axes)(train_state.params, *last_obs)

        def _calculate_gae(traj_batch, last_val):
            if config.GAE_LAMBDA is None:
                # Multiply by 3 so that more time spent in high lambda at end of training
                frac = 3 * train_state.update_step / (config.NUM_UPDATES * config.LAMBDA_SCHEDULE_MULTIPLIER)
                sech_frac = 1 - 1/jnp.cosh(frac)
                lambda_delta = config.FINAL_LAMBDA - config.INITIAL_LAMBDA
                current_lambda = config.INITIAL_LAMBDA + (sech_frac * lambda_delta)
            else:
                current_lambda = config.GAE_LAMBDA
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                centered_reward = reward - train_state.avg_reward if config.REWARD_CENTERING else reward
                delta = centered_reward + config.GAMMA * next_value * (1 - done) - value
                gae = (
                    delta
                    + config.GAMMA * current_lambda * (1 - done) * gae
                )
                return (gae, value), (gae, delta)

            _, (advantages, deltas) = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=True,
            )
            return advantages, advantages + traj_batch.value, deltas

        advantages, targets, deltas = _calculate_gae(traj_batch, last_val)

        if config.REWARD_CENTERING:
            train_state = train_state.update_step_size()
            # Extract the one-step TD errors (deltas) from your GAE calculation
            updated_avg_reward = train_state.avg_reward + train_state.reward_stepsize * jnp.mean(deltas)
            # This makes the estimate robust to initialization
            adjustment = train_state.avg_reward - updated_avg_reward
            targets = targets + adjustment
            train_state = train_state.replace(avg_reward=updated_avg_reward)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):

            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    axes = (None, 0, None) if config.USE_GNN else (None, 0)
                    pi, value = jax.vmap(train_state.apply_fn, in_axes=axes)(params, *traj_batch.obs)

                    if config.env_type.lower() == "vone":
                        # TODO - change this to use logits not separate pi
                        pi_source = distrax.Categorical(
                            logits=jnp.where(traj_batch.action_mask_s, pi._logits, -1e8))
                        pi_path = distrax.Categorical(
                            logits=jnp.where(traj_batch.action_mask_p, pi._logits, -1e8))
                        pi_dest = distrax.Categorical(
                            logits=jnp.where(traj_batch.action_mask_d, pi._logits, -1e8))
                        action_s = traj_batch.action[:, 0]
                        action_p = traj_batch.action[:, 1]
                        action_d = traj_batch.action[:, 2]
                        log_prob_source = pi_source.log_prob(action_s)
                        log_prob_path = pi_path.log_prob(action_p)
                        log_prob_dest = pi_dest.log_prob(action_d)
                        log_prob = log_prob_source + log_prob_path + log_prob_dest
                        entropy = pi_source.entropy().mean() + pi_path.entropy().mean() + pi_dest.entropy().mean()

                    elif config.ACTION_MASKING:
                        pi_masked = distrax.Categorical(logits=jnp.where(traj_batch.action_mask, pi[0]._logits, -1e8))
                        log_prob = pi_masked.log_prob(traj_batch.action)
                        entropy = pi_masked.entropy().mean()

                    elif config.env_type.lower() == "rsa_gn_model" and config.launch_power_type == "rl":
                        path_actions = traj_batch.action[..., 0]
                        power_actions = traj_batch.action[..., 1]
                        path_dist, power_dist = pi
                        path_log_prob = path_entropy = 0.0
                        if config.GNN_OUTPUT_RSA:
                            pi_masked = distrax.Categorical(logits=jnp.where(traj_batch.action_mask, path_dist._logits, -1e8))
                            path_log_prob = pi_masked.log_prob(path_actions)
                            path_entropy = pi_masked.entropy().mean()


                        path_indices = jax.vmap(process_path_action, in_axes=(0, None, 0))(traj_batch.obs[0], env_params, path_actions)[0]
                        # Re-scale action from [min_power, max_power] to [0, 1]
                        power_actions = jnp.astype(
                            (to_dbm(power_actions) - env_params.min_power) / env_params.step_power,
                            jnp.int32
                        )
                        # Repeat the power action along the last axis K-paths time
                        power_actions = jnp.tile(power_actions[..., None], (1, env_params.k_paths))
                        power_log_prob = power_dist.log_prob(power_actions)
                        # Slice log prob to just take the path index
                        power_log_prob = jax.vmap(lambda x, i: jax.lax.dynamic_slice(x, (i,), (1,)))(power_log_prob, path_indices)
                        power_entropy = power_dist.entropy().mean()


                        #power_log_prob = power_dist.log_prob(power_actions)
                        #power_entropy = power_dist.entropy().mean()
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
                        log_prob = pi.log_prob(traj_batch.action)
                        entropy = pi.entropy().mean()

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config.CLIP_EPS, config.CLIP_EPS)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.CLIP_EPS,
                            1.0 + config.CLIP_EPS,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()

                    total_loss = (
                        loss_actor
                        + config.VF_COEF * value_loss
                        - config.ENT_COEF * entropy
                    )

                    if config.DEBUG:
                        jax.debug.print("log_prob {}", log_prob, ordered=config.ORDERED)
                        jax.debug.print("entropy {}", entropy, ordered=config.ORDERED)
                        jax.debug.print("ratio {}", ratio, ordered=config.ORDERED)
                        jax.debug.print("gae {}", gae, ordered=config.ORDERED)
                        jax.debug.print("loss_actor1 {}", loss_actor1, ordered=config.ORDERED)
                        jax.debug.print("loss_actor2 {}", loss_actor2, ordered=config.ORDERED)
                        jax.debug.print("value_loss {}", value_loss, ordered=config.ORDERED)
                        jax.debug.print("loss_actor {}", loss_actor, ordered=config.ORDERED)
                        jax.debug.print("entropy {}", entropy, ordered=config.ORDERED)
                        jax.debug.print("total_loss {}", total_loss, ordered=config.ORDERED)

                    return total_loss, (log_prob.mean(), ratio.mean(), gae.mean(), value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                if config.DEBUG:
                    grad_norm = optax.global_norm(grads)
                    jax.debug.print("gradient_norm {}", grad_norm)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng_step, rng_epoch = update_state
            rng_epoch, perm_key = jax.random.split(rng_epoch, 2)
            batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
            assert (
                batch_size == config.ROLLOUT_LENGTH * config.NUM_ENVS
            ), "batch size must be equal to number of steps * number of envs * number of devices"
            permutation = jax.random.permutation(perm_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree.map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            runner_state = (train_state, traj_batch, advantages, targets, rng_step, rng_epoch)
            return runner_state, total_loss

        update_state = (train_state, traj_batch, advantages, targets, rng_step, rng_epoch)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.UPDATE_EPOCHS
        )
        train_state = update_state[0]
        metric = traj_batch.info
        rng_step = update_state[4]
        rng_epoch = update_state[5]
        runner_state = (train_state, env_state, last_obs, rng_step, rng_epoch)
        loss_info = {
            "loss/total_loss": loss_info[0].reshape(-1),
            "loss/log_prob": loss_info[1][0].reshape(-1),
            "loss/ratio": loss_info[1][1].reshape(-1),
            "loss/gae": loss_info[1][2].reshape(-1),
            "loss/value_loss": loss_info[1][3].reshape(-1),
            "loss/loss_actor": loss_info[1][4].reshape(-1),
            "loss/entropy": loss_info[1][5].reshape(-1),
        }

        if config.DEBUG:
            jax.debug.print("metric {}", metric, ordered=config.ORDERED)

        return runner_state, (metric, loss_info)

    def learner_fn(update_state):

        train_state, (metric_info, loss_info) = jax.lax.scan(
            _update_step, update_state, None, config.NUM_UPDATES
        )
        return {"runner_state": train_state, "metrics": metric_info, "loss_info": loss_info}

    return learner_fn
