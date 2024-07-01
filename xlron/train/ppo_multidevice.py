import os
import math
import absl
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import distrax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple
from flax.training.train_state import TrainState
from xlron.environments.env_funcs import *
from xlron.environments.wrappers import LogWrapper
from xlron.environments.vone import make_vone_env
from xlron.environments.rsa import make_rsa_env
from xlron.models.models import ActorCriticGNN, ActorCriticMLP
from xlron.environments.dataclasses import EnvState, EnvParams, VONETransition, RSATransition
from xlron.train.train_utils import *


def get_learner_fn(
    env: environment.Environment,
    env_params: EnvParams,
    train_state: TrainState,
    config: absl.flags.FlagValues,
) -> Callable:

    # TRAIN LOOP
    def _update_step(runner_state, unused):
        # COLLECT TRAJECTORIES

        runner_state, env_state, last_obs, rng = runner_state
        rng, rng_epoch, rng_step = jax.random.split(rng, 3)

        # Add rngs to runner_state tuple
        runner_state = (runner_state, env_state, last_obs, rng, rng_epoch, rng_step)

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, rng, rng_epoch, rng_step = runner_state

            rng_step, action_key, step_key = jax.random.split(rng_step, 3)

            # SELECT ACTION
            action_key = jax.random.split(action_key, config.NUM_DEVICES*config.NUM_ENVS)
            action_key = reshape_keys(action_key, config.NUM_DEVICES, config.NUM_ENVS)
            select_action_fn = lambda x: select_action(x, env, env_params, train_state, config)
            select_action_fn = jax.vmap(select_action_fn, axis_name='env')
            select_action_fn = jax.pmap(select_action_fn, axis_name='device')
            select_action_state = (action_key, env_state, last_obs)
            action, log_prob, value = select_action_fn(select_action_state)

            # STEP ENV
            step_key = jax.random.split(step_key, config.NUM_DEVICES*config.NUM_ENVS)
            step_key = reshape_keys(step_key, config.NUM_DEVICES, config.NUM_ENVS)
            step_fn = lambda x, y, z: env.step(x, y, z, env_params)
            step_fn = jax.vmap(step_fn, axis_name='env')
            step_fn = jax.pmap(step_fn, axis_name='device')
            obsv, env_state, reward, done, info = step_fn(step_key, env_state, action)

            obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
            transition = VONETransition(
                done, action, value, reward, log_prob, last_obs, info, env_state.env_state.node_mask_s,
                env_state.env_state.link_slot_mask,
                env_state.env_state.node_mask_d
            ) if config.env_type.lower() == "vone" else RSATransition(
                done, action, value, reward, log_prob, last_obs, info, env_state.env_state.link_slot_mask
            )
            runner_state = (train_state, env_state, obsv, rng, rng_epoch, rng_step)

            if config.DEBUG:
                jax.debug.print("log_prob {}", log_prob, ordered=config.ORDERED)
                jax.debug.print("action {}", action, ordered=config.ORDERED)
                jax.debug.print("reward {}", reward, ordered=config.ORDERED)
                #jax.debug.print("link_slot_array {}", env_state.env_state.link_slot_array, ordered=config.ORDERED)
                #jax.debug.print("link_slot_mask {}", env_state.env_state.link_slot_mask, ordered=config.ORDERED)
                if config.env_type.lower() == "vone":
                    jax.debug.print("node_mask_s {}", env_state.env_state.node_mask_s, ordered=config.ORDERED)
                    jax.debug.print("node_mask_d {}", env_state.env_state.node_mask_d, ordered=config.ORDERED)
                    jax.debug.print("action_history {}", env_state.env_state.action_history, ordered=config.ORDERED)
                    jax.debug.print("action_counter {}", env_state.env_state.action_counter, ordered=config.ORDERED)
                    jax.debug.print("request_array {}", env_state.env_state.request_array, ordered=config.ORDERED)
                    jax.debug.print("node_capacity_array {}", env_state.env_state.node_capacity_array, ordered=config.ORDERED)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config.ROLLOUT_LENGTH
        )
        if config.DEBUG:
            jax.debug.print("traj_batch.info {}", traj_batch.info, ordered=config.ORDERED)

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, rng, rng_epoch, rng_step = runner_state
        last_obs = (env_state.env_state, env_params) if config.USE_GNN else last_obs
        axes = (None, 0, None) if config.USE_GNN else (None, 0)
        _, last_val = jax.pmap(
            jax.vmap(train_state.apply_fn, in_axes=axes, axis_name='env'),
            in_axes=axes, axis_name='device'
        )(train_state.params, *last_obs)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config.GAMMA * next_value * (1 - done) - value
                gae = (
                    delta
                    + config.GAMMA * config.GAE_LAMBDA * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):

            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    axes = (None, 0, None) if config.USE_GNN else (None, 0)
                    pi, value = jax.vmap(train_state.apply_fn, in_axes=axes)(params, *traj_batch.obs)

                    if config.env_type.lower() == "vone":
                        pi_source = distrax.Categorical(
                            logits=jnp.where(traj_batch.action_mask_s, pi[0]._logits, -1e8))
                        pi_path = distrax.Categorical(
                            logits=jnp.where(traj_batch.action_mask_p, pi[1]._logits, -1e8))
                        pi_dest = distrax.Categorical(
                            logits=jnp.where(traj_batch.action_mask_d, pi[2]._logits, -1e8))
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

                    else:
                        log_prob = pi[0].log_prob(traj_batch.action)
                        entropy = pi[0].entropy().mean()

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
                        jax.debug.print("entropy {}", entropy, ordered=config.ORDERED)
                        jax.debug.print("ratio {}", ratio, ordered=config.ORDERED)
                        jax.debug.print("gae {}", gae, ordered=config.ORDERED)
                        jax.debug.print("loss_actor1 {}", loss_actor1, ordered=config.ORDERED)
                        jax.debug.print("loss_actor2 {}", loss_actor2, ordered=config.ORDERED)
                        jax.debug.print("value_loss {}", value_loss, ordered=config.ORDERED)
                        jax.debug.print("loss_actor {}", loss_actor, ordered=config.ORDERED)
                        jax.debug.print("entropy {}", entropy, ordered=config.ORDERED)
                        jax.debug.print("total_loss {}", total_loss, ordered=config.ORDERED)

                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng, rng_epoch, rng_step = update_state
            rng_epoch, perm_key = jax.random.split(rng, 2)
            batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
            assert (
                batch_size == config.ROLLOUT_LENGTH * config.NUM_ENVS * config.NUM_DEVICES
            ), "batch size must be equal to number of steps * number of envs * number of devices"
            permutation = jax.random.permutation(perm_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree.map(
                lambda x: x.reshape((batch_size,) + x.shape[3:]), batch
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
            update_state = (train_state, traj_batch, advantages, targets, rng, rng_epoch, rng_step)
            return update_state, total_loss

        update_state = (train_state, traj_batch, advantages, targets, rng, rng_epoch, rng_step)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.UPDATE_EPOCHS
        )
        train_state = update_state[0]
        metric = traj_batch.info
        rng = update_state[-3]
        runner_state = (train_state, env_state, last_obs, rng)

        if config.DEBUG:
            jax.debug.print("metric {}", metric, ordered=config.ORDERED)

        return runner_state, (metric, loss_info)

    def learner_fn(update_state):

        train_state, (metric_info, loss_info) = jax.lax.scan(
            _update_step, update_state, None, config.NUM_UPDATES
        )
        return {"runner_state": train_state, "metrics": metric_info, "loss_info": loss_info}

    return learner_fn


def learner_data_setup(config: absl.flags.FlagValues, rng: chex.PRNGKey) -> Tuple:

    env, env_params = define_env(config)
    rng, rng_epoch, rng_step, reset_key, network_key, warmup_key = jax.random.split(rng, 6)
    reset_key = jax.random.split(reset_key, config.NUM_DEVICES * config.NUM_ENVS)
    reset_key = reshape_keys(reset_key, config.NUM_DEVICES, config.NUM_ENVS)
    obsv, env_state = jax.pmap(
        jax.vmap(env.reset, in_axes=(0, None), axis_name='env'),
        axis_name='device'
    )(reset_key, env_params)
    obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])

    # INIT NETWORK
    network, init_x = init_network(config, env, env_state, env_params)
    init_x = (jax.tree.map(lambda x: x[0], init_x[0]), init_x[1]) if config.USE_GNN else init_x

    if config.LOAD_MODEL:
        network_params = config.model["model"]["params"]
        print('Retraining model')
    else:
        network_params = network.init(network_key, *init_x)

    # INIT LEARNING RATE SCHEDULE
    lr_schedule = make_lr_schedule(config)

    tx = optax.chain(
        optax.clip_by_global_norm(config.MAX_GRAD_NORM),
        optax.adam(learning_rate=lr_schedule, eps=config.ADAM_EPS, b1=config.ADAM_BETA1, b2=config.ADAM_BETA2),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # Recreate DeepRMSA warmup period
    if config.ENV_WARMUP_STEPS:
        warmup_key = jax.random.split(warmup_key, config.NUM_DEVICES * config.NUM_ENVS)
        warmup_key = reshape_keys(warmup_key, config.NUM_DEVICES, config.NUM_ENVS)
        warmup_state = (warmup_key, env_state, obsv)
        warmup_fn = get_warmup_fn(warmup_state, env, env_params, train_state, config)
        warmup_fn = jax.vmap(warmup_fn, axis_name="env")
        warmup_fn = jax.pmap(warmup_fn, axis_name="device")
        env_state, obsv = warmup_fn(warmup_state)

    # Initialise learner state.
    init_train_state = (train_state, env_state, obsv, rng)

    return init_train_state, env, env_params
