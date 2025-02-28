import os
import math
import absl
import chex
import jax
import flax
import jax.numpy as jnp
import optax
import distrax
from typing import Sequence, NamedTuple, Any, Tuple, Callable
from gymnax.environments import environment
from xlron.environments.env_funcs import *
from xlron.train.train_utils import *


def get_warmup_fn(warmup_state, env, params, config) -> Tuple[EnvState, chex.Array]:
    """Warmup period for DeepRMSA."""

    def warmup_fn(warmup_state):

        rng, state, learner, last_obs = warmup_state

        def warmup_step(i, val):
            _rng, _state, _learner, _last_obs = val
            # SELECT ACTION
            _rng, action_key, step_key = jax.random.split(_rng, 3)
            select_action_state = action_key, _state, _last_obs
            action, log_prob, value = select_action(
                select_action_state, env, params, _learner, config,
            )
            # STEP ENV
            obsv, _state, reward, done, info = env.step(
                step_key, _state, action, params
            )
            obsv = (_state.env_state, params) if config.USE_GNN else tuple([obsv])
            return _rng, _state, _learner, obsv

        batched_warmup_step = jax.vmap(warmup_step, in_axes=(None, 0))

        vals = jax.lax.fori_loop(0, config.ENV_WARMUP_STEPS, batched_warmup_step,
                                 (rng, state, learner, last_obs))
        return vals[1], vals[3]

    return warmup_fn


def get_learner_fn(
    env: environment.Environment,
    env_params: EnvParams,
    train_state: TrainState,
    config: flags,
) -> Callable:
    """Get the learner function."""

    def _update_step(runner_state, _) -> Tuple:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            train_state (TrainState): The current training state.
            _ (Any): The current metrics info.
        """

        train_state, env_state, last_obs, rng = runner_state
        rng, rng_epoch, rng_step = jax.random.split(rng, 3)

        # Add rngs to runner_state tuple
        runner_state = (train_state, env_state, last_obs, rng, rng_epoch, rng_step)

        @partial(jax.jit, donate_argnums=(0, 1))
        @jax.checkpoint
        def _env_step(runner_state, _) -> Tuple:
            """Step the environment."""
            train_state, env_state, last_obs, rng, rng_epoch, rng_step = runner_state

            rng_step, action_key, step_key = jax.random.split(rng_step, 3)

            # SELECT ACTION
            select_action_state = (action_key, env_state, last_obs)
            action, log_prob, value = select_action(
                select_action_state, env, env_params, train_state, config
            )

            # STEP ENVIRONMENT
            obsv, env_state, reward, done, info = env.step(step_key, env_state, action, env_params)
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
                # jax.debug.print("link_slot_array {}", env_state.env_state.link_slot_array, ordered=config.ORDERED)
                # jax.debug.print("link_slot_mask {}", env_state.env_state.link_slot_mask, ordered=config.ORDERED)
                if config.env_type.lower() == "vone":
                    jax.debug.print("node_mask_s {}", env_state.env_state.node_mask_s, ordered=config.ORDERED)
                    jax.debug.print("node_mask_d {}", env_state.env_state.node_mask_d, ordered=config.ORDERED)
                    jax.debug.print("action_history {}", env_state.env_state.action_history, ordered=config.ORDERED)
                    jax.debug.print("action_counter {}", env_state.env_state.action_counter, ordered=config.ORDERED)
                    jax.debug.print("request_array {}", env_state.env_state.request_array, ordered=config.ORDERED)
                    jax.debug.print("node_capacity_array {}", env_state.env_state.node_capacity_array,
                                    ordered=config.ORDERED)
            return runner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config.ROLLOUT_LENGTH
        )
        if config.DEBUG:
            jax.debug.print("traj_batch.info {}", traj_batch.info, ordered=config.ORDERED)

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, rng, rng_epoch, rng_step = runner_state
        last_obs = (env_state.env_state, env_params) if config.USE_GNN else last_obs
        _, last_val = train_state.apply_fn(train_state.params, *last_obs)

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

        def _update_epoch(update_state, unused):

            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RUN NETWORKS ACROSS TRAJECTORY DIMENSION
                    in_axes = (None, 0, None) if len(traj_batch.obs) == 2 else (None, 0)
                    pi, value = jax.vmap(train_state.apply_fn, in_axes=in_axes)(params, *traj_batch.obs)

                    # pi, value = train_state.apply_fn(params, *traj_batch.obs)

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

                # CALCULATE LOSS
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                total_loss, grads = jax.lax.pmean(
                    (total_loss, grads), axis_name="batch"
                )
                # pmean over devices.
                total_loss, grads = jax.lax.pmean(
                    (total_loss, grads), axis_name="device"
                )
                # UPDATE ACTOR-CRITIC PARAMS AND OPTIMISER STATE
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng, rng_epoch, rng_step = update_state
            rng_epoch, perm_key = jax.random.split(rng_epoch)

            # SHUFFLE MINIBATCHES
            permutation = jax.random.permutation(perm_key, config.ROLLOUT_LENGTH)
            batch = (traj_batch, advantages, targets)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            runner_state = (train_state, traj_batch, advantages, targets, rng, rng_epoch, rng_step)
            return runner_state, total_loss

        update_state = (train_state, traj_batch, advantages, targets, rng, rng_epoch, rng_step)

        # UPDATE EPOCHS

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
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            update_state: (replicated_learner, env_states, init_obs, step_keys)
        """

        batched_update_step = jax.vmap(jax.checkpoint(_update_step), in_axes=(0, None), axis_name="batch")

        train_state, (metric_info, loss_info) = jax.lax.scan(
            batched_update_step, update_state, None, config.NUM_UPDATES
        )
        return {"runner_state": train_state, "metrics": metric_info, "loss_info": loss_info}

    return learner_fn


def learner_setup(env, env_params, config):
    """Initialise learner_fn, network, optimiser, environment and states."""

    num_devices = len(jax.devices())
    rng = jax.random.PRNGKey(config.SEED)
    rng, step_rng, reset_rng, network_key, warmup_key = jax.random.split(rng, 5)

    @partial(jax.jit, static_argnums=(1,))
    def init_obsv_and_state(_reset_rng, _env_params):
        obsv, env_state = env.reset(_reset_rng, _env_params)
        obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
        return obsv, env_state

    # INIT NETWORK
    _, dummy_state = init_obsv_and_state(reset_rng, env_params)
    network, init_x = init_network(config, env, dummy_state, env_params)

    if config.LOAD_MODEL:
        network_params = config.model["model"]["params"]
        print('Retraining model')
    else:
        network_params = network.init(network_key, *init_x)
    # INIT LEARNING RATE SCHEDULE
    lr_schedule = make_lr_schedule(config)
    # INIT OPTIMISER
    tx = optax.chain(
        optax.clip_by_global_norm(config.MAX_GRAD_NORM),
        optax.adam(learning_rate=lr_schedule, eps=config.ADAM_EPS, b1=config.ADAM_BETA1, b2=config.ADAM_BETA2),
    )
    train_state = TrainState.create(
        apply_fn=jax.checkpoint(network.apply),
        params=network_params,
        tx=tx,
    )

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, env_params, train_state, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    dimensions = (num_devices, config.NUM_ENVS)
    reshape = lambda x: x.reshape(dimensions + x.shape[1:])
    broadcast = lambda x: jnp.broadcast_to(x, (dimensions[1],) + x.shape)

    reset_keys = jax.random.split(reset_rng, math.prod(dimensions))
    observations, env_states = jax.vmap(init_obsv_and_state, in_axes=(0, None))(jnp.stack(reset_keys), env_params)

    env_states = jax.tree.map(reshape, env_states)
    init_obs = jax.tree.map(reshape, observations)

    step_keys = jax.random.split(step_rng, math.prod(dimensions))
    step_keys = reshape(jnp.stack(step_keys))

    # Duplicate learner for update_batch_size.
    replicated_learner = jax.tree.map(broadcast, train_state)

    # Duplicate learner across devices.
    replicated_learner = flax.jax_utils.replicate(replicated_learner, devices=jax.devices())

    # # Recreate DeepRMSA warmup period
    if config.ENV_WARMUP_STEPS:
        warmup_keys = jax.random.split(warmup_key, math.prod(dimensions))
        warmup_keys = reshape(jnp.stack(warmup_keys))
        warmup_state = (warmup_keys, env_states, replicated_learner, init_obs)
        warmup_fn = get_warmup_fn(warmup_state, env, env_params, config)
        warmup_fn = jax.pmap(warmup_fn, axis_name="device")
        env_states, init_obs = warmup_fn(warmup_state)

    # Initialise learner state.
    init_train_state = (replicated_learner, env_states, init_obs, step_keys)

    return learn, init_train_state


def setup_experiment(config: flags) -> Tuple:
    """Runs experiment."""

    # Create the environments for train and eval.
    env, env_params = define_env(config)

    # Calculate number of updates per environment.
    NUM_UPDATES = (
            config.TOTAL_TIMESTEPS // config.ROLLOUT_LENGTH // config.NUM_ENVS
    )
    config.__setattr__("NUM_UPDATES", NUM_UPDATES)

    # Setup learner.
    learn, learner_state = learner_setup(
        env, env_params, config
    )

    return learn, learner_state
