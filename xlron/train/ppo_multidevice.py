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
from xlron.train.jax_utils import *


class VONETransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    action_mask_s: jnp.ndarray
    action_mask_p: jnp.ndarray
    action_mask_d: jnp.ndarray


class RSATransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    action_mask: jnp.ndarray


def define_env(config: absl.flags.FlagValues):
    config_dict = {k: v.value for k, v in config.__flags.items()}
    if config.env_type.lower() == "vone":
        env, env_params = make_vone_env(config_dict)
    elif config.env_type.lower() in ["rsa", "rmsa", "rwa", "deeprmsa", "rwa_lightpath_reuse"]:
        env, env_params = make_rsa_env(config_dict)
    else:
        raise ValueError(f"Invalid environment type {config.env_type}")
    env = LogWrapper(env)
    return env, env_params


def init_network(config, env, env_state, env_params):
    if config.env_type.lower() == "vone":
        network = ActorCriticMLP([space.n for space in env.action_space(env_params).spaces],
                                 activation=config.ACTIVATION,
                                 num_layers=config.NUM_LAYERS,
                                 num_units=config.NUM_UNITS,
                                 layer_norm=config.LAYER_NORM,)
        init_x = tuple([jnp.zeros(env.observation_space(env_params).n)])
    elif config.env_type.lower() in ["rsa", "rmsa", "rwa", "deeprmsa", "rwa_lightpath_reuse"]:
        if config.USE_GNN:
            network = ActorCriticGNN(
                activation=config.ACTIVATION,
                num_layers=config.NUM_LAYERS,
                num_units=config.NUM_UNITS,
                gnn_latent=config.gnn_latent,
                message_passing_steps=config.message_passing_steps,
                # output_edges_size must equal number of slot actions
                output_edges_size=math.ceil(env_params.link_resources / env_params.aggregate_slots),
                output_nodes_size=config.output_nodes_size,
                output_globals_size=config.output_globals_size,
                gnn_mlp_layers=config.gnn_mlp_layers,
                normalise_by_link_length=config.normalize_by_link_length,
                mlp_layer_norm=config.LAYER_NORM,
                vmap=False,
            )
            init_x = (env_state.env_state, env_params)
        else:
            network = ActorCriticMLP([env.action_space(env_params).n],
                                     activation=config.ACTIVATION,
                                     num_layers=config.NUM_LAYERS,
                                     num_units=config.NUM_UNITS,
                                     layer_norm=config.LAYER_NORM,)

            init_x = tuple([jnp.zeros(env.observation_space(env_params).n)])
    else:
        raise ValueError(f"Invalid environment type {config.env_type}")
    return network, init_x


def select_action(select_action_state, env, env_params, train_state, config):
    """Select an action from the policy.
    If using VONE, the action is a tuple of (source, path, destination).
    Otherwise, the action is a single lightpath.
    Args:
        rng: jax.random.PRNGKey
        env: Environment
        env_state: Environment state
        env_params: Environment parameters
        network: Policy and value network
        network_params: Policy and value network parameters
        config: Config
        last_obs: Last observation
        deterministic: Whether to use the mode of the action distribution
    Returns:
        action: Action
        log_prob: Log probability of action
        value: Value of state
    """
    rng_key, env_state, last_obs = select_action_state
    last_obs = (env_state.env_state, env_params) if config.USE_GNN else last_obs
    pi, value = train_state.apply_fn(train_state.params, *last_obs)
    action_keys = jax.random.split(rng_key, len(pi))

    # Always do action masking with VONE
    if config.env_type.lower() == "vone":
        vmap_mask_nodes = jax.vmap(env.action_mask_nodes, in_axes=(0, None))
        vmap_mask_slots = jax.vmap(env.action_mask_slots, in_axes=(0, None, 0))
        vmap_mask_dest_node = jax.vmap(env.action_mask_dest_node, in_axes=(0, None, 0))

        env_state = env_state.replace(env_state=vmap_mask_nodes(env_state.env_state, env_params))
        pi_source = distrax.Categorical(logits=jnp.where(env_state.env_state.node_mask_s, pi[0]._logits, -1e8))

        action_s = pi_source.sample(seed=action_keys[0]) if not config.deterministic else pi_source.mode()

        # Update destination mask now source has been selected
        env_state = env_state.replace(env_state=vmap_mask_dest_node(env_state.env_state, env_params, action_s))
        pi_dest = distrax.Categorical(
            logits=jnp.where(env_state.env_state.node_mask_d, pi[2]._logits, -1e8))

        action_p = jnp.full(action_s.shape, 0)
        action_d = pi_dest.sample(seed=action_keys[2]) if not config.deterministic else pi_dest.mode()
        action = jnp.stack((action_s, action_p, action_d), axis=1)

        env_state = env_state.replace(env_state=vmap_mask_slots(env_state.env_state, env_params, action))
        pi_path = distrax.Categorical(logits=jnp.where(env_state.env_state.link_slot_mask, pi[1]._logits, -1e8))
        action_p = pi_path.sample(seed=action_keys[1]) if not config.deterministic else pi_path.mode()
        action = jnp.stack((action_s, action_p, action_d), axis=1)

        log_prob_source = pi_source.log_prob(action_s)
        log_prob_path = pi_path.log_prob(action_p)
        log_prob_dest = pi_dest.log_prob(action_d)
        log_prob = log_prob_dest + log_prob_path + log_prob_source

    elif config.ACTION_MASKING:
        env_state = env_state.replace(env_state=env.action_mask(env_state.env_state, env_params))
        pi_masked = distrax.Categorical(logits=jnp.where(env_state.env_state.link_slot_mask, pi[0]._logits, -1e8))
        if config.DEBUG:
            jax.debug.print("pi {}", pi[0]._logits, ordered=config.ORDERED)
            jax.debug.print("pi_masked {}", pi_masked._logits, ordered=config.ORDERED)
            jax.debug.print("last_obs {}", last_obs[0].graph.edges, ordered=config.ORDERED)
        action = pi_masked.sample(seed=action_keys[0]) if not config.deterministic else pi[0].mode()
        log_prob = pi_masked.log_prob(action)

    else:
        action = pi[0].sample(seed=action_keys[0]) if not config.deterministic else pi[0].mode()
        log_prob = pi[0].log_prob(action)

    return action, log_prob, value


def warmup_period(warmup_state, env, params, train_state, config) -> EnvState:
    """Warmup period for DeepRMSA."""

    rng, state, last_obs = warmup_state
    def body_fn(i, val):
        _rng, _state, _params, _train_state, _last_obs = val
        _rng, action_key, step_key = jax.random.split(_rng, 3)
        # SELECT ACTION
        action, log_prob, value = select_action(action_key, env, _state, _params, _train_state, config, _last_obs)
        # STEP ENV
        obsv, _state, reward, done, info = env.step(step_key, _state, action, _params)
        obsv = (_state.env_state, _params) if config.USE_GNN else tuple([obsv])
        return _rng, _state, _params, _train_state, obsv

    vals = jax.lax.fori_loop(0, config.ENV_WARMUP_STEPS, body_fn,
                             (rng, state, params, train_state, last_obs))

    return vals[1]


def get_warmup_fn(warmup_state, env, params, train_state, config) -> Tuple[EnvState, chex.Array]:
    """Warmup period for DeepRMSA."""

    def warmup_fn(warmup_state):

        rng, state, last_obs = warmup_state

        def warmup_step(i, val):
            _rng, _state, _params, _train_state, _last_obs = val
            # SELECT ACTION
            _rng, action_key, step_key = jax.random.split(_rng, 3)
            select_action_state = (_rng, _state, _last_obs)
            action, log_prob, value = select_action(select_action_state, env, _params, _train_state, config)
            # STEP ENV
            obsv, _state, reward, done, info = env.step(
                step_key, _state, action, params
            )
            obsv = (_state.env_state, params) if config.USE_GNN else tuple([obsv])
            return _rng, _state, _params, _train_state, obsv

        vals = jax.lax.fori_loop(0, config.ENV_WARMUP_STEPS, warmup_step,
                                 (rng, state, params, train_state, last_obs))

        return vals[1]

    return warmup_fn


def reshape_keys(keys, size1, size2):
    dimensions = (size1, size2)
    reshape = lambda x: x.reshape(dimensions + x.shape[1:])
    return reshape(jnp.stack(keys))


def make_train(config):

    env, env_params = define_env(config)

    def linear_schedule(count):
        frac = (1.0 - (count // (config.NUM_MINIBATCHES * config.UPDATE_EPOCHS)) /
                (config.NUM_UPDATES * config.SCHEDULE_MULTIPLIER))
        return config.LR * frac

    def lr_schedule(count):
        total_steps = config.NUM_UPDATES * config.UPDATE_EPOCHS * config.NUM_MINIBATCHES * config.SCHEDULE_MULTIPLIER
        if config.LR_SCHEDULE == "warmup_cosine":
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=config.LR,
                peak_value=config.LR*config.WARMUP_PEAK_MULTIPLIER,
                warmup_steps=int(total_steps * config.WARMUP_STEPS_FRACTION),
                decay_steps=total_steps,
                end_value=config.LR*config.WARMUP_END_FRACTION)
        elif config.LR_SCHEDULE == "linear":
            schedule = linear_schedule
        elif config.LR_SCHEDULE == "constant":
            schedule = lambda x: config.LR
        else:
            raise ValueError(f"Invalid LR schedule {config.LR_SCHEDULE}")
        return schedule(count)

    def train(rng):
        # INIT ENV
        rng_epoch, rng_step, reset_key, network_key, warmup_key = jax.random.split(rng, 5)
        reset_key = jax.random.split(reset_key, config.NUM_DEVICES*config.NUM_ENVS)
        reset_key = reshape_keys(reset_key, config.NUM_DEVICES, config.NUM_ENVS)
        obsv, env_state = jax.pmap(
            jax.vmap(env.reset, in_axes=(0, None), axis_name='env'),
            axis_name='device'
        )(reset_key, env_params)
        obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])

        # INIT NETWORK
        network, init_x = init_network(config, env, env_state, env_params)
        init_x = (jax.tree.map(lambda x: merge_leading_dims(x, 2)[0], init_x[0]), init_x[1]) if config.USE_GNN else init_x
        if config.LOAD_MODEL:
            network_params = config.model["model"]["params"]
            print('Retraining model')
        else:
            network_params = network.init(network_key, *init_x)
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
            env_state = warmup_fn(warmup_state)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng_epoch, rng_step = runner_state

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
                runner_state = (train_state, env_state, obsv, rng_epoch, rng_step)

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
            train_state, env_state, last_obs, rng_epoch, rng_step = runner_state
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

                train_state, traj_batch, advantages, targets, rng_epoch, rng_step = update_state
                rng_epoch, perm_rng = jax.random.split(rng)
                batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
                assert (
                    batch_size == config.ROLLOUT_LENGTH * config.NUM_ENVS * config.NUM_DEVICES
                ), "batch size must be equal to number of steps * number of envs * number of devices"
                permutation = jax.random.permutation(perm_rng, batch_size)
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
                update_state = (train_state, traj_batch, advantages, targets, rng_epoch, rng_step)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng_epoch, rng_step)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.UPDATE_EPOCHS
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng_epoch = update_state[-2]
            rng_step = update_state[-1]
            runner_state = (train_state, env_state, last_obs, rng_epoch, rng_step)

            if config.DEBUG:
                jax.debug.print("metric {}", metric, ordered=config.ORDERED)

            return runner_state, (metric, loss_info)

        runner_state = (train_state, env_state, obsv, rng_epoch, rng_step)
        runner_state, (metric_info, loss_info) = jax.lax.scan(
            _update_step, runner_state, None, config.NUM_UPDATES
        )

        return {"runner_state": runner_state, "metrics": metric_info, "loss_info": loss_info}

    return train
