import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import distrax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple
from flax.training.train_state import TrainState
from xlron.environments.env_funcs import *
from xlron.environments.vone import make_vone_env
from xlron.environments.rsa import make_rsa_env


# TODO - Remove request from observation that is fed to state value function
#  requires separation of actor and critic into separate classes and modification of flax train_state
#  to hold two sets of params
#  https://flax.readthedocs.io/en/latest/_modules/flax/training/train_state.html
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    num_layers: int = 2
    num_units: int = 64

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        def make_layers(x):
            layer = nn.Dense(
                self.num_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            layer = activation(layer)
            for _ in range(self.num_layers-1):
                layer = nn.Dense(
                    self.num_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
                )(layer)
                layer = activation(layer)
            return layer

        actor_mean = make_layers(x)
        action_dists = []
        for dim in self.action_dim:
            actor_mean_dim = nn.Dense(
                dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
            pi_dim = distrax.Categorical(logits=actor_mean_dim)
            action_dists.append(pi_dim)

        critic = make_layers(x)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return action_dists, jnp.squeeze(critic, axis=-1)


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


def make_train(config):
    NUM_UPDATES = (
        config.TOTAL_TIMESTEPS // config.NUM_STEPS // config.NUM_ENVS
    )
    MINIBATCH_SIZE = (
        config.NUM_ENVS * config.NUM_STEPS // config.NUM_MINIBATCHES
    )
    config_dict = {k: v.value for k, v in config.__flags.items()}
    if config.env_type.lower() == "vone":
        env, env_params = make_vone_env(config_dict)
    elif config.env_type.lower() in ["rsa", "rmsa", "rwa", "deeprmsa"]:
        env, env_params = make_rsa_env(config_dict)
    else:
        raise ValueError(f"Invalid environment type {config.env_type}")
    env = LogWrapper(env)

    # TODO - Does it matter if lr changes slightly between each minibatch? Linear handles this but optax built-ins don't
    def linear_schedule(count):
        frac = (1.0 - (count // (config.NUM_MINIBATCHES * config.UPDATE_EPOCHS)) /
                (NUM_UPDATES * config.SCHEDULE_MULTIPLIER))
        return config.LR * frac

    def lr_schedule(count):
        total_steps = NUM_UPDATES * config.UPDATE_EPOCHS * config.NUM_MINIBATCHES * config.SCHEDULE_MULTIPLIER
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

        # INIT NETWORK
        if config.env_type.lower() == "vone":
            network = ActorCritic([space.n for space in env.action_space(env_params).spaces],
                                  activation=config.ACTIVATION,
                                  num_layers=config.NUM_LAYERS,
                                  num_units=config.NUM_UNITS)
        elif config.env_type.lower() in ["rsa", "rmsa", "rwa", "deeprmsa"]:
            network = ActorCritic([env.action_space(env_params).n],
                                  activation=config.ACTIVATION,
                                  num_layers=config.NUM_LAYERS,
                                  num_units=config.NUM_UNITS)
        else:
            raise ValueError(f"Invalid environment type {config.env_type}")
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).n)
        network_params = network.init(_rng, init_x)
        tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(learning_rate=lr_schedule, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                pi, value = network.apply(train_state.params, last_obs)
                rng = jax.random.split(rng, 1 + len(pi))

                # Always do action masking with VONE
                if config.env_type.lower() == "vone":
                    vmap_mask_nodes = jax.vmap(env.action_mask_nodes, in_axes=(0, None))
                    vmap_mask_slots = jax.vmap(env.action_mask_slots, in_axes=(0, None, 0))
                    vmap_mask_dest_node = jax.vmap(env.action_mask_dest_node, in_axes=(0, None, 0))

                    env_state = env_state.replace(env_state=vmap_mask_nodes(env_state.env_state, env_params))
                    pi_source = distrax.Categorical(logits=jnp.where(env_state.env_state.node_mask_s, pi[0]._logits, -1e8))

                    action_s = pi_source.sample(seed=rng[1])

                    # Update destination mask now source has been selected
                    env_state = env_state.replace(env_state=vmap_mask_dest_node(env_state.env_state, env_params, action_s))
                    pi_dest = distrax.Categorical(
                        logits=jnp.where(env_state.env_state.node_mask_d, pi[2]._logits, -1e8))

                    action_p = jnp.full(action_s.shape, 0)
                    action_d = pi_dest.sample(seed=rng[3])
                    action = jnp.stack((action_s, action_p, action_d), axis=1)

                    env_state = env_state.replace(env_state=vmap_mask_slots(env_state.env_state, env_params, action))
                    pi_path = distrax.Categorical(logits=jnp.where(env_state.env_state.link_slot_mask, pi[1]._logits, -1e8))
                    action_p = pi_path.sample(seed=rng[2])
                    action = jnp.stack((action_s, action_p, action_d), axis=1)

                    log_prob_source = pi_source.log_prob(action_s)
                    log_prob_path = pi_path.log_prob(action_p)
                    log_prob_dest = pi_dest.log_prob(action_d)

                    log_prob = log_prob_dest + log_prob_path + log_prob_source

                elif config.ACTION_MASKING:
                    vmap_mask_slots = jax.vmap(env.action_mask, in_axes=(0, None))
                    env_state = env_state.replace(env_state=vmap_mask_slots(env_state.env_state, env_params))
                    pi_masked = distrax.Categorical(logits=jnp.where(env_state.env_state.link_slot_mask, pi[0]._logits, -1e8))
                    action = pi_masked.sample(seed=rng[1])
                    log_prob = pi_masked.log_prob(action)

                else:
                    action = pi[0].sample(seed=rng[1])
                    log_prob = pi[0].log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng[0])
                rng_step = jax.random.split(_rng, config.NUM_ENVS)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                transition = VONETransition(
                    done, action, value, reward, log_prob, last_obs, info, env_state.env_state.node_mask_s,
                    env_state.env_state.link_slot_mask,
                    env_state.env_state.node_mask_d
                ) if config.env_type.lower() == "vone" else RSATransition(
                    done, action, value, reward, log_prob, last_obs, info, env_state.env_state.link_slot_mask
                )
                runner_state = (train_state, env_state, obsv, rng)

                if config.DEBUG:
                    jax.debug.print("log_prob {}", log_prob, ordered=config.ORDERED)
                    jax.debug.print("action {}", action, ordered=config.ORDERED)
                    jax.debug.print("reward {}", reward, ordered=config.ORDERED)
                    jax.debug.print("link_slot_array {}", env_state.env_state.link_slot_array, ordered=config.ORDERED)
                    jax.debug.print("link_slot_mask {}", env_state.env_state.link_slot_mask, ordered=config.ORDERED)
                    if config.env_type.lower() == "vone":
                        jax.debug.print("node_mask_s {}", env_state.env_state.node_mask_s, ordered=config.ORDERED)
                        jax.debug.print("node_mask_d {}", env_state.env_state.node_mask_d, ordered=config.ORDERED)
                        jax.debug.print("action_history {}", env_state.env_state.action_history, ordered=config.ORDERED)
                        jax.debug.print("action_counter {}", env_state.env_state.action_counter, ordered=config.ORDERED)
                        jax.debug.print("request_array {}", env_state.env_state.request_array, ordered=config.ORDERED)
                        jax.debug.print("node_capacity_array {}", env_state.env_state.node_capacity_array, ordered=config.ORDERED)

                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.NUM_STEPS
            )
            if config.DEBUG:
                jax.debug.print("traj_batch.info {}", traj_batch.info, ordered=config.ORDERED)

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

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
                        pi, value = network.apply(params, traj_batch.obs)

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

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = MINIBATCH_SIZE * config.NUM_MINIBATCHES
                assert (
                    batch_size == config.NUM_STEPS * config.NUM_ENVS
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.UPDATE_EPOCHS
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            runner_state = (train_state, env_state, last_obs, rng)

            if config.DEBUG:
                jax.debug.print("metric {}", metric, ordered=config.ORDERED)

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
