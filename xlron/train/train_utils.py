import json
import math
import os
import pathlib
import pickle
from typing import Any, Callable, Dict, Tuple, Union

import absl
import box
import chex
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint
import pandas as pd
from box import Box
from flax.training import orbax_utils
from jax import Array
from optax import Schedule

import wandb
from xlron import dtype_config
from xlron.environments.dataclasses import EnvState
from xlron.environments.env_funcs import (
    get_launch_power,
    get_paths,
    init_link_length_array,
    make_graph,
    process_path_action,
)
from xlron.environments.make_env import make
from xlron.environments.wrappers import TimeIt
from xlron.heuristics.heuristics import (
    bf_ksp,
    ff_ksp,
    kca_ff,
    kmc_ff,
    kme_ff,
    kmf_ff,
    ksp_bf,
    ksp_ff,
    ksp_lf,
    ksp_mu,
    mu_ksp,
)
from xlron.models.gnn import ActorCriticGNN
from xlron.models.mlp import ActorCriticMLP, LaunchPowerActorCriticMLP
from xlron.models.transformer import ActorCriticTransformer

# TODO - Add all possible metrics here (they will all be registered in wandb) then just add a try except when adding them to processed data
metrics = [
    "returns",
    "lengths",
    "cum_returns",
    "accepted_services",
    "accepted_bitrate",
    "total_bitrate",
    "utilisation",
    "service_blocking_probability",
    "bitrate_blocking_probability",
    "throughput",  # Only for RSA GN Model
    "launch_power",
    "path_snr",
    "request_source",
    "request_dest",
    "request_data_rate",
    "arrival_time",
    "departure_time",
    "path_indices",
    "slot_indices",
    "returns",
    "path_links",
    "path_spectral_efficiency",
    "required_slots",
    "path_length",
    "num_hops",
]
loss_metrics = [
    "loss/total_loss",
    "loss/actor_loss",
    "loss/value_loss",
    "loss/entropy",
    "loss/gae",
    "loss/ratio",
    "loss/log_prob",
    "loss/entropy_loss_scaled",
    "loss/value_loss_scaled",
]


class TrainState(eqx.Module):
    """Train state for Equinox models.

    The model is stored but marked as non-pytree so JAX doesn't try to trace it.
    """

    step: Array
    model_params: eqx.Module
    model_static: eqx.Module = eqx.field(static=True)  # Mark as static/non-pytree
    tx: optax.GradientTransformation = eqx.field(static=True)
    opt_state: optax.OptState
    lr_schedule: Schedule = eqx.field(static=True)
    ent_schedule: Schedule = eqx.field(static=True)
    avg_reward: Array
    reward_stepsize: Array
    reward_stepsize_init: Array
    reward_stepsize_offset: Array
    prio_alpha: Array
    prio_beta0: Array
    prio_beta: Array

    def apply_gradients(self, grads: Any) -> "TrainState":
        """Updates model parameters and opt_state."""
        model = eqx.combine(self.model_params, self.model_static)
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.model_params)
        new_model = eqx.apply_updates(model, updates)
        new_model_params, new_model_static = eqx.partition(new_model, eqx.is_inexact_array)
        # Can't use eqx.tree_at for static fields, create new instance
        return TrainState(
            step=self.step,
            model_params=new_model_params,
            model_static=new_model_static,
            tx=self.tx,
            opt_state=new_opt_state,
            lr_schedule=self.lr_schedule,
            ent_schedule=self.ent_schedule,
            avg_reward=self.avg_reward,
            reward_stepsize=self.reward_stepsize,
            reward_stepsize_init=self.reward_stepsize_init,
            reward_stepsize_offset=self.reward_stepsize_offset,
            prio_alpha=self.prio_alpha,
            prio_beta0=self.prio_beta0,
            prio_beta=self.prio_beta,
        )

    def update_step_size(self) -> "TrainState":
        """Updates the step size used for reward centering."""
        reward_stepsize_offset = self.reward_stepsize_offset + self.reward_stepsize_init * (
            1 - self.reward_stepsize_offset
        )
        reward_stepsize = self.reward_stepsize_init / reward_stepsize_offset
        return eqx.tree_at(
            lambda state: (state.reward_stepsize, state.reward_stepsize_offset),
            self,
            (reward_stepsize, reward_stepsize_offset),
        )

    @staticmethod
    def create(
        model: eqx.Module | None,
        tx: optax.GradientTransformation,
        lr_schedule: Schedule = lambda x: jnp.array(0.0),
        ent_schedule: Schedule = lambda x: jnp.array(0.0),
        prio_alpha: float = 0.0,
        prio_beta0: float = 1.0,
        prio_beta: float = 1.0,
        reward_stepsize_init: float = 0.01,
    ) -> "TrainState":
        """Creates a new instance with step=0 and initialized opt_state."""
        opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))
        model_params, model_static = eqx.partition(model, eqx.is_inexact_array)
        return TrainState(
            step=jnp.array(0),
            model_params=model_params,
            model_static=model_static,
            tx=tx,
            opt_state=opt_state,
            lr_schedule=lr_schedule,
            ent_schedule=ent_schedule,
            avg_reward=jnp.array(0.0, dtype=dtype_config.REWARD_DTYPE),
            reward_stepsize=jnp.array(reward_stepsize_init, dtype=dtype_config.REWARD_DTYPE),
            reward_stepsize_init=jnp.array(reward_stepsize_init, dtype=dtype_config.REWARD_DTYPE),
            reward_stepsize_offset=jnp.array(1.0, dtype=dtype_config.REWARD_DTYPE),
            prio_alpha=jnp.array(prio_alpha, dtype=dtype_config.REWARD_DTYPE),
            prio_beta0=jnp.array(prio_beta0, dtype=dtype_config.REWARD_DTYPE),
            prio_beta=jnp.array(prio_beta, dtype=dtype_config.REWARD_DTYPE),
        )


def scale_gradient(g: chex.Array, scale: float = 1) -> chex.Array:
    """Scales the gradient of `g` by `scale` but keeps the original value unchanged."""
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)


def count_parameters(params: chex.ArrayTree) -> int:
    """Counts the number of parameters in a parameter tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def ndim_at_least(x: chex.Array, num_dims: chex.Numeric) -> jax.Array:
    """Check if the number of dimensions of `x` is at least `num_dims`."""
    if not (isinstance(x, jax.Array) or isinstance(x, np.ndarray)):
        x = jnp.asarray(x)
    return x.ndim >= num_dims


def merge_leading_dims(x: chex.Array, num_dims: chex.Numeric) -> chex.Array:
    """Merge leading dimensions.

    Note:
        This implementation is a generic function for merging leading dimensions
        extracted from Haiku.
        For the original implementation, please refer to the following link:
        (https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/basic.py#L207)
    """
    # Don't merge if there aren't dimensions to merge.
    if not ndim_at_least(x, num_dims):
        return x

    new_shape = (np.prod(x.shape[:num_dims]),) + x.shape[num_dims:]
    return x.reshape(new_shape)


def unreplicate_n_dims(x: chex.ArrayTree, unreplicate_depth: int = 2) -> chex.ArrayTree:
    """Unreplicates a pytree by removing the first `unreplicate_depth` axes.

    This function takes a pytree and removes some number of axes, associated with parameter
    duplication for running multiple updates across devices and in parallel with `vmap`.
    This is typically one axis for device replication, and one for the `update batch size`.
    """
    return jax.tree_util.tree_map(lambda x: x[(0,) * unreplicate_depth], x)


def unreplicate_batch_dim(x: chex.ArrayTree) -> chex.ArrayTree:
    """Unreplicated just the update batch dimension.
    (The dimension that is vmapped over when acting and learning)

    In stoix's case it is always the second dimension, after the device dimension.
    We simply take element 0 as the params are identical across this dimension.
    """
    return jax.tree_util.tree_map(lambda x: x[:, 0, ...], x)


def moving_average(x, w):
    return jnp.convolve(x, jnp.ones(w), "valid") / w


def save_model(train_state: TrainState, run_name, config: Union[box.Box, absl.flags.FlagValues]):
    config_dict = config.to_dict() if isinstance(config, box.Box) else config
    save_data = {"model": train_state, "config": config_dict}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(save_data)
    # Get path to current file
    model_path = (
        pathlib.Path(config.MODEL_PATH)
        if config.MODEL_PATH is not None
        else (pathlib.Path(__file__).resolve().parents[2] / "models" / run_name)
    )
    # If model_path dir already exists, append a number to the end
    i = 1
    model_path_og = model_path
    while model_path.exists():
        # Add index to end of model_path
        model_path = (
            pathlib.Path(str(model_path_og) + f"_{i}")
            if config.MODEL_PATH
            else model_path_og.parent / (model_path_og.name + f"_{i}")
        )
        i += 1
    print(f"Saving model to {model_path.absolute()}")
    orbax_checkpointer.save(model_path.absolute(), save_data, save_args=save_args)
    # Upload model to wandb
    if config.WANDB:
        print((model_path / "*").absolute())
        wandb.save(str((model_path / "*").absolute()), base_path=str(model_path.parent))


def init_network(config: Box, key: chex.PRNGKey) -> eqx.Module:
    if config.env_type.lower() == "vone":
        network = ActorCriticMLP(
            config.ACTION_DIM + (1*config.include_no_op), # +1 for "no op"
            config.INPUT_DIM,
            activation=config.ACTIVATION,
            num_layers=config.NUM_LAYERS,
            num_units=config.NUM_UNITS,
            layer_norm=config.mlp_layer_norm,
            key=key,
        )
    elif config.env_type.lower() in [
        "rsa",
        "rmsa",
        "rwa",
        "deeprmsa",
        "rwa_lightpath_reuse",
        "rsa_gn_model",
        "rmsa_gn_model",
        "rsa_multiband",
    ]:
        if config.USE_TRANSFORMER:
            # For transformer: input_size is the per-token feature dimension
            # This includes: wire_features + edge_features (link_slot_array or departure times)
            # The link_slot_array has shape (num_links, link_resources) so per-link features
            input_size = config.num_wire_features + config.link_resources + 1 # 1 for link relevance
            network = ActorCriticTransformer(
                input_size=input_size,
                embedding_size=config.transformer_embedding_size,
                intermediate_size=config.transformer_intermediate_size,
                num_slot_actions=config.ACTION_DIM // config.k,  # Number of slot actions per path
                num_layers=config.transformer_num_layers,
                num_heads=config.transformer_num_heads,
                enable_dropout=config.transformer_enable_dropout,
                dropout_rate=config.transformer_dropout_rate,
                attention_dropout_rate=config.transformer_attention_dropout_rate,
                share_layers=config.transformer_share_layers,
                num_wire_features=config.num_wire_features,
                actor_mlp_width=config.transformer_actor_mlp_width,
                critic_mlp_width=config.transformer_critic_mlp_width,
                actor_mlp_depth=config.transformer_actor_mlp_depth,
                critic_mlp_depth=config.transformer_critic_mlp_depth,
                key=key,
            )
        elif config.USE_GNN:
            if "gn_model" in config.env_type.lower() and config.output_globals_size_actor > 0:
                global_output_size_actor = (
                    int((config.max_power - config.min_power) / config.step_power) + 1
                    if config.discrete_launch_power
                    else 1
                )
            else:
                global_output_size_actor = config.global_output_size_actor
            input_node_feature_size = (
                1
                if config.DISABLE_NODE_FEATURES
                else config.num_spectral_features + 2  # 2 for source/dest indicators
            )
            network = ActorCriticGNN(
                config.link_resources,
                input_node_feature_size,
                1,  # Global input feature is just normalized requested datarate
                activation=config.ACTIVATION,
                num_layers=config.NUM_LAYERS,
                num_units=config.NUM_UNITS,
                message_passing_steps=config.message_passing_steps,
                # output_edges_size must equal number of slot actions
                mlp_layers=config.mlp_layers,
                mlp_latent=config.mlp_latent,
                edge_embedding_size=config.edge_embedding_size,
                edge_mlp_layers=config.edge_mlp_layers,
                edge_mlp_latent=config.edge_mlp_latent,
                edge_output_size_actor=math.ceil(config.link_resources / config.aggregate_slots),
                edge_output_size_critic=config.edge_output_size_critic,
                global_embedding_size=config.global_embedding_size,
                global_mlp_layers=config.global_mlp_layers,
                global_mlp_latent=config.global_mlp_latent,
                global_output_size_actor=global_output_size_actor,
                global_output_size_critic=config.global_output_size_critic,
                node_embedding_size=config.node_embedding_size,
                node_mlp_layers=config.node_mlp_layers,
                node_mlp_latent=config.node_mlp_latent,
                node_output_size_actor=config.node_output_size_actor,
                node_output_size_critic=config.node_output_size_critic,
                attn_mlp_layers=config.attn_mlp_layers,
                attn_mlp_latent=config.attn_mlp_latent,
                use_attention=config.attn_mlp_layers > 0,
                normalise_by_link_length=config.normalize_by_link_length,
                gnn_layer_norm=config.gnn_layer_norm,
                mlp_layer_norm=config.mlp_layer_norm,
                temperature=config.temperature,
                min_power_dbm=config.min_power,
                max_power_dbm=config.max_power,
                step_power_dbm=config.step_power,
                discrete=config.discrete_launch_power,
                min_concentration=config.min_concentration,
                max_concentration=config.max_concentration,
                epsilon=config.EPSILON,
                vmap=False,
                key=key,
            )
        elif "gn_model" in config.env_type.lower() and config.launch_power_type == 3:
            network = LaunchPowerActorCriticMLP(
                config.INPUT_DIM,
                config.ACTION_DIM + (1*config.include_no_op), # +1 for "no op"
                activation=config.ACTIVATION,
                num_layers=config.NUM_LAYERS,
                num_units=config.NUM_UNITS,
                layer_norm=config.mlp_layer_norm,
                discrete=config.discrete_launch_power,
                min_power_dbm=config.min_power,
                max_power_dbm=config.max_power,
                step_power_dbm=config.step_power,
                k_paths=config.k_paths,
                key=key,
            )
        else:
            network = ActorCriticMLP(
                config.ACTION_DIM,
                config.INPUT_DIM + (1*config.include_no_op),# +1 for "no op"
                activation=config.ACTIVATION,
                num_layers=config.NUM_LAYERS,
                num_units=config.NUM_UNITS,
                layer_norm=config.mlp_layer_norm,
                key=key,
            )
    else:
        raise ValueError(f"Invalid environment type {config.env_type}")
    return network


def load_model(config: Box, key: chex.PRNGKey) -> eqx.Module:
    # N.B. that this just restores the model weights but not the optimizer state
    try:
        with open(config.MODEL_PATH, "rb") as f:
            # First line of the file is hyperparams of model
            hyperparams = Box(json.loads(f.readline().decode()))
            model = init_network(config=hyperparams, key=key)
            model_loaded = eqx.tree_deserialise_leaves(f, model)
    except UnicodeDecodeError:
        print("No hyperparameters in file")
        model = init_network(config=config, key=key)
        model_loaded = eqx.tree_deserialise_leaves(config.MODEL_PATH, model)
    if config.KEEP_VF:
        # Replace critic layers with loaded model's critic layers
        model = eqx.tree_at(lambda m: m.critic_layers, model, model_loaded.critic_layers)
        model = eqx.tree_at(lambda m: m.critic_output, model, model_loaded.critic_output)
    else:
        model = model_loaded
    print(f"Loaded model: {config.MODEL_PATH}")
    return model


def experiment_data_setup(config: Box, rng: chex.PRNGKey) -> Tuple:
    # INIT ENV
    env, env_params = make(config)
    config.ACTION_DIM = env.num_actions(env_params)
    config.INPUT_DIM = int(env.observation_space(env_params).n)
    config.NUM_NODES = env_params.num_nodes
    config.NUM_LINKS = env_params.num_links
    rng, rng_step, rng_epoch, warmup_key, reset_key, network_key = jax.random.split(rng, 6)
    reset_key = jax.random.split(reset_key, config.NUM_ENVS) if config.NUM_ENVS > 1 else reset_key
    obsv, env_state = (
        jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        if config.NUM_ENVS > 1
        else env.reset(reset_key, env_params)
    )
    obsv = (env_state.env_state, env_params) if config.USE_GNN or config.USE_TRANSFORMER else tuple([obsv])

    # TRAINING MODE
    if config.RETRAIN_MODEL or config.EVAL_MODEL:
        # N.B. that this just restores the model weights but not the optimizer state
        model = load_model(config, network_key)
    elif config.EVAL_HEURISTIC or config.ACTION_OPTIMIZATION:
        model = None
    else:
        model = init_network(config, network_key)

    # INIT LEARNING RATE SCHEDULE AND OPTIMIZER
    lr_schedule = make_lr_schedule(config)
    ent_schedule = make_ent_schedule(config)

    # Use AdamW optimizer
    optimizer = optax.adamw(
        learning_rate=lr_schedule,
        eps=config.ADAM_EPS,
        b1=config.ADAM_BETA1,
        b2=config.ADAM_BETA2,
        weight_decay=config.WEIGHT_DECAY,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.MAX_GRAD_NORM),
        optimizer,
    )

    runner_state = TrainState.create(
        model=model,
        tx=tx,
        lr_schedule=lr_schedule,
        ent_schedule=ent_schedule,
        prio_alpha=config.PRIO_ALPHA,
        prio_beta0=config.PRIO_BETA0,
        prio_beta=config.PRIO_BETA0,
    )

    # Recreate DeepRMSA warmup period
    warmup_key = (
        jax.random.split(warmup_key, config.NUM_ENVS) if config.NUM_ENVS > 1 else warmup_key
    )
    warmup_state = (warmup_key, env_state, obsv)
    warmup_fn = get_warmup_fn(warmup_state, env, env_params, runner_state, config)
    warmup_fn = jax.vmap(warmup_fn) if config.NUM_ENVS > 1 else warmup_fn
    env_state, obsv = warmup_fn(warmup_state)

    # Initialise eval state
    init_runner_state = (runner_state, env_state, obsv, rng_step, rng_epoch)

    return init_runner_state, env, env_params


def select_action(select_action_state, env, env_params, train_state, config):
    """Select an action from the policy.
    If using VONE, the action is a tuple of (source, path, destination).
    Otherwise, the action is a single lightpath.
    Args:
        select_action_state: Tuple of (rng_key, env_state, last_obs)
        env: Environment
        env_params: Environment parameters
        train_state: TrainState
        config: Configuration
    Returns:
        env_state: Environment state
        action: Action
        log_prob: Log probability of action
        value: Value of state
    """
    action_key, env_state, last_obs = select_action_state
    if config.USE_GNN or config.USE_TRANSFORMER:
        last_obs = (env_state.env_state, env_params)
    model = eqx.combine(train_state.model_params, train_state.model_static)
    pi, value = model(*last_obs)
    # Action masking
    env_state = env_state.replace(env_state=env.action_mask(env_state.env_state, env_params))

    # Always do action masking with VONE
    if config.env_type.lower() == "vone":
        # TODO - change this to work with single set of logits (probably just slice them)
        vmap_mask_nodes = jax.vmap(env.action_mask_nodes, in_axes=(0, None))
        vmap_mask_slots = jax.vmap(env.action_mask_slots, in_axes=(0, None, 0))
        vmap_mask_dest_node = jax.vmap(env.action_mask_dest_node, in_axes=(0, None, 0))

        env_state = env_state.replace(env_state=vmap_mask_nodes(env_state.env_state, env_params))
        pi_source = distrax.Categorical(
            logits=jnp.where(env_state.env_state.node_mask_s, pi[0]._logits, -1e8)
        )

        action_s = (
            pi_source.sample(seed=action_key) if not config.deterministic else pi_source.mode()
        )

        # Update destination mask now source has been selected
        env_state = env_state.replace(
            env_state=vmap_mask_dest_node(env_state.env_state, env_params, action_s)
        )
        pi_dest = distrax.Categorical(
            logits=jnp.where(env_state.env_state.node_mask_d, pi[0]._logits, -1e8)
        )

        action_p = jnp.full(action_s.shape, 0)
        action_d = pi_dest.sample(seed=action_key) if not config.deterministic else pi_dest.mode()
        action = jnp.stack((action_s, action_p, action_d), axis=1)

        env_state = env_state.replace(
            env_state=vmap_mask_slots(env_state.env_state, env_params, action)
        )
        pi_path = distrax.Categorical(
            logits=jnp.where(env_state.env_state.link_slot_mask, pi[0]._logits, -1e8)
        )
        action_p = pi_path.sample(seed=action_key) if not config.deterministic else pi_path.mode()
        action = jnp.stack((action_s, action_p, action_d), axis=1)

        log_prob_source = pi_source.log_prob(action_s)
        log_prob_path = pi_path.log_prob(action_p)
        log_prob_dest = pi_dest.log_prob(action_d)
        log_prob = log_prob_dest + log_prob_path + log_prob_source

    elif "gn_model" in config.env_type.lower() and config.launch_power_type == "rl":
        pi_masked = distrax.Categorical(
            logits=jnp.where(env_state.env_state.link_slot_mask, pi[0]._logits, -1e8)
        )
        if config.GNN_OUTPUT_RSA and not config.GNN_OUTPUT_LP:
            path_action, log_prob = train_state.sample_fn(
                action_key, pi_masked, log_prob=True, deterministic=config.deterministic
            )
            power_action = jnp.array([env_params.default_launch_power])
        elif config.GNN_OUTPUT_RSA and config.GNN_OUTPUT_LP:
            path_action, power_action, log_prob = train_state.sample_fn(
                action_key,
                (pi_masked, pi[1]),
                log_prob=True,
                deterministic=config.deterministic,
            )
        else:
            power_action, log_prob = train_state.sample_fn(
                action_key, pi[1], log_prob=True, deterministic=config.deterministic
            )
            inner_state = env_state.env_state.replace(launch_power_array=power_action)
            env_state = env_state.replace(env_state=inner_state)
            path_action = (
                ksp_lf(env_state.env_state, env_params)
                if env_params.last_fit is True
                else ksp_ff(env_state.env_state, env_params)
            )
        inner_state = env_state.env_state.replace(launch_power_array=power_action)
        env_state = env_state.replace(env_state=inner_state)
        if config.output_globals_size_actor == 0:
            path_index, _ = process_path_action(env_state.env_state, env_params, path_action)
            power_action, log_prob = power_action[path_index], log_prob[path_index]
        action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))], axis=0)

    else:
        env_state = env_state.replace(env_state=env.action_mask(env_state.env_state, env_params))
        pi_masked = distrax.Categorical(
            logits=jnp.where(env_state.env_state.link_slot_mask, pi[0]._logits, -1e8)
        )
        action = pi_masked.sample(seed=action_key) if not config.deterministic else pi_masked.mode()
        log_prob = pi_masked.log_prob(action)

    return env_state, action, log_prob, value


def select_action_eval(select_action_state, env, env_params, eval_state, config):
    rng, env_state, last_obs = select_action_state

    if config.EVAL_HEURISTIC:
        # Action masking
        env_state = env_state.replace(env_state=env.action_mask(env_state.env_state, env_params))

        if config.env_type.lower() == "vone":
            raise NotImplementedError("VONE heuristics not yet implemented")

        elif config.env_type.lower() in [
            "rsa",
            "rwa",
            "rmsa",
            "deeprmsa",
            "rwa_lightpath_reuse",
            "rsa_gn_model",
            "rmsa_gn_model",
            "rsa_multiband",
        ]:
            if config.path_heuristic.lower() == "ksp_ff":
                action = ksp_ff(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "ff_ksp":
                action = ff_ksp(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "kmc_ff":
                action = kmc_ff(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "kmf_ff":
                action = kmf_ff(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "ksp_mu":
                action = ksp_mu(env_state.env_state, env_params, False, True)
            elif config.path_heuristic.lower() == "ksp_mu_nonrel":
                action = ksp_mu(env_state.env_state, env_params, False, False)
            elif config.path_heuristic.lower() == "ksp_mu_unique":
                action = ksp_mu(env_state.env_state, env_params, True, True)
            elif config.path_heuristic.lower() == "mu_ksp":
                action = mu_ksp(env_state.env_state, env_params, False, True)
            elif config.path_heuristic.lower() == "mu_ksp_nonrel":
                action = mu_ksp(env_state.env_state, env_params, False, False)
            elif config.path_heuristic.lower() == "mu_ksp_unique":
                action = mu_ksp(env_state.env_state, env_params, True, True)
            elif config.path_heuristic.lower() == "kca_ff":
                action = kca_ff(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "kme_ff":
                action = kme_ff(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "ksp_bf":
                action = ksp_bf(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "bf_ksp":
                action = bf_ksp(env_state.env_state, env_params)
            elif config.path_heuristic.lower() == "ksp_lf":
                action = ksp_lf(env_state.env_state, env_params)
            else:
                raise ValueError(f"Invalid path heuristic {config.path_heuristic}")
            if env_params.__class__.__name__ in ["RSAGNModelEnvParams", "RMSAGNModelEnvParams"]:
                if config.launch_power_type == "rl":
                    raise ValueError(
                        "launch_power_type cannot be 'rl' when --EVAL_HEURISTIC flag is True"
                    )
                launch_power = get_launch_power(env_state.env_state, action, action, env_params)
                action = jnp.concatenate([action.reshape((1,)), launch_power.reshape((1,))], axis=0)
        else:
            raise ValueError(f"Invalid environment type {config.env_type}")
        # For DeepRMSA env, the action can only be the path index (first-fit for spectrum always)
        if config.env_type.lower() == "deeprmsa":
            action = process_path_action(env_state.env_state, env_params, action)[0]
    else:
        env_state, action, _, _ = select_action(
            select_action_state, env, env_params, eval_state, config
        )
    return env_state, action, None, None


def get_warmup_fn(warmup_state, env, params, train_state, config) -> Callable[[Tuple], Tuple]:
    """Warmup period for DeepRMSA."""

    def warmup_fn(warmup_state) -> Tuple[EnvState, chex.Array]:
        rng, state, last_obs = warmup_state

        def warmup_step(i, val) -> Tuple:
            _rng, _state, _params, _train_state, _last_obs = val
            # SELECT ACTION
            _rng, action_key, step_key = jax.random.split(_rng, 3)
            select_action_state = (_rng, _state, _last_obs)
            action_fn = select_action if not config.EVAL_HEURISTIC else select_action_eval
            _state, action, log_prob, value = action_fn(
                select_action_state, env, _params, _train_state, config
            )
            if "gn_model" in config.env_type.lower() and config.launch_power_type == "rl":
                # If the action is launch power, the action is this shape:
                # jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))], axis=0)
                # We want to overwrite the launch power with a default launch_power
                path_action = (
                    ksp_lf(_state.env_state, _params)
                    if _params.last_fit is True
                    else ksp_ff(_state.env_state, _params)
                )
                action = jnp.concatenate(
                    [
                        path_action.reshape((1,)),
                        jnp.array(
                            [
                                params.default_launch_power,
                            ]
                        ),
                    ],
                    axis=0,
                )
            elif (
                "gn_model" in config.env_type.lower()
                and config.launch_power_type != "rl"
                and not config.EVAL_HEURISTIC
            ):
                raise ValueError("Check that EVAL_HEURISTIC is set to True if using a heuristic")
            # STEP ENV
            obsv, _state, reward, terminal, truncated, info = env.step(
                step_key, _state, action, params
            )
            obsv = (_state.env_state, params) if config.USE_GNN or config.USE_TRANSFORMER else tuple([obsv])
            return _rng, _state, _params, _train_state, obsv

        vals = jax.lax.fori_loop(
            0, config.ENV_WARMUP_STEPS, warmup_step, (rng, state, params, train_state, last_obs)
        )

        return vals[1], vals[4]

    return warmup_fn


def make_lr_schedule(config: Box) -> optax.Schedule:
    """Create a learning rate schedule based on the configuration."""

    LR = config.LR
    LR_END_FRACTION = config.LR_END_FRACTION
    NUM_MINIBATCHES = config.NUM_MINIBATCHES
    NUM_UPDATES = config.NUM_UPDATES * config.NUM_INCREMENTS
    UPDATE_EPOCHS = config.UPDATE_EPOCHS
    SCHEDULE_MULTIPLIER = config.SCHEDULE_MULTIPLIER
    WARMUP_MULTIPLIER = config.WARMUP_MULTIPLIER
    WARMUP_STEPS_FRACTION = config.WARMUP_STEPS_FRACTION
    end_value = LR * LR_END_FRACTION

    def lr_schedule(count: chex.Numeric) -> chex.Numeric:
        total_steps = NUM_UPDATES * UPDATE_EPOCHS * NUM_MINIBATCHES * SCHEDULE_MULTIPLIER
        if config.LR_SCHEDULE == "warmup_cosine":
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=LR,
                peak_value=LR * WARMUP_MULTIPLIER,
                warmup_steps=total_steps * WARMUP_STEPS_FRACTION,
                decay_steps=total_steps,
                end_value=end_value,
            )
        elif config.LR_SCHEDULE == "cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=LR,
                decay_steps=total_steps,
                alpha=end_value,
            )
        elif config.LR_SCHEDULE == "linear":
            schedule = optax.linear_schedule(
                init_value=LR,
                end_value=end_value,
                transition_steps=total_steps,
            )
        elif config.LR_SCHEDULE == "constant":
            schedule = optax.constant_schedule(LR)
        else:
            raise ValueError(f"Invalid LR schedule {config.LR_SCHEDULE}")
        return schedule(count)

    return lr_schedule


def make_ent_schedule(config: Box) -> optax.Schedule:
    """Create an entropy coefficient schedule based on the configuration."""

    ENT_COEF = config.ENT_COEF
    ENT_END_FRACTION = config.ENT_END_FRACTION
    NUM_MINIBATCHES = config.NUM_MINIBATCHES
    NUM_UPDATES = config.NUM_UPDATES * config.NUM_INCREMENTS
    UPDATE_EPOCHS = config.UPDATE_EPOCHS
    SCHEDULE_MULTIPLIER = config.SCHEDULE_MULTIPLIER
    end_value = ENT_COEF * ENT_END_FRACTION

    def ent_schedule(count: chex.Numeric) -> chex.Numeric:
        total_steps = NUM_UPDATES * UPDATE_EPOCHS * NUM_MINIBATCHES * SCHEDULE_MULTIPLIER
        if config.ENT_SCHEDULE == "cosine":
            schedule = optax.cosine_decay_schedule(
                init_value=ENT_COEF,
                decay_steps=total_steps,
                alpha=end_value,
            )
        elif config.ENT_SCHEDULE == "linear":
            schedule = optax.linear_schedule(
                init_value=ENT_COEF,
                end_value=end_value,
                transition_steps=total_steps,
            )
        elif config.ENT_SCHEDULE == "constant":
            schedule = optax.constant_schedule(ENT_COEF)
        else:
            raise ValueError(f"Invalid entropy schedule {config.ENT_SCHEDULE}")
        return schedule(count)

    return ent_schedule


def reshape_keys(keys, size1, size2):
    dimensions = (size1, size2)

    def reshape(x):
        return x.reshape(dimensions + x.shape[1:])

    return reshape(jnp.stack(keys))


def setup_wandb(config, project_name, experiment_name):
    wandb.setup(wandb.Settings(program="train.py", program_relpath="train.py"))
    run = wandb.init(
        project=project_name,
        save_code=True,  # optional
    )
    wandb.config.update(config)
    run.name = experiment_name
    wandb.define_metric("episode_count")
    wandb.define_metric("env_step")
    for metric in metrics:
        for agg in ["mean", "std", "iqr_upper", "iqr_lower"]:
            wandb.define_metric(f"{metric}_{agg}", step_metric="env_step")
            wandb.define_metric(
                f"{metric}_episode_end_{agg}", step_metric="episode_count", summary="last"
            )
            wandb.define_metric(
                f"{metric}_episode_end_{agg}", step_metric="episode_count", summary="mean"
            )
            wandb.define_metric(
                f"{metric}_episode_end_{agg}", step_metric="episode_count", summary="min"
            )
            wandb.define_metric(
                f"{metric}_episode_end_{agg}", step_metric="episode_count", summary="max"
            )
    for metric in loss_metrics:
        wandb.define_metric(f"{metric}", step_metric="update_epoch")
    wandb.define_metric("training_time", step_metric="env_step")


def get_mean_std_iqr(x, y):
    _mean = x[y].mean(axis=0).reshape(-1)
    _std = x[y].std(axis=0).reshape(-1)
    _iqr_upper = jnp.percentile(x[y], 75, axis=0).reshape(-1)
    _iqr_lower = jnp.percentile(x[y], 25, axis=0).reshape(-1)
    return jnp.array(_mean), jnp.array(_std), jnp.array(_iqr_upper), jnp.array(_iqr_lower)


def get_episode_end_mean_std_iqr(
    data: Array, episode_ends: Array, num_envs: int
) -> Tuple[Array, Array, Array, Array]:
    # Reshape to combine rollout and step dimensions
    data = data.reshape(num_envs, -1)
    episode_ends = episode_ends.reshape(num_envs, -1)
    # Count True values per environment
    counts = jnp.sum(episode_ends, axis=1)
    max_length = jnp.max(counts)
    diffs = max_length - counts
    # Maximum difference in episode ends between envs
    max_diff = jnp.max(diffs)
    # Append nans equal to max diffs
    data = jnp.hstack((data, jnp.full((num_envs, max_diff), jnp.nan)))
    # Append trues equal to diffs
    trues = jnp.tile(jnp.arange(max_diff), (num_envs, 1)) < diffs[:, None]
    episode_ends = jnp.hstack((episode_ends, trues))
    episode_ends = episode_ends.reshape(data.shape)
    episode_end_values = data[episode_ends]
    episode_end_values = episode_end_values.reshape((num_envs, -1))
    # Calculate statistics efficiently using JAX's nanops
    _end_mean = jnp.nanmean(episode_end_values, axis=0, dtype=jnp.float32)
    _end_std = jnp.nanstd(episode_end_values, axis=0, dtype=jnp.float32)
    _end_iqr_upper = jnp.nanpercentile(episode_end_values, 75, axis=0).astype(jnp.float32)
    _end_iqr_lower = jnp.nanpercentile(episode_end_values, 25, axis=0).astype(jnp.float32)

    return _end_mean, _end_std, _end_iqr_upper, _end_iqr_lower


def process_metrics(config, out, total_time, merge_func):
    """Calculate statistics from training or evaluation run."""
    merged_out = {k: jax.tree.map(merge_func, v) for k, v in out["metrics"].items()}
    if config.EVAL_HEURISTIC or config.EVAL_MODEL:
        merged_out_loss = None
    else:
        # Average over minibatches and epochs to get one value per update
        num_learners_or_1 = config.NUM_LEARNERS if config.NUM_LEARNERS > 1 else 1
        merged_out_loss = {
            k: jax.tree.map(
                lambda x: x.reshape((num_learners_or_1, config.NUM_UPDATES, -1))
                .mean(axis=-1)
                .reshape((-1,)),
                v,
            )
            for k, v in out.get("loss_info", {}).items()
        }

    # Calculate blocking probabilities
    merged_out["service_blocking_probability"] = 1 - (
        merged_out["accepted_services"]
        / jnp.where(merged_out["lengths"] == 0, 1, merged_out["lengths"])
    )
    merged_out["bitrate_blocking_probability"] = 1 - (
        merged_out["accepted_bitrate"]
        / jnp.where(merged_out["total_bitrate"] == 0, 1, merged_out["total_bitrate"])
    )

    # Calculate episode ends
    merged_out["done"] = jnp.logical_or(merged_out["terminal"], merged_out["truncated"])
    episode_ends = merged_out["done"]
    # Instead of flattening, create a boolean mask of where episodes end
    # This preserves the structure across environments
    episode_ends = episode_ends.reshape(episode_ends.shape[0], -1)

    episode_ends = jnp.hstack((episode_ends[:, 1:], jnp.full((episode_ends.shape[0], 1), False)))

    # Reshape episode_ends to match the original shape
    episode_ends = episode_ends.reshape(merged_out["done"].shape)

    print(f"Created episode end mask with {np.sum(episode_ends)} episode endings")

    processed_data = {}
    print("Processing output metrics")
    for metric in metrics:
        if metric == "throughput":
            # Shift values down one index position
            ends = jnp.concatenate([jnp.array([False]), episode_ends.flatten()[:-1]]).reshape(
                episode_ends.shape
            )
        else:
            ends = episode_ends
        try:
            episode_end_mean, episode_end_std, episode_end_iqr_upper, episode_end_iqr_lower = (
                get_episode_end_mean_std_iqr(merged_out[metric], ends, config.NUM_ENVS)
            )
            mean, std, iqr_upper, iqr_lower = get_mean_std_iqr(merged_out, metric)
            processed_data[metric] = {
                "mean": mean,
                "std": std,
                "iqr_upper": iqr_upper,
                "iqr_lower": iqr_lower,
                "episode_end_mean": episode_end_mean,
                "episode_end_std": episode_end_std,
                "episode_end_iqr_upper": episode_end_iqr_upper,
                "episode_end_iqr_lower": episode_end_iqr_lower,
            }
        except KeyError:
            continue
    return merged_out, merged_out_loss, processed_data, episode_ends


def plot_metrics(
    experiment_name: str, processed_data: Dict[str, Any], config: Union[Box, Dict[str, Any]]
) -> None:
    print("Plotting metrics")
    if config.incremental_loading:
        plot_metric = processed_data["accepted_services"]["episode_end_mean"]
        plot_metric_upper = processed_data["accepted_services"]["episode_end_iqr_upper"]
        plot_metric_lower = processed_data["accepted_services"]["episode_end_iqr_lower"]
        plot_metric_name = "Accepted Services"
    elif config.end_first_blocking:
        plot_metric = processed_data["lengths"]["episode_end_mean"]
        plot_metric_upper = processed_data["lengths"]["episode_end_iqr_upper"]
        plot_metric_lower = processed_data["lengths"]["episode_end_iqr_lower"]
        plot_metric_name = "Episode Length"
    elif config.reward_type == "service":
        plot_metric = processed_data["service_blocking_probability"]["mean"]
        plot_metric_upper = processed_data["service_blocking_probability"]["iqr_upper"]
        plot_metric_lower = processed_data["service_blocking_probability"]["iqr_lower"]
        plot_metric_name = "Service Blocking Probability"
    else:
        plot_metric = processed_data["bitrate_blocking_probability"]["mean"]
        plot_metric_upper = processed_data["bitrate_blocking_probability"]["iqr_upper"]
        plot_metric_lower = processed_data["bitrate_blocking_probability"]["iqr_lower"]
        plot_metric_name = "Bitrate Blocking Probability"

    smoothing_factor =  min(100, int(len(plot_metric) / 2))
    step_factor = int(len(plot_metric)) / smoothing_factor
    plot_metric = moving_average(plot_metric, smoothing_factor)
    plot_metric_upper = moving_average(plot_metric_upper, smoothing_factor)
    plot_metric_lower = moving_average(plot_metric_lower, smoothing_factor)
    plt.plot(jnp.arange(len(plot_metric))*config.DOWNSAMPLE_FACTOR*step_factor, plot_metric)
    plt.fill_between(range(len(plot_metric)), plot_metric_lower, plot_metric_upper, alpha=0.2)
    plt.xlabel("Environment Step" if not config.incremental_loading else "Episode Count")
    plt.ylabel(plot_metric_name)
    plt.title(experiment_name)
    plt.show()


def log_actions(merged_out, processed_data, config):
    print(
        f"Logging actions. \
        N.B. data is only logged from most recent increment. \
        Total increments: {config.NUM_INCREMENTS}"
    )
    env, params = make(config)
    request_source = merged_out["source"]
    request_dest = merged_out["dest"]
    request_data_rate = merged_out["data_rate"]
    path_indices = merged_out["path_index"]
    slot_indices = merged_out["slot_index"]
    returns = merged_out["returns"]
    arrival_time = merged_out["arrival_time"]
    departure_time = merged_out["departure_time"]

    # Reshape to combine episodes into a single trajectory. Only keep the first environment's output.
    # TODO - keep all the actions from every episode
    request_source = request_source.reshape((request_source.shape[0], -1))[0]
    request_dest = request_dest.reshape((request_dest.shape[0], -1))[0]
    request_data_rate = request_data_rate.reshape((request_data_rate.shape[0], -1))[0]
    path_indices = path_indices.reshape((path_indices.shape[0], -1))[0]
    slot_indices = slot_indices.reshape((slot_indices.shape[0], -1))[0]
    arrival_time = arrival_time.reshape((arrival_time.shape[0], -1))[0]
    departure_time = departure_time.reshape((departure_time.shape[0], -1))[0]
    returns = returns.reshape((returns.shape[0], -1))[0]

    # Get the link length array
    topology_name = config.topology_name
    graph = make_graph(topology_name, topology_directory=config.topology_directory)
    link_length_array = init_link_length_array(graph)
    # Get path, path lengths, number of hops
    paths = jnp.take(params.path_link_array.val, path_indices, axis=0)
    path_lengths = jax.vmap(lambda x: jnp.dot(x, link_length_array), in_axes=(0))(paths)
    num_hops = jnp.sum(paths, axis=-1)

    paths_list = []
    spectral_efficiency_list = []
    required_slots_list = []

    for path_index, slot_index, source, dest, data_rate in zip(
        path_indices, slot_indices, request_source, request_dest, request_data_rate
    ):
        source, dest = source.reshape(1), dest.reshape(1)
        path_links = get_paths(params, jnp.concatenate([source, dest]))[path_index % params.k_paths]
        # Make path links into a string
        path_str = "".join([str(x.astype(dtype_config.LARGE_INT_DTYPE)) for x in path_links])
        paths_list.append(path_str)
        path_spectral_efficiency = params.path_se_array.val[path_index]
        required_slots = int(jnp.ceil(data_rate / (path_spectral_efficiency * params.slot_size)))
        required_slots_list.append(required_slots)
        spectral_efficiency_list.append(path_spectral_efficiency)

    if config.TRAJ_DATA_OUTPUT_FILE:
        print(f"Saving trajectory metrics to {config.TRAJ_DATA_OUTPUT_FILE}")
        # Save episode end metrics to file
        log_dict = {
            "request_source": request_source,
            "request_dest": request_dest,
            "request_data_rate": request_data_rate,
            "arrival_time": arrival_time,
            "departure_time": departure_time,
            "path_indices": path_indices,
            "slot_indices": slot_indices,
            "returns": returns,
            "path_links": paths_list,
            "path_spectral_efficiency": spectral_efficiency_list,
            "required_slots": required_slots_list,
            "utilization": processed_data["utilisation"]["mean"],
            "bitrate_blocking_probability": processed_data["bitrate_blocking_probability"]["mean"],
            "service_blocking_probability": processed_data["service_blocking_probability"]["mean"],
            "path_length": path_lengths,
            "num_hops": num_hops,
        }
        if "gn_model" in config.env_type.lower():
            log_dict["launch_power"] = processed_data["launch_power"]["mean"]
            log_dict["path_snr"] = processed_data["path_snr"]["mean"]
        df = pd.DataFrame(log_dict)
        df.to_csv(config.TRAJ_DATA_OUTPUT_FILE)

    if config.log_path_lengths:
        path_lengths_mean = path_lengths.mean()
        path_lengths_std = path_lengths.std()
        path_lengths_iqr_upper = jnp.percentile(path_lengths, 75)
        path_lengths_iqr_lower = jnp.percentile(path_lengths, 25)
        num_hops_mean = num_hops.mean()
        num_hops_std = num_hops.std()
        print(f"Average path length mean: {path_lengths_mean:.0f}")
        print(f"Average path length std: {path_lengths_std:.0f}")
        print(f"Average path length IQR upper: {path_lengths_iqr_upper:.0f}")
        print(f"Average path length IQR lower: {path_lengths_iqr_lower:.0f}")
        print(f"Average number of hops mean: {num_hops_mean:.2f}")
        print(f"Average number of hops std: {num_hops_std:.2f}")
        print(f"Average number of hops IQR upper: {jnp.percentile(num_hops, 75):.2f}")
        print(f"Average number of hops IQR lower: {jnp.percentile(num_hops, 25):.2f}")
        # Get path lengths where returns are positive, 0 otherwise
        utilised_path_lengths = jnp.where(returns > 0, jnp.take(path_lengths, path_indices), 0)
        utilised_path_hops = jnp.where(returns > 0, jnp.take(num_hops, path_indices), 0)
        print(
            f"Average path length for successful actions mean: {utilised_path_lengths.mean():.0f}"
        )
        print(f"Average path length for successful actions std: {utilised_path_lengths.std():.0f}")
        print(
            f"Average path length for successful actions IQR upper: {jnp.percentile(utilised_path_lengths, 75):.0f}"
        )
        print(
            f"Average path length for successful actions IQR lower: {jnp.percentile(utilised_path_lengths, 25):.0f}"
        )
        print(
            f"Average number of hops for successful actions mean: {utilised_path_hops.mean():.2f}"
        )
        print(f"Average number of hops for successful actions std: {utilised_path_hops.std():.2f}")
        print(
            f"Average number of hops for successful actions IQR upper: {jnp.percentile(utilised_path_hops, 75):.2f}"
        )
        print(
            f"Average number of hops for successful actions IQR lower: {jnp.percentile(utilised_path_hops, 25):.2f}"
        )

        request_source = jnp.squeeze(merged_out["source"])
        request_dest = jnp.squeeze(merged_out["dest"])
        request_data_rate = jnp.squeeze(merged_out["data_rate"])
        path_indices = jnp.squeeze(merged_out["path_index"])
        slot_indices = jnp.squeeze(merged_out["slot_index"])

        # Compare the available paths
        df_path_links = pd.DataFrame(params.path_link_array.val).reset_index(drop=True)
        # Set config.weight = "weight" to use the length of the path for ordering else no. of hops
        config.weight = "weight" if not config.weight else None
        env, params = make(config)
        df_path_links_alt = pd.DataFrame(params.path_link_array.val).reset_index(drop=True)
        # Find rows that are unique to each dataframe
        # First, make a unique identifer for each row
        df_path_id = df_path_links.apply(lambda x: hash(tuple(x)), axis=1).reset_index(drop=True)
        df_path_id_alt = df_path_links_alt.apply(lambda x: hash(tuple(x)), axis=1).reset_index(
            drop=True
        )
        # Then check uniqueness (unique to the path ordering compared to alternate ordering)
        unique_paths = df_path_id[~df_path_id.isin(df_path_id_alt)]
        print(
            f"Fraction of paths that are unique to ordering: {len(unique_paths) / len(df_path_id):.2f}"
        )
        # Then for each path index we have, see if it corresponds to a unique path
        # Get indices of unique paths
        unique_path_indices = jnp.array(unique_paths.index)
        # Get the path indices of the requests
        unique_paths_used = jnp.isin(path_indices, unique_path_indices)
        # Remove elements from unique_paths_used that have negative returns
        unique_paths_used = jnp.where(returns > 0, unique_paths_used, 0)
        unique_paths_used_count = jnp.count_nonzero(unique_paths_used, axis=-1)
        positive_return_count = jnp.count_nonzero(jnp.where(returns > 0, returns, 0), axis=-1)
        unique_paths_used_mean = (
            (unique_paths_used_count / positive_return_count).reshape(-1).mean()
        )
        unique_paths_used_std = (unique_paths_used_count / positive_return_count).reshape(-1).std()
        unique_paths_used_iqr_upper = jnp.percentile(
            unique_paths_used_count / positive_return_count, 75
        )
        unique_paths_used_iqr_lower = jnp.percentile(
            unique_paths_used_count / positive_return_count, 25
        )
        print(
            f"Fraction of successful actions that use unique paths mean: {unique_paths_used_mean:.3f}"
        )
        print(
            f"Fraction of successful actions that use unique paths std: {unique_paths_used_std:.3f}"
        )
        print(
            f"Fraction of successful actions that use unique paths IQR upper: {unique_paths_used_iqr_upper:.3f}"
        )
        print(
            f"Fraction of successful actions that use unique paths IQR lower: {unique_paths_used_iqr_lower:.3f}"
        )


def print_metrics(
    processed_data: Dict[str, Dict[str, Array]], config: Union[Box, Dict[str, Any]]
) -> None:
    # Print the final metrics to console
    for metric in processed_data.keys():
        if config.get("continuous_operation", False):
            print(
                f"{metric}: {processed_data[metric]['mean'][-1].astype(np.float32):.5f} ± {processed_data[metric]['std'][-1].astype(np.float32):.5f}"
            )
            print(f"{metric} mean: {processed_data[metric]['mean'][-1].astype(np.float32):.5f}")
            print(f"{metric} std: {processed_data[metric]['std'][-1].astype(np.float32):.5f}")
            print(
                f"{metric} IQR lower: {processed_data[metric]['iqr_lower'][-1].astype(np.float32):.5f}"
            )
            print(
                f"{metric} IQR upper: {processed_data[metric]['iqr_upper'][-1].astype(np.float32):.5f}"
            )
        else:
            print(
                f"{metric}: {processed_data[metric]['episode_end_mean'].mean():.5f} ± {processed_data[metric]['episode_end_std'].mean():.5f}"
            )
            print(f"{metric} mean: {processed_data[metric]['episode_end_mean'].mean():.5f}")
            print(f"{metric} std: {processed_data[metric]['episode_end_std'].mean():.5f}")
            print(
                f"{metric} IQR lower: {processed_data[metric]['episode_end_iqr_lower'].mean():.5f}"
            )
            print(
                f"{metric} IQR upper: {processed_data[metric]['episode_end_iqr_upper'].mean():.5f}"
            )


def log_metrics(
    config: Box,
    out: Dict[str, Dict[str, Array]],
    total_time: float,
    merge_func: Callable,
    episode_count: int = 0,
    update_count: int = 0,
    step_count: int = 0,
) -> Tuple[Dict, Dict]:
    """Log metrics to wandb and/or save episode end metrics to CSV."""

    with TimeIt("Processing metrics"):
        merged_out, merged_out_loss, processed_data, episode_ends = process_metrics(
            config, out, total_time, merge_func
        )

    all_metrics = list(processed_data.keys())
    if not config.LOG_ALL_INFO:
        all_metrics = [
            "service_blocking_probability",
            "bitrate_blocking_probability",
            "accepted_services",
            "accepted_bitrate",
        ]

    with TimeIt("Logging metrics"):
        if config.DATA_OUTPUT_FILE:
            print("Saving metrics to file")
            # Save episode end metrics to file
            episode_end_df = pd.DataFrame(
                {
                    f"{metric}_{stat}": processed_data[metric][stat]
                    for metric in all_metrics
                    for stat in [
                        "episode_end_mean",
                        "episode_end_std",
                        "episode_end_iqr_upper",
                        "episode_end_iqr_lower",
                    ]
                }
            )
            # Check if data output file exists
            write_headers = not os.path.exists(config.DATA_OUTPUT_FILE)
            episode_end_df.to_csv(
                config.DATA_OUTPUT_FILE, mode="a", header=write_headers, index=False
            )
            # Pickle merged_out for further analysis
            with open(config.DATA_OUTPUT_FILE.replace(".csv", ".pkl"), "wb") as f:
                pickle.dump(merged_out, f)

        if config.WANDB:
            print("Logging metrics to wandb")

            if not config.continuous_operation:
                # Log episode end metrics
                print(f"Logging episode end metrics for {np.sum(episode_ends)} episodes")
                for i in range(len(processed_data[all_metrics[0]]["episode_end_mean"])):
                    log_dict = {
                        f"{metric}_{stat}": processed_data[metric][stat][i]
                        for metric in all_metrics
                        for stat in [
                            "episode_end_mean",
                            "episode_end_std",
                            "episode_end_iqr_upper",
                            "episode_end_iqr_lower",
                        ]
                    }
                    log_dict["episode_count"] = i + episode_count
                    wandb.log(log_dict)

            else:
                # Log metrics from every step
                # Define the downsample factor to speed up upload to wandb
                # Then reshape the array and compute the mean
                training_time = (
                    jnp.arange(len(processed_data[all_metrics[0]]["mean"]))
                    / len(processed_data[all_metrics[0]]["mean"])
                    * total_time
                )

                chop = len(processed_data[all_metrics[0]]["mean"]) % config.DOWNSAMPLE_FACTOR

                def downsample_mean(x: Array) -> Array:
                    x = jnp.asarray(x)
                    return x[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)

                for key in all_metrics:
                    processed_data[key]["mean"] = downsample_mean(processed_data[key]["mean"])
                    processed_data[key]["std"] = downsample_mean(processed_data[key]["std"])
                    processed_data[key]["iqr_upper"] = downsample_mean(
                        processed_data[key]["iqr_upper"]
                    )
                    processed_data[key]["iqr_lower"] = downsample_mean(
                        processed_data[key]["iqr_lower"]
                    )
                training_time = downsample_mean(training_time)

                # Log per step metrics
                print("Logging per step metrics")
                for i in range(len(processed_data[all_metrics[0]]["mean"])):
                    log_dict = {
                        f"{metric}_{agg}": processed_data[metric][agg][i]
                        for metric in all_metrics
                        for agg in ["mean", "std", "iqr_upper", "iqr_lower"]
                    }
                    log_dict["training_time"] = training_time[i]
                    log_dict["env_step"] = (i*config.DOWNSAMPLE_FACTOR) + step_count
                    wandb.log(log_dict)

            if config.LOG_LOSS_INFO and merged_out_loss is not None:
                print("Logging loss info")
                for i in range(len(merged_out_loss["loss/total_loss"])):
                    log_dict = {f"{metric}": merged_out_loss[metric][i] for metric in loss_metrics}
                    log_dict["update_epoch"] = i + update_count
                    wandb.log(log_dict)

    return merged_out, processed_data
