import chex
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Any, Callable, Union, Tuple, Dict
import absl
import box
import optax
from box import Box
from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training import orbax_utils
import orbax.checkpoint
import pathlib
import wandb
import distrax
import math
import pickle
import subprocess
import matplotlib.pyplot as plt
from jax import Array

from xlron.environments.gn_model import *
from xlron.environments.make_env import make
from xlron.models.models import ActorCriticGNN, ActorCriticMLP, LaunchPowerActorCriticMLP
from xlron.environments.dataclasses import EnvState, EvalState
from xlron.environments.wrappers import TimeIt
from xlron.environments.env_funcs import init_link_length_array, make_graph, process_path_action, get_launch_power, get_paths
from xlron.heuristics.heuristics import ksp_ff, ff_ksp, kmc_ff, kmf_ff, ksp_mu, mu_ksp, kca_ff, kme_ff, ksp_bf, bf_ksp, ksp_lf
from xlron.dtype_config import COMPUTE_DTYPE, PARAMS_DTYPE, LARGE_INT_DTYPE, LARGE_FLOAT_DTYPE, \
    SMALL_INT_DTYPE, SMALL_FLOAT_DTYPE, MED_INT_DTYPE

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

class TrainState(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Args:
        step: Counter starts at 0 and is incremented by every call to ``.apply_gradients()``.
        apply_fn: Usually set to ``model.apply()``. Kept in this dataclass for convenience
        sample_fn: A function that samples actions from the policy.
        to have a shorter params list for the ``train_step()`` function in your training loop.
        params: The parameters to be updated by ``tx`` and used by ``apply_fn``.
        tx: An Optax gradient transformation.
        opt_state: The state for ``tx``.
    """

    step: Union[int, jax.Array]
    apply_fn: Callable = struct.field(pytree_node=False)
    sample_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    ent_schedule: Callable = struct.field(pytree_node=False)
    avg_reward: jax.Array = struct.field(pytree_node=True, default=0.0)
    reward_stepsize: jax.Array = struct.field(pytree_node=True, default=0.01)
    reward_stepsize_init: jax.Array = struct.field(pytree_node=True, default=0.01)
    reward_stepsize_offset: jax.Array = struct.field(pytree_node=True, default=1.0)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.

        Note that internally this function calls ``.tx.update()`` followed by a call
        to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

        Args:
          grads: Gradients that have the same pytree structure as ``.params``.
          **kwargs: Additional dataclass attributes that should be ``.replace()``-ed.

        Returns:
          An updated instance of ``self`` with ``step`` incremented by one, ``params``
          and ``opt_state`` updated by applying ``grads``, and additional attributes
          replaced as specified by ``kwargs``.
        """
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads['params']
            params_with_opt = self.params['params']
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)

        # As implied by the OWG name, the gradients are used directly to update the
        # parameters.
        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                'params': new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, sample_fn=lambda x: x, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.
        params_with_opt = (
            params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        )
        opt_state = tx.init(params_with_opt)
        return cls(
            step=jnp.array(0),
            apply_fn=apply_fn,
            sample_fn=sample_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def update_step_size(self):
        """Updates the step size used for reward centering."""
        reward_stepsize_offset = self.reward_stepsize_offset + self.reward_stepsize_init * (1 - self.reward_stepsize_offset)
        reward_stepsize = self.reward_stepsize_init / reward_stepsize_offset
        return self.replace(
            reward_stepsize=reward_stepsize,
            reward_stepsize_offset=reward_stepsize_offset,
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
    return jax.tree_util.tree_map(lambda x: x[(0,) * unreplicate_depth], x)  # type: ignore


def unreplicate_batch_dim(x: chex.ArrayTree) -> chex.ArrayTree:
    """Unreplicated just the update batch dimension.
    (The dimension that is vmapped over when acting and learning)

    In stoix's case it is always the second dimension, after the device dimension.
    We simply take element 0 as the params are identical across this dimension.
    """
    return jax.tree_util.tree_map(lambda x: x[:, 0, ...], x)  # type: ignore


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def save_model(train_state: TrainState, run_name, config: Union[box.Box, absl.flags.FlagValues]):
    config_dict = config.to_dict() if isinstance(config, box.Box) else config
    save_data = {"model": train_state, "config": config_dict}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(save_data)
    # Get path to current file
    model_path = pathlib.Path(config.MODEL_PATH) if config.MODEL_PATH is not None else (
            pathlib.Path(__file__).resolve().parents[2] / "models" / run_name)
    # If model_path dir already exists, append a number to the end
    i = 1
    model_path_og = model_path
    while model_path.exists():
        # Add index to end of model_path
        model_path = pathlib.Path(str(model_path_og) + f"_{i}") if config.MODEL_PATH else model_path_og.parent / (
                model_path_og.name + f"_{i}")
        i += 1
    print(f"Saving model to {model_path.absolute()}")
    orbax_checkpointer.save(model_path.absolute(), save_data, save_args=save_args)
    # Upload model to wandb
    if config.WANDB:
        print((model_path / "*").absolute())
        wandb.save(str((model_path / "*").absolute()), base_path=str(model_path.parent))


def init_network(config, env, env_state, env_params):
    if config.env_type.lower() == "vone":
        network = ActorCriticMLP(env.action_space(env_params).n,
                                 activation=config.ACTIVATION,
                                 num_layers=config.NUM_LAYERS,
                                 num_units=config.NUM_UNITS,
                                 layer_norm=config.mlp_layer_norm, )
        init_x = tuple([jnp.zeros(env.observation_space(env_params).n, dtype=PARAMS_DTYPE)])
    elif config.env_type.lower() in ["rsa", "rmsa", "rwa", "deeprmsa", "rwa_lightpath_reuse", "rsa_gn_model", "rmsa_gn_model", "rsa_multiband"]:
        if config.USE_GNN:
            if "gn_model" in config.env_type.lower() and config.output_globals_size_actor > 0:
                global_output_size_actor = int((env_params.max_power - env_params.min_power) / env_params.step_power) + 1 if config.discrete_launch_power else 1
            else:
                global_output_size_actor = config.global_output_size_actor
            network = ActorCriticGNN(
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
                edge_output_size_actor=math.ceil(env_params.link_resources / env_params.aggregate_slots),
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
                gnn_layer_norm=config.gnn_layer_norm,
                mlp_layer_norm=config.mlp_layer_norm,
                normalise_by_link_length=config.normalize_by_link_length,
                vmap=False,
                discrete=config.discrete_launch_power,
                min_power_dbm=config.min_power,
                max_power_dbm=config.max_power,
                step_power_dbm=config.step_power,
                # Bools to determine which actions to output
                output_path = config.GNN_OUTPUT_RSA,
                output_power = config.GNN_OUTPUT_LP,
            )
            init_x = (env_state.env_state, env_params)
        elif "gn_model" in config.env_type.lower() and env_params.launch_power_type == 3:
            network = LaunchPowerActorCriticMLP(
                action_dim=env.action_space(env_params).n,
                activation=config.ACTIVATION,
                num_layers=config.NUM_LAYERS,
                num_units=config.NUM_UNITS,
                layer_norm=config.mlp_layer_norm,
                discrete=config.discrete_launch_power,
                min_power_dbm=config.min_power,
                max_power_dbm=config.max_power,
                step_power_dbm=config.step_power,
                k_paths=env_params.k_paths,
            )
            init_x = tuple([jnp.zeros(env.observation_space(env_params).n, dtype=PARAMS_DTYPE)])
        else:
            network = ActorCriticMLP(env.action_space(env_params).n,
                                     activation=config.ACTIVATION,
                                     num_layers=config.NUM_LAYERS,
                                     num_units=config.NUM_UNITS,
                                     layer_norm=config.mlp_layer_norm, )

            init_x = tuple([jnp.zeros(env.observation_space(env_params).n, dtype=PARAMS_DTYPE)])
    else:
        raise ValueError(f"Invalid environment type {config.env_type}")
    return network, init_x


def experiment_data_setup(config: absl.flags.FlagValues, rng: chex.PRNGKey) -> Tuple:
    # INIT ENV
    env, env_params = make(config)
    rng, rng_step, rng_epoch, warmup_key, reset_key, network_key = jax.random.split(rng, 6)
    reset_key = jax.random.split(reset_key, config.NUM_ENVS) if config.NUM_ENVS > 1 else reset_key
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params) if config.NUM_ENVS > 1 else env.reset(reset_key, env_params)
    obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])

    # TRAINING MODE
    if not config.EVAL_HEURISTIC and not config.EVAL_MODEL:

        # INIT NETWORK
        network, init_x = init_network(config, env, env_state, env_params)
        init_x = (jax.tree.map(lambda x: x[0], init_x[0]), init_x[1]) if config.USE_GNN else init_x

        if config.RETRAIN_MODEL:
            config_dict = config.to_dict() if isinstance(config, box.Box) else config
            network_params = config_dict["model"]["model"]["params"]
            print('Retraining model')
        else:
            network_params = network.init(network_key, *init_x)

        # INIT LEARNING RATE SCHEDULE AND OPTIMIZER
        lr_schedule = make_lr_schedule(config)
        ent_schedule = make_ent_schedule(config)
        tx = optax.chain(
            optax.clip_by_global_norm(config.MAX_GRAD_NORM),
            optax.adam(learning_rate=lr_schedule, eps=config.ADAM_EPS, b1=config.ADAM_BETA1, b2=config.ADAM_BETA2, mu_dtype=COMPUTE_DTYPE),
        )

        runner_state = TrainState.create(
            apply_fn=network.apply,
            sample_fn=network.sample_action,
            params=network_params.to_dict() if isinstance(network_params, box.Box) else network_params,
            tx=tx,
            ent_schedule=ent_schedule,
            avg_reward=jnp.array(config.INITIAL_AVERAGE_REWARD, dtype=LARGE_FLOAT_DTYPE),
        )

    # EVALUATION MODE
    else:
        # LOAD MODEL
        if config.EVAL_HEURISTIC:
            network_params = apply = sample = None

        elif config.EVAL_MODEL:
            network, last_obs = init_network(config, env, env_state, env_params)
            network_params = config.model.to_dict()["model"]["params"]
            apply = network.apply
            sample = network.sample_action
            print('Evaluating model')

        runner_state = EvalState(apply_fn=apply, sample_fn=sample, params=network_params)

    # Recreate DeepRMSA warmup period
    warmup_key = jax.random.split(warmup_key, config.NUM_ENVS) if config.NUM_ENVS > 1 else warmup_key
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
    last_obs = (env_state.env_state, env_params) if config.USE_GNN else last_obs
    pi, value = train_state.apply_fn(train_state.params, *last_obs)
    # Action masking
    env_state = env_state.replace(env_state=env.action_mask(env_state.env_state, env_params))

    # Always do action masking with VONE
    if config.env_type.lower() == "vone":
        # TODO - change this to work with single set of logits (probably just slice them)
        vmap_mask_nodes = jax.vmap(env.action_mask_nodes, in_axes=(0, None))
        vmap_mask_slots = jax.vmap(env.action_mask_slots, in_axes=(0, None, 0))
        vmap_mask_dest_node = jax.vmap(env.action_mask_dest_node, in_axes=(0, None, 0))

        env_state = env_state.replace(env_state=vmap_mask_nodes(env_state.env_state, env_params))
        pi_source = distrax.Categorical(logits=jnp.where(env_state.env_state.node_mask_s, pi[0]._logits, -1e8))

        action_s = pi_source.sample(seed=action_key) if not config.deterministic else pi_source.mode()

        # Update destination mask now source has been selected
        env_state = env_state.replace(env_state=vmap_mask_dest_node(env_state.env_state, env_params, action_s))
        pi_dest = distrax.Categorical(logits=jnp.where(env_state.env_state.node_mask_d, pi[0]._logits, -1e8))

        action_p = jnp.full(action_s.shape, 0)
        action_d = pi_dest.sample(seed=action_key) if not config.deterministic else pi_dest.mode()
        action = jnp.stack((action_s, action_p, action_d), axis=1)

        env_state = env_state.replace(env_state=vmap_mask_slots(env_state.env_state, env_params, action))
        pi_path = distrax.Categorical(logits=jnp.where(env_state.env_state.link_slot_mask, pi[0]._logits, -1e8))
        action_p = pi_path.sample(seed=action_key) if not config.deterministic else pi_path.mode()
        action = jnp.stack((action_s, action_p, action_d), axis=1)

        log_prob_source = pi_source.log_prob(action_s)
        log_prob_path = pi_path.log_prob(action_p)
        log_prob_dest = pi_dest.log_prob(action_d)
        log_prob = log_prob_dest + log_prob_path + log_prob_source

    elif "gn_model" in config.env_type.lower() and config.launch_power_type == "rl":
        pi_masked = distrax.Categorical(logits=jnp.where(env_state.env_state.link_slot_mask, pi[0]._logits, -1e8))
        if config.GNN_OUTPUT_RSA and not config.GNN_OUTPUT_LP:
            path_action, log_prob = train_state.sample_fn(action_key, pi_masked, log_prob=True, deterministic=config.deterministic)
            power_action = jnp.array([env_params.default_launch_power])
        else:
            if config.GNN_OUTPUT_RSA and config.GNN_OUTPUT_LP:
                path_action, power_action, log_prob = train_state.sample_fn(action_key, (pi_masked, pi[1]), log_prob=True, deterministic=config.deterministic)
            else:
                power_action, log_prob = train_state.sample_fn(action_key, pi[1], log_prob=True, deterministic=config.deterministic)
        inner_state = env_state.env_state.replace(launch_power_array=power_action)
        env_state = env_state.replace(env_state=inner_state)
        if not config.GNN_OUTPUT_RSA:
            path_action = ksp_lf(env_state.env_state, env_params) if env_params.last_fit is True else ksp_ff(env_state.env_state, env_params)
        if config.output_globals_size_actor == 0:
            path_index, _ = process_path_action(env_state.env_state, env_params, path_action)
            power_action, log_prob = power_action[path_index], log_prob[path_index]
        action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))], axis=0)
        # if config.GNN_OUTPUT_RSA:
        #     if config.GNN_OUTPUT_LP:
        #         path_action, power_action, log_prob = train_state.sample_fn(action_key, (pi_masked, pi[1]), log_prob=True, deterministic=config.deterministic)
        #     else:
        #         path_action, log_prob = train_state.sample_fn(action_key, pi_masked, log_prob=True, deterministic=config.deterministic)
        #         power_action = jnp.array([env_params.default_launch_power])
        # else:
        #     # TODO(GNN_LP) - modify this so its possible to have routing from GNN and LP from another source (maybe)
        #     # TODO(GNN_LP) - do a foriloop to mark each of the selected path links, get a power action (and log prob) for each,
        #     #  then select a path action and then select the power action corresponding to that path action
        #     power_action, log_prob = train_state.sample_fn(action_key, pi[1], log_prob=True, deterministic=config.deterministic)
        #
        # if config.GNN_OUTPUT_RSA and config.GNN_OUTPUT_LP:
        # inner_state = env_state.env_state.replace(launch_power_array=power_action)
        # env_state = env_state.replace(env_state=inner_state)
        #
        # path_action = ksp_lf(env_state.env_state, env_params) if env_params.last_fit is True else ksp_ff(env_state.env_state, env_params)
        #
        # if config.output_globals_size_actor == 0:
        #     path_index, _ = process_path_action(env_state.env_state, env_params, path_action)
        #     power_action, log_prob = power_action[path_index], log_prob[path_index]
        #
        # action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))], axis=0)

    else:
        env_state = env_state.replace(env_state=env.action_mask(env_state.env_state, env_params))
        pi_masked = distrax.Categorical(logits=jnp.where(env_state.env_state.link_slot_mask, pi[0]._logits, -1e8))
        action = pi_masked.sample(seed=action_key) if not config.deterministic else pi_masked.mode()
        log_prob = pi_masked.log_prob(action)

    return env_state, action, log_prob, value


def select_action_eval(select_action_state, env, env_params, eval_state, config):

    rng, env_state, last_obs = select_action_state

    if config.EVAL_HEURISTIC:

        # Action masking
        env_state = env_state.replace(env_state=env.action_mask(env_state.env_state, env_params))

        if config.env_type.lower() == "vone":
            raise NotImplementedError(f"VONE heuristics not yet implemented")

        elif config.env_type.lower() in ["rsa", "rwa", "rmsa", "deeprmsa", "rwa_lightpath_reuse", "rsa_gn_model", "rmsa_gn_model", "rsa_multiband"]:
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
                    raise ValueError("launch_power_type cannot be 'rl' when --EVAL_HEURISTIC flag is True")
                launch_power = get_launch_power(env_state.env_state, action, action, env_params)
                action = jnp.concatenate([action.reshape((1,)), launch_power.reshape((1,))], axis=0)
        else:
            raise ValueError(f"Invalid environment type {config.env_type}")
        # For DeepRMSA env, the action can only be the path index (first-fit for spectrum always)
        if config.env_type.lower() == "deeprmsa":
            action = process_path_action(env_state.env_state, env_params, action)[0]
    else:
        env_state, action, _, _ = select_action(select_action_state, env, env_params, eval_state, config)
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
            _state, action, log_prob, value = action_fn(select_action_state, env, _params, _train_state, config)
            if "gn_model" in config.env_type.lower() and config.launch_power_type == "rl":
                # If the action is launch power, the action is this shape:
                # jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))], axis=0)
                # We want to overwrite the launch power with a default launch_power
                path_action = ksp_lf(_state.env_state, _params) if _params.last_fit is True else ksp_ff(_state.env_state, _params)
                action = jnp.concatenate([path_action.reshape((1,)), jnp.array([params.default_launch_power,])], axis=0)
            elif "gn_model" in config.env_type.lower() and config.launch_power_type != "rl" and not config.EVAL_HEURISTIC:
                raise ValueError("Check that EVAL_HEURISTIC is set to True if using a heuristic")
            # STEP ENV
            obsv, _state, reward, done, info = env.step(
                step_key, _state, action, params
            )
            obsv = (_state.env_state, params) if config.USE_GNN else tuple([obsv])
            return _rng, _state, _params, _train_state, obsv

        vals = jax.lax.fori_loop(0, config.ENV_WARMUP_STEPS, warmup_step,
                                 (rng, state, params, train_state, last_obs))

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
    wandb.define_metric('episode_count')
    wandb.define_metric("env_step")
    for metric in metrics:
        for agg in [
            "mean",
            "std",
            "iqr_upper",
            "iqr_lower"
        ]:
            wandb.define_metric(f"{metric}_{agg}", step_metric="env_step")
            wandb.define_metric(f"{metric}_episode_end_{agg}", step_metric="episode_count", summary="last")
            wandb.define_metric(f"{metric}_episode_end_{agg}", step_metric="episode_count", summary="mean")
            wandb.define_metric(f"{metric}_episode_end_{agg}", step_metric="episode_count", summary="min")
            wandb.define_metric(f"{metric}_episode_end_{agg}", step_metric="episode_count", summary="max")
    for metric in loss_metrics:
        wandb.define_metric(f"{metric}", step_metric="update_epoch")
    wandb.define_metric("training_time", step_metric="env_step")


def get_mean_std_iqr(x, y):
    _mean = x[y].mean(axis=0).reshape(-1)
    _std = x[y].std(axis=0).reshape(-1)
    _iqr_upper = jnp.percentile(x[y], 75, axis=0).reshape(-1)
    _iqr_lower = jnp.percentile(x[y], 25, axis=0).reshape(-1)
    return np.array(_mean), np.array(_std), np.array(_iqr_upper), np.array(_iqr_lower)


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
    if not config.EVAL_HEURISTIC or not config.EVAL_MODEL:
        merged_out_loss = {k: jax.tree.map(lambda x: x.reshape((-1,)), v) for k, v in out.get("loss_info", {}).items()}
    else:
        merged_out_loss = None

    # Calculate blocking probabilities
    merged_out["service_blocking_probability"] = 1 - (
            merged_out["accepted_services"] / jnp.where(merged_out["lengths"] == 0, 1, merged_out["lengths"])
    )
    merged_out["bitrate_blocking_probability"] = 1 - (
            merged_out["accepted_bitrate"] / jnp.where(merged_out["total_bitrate"] == 0, 1, merged_out["total_bitrate"])
    )

    # Calculate episode ends
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
            ends = np.concatenate([[False], episode_ends.flatten()[:-1]]).reshape(episode_ends.shape)
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
            print(f"KeyError for metric {metric}, skipping")
            continue
    return merged_out, merged_out_loss, processed_data, episode_ends
    
    
def plot_metrics(experiment_name: str, processed_data: Dict[str, Any], config: Union[Box, Dict[str, Any]]) -> None:
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

    plot_metric = moving_average(plot_metric, min(100, int(len(plot_metric) / 2)))
    plot_metric_upper = moving_average(plot_metric_upper, min(100, int(len(plot_metric_upper) / 2)))
    plot_metric_lower = moving_average(plot_metric_lower, min(100, int(len(plot_metric_lower) / 2)))
    plt.plot(plot_metric)
    plt.fill_between(
        range(len(plot_metric)),
        plot_metric_lower,
        plot_metric_upper,
        alpha=0.2
    )
    plt.xlabel("Environment Step" if not config.incremental_loading else "Episode Count")
    plt.ylabel(plot_metric_name)
    plt.title(experiment_name)
    plt.show()


def log_actions(merged_out, processed_data, config):
    print(f"Logging actions. \
        N.B. data is only logged from most recent increment. \
        Total increments: {config.NUM_INCREMENTS}")
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

    for path_index, slot_index, source, dest, data_rate in zip(path_indices, slot_indices, request_source, request_dest, request_data_rate):
        source, dest = source.reshape(1), dest.reshape(1)
        path_links = get_paths(params, jnp.concatenate([source, dest]))[path_index % params.k_paths]
        # Make path links into a string
        path_str = "".join([str(x.astype(LARGE_INT_DTYPE)) for x in path_links])
        paths_list.append(path_str)
        path_spectral_efficiency = params.path_se_array.val[path_index]
        required_slots = int(jnp.ceil(data_rate / (path_spectral_efficiency*params.slot_size)))
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
        path_lengths_iqr_upper = np.percentile(path_lengths, 75)
        path_lengths_iqr_lower = np.percentile(path_lengths, 25)
        num_hops_mean = num_hops.mean()
        num_hops_std = num_hops.std()
        print(f"Average path length mean: {path_lengths_mean:.0f}")
        print(f"Average path length std: {path_lengths_std:.0f}")
        print(f"Average path length IQR upper: {path_lengths_iqr_upper:.0f}")
        print(f"Average path length IQR lower: {path_lengths_iqr_lower:.0f}")
        print(f"Average number of hops mean: {num_hops_mean:.2f}")
        print(f"Average number of hops std: {num_hops_std:.2f}")
        print(f"Average number of hops IQR upper: {np.percentile(num_hops, 75):.2f}")
        print(f"Average number of hops IQR lower: {np.percentile(num_hops, 25):.2f}")
        # Get path lengths where returns are positive, 0 otherwise
        utilised_path_lengths = jnp.where(returns > 0, jnp.take(path_lengths, path_indices), 0)
        utilised_path_hops = jnp.where(returns > 0, jnp.take(num_hops, path_indices), 0)
        print(f"Average path length for successful actions mean: {utilised_path_lengths.mean():.0f}")
        print(f"Average path length for successful actions std: {utilised_path_lengths.std():.0f}")
        print(f"Average path length for successful actions IQR upper: {np.percentile(utilised_path_lengths, 75):.0f}")
        print(f"Average path length for successful actions IQR lower: {np.percentile(utilised_path_lengths, 25):.0f}")
        print(f"Average number of hops for successful actions mean: {utilised_path_hops.mean():.2f}")
        print(f"Average number of hops for successful actions std: {utilised_path_hops.std():.2f}")
        print(f"Average number of hops for successful actions IQR upper: {np.percentile(utilised_path_hops, 75):.2f}")
        print(f"Average number of hops for successful actions IQR lower: {np.percentile(utilised_path_hops, 25):.2f}")

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
        df_path_id_alt = df_path_links_alt.apply(lambda x: hash(tuple(x)), axis=1).reset_index(drop=True)
        # Then check uniqueness (unique to the path ordering compared to alternate ordering)
        unique_paths = df_path_id[~df_path_id.isin(df_path_id_alt)]
        print(f"Fraction of paths that are unique to ordering: {len(unique_paths) / len(df_path_id):.2f}")
        # Then for each path index we have, see if it corresponds to a unique path
        # Get indices of unique paths
        unique_path_indices = jnp.array(unique_paths.index)
        # Get the path indices of the requests
        unique_paths_used = jnp.isin(path_indices, unique_path_indices)
        # Remove elements from unique_paths_used that have negative returns
        unique_paths_used = jnp.where(returns > 0, unique_paths_used, 0)
        unique_paths_used_count = jnp.count_nonzero(unique_paths_used, axis=-1)
        positive_return_count = jnp.count_nonzero(jnp.where(returns > 0, returns, 0), axis=-1)
        unique_paths_used_mean = (unique_paths_used_count / positive_return_count).reshape(-1).mean()
        unique_paths_used_std = (unique_paths_used_count / positive_return_count).reshape(-1).std()
        unique_paths_used_iqr_upper = np.percentile(unique_paths_used_count / positive_return_count, 75)
        unique_paths_used_iqr_lower = np.percentile(unique_paths_used_count / positive_return_count, 25)
        print(f"Fraction of successful actions that use unique paths mean: {unique_paths_used_mean:.3f}")
        print(f"Fraction of successful actions that use unique paths std: {unique_paths_used_std:.3f}")
        print(f"Fraction of successful actions that use unique paths IQR upper: {unique_paths_used_iqr_upper:.3f}")
        print(f"Fraction of successful actions that use unique paths IQR lower: {unique_paths_used_iqr_lower:.3f}")


def print_metrics(processed_data: Dict[str, Dict[str, Array]], config: Union[Box, Dict[str, Any]]) -> None:
    # Print the final metrics to console
    for metric in processed_data.keys():
        if config.continuous_operation:
            print(f"{metric}: {processed_data[metric]['mean'][-1].astype(np.float32):.5f} ± {processed_data[metric]['std'][-1].astype(np.float32):.5f}")
            print(f"{metric} mean: {processed_data[metric]['mean'][-1].astype(np.float32):.5f}")
            print(f"{metric} std: {processed_data[metric]['std'][-1].astype(np.float32):.5f}")
            print(f"{metric} IQR lower: {processed_data[metric]['iqr_lower'][-1].astype(np.float32):.5f}")
            print(f"{metric} IQR upper: {processed_data[metric]['iqr_upper'][-1].astype(np.float32):.5f}")
        else:
            print(f"{metric}: {processed_data[metric]['episode_end_mean'].mean():.5f} ± {processed_data[metric]['episode_end_std'].mean():.5f}")
            print(f"{metric} mean: {processed_data[metric]['episode_end_mean'].mean():.5f}")
            print(f"{metric} std: {processed_data[metric]['episode_end_std'].mean():.5f}")
            print(f"{metric} IQR lower: {processed_data[metric]['episode_end_iqr_lower'].mean():.5f}")
            print(f"{metric} IQR upper: {processed_data[metric]['episode_end_iqr_upper'].mean():.5f}")


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
            episode_end_df.to_csv(config.DATA_OUTPUT_FILE, mode='a', header=write_headers, index=False)
            # Pickle merged_out for further analysis
            with open(config.DATA_OUTPUT_FILE.replace(".csv", ".pkl"), "wb") as f:
                pickle.dump(merged_out, f)
    
        if config.WANDB:
            print("Logging metrics to wandb")
    
            if not config.continuous_operation:
                # Log episode end metrics
                print(f"Logging episode end metrics for {np.sum(episode_ends)} episodes")
                for i in range(len(processed_data["returns"]["episode_end_mean"])):
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
                    jnp.arange(len(processed_data["returns"]["mean"]))
                    / len(processed_data["returns"]["mean"])
                    * total_time
                )
    
                chop = len(processed_data["returns"]["mean"]) % config.DOWNSAMPLE_FACTOR
    
                def downsample_mean(x: ArrayLike) -> Array:
                    x = jnp.asarray(x)
                    return x[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
    
                for key in all_metrics:
                    processed_data[key]["mean"] = downsample_mean(processed_data[key]["mean"])
                    processed_data[key]["std"] = downsample_mean(processed_data[key]["std"])
                    processed_data[key]["iqr_upper"] = downsample_mean(processed_data[key]["iqr_upper"])
                    processed_data[key]["iqr_lower"] = downsample_mean(processed_data[key]["iqr_lower"])
                training_time = downsample_mean(training_time)
    
                # Log per step metrics
                print("Logging per step metrics")
                for i in range(len(processed_data["returns"]["mean"])):
                    log_dict = {
                        f"{metric}_{agg}": processed_data[metric][agg][i]
                        for metric in all_metrics
                        for agg in ["mean", "std", "iqr_upper", "iqr_lower"]
                    }
                    log_dict["training_time"] = training_time[i]
                    log_dict["env_step"] = i + step_count
                    wandb.log(log_dict)
    
            if config.LOG_LOSS_INFO and merged_out_loss is not None:
                print("Logging loss info")
                for i in range(len(merged_out_loss["loss/total_loss"])):
                    log_dict = {f"{metric}": merged_out_loss[metric][i] for metric in loss_metrics}
                    log_dict["update_epoch"] = i + update_count
                    wandb.log(log_dict)

    return merged_out, processed_data
