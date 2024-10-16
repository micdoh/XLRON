import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Any, Callable, Union, Sequence, NamedTuple, Any, Tuple
import absl
import optax
from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training import orbax_utils
import orbax.checkpoint
import pathlib
import wandb
import distrax
import math
import matplotlib.pyplot as plt
from xlron.environments.wrappers import LogWrapper
from xlron.environments.vone import make_vone_env
from xlron.environments.rsa import make_rsa_env
from xlron.models.models import ActorCriticGNN, ActorCriticMLP
from xlron.environments.dataclasses import EnvState
from xlron.environments.env_funcs import init_link_length_array, make_graph


class TrainState(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Args:
        step: Counter starts at 0 and is incremented by every call to ``.apply_gradients()``.
        apply_fn: Usually set to ``model.apply()``. Kept in this dataclass for convenience
        to have a shorter params list for the ``train_step()`` function in your training loop.
        params: The parameters to be updated by ``tx`` and used by ``apply_fn``.
        tx: An Optax gradient transformation.
        opt_state: The state for ``tx``.
    """

    step: Union[int, jax.Array]
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)

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
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.
        params_with_opt = (
            params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        )
        opt_state = tx.init(params_with_opt)
        return cls(
            step=jnp.array(0),
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


def scale_gradient(g: chex.Array, scale: float = 1) -> chex.Array:
    """Scales the gradient of `g` by `scale` but keeps the original value unchanged."""
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)


def count_parameters(params: chex.ArrayTree) -> int:
    """Counts the number of parameters in a parameter tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def ndim_at_least(x: chex.Array, num_dims: chex.Numeric) -> chex.Array:
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


def save_model(train_state: TrainState, run_name, flags: absl.flags.FlagValues):
    save_data = {"model": train_state, "config": flags}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(save_data)
    # Get path to current file
    model_path = pathlib.Path(flags.MODEL_PATH) if flags.MODEL_PATH else pathlib.Path(__file__).resolve().parents[
                                                                             2] / "models" / run_name
    # If model_path dir already exists, append a number to the end
    i = 1
    model_path_og = model_path
    while model_path.exists():
        # Add index to end of model_path
        model_path = pathlib.Path(str(model_path_og) + f"_{i}") if flags.MODEL_PATH else model_path_og.parent / (
                model_path_og.name + f"_{i}")
        i += 1
    print(f"Saving model to {model_path.absolute()}")
    orbax_checkpointer.save(model_path.absolute(), save_data, save_args=save_args)
    # Upload model to wandb
    if flags.WANDB:
        print((model_path / "*").absolute())
        wandb.save(str((model_path / "*").absolute()), base_path=str(model_path.parent))


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
                                 layer_norm=config.LAYER_NORM, )
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
                                     layer_norm=config.LAYER_NORM, )

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

        return vals[1], vals[4]

    return warmup_fn


def make_lr_schedule(config):
    def linear_schedule(count):
        frac = (1.0 - (count // (config.NUM_MINIBATCHES * config.UPDATE_EPOCHS)) /
                (config.NUM_UPDATES * config.SCHEDULE_MULTIPLIER))
        return config.LR * frac

    def lr_schedule(count):
        total_steps = config.NUM_UPDATES * config.UPDATE_EPOCHS * config.NUM_MINIBATCHES * config.SCHEDULE_MULTIPLIER
        if config.LR_SCHEDULE == "warmup_cosine":
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=config.LR,
                peak_value=config.LR * config.WARMUP_PEAK_MULTIPLIER,
                warmup_steps=int(total_steps * config.WARMUP_STEPS_FRACTION),
                decay_steps=total_steps,
                end_value=config.LR * config.WARMUP_END_FRACTION)
        elif config.LR_SCHEDULE == "linear":
            schedule = linear_schedule
        elif config.LR_SCHEDULE == "constant":
            schedule = lambda x: config.LR
        else:
            raise ValueError(f"Invalid LR schedule {config.LR_SCHEDULE}")
        return schedule(count)

    return lr_schedule


def reshape_keys(keys, size1, size2):
    dimensions = (size1, size2)
    reshape = lambda x: x.reshape(dimensions + x.shape[1:])
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
    wandb.define_metric("lengths", step_metric="env_step")
    wandb.define_metric("returns", step_metric="env_step")
    wandb.define_metric("cum_returns", step_metric="update_step")
    wandb.define_metric("episode_accepted_services", step_metric="episode_count")
    wandb.define_metric("episode_accepted_services_std", step_metric="episode_count")
    wandb.define_metric("episode_accepted_bitrate", step_metric="episode_count")
    wandb.define_metric("episode_accepted_bitrate_std", step_metric="episode_count")
    wandb.define_metric("episode_end_training_time", step_metric="episode_count")


def log_metrics(config, out, experiment_name, total_time, merge_func):
    merged_out = {k: jax.tree.map(merge_func, v) for k, v in out["metrics"].items()}
    get_mean = lambda x, y: x[y].mean(0).reshape(-1)
    get_std = lambda x, y: x[y].std(0).reshape(-1)

    if config.end_first_blocking:
        # Episode lengths are variable so return the episode end values and the std of the episode end values
        episode_ends = np.where(merged_out["done"].reshape(-1) == 1)[0] - 1
        get_episode_end_mean = lambda x: x.reshape(-1)[episode_ends]
        get_episode_end_std = lambda x: jnp.full(x.reshape(-1)[episode_ends].shape, x.reshape(-1)[episode_ends].std())
    else:
        # Episode lengths are uniform so return the mean and std across envs at each episode end
        episode_ends = np.where(merged_out["done"].mean(0).reshape(-1) == 1)[0] - 1 \
            if not config.continuous_operation else np.arange(0, config.TOTAL_TIMESTEPS, config.max_timesteps)[
                                                    1:].astype(int) - 1
        get_episode_end_mean = lambda x: x.mean(0).reshape(-1)[episode_ends]
        get_episode_end_std = lambda x: x.std(0).reshape(-1)[episode_ends]

    # Get episode end metrics
    returns_mean_episode_end = get_episode_end_mean(merged_out["returns"])
    returns_std_episode_end = get_episode_end_std(merged_out["returns"])
    lengths_mean_episode_end = get_episode_end_mean(merged_out["lengths"])
    lengths_std_episode_end = get_episode_end_std(merged_out["lengths"])
    cum_returns_mean_episode_end = get_episode_end_mean(merged_out["cum_returns"])
    cum_returns_std_episode_end = get_episode_end_std(merged_out["cum_returns"])
    accepted_services_mean_episode_end = get_episode_end_mean(merged_out["accepted_services"])
    accepted_services_std_episode_end = get_episode_end_std(merged_out["accepted_services"])
    accepted_bitrate_mean_episode_end = get_episode_end_mean(merged_out["accepted_bitrate"])
    accepted_bitrate_std_episode_end = get_episode_end_std(merged_out["accepted_bitrate"])
    total_bitrate_mean_episode_end = get_episode_end_mean(merged_out["total_bitrate"])
    total_bitrate_std_episode_end = get_episode_end_std(merged_out["total_bitrate"])
    utilisation_mean_episode_end = get_episode_end_mean(merged_out["utilisation"])
    utilisation_std_episode_end = get_episode_end_std(merged_out["utilisation"])
    service_blocking_probability_episode_end = 1 - (accepted_services_mean_episode_end / lengths_mean_episode_end)
    service_blocking_probability_std_episode_end = accepted_services_std_episode_end / lengths_mean_episode_end
    bitrate_blocking_probability_episode_end = 1 - (accepted_bitrate_mean_episode_end / total_bitrate_mean_episode_end)
    bitrate_blocking_probability_std_episode_end = accepted_bitrate_std_episode_end / total_bitrate_mean_episode_end
    training_time_episode_end = np.arange(len(returns_mean_episode_end)) / returns_mean_episode_end * total_time

    returns_mean = get_mean(merged_out, "returns") if not config.end_first_blocking else returns_mean_episode_end
    returns_std = get_std(merged_out, "returns") if not config.end_first_blocking else returns_std_episode_end
    lengths_mean = get_mean(merged_out, "lengths") if not config.end_first_blocking else lengths_mean_episode_end
    lengths_std = get_std(merged_out, "lengths") if not config.end_first_blocking else lengths_std_episode_end
    cum_returns_mean = get_mean(merged_out,
                                "cum_returns") if not config.end_first_blocking else cum_returns_mean_episode_end
    cum_returns_std = get_std(merged_out,
                              "cum_returns") if not config.end_first_blocking else cum_returns_std_episode_end
    accepted_services_mean = get_mean(merged_out,
                                      "accepted_services") if not config.end_first_blocking else accepted_services_mean_episode_end
    accepted_services_std = get_std(merged_out,
                                    "accepted_services") if not config.end_first_blocking else accepted_services_std_episode_end
    accepted_bitrate_mean = get_mean(merged_out,
                                     "accepted_bitrate") if not config.end_first_blocking else accepted_bitrate_mean_episode_end
    accepted_bitrate_std = get_std(merged_out,
                                   "accepted_bitrate") if not config.end_first_blocking else accepted_bitrate_std_episode_end
    total_bitrate_mean = get_mean(merged_out,
                                  "total_bitrate") if not config.end_first_blocking else total_bitrate_mean_episode_end
    total_bitrate_std = get_std(merged_out,
                                "total_bitrate") if not config.end_first_blocking else total_bitrate_std_episode_end
    utilisation_mean = get_mean(merged_out,
                                "utilisation") if not config.end_first_blocking else utilisation_mean_episode_end
    utilisation_std = get_std(merged_out,
                              "utilisation") if not config.end_first_blocking else utilisation_std_episode_end
    training_time = np.arange(len(returns_mean)) / returns_mean * total_time
    # get values of service and bitrate blocking probs
    service_blocking_probability = 1 - (
                accepted_services_mean / lengths_mean) if not config.end_first_blocking else service_blocking_probability_episode_end
    service_blocking_probability_std = accepted_services_std / lengths_mean if not config.end_first_blocking else service_blocking_probability_std_episode_end
    bitrate_blocking_probability = 1 - (
                accepted_bitrate_mean / total_bitrate_mean) if not config.end_first_blocking else bitrate_blocking_probability_episode_end
    bitrate_blocking_probability_std = accepted_bitrate_std / total_bitrate_mean if not config.end_first_blocking else bitrate_blocking_probability_std_episode_end

    if config.PLOTTING:
        if config.incremental_loading:
            plot_metric = accepted_services_mean
            plot_metric_std = accepted_services_std
            plot_metric_name = "Accepted Services"
        elif config.end_first_blocking:
            plot_metric = lengths_mean_episode_end
            plot_metric_std = lengths_std_episode_end
            plot_metric_name = "Episode Length"
        elif config.reward_type == "service":
            plot_metric = service_blocking_probability
            plot_metric_std = service_blocking_probability_std
            plot_metric_name = "Service Blocking Probability"
        else:
            plot_metric = bitrate_blocking_probability
            plot_metric_std = bitrate_blocking_probability_std
            plot_metric_name = "Bitrate Blocking Probability"

        # Do box and whisker plot of accepted services and bitrate at episode ends
        plt.boxplot(accepted_services_mean)
        plt.ylabel("Accepted Services")
        plt.title(experiment_name)
        plt.show()

        plt.boxplot(accepted_bitrate_mean)
        plt.ylabel("Accepted Bitrate")
        plt.title(experiment_name)
        plt.show()

        plot_metric = moving_average(plot_metric, min(100, int(len(plot_metric) / 2)))
        plot_metric_std = moving_average(plot_metric_std, min(100, int(len(plot_metric_std) / 2)))
        plt.plot(plot_metric)
        plt.fill_between(
            range(len(plot_metric)),
            plot_metric - plot_metric_std,
            plot_metric + plot_metric_std,
            alpha=0.2
        )
        plt.xlabel("Environment Step" if not config.end_first_blocking else "Episode Count")
        plt.ylabel(plot_metric_name)
        plt.title(experiment_name)
        plt.show()

    if config.DATA_OUTPUT_FILE:
        # Save episode end metrics to file
        df = pd.DataFrame({
            "accepted_services": accepted_services_mean_episode_end,
            "accepted_services_std": accepted_services_std_episode_end,
            "accepted_bitrate": accepted_bitrate_mean_episode_end,
            "accepted_bitrate_std": accepted_bitrate_std_episode_end,
            "service_blocking_probability": service_blocking_probability_episode_end,
            "service_blocking_probability_std": service_blocking_probability_std_episode_end,
            "bitrate_blocking_probability": bitrate_blocking_probability_episode_end,
            "bitrate_blocking_probability_std": bitrate_blocking_probability_std_episode_end,
            "total_bitrate": total_bitrate_mean_episode_end,
            "total_bitrate_std": total_bitrate_std_episode_end,
            "utilisation_mean": utilisation_mean_episode_end,
            "utilisation_std": utilisation_std_episode_end,
            "returns": returns_mean_episode_end,
            "returns_std": returns_std_episode_end,
            "cum_returns": cum_returns_mean_episode_end,
            "cum_returns_std": cum_returns_std_episode_end,
            "lengths": lengths_mean_episode_end,
            "lengths_std": lengths_std_episode_end,
            "training_time": training_time_episode_end,
        })
        df.to_csv(config.DATA_OUTPUT_FILE)

    if config.WANDB:
        # Log the data to wandb
        # Define the downsample factor to speed up upload to wandb
        # Then reshape the array and compute the mean
        chop = len(returns_mean) % config.DOWNSAMPLE_FACTOR
        cum_returns_mean = cum_returns_mean[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        cum_returns_std = cum_returns_std[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        returns_mean = returns_mean[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        returns_std = returns_std[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        lengths_mean = lengths_mean[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        lengths_std = lengths_std[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        total_bitrate_mean = total_bitrate_mean[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        total_bitrate_std = total_bitrate_std[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        service_blocking_probability = service_blocking_probability[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(
            axis=1)
        service_blocking_probability_std = service_blocking_probability_std[chop:].reshape(-1,
                                                                                           config.DOWNSAMPLE_FACTOR).mean(
            axis=1)
        bitrate_blocking_probability = bitrate_blocking_probability[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(
            axis=1)
        bitrate_blocking_probability_std = bitrate_blocking_probability_std[chop:].reshape(-1,
                                                                                           config.DOWNSAMPLE_FACTOR).mean(
            axis=1)
        accepted_services_mean = accepted_services_mean[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_services_std = accepted_services_std[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_bitrate_mean = accepted_bitrate_mean[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_bitrate_std = accepted_bitrate_std[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        utilisation_mean = utilisation_mean[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        utilisation_std = utilisation_std[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)
        training_time = training_time[chop:].reshape(-1, config.DOWNSAMPLE_FACTOR).mean(axis=1)

        for i in range(len(episode_ends)):
            log_dict = {
                "episode_count": i,
                "episode_end_accepted_services": accepted_services_mean_episode_end[i],
                "episode_end_accepted_services_std": accepted_services_std_episode_end[i],
                "episode_end_accepted_bitrate": accepted_bitrate_mean_episode_end[i],
                "episode_end_accepted_bitrate_std": accepted_bitrate_std_episode_end[i],
                "episode_end_training_time": training_time_episode_end[i],
            }
            wandb.log(log_dict)

        for i in range(len(returns_mean)):
            # Log the data
            log_dict = {
                "update_step": i * config.DOWNSAMPLE_FACTOR,
                "cum_returns_mean": cum_returns_mean[i],
                "cum_returns_std": cum_returns_std[i],
                "returns_mean": returns_mean[i],
                "returns_std": returns_std[i],
                "lengths_mean": lengths_mean[i],
                "lengths_std": lengths_std[i],
                "service_blocking_probability": service_blocking_probability[i],
                "service_blocking_probability_std": service_blocking_probability_std[i],
                "bitrate_blocking_probability": bitrate_blocking_probability[i],
                "bitrate_blocking_probability_std": bitrate_blocking_probability_std[i],
                "accepted_services_mean": accepted_services_mean[i],
                "accepted_services_std": accepted_services_std[i],
                "accepted_bitrate_mean": accepted_bitrate_mean[i],
                "accepted_bitrate_std": accepted_bitrate_std[i],
                "total_bitrate_mean": total_bitrate_mean[i],
                "total_bitrate_std": total_bitrate_std[i],
                "utilisation_mean": utilisation_mean[i],
                "utilisation_std": utilisation_std[i],
                "training_time": training_time[i],
            }
            wandb.log(log_dict)

    print(f"Service Blocking Probability: "
          f"{service_blocking_probability[-1] if config.continuous_operation else service_blocking_probability_episode_end.mean():.5f}"
          f" ± {service_blocking_probability_std[-1] if config.continuous_operation else service_blocking_probability_std_episode_end.mean():.5f}")
    print(f"Bitrate Blocking Probability: "
          f"{bitrate_blocking_probability[-1] if config.continuous_operation else bitrate_blocking_probability_episode_end.mean():.5f}"
          f" ± {bitrate_blocking_probability_std[-1] if config.continuous_operation else bitrate_blocking_probability_std_episode_end.mean():.5f}")
    print(f"Accepted Services Episode: "
          f"{accepted_services_mean[-1] if config.continuous_operation else accepted_services_mean_episode_end.mean():.0f}"
          f" ± {accepted_services_std[-1] if config.continuous_operation else accepted_services_std_episode_end.mean():.0f}")
    print(f"Accepted Bitrate Episode: "
          f"{accepted_bitrate_mean[-1] if config.continuous_operation else accepted_bitrate_mean_episode_end.mean():.0f}"
          f" ± {accepted_bitrate_std[-1] if config.continuous_operation else accepted_bitrate_std_episode_end.mean():.0f}")

    if config.log_actions:
        env, params = define_env(config)
        request_source = merged_out["source"]
        request_dest = merged_out["dest"]
        request_data_rate = merged_out["data_rate"]
        path_indices = merged_out["path_index"]
        slot_indices = merged_out["slot_index"]
        returns = merged_out["returns"]
        # Get the link length array
        topology_name = config.topology_name
        graph = make_graph(topology_name, topology_directory=config.topology_directory)
        link_length_array = init_link_length_array(graph)
        # Get path, path lengths, number of hops
        paths = jnp.take(params.path_link_array.val, path_indices, axis=0)
        path_lengths = jax.vmap(lambda x: jnp.dot(x, link_length_array), in_axes=(0))(paths)
        num_hops = jnp.sum(paths, axis=-1)
        path_lengths_mean = path_lengths.mean()
        path_lengths_std = path_lengths.std()
        num_hops_mean = num_hops.mean()
        num_hops_std = num_hops.std()
        print(f"Average path length: {path_lengths_mean:.0f} ± {path_lengths_std:.0f}")
        print(f"Average number of hops: {num_hops_mean:.2f} ± {num_hops_std:.2f}")

        request_source = jnp.squeeze(merged_out["source"])
        request_dest = jnp.squeeze(merged_out["dest"])
        request_data_rate = jnp.squeeze(merged_out["data_rate"])
        path_indices = jnp.squeeze(merged_out["path_index"])
        slot_indices = jnp.squeeze(merged_out["slot_index"])

        # Compare the available paths
        df_path_links = pd.DataFrame(params.path_link_array.val).reset_index(drop=True)
        # Set config.weight = "weight" to use the length of the path for ordering else no. of hops
        config.weight = "weight" if not config.weight else None
        env, params = define_env(config)
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
        print(f"Fraction of successful actions that use unique paths: {unique_paths_used_mean:.3f} ± {unique_paths_used_std:.3f}")

