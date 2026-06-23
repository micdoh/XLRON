import collections
from typing import Callable, Optional, Sequence, Tuple, Union

import chex
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax._src.typing import ArrayLike

from xlron import dtype_config
from xlron.environments.gn_model.isrs_gn_model import from_dbm

# Immutable class for storing nested node/edge features containing an embedding and a recurrent state.
StatefulField = collections.namedtuple("StatefulField", ["embedding", "state"])


def orthogonal_init(
    key: Array, shape: Tuple[int, ...], scale: float = 1.0, dtype: jnp.dtype = jnp.float32
) -> Array:
    """
    Orthogonal initializer that is safe for bfloat16 and other dtypes.
    Based on JAX/Flax orthogonal initializer.

    Args:
        key: PRNGKey for initialization
        shape: Shape of the weight matrix
        scale: Scaling factor for the orthogonal matrix
        dtype: Target dtype (will init in float32 then cast)

    Returns:
        Orthogonally initialized weight matrix
    """
    # Always initialize in float32 to avoid dtype issues
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least 2D shape")

    num_rows, num_cols = shape[-2], shape[-1]
    flat_shape = (max(num_rows, num_cols), min(num_rows, num_cols))

    # Generate random matrix
    a = jax.random.normal(key, flat_shape, dtype=jnp.float32)

    # QR decomposition
    q, r = jnp.linalg.qr(a)
    d = jnp.diag(r)
    q = q * jnp.sign(d)

    # If num_rows < num_cols, we need to transpose
    if num_rows < num_cols:
        q = q.T

    # Take the slice we need
    q = q[:num_rows, :num_cols]

    # Reshape if needed
    if len(shape) > 2:
        q = q.reshape(shape)

    # Scale and cast
    return (scale * q).astype(dtype)


def crelu(x: ArrayLike) -> Array:
    """Computes the Concatenated ReLU (CReLU) activation function."""
    x = jnp.concatenate([x, -x], axis=-1)
    return jax.nn.relu(x)


def select_activation(activation: str) -> Callable[[ArrayLike], Array]:
    """Selects the activation function based on the provided string."""
    if activation == "relu":
        return jax.nn.relu
    elif activation == "crelu":
        return crelu
    else:
        return jax.nn.tanh  # Default to tanh if no valid activation is specified


def bfloat16_safe_orthogonal(scale: float = 1.0) -> Callable:
    """Returns an orthogonal initializer that is safe for bfloat16."""

    def init(key: Array, shape: Sequence[int], dtype: jnp.dtype = jnp.float32) -> Array:
        return orthogonal_init(key, tuple(shape), scale=scale, dtype=dtype)

    return init


def constant(value: float) -> Callable:
    """Returns a constant initializer."""

    def init(key: Array, shape: Sequence[int], dtype: jnp.dtype = jnp.float32) -> Array:
        return jnp.full(shape, value, dtype=dtype)

    return init


def make_linear_with_orthogonal_init(
    in_features: int,
    out_features: int,
    key: Array,
    scale: float = 1.0,
    dtype: jnp.dtype = None,
) -> eqx.nn.Linear:
    """Create a Linear layer with orthogonal initialization."""
    if dtype is None:
        dtype = dtype_config.PARAMS_DTYPE

    key1, key2 = jax.random.split(key)
    weight = orthogonal_init(key1, (out_features, in_features), scale=scale, dtype=dtype)
    bias = jnp.zeros(out_features, dtype=dtype)

    linear = eqx.nn.Linear(in_features, out_features, key=key2, dtype=dtype)
    linear = eqx.tree_at(lambda layer: (layer.weight, layer.bias), linear, (weight, bias))
    return linear


class MLP(eqx.Module):
    """Simple MLP module using Equinox."""

    layers: tuple
    activation_fn: Callable = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)

    def __init__(
        self,
        features: Sequence[int],
        in_features: int,
        activation: str = "tanh",
        dropout_rate: float = 0.0,
        deterministic: bool = True,
        layer_norm: bool = False,
        *,
        key: Array,
    ):
        self.activation_fn = select_activation(activation)
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic

        layers_list = []
        keys = jax.random.split(key, len(features))
        current_in = in_features

        for i, out_features in enumerate(features):
            linear = make_linear_with_orthogonal_init(
                current_in, out_features, keys[i], scale=np.sqrt(2)
            )
            layers_list.append(linear)
            if layer_norm and i < len(features) - 1:
                layers_list.append(eqx.nn.LayerNorm(out_features))
            current_in = out_features

        self.layers = tuple(layers_list)

    def __call__(self, x: Array, *, key: Optional[Array] = None) -> Array:
        for i, layer in enumerate(self.layers):
            if isinstance(layer, eqx.nn.Linear):
                x = layer(x)
                # Apply activation for all but the last linear layer
                if i < len(self.layers) - 1:
                    x = self.activation_fn(x)
                    if not self.deterministic and self.dropout_rate > 0 and key is not None:
                        key, subkey = jax.random.split(key)
                        x = eqx.nn.Dropout(self.dropout_rate)(x, key=subkey)
            elif isinstance(layer, eqx.nn.LayerNorm):
                x = layer(x)
        return x


class ActorCriticMLP(eqx.Module):
    """Actor-Critic MLP using Equinox."""

    actor: eqx.Module
    critic: eqx.Module
    activation_fn: Callable
    temperature: float

    def __init__(
        self,
        action_dim: int,
        input_dim: int,
        activation: str = "tanh",
        num_layers: int = 2,
        num_units: int = 64,
        layer_norm: bool = False,  # Not used for now, can add eqx.nn.LayerNorm later
        temperature: float = 1.0,
        dropout_rate: float = 0.0,
        deterministic: bool = True,
        *,
        key: Array,
    ):
        actor_key, critic_key = jax.random.split(key)
        self.activation_fn = select_activation(activation)
        self.temperature = temperature

        # Build actor layers
        actor_features = [num_units] * num_layers + [action_dim]
        self.actor = MLP(
            actor_features,
            input_dim,
            activation=activation,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            deterministic=deterministic,
            key=actor_key,
        )
        critic_features = [num_units] * num_layers + [1]
        self.critic = MLP(
            critic_features,
            input_dim,
            activation=activation,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            deterministic=deterministic,
            key=actor_key,
        )

    def __call__(self, x: Array, key: Optional[Array] = None) -> Tuple[distrax.Categorical, Array]:
        # Cast the (possibly low-precision under mixed precision) observation up to the NN compute
        # dtype so weights, activations and the optimizer stay at full precision for stability.
        x = x.astype(dtype_config.COMPUTE_DTYPE)
        # Actor forward pass
        actor_key, critic_key = jax.random.split(key) if key else (None, None)
        logits = self.actor(x, key=actor_key) / self.temperature
        # Cast outputs back to PARAMS_DTYPE (float32) so bf16 stays inside the model and the
        # downstream PPO math / f32 env-state stay float32 (see transformer model for rationale).
        action_dist = distrax.Categorical(logits=logits.astype(dtype_config.PARAMS_DTYPE))
        value = self.critic(x, key=critic_key).astype(dtype_config.PARAMS_DTYPE)
        return action_dist, jnp.squeeze(value, axis=-1)

    def sample_action(
        self,
        seed: chex.PRNGKey,
        dist: distrax.Categorical,
        log_prob: bool = False,
        deterministic: bool = False,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Sample an action from the distribution"""
        action = jnp.argmax(dist.probs()) if deterministic else dist.sample(seed=seed)
        if log_prob:
            return action, dist.log_prob(action)
        return action


class LaunchPowerActorCriticMLP(eqx.Module):
    """Actor-Critic MLP for launch power optimization.

    Takes an observation of the current request + statistics on each of the K candidate paths.
    Makes K forward passes, one for each path, and outputs a distribution over power levels for each path.
    """

    # For continuous action space (Beta distribution)
    alpha_out: Optional[eqx.nn.Linear]
    beta_out: Optional[eqx.nn.Linear]

    # Static configuration
    activation: str = eqx.field(static=True)
    layer_norm: bool = eqx.field(static=True)
    min_power_dbm: float = eqx.field(static=True)
    max_power_dbm: float = eqx.field(static=True)
    step_power_dbm: float = eqx.field(static=True)
    discrete: bool = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    k_paths: int = eqx.field(static=True)
    num_base_features: int = eqx.field(static=True)
    num_path_features: int = eqx.field(static=True)
    min_concentration: float = eqx.field(static=True)
    max_concentration: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        action_dim: Sequence[int],
        input_dim: int,
        activation: str = "tanh",
        num_layers: int = 2,
        num_units: int = 64,
        layer_norm: bool = False,
        min_power_dbm: float = 0.0,
        max_power_dbm: float = 2.0,
        step_power_dbm: float = 0.1,
        discrete: bool = True,
        temperature: float = 1.0,
        k_paths: int = 5,
        num_base_features: int = 4,
        num_path_features: int = 7,
        min_concentration: float = 0.1,
        max_concentration: float = 20.0,
        epsilon: float = 1e-6,
        *,
        key: Array,
    ):
        super().__init__()
        self.activation = activation
        self.layer_norm = layer_norm
        self.min_power_dbm = min_power_dbm
        self.max_power_dbm = max_power_dbm
        self.step_power_dbm = step_power_dbm
        self.discrete = discrete
        self.temperature = temperature
        self.k_paths = k_paths
        self.num_base_features = num_base_features
        self.num_path_features = num_path_features
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.epsilon = epsilon

        actor_key, critic_key, output_key = jax.random.split(key, 3)

        # Build actor layers
        actor_keys = jax.random.split(actor_key, num_layers)
        actor_layers_list = []
        # Input is base features + path features
        current_in = num_base_features + num_path_features
        for i in range(num_layers):
            linear = make_linear_with_orthogonal_init(
                current_in, num_units, actor_keys[i], scale=np.sqrt(2)
            )
            actor_layers_list.append(linear)
            if layer_norm:
                actor_layers_list.append(eqx.nn.LayerNorm(num_units))
            current_in = num_units
        self.actor_layers = tuple(actor_layers_list)

        # Actor output
        out_key1, out_key2, out_key3 = jax.random.split(output_key, 3)
        if discrete:
            num_power_levels = int((max_power_dbm - min_power_dbm) / step_power_dbm) + 1
            self.actor_output = make_linear_with_orthogonal_init(
                num_units, num_power_levels, out_key1, scale=0.01
            )
            self.alpha_out = None
            self.beta_out = None
        else:
            self.actor_output = None  # Not used for continuous
            self.alpha_out = make_linear_with_orthogonal_init(num_units, 1, out_key2, scale=0.01)
            self.beta_out = make_linear_with_orthogonal_init(num_units, 1, out_key3, scale=0.01)

    @property
    def num_power_levels(self):
        """Calculate number of power levels dynamically"""
        return int((self.max_power_dbm - self.min_power_dbm) / self.step_power_dbm) + 1

    @property
    def power_levels(self):
        """Calculate power levels dynamically"""
        return jnp.linspace(
            self.min_power_dbm,
            self.max_power_dbm,
            self.num_power_levels,
            dtype=dtype_config.SMALL_FLOAT_DTYPE,
        )

    def _activate(self, x):
        if self.activation == "relu":
            return jax.nn.relu(x)
        elif self.activation == "crelu":
            return crelu(x)
        return jnp.tanh(x)

    def _forward_layers(self, x, layers):
        for layer in layers:
            x = layer(x)
            if isinstance(layer, eqx.nn.Linear):
                x = self._activate(x)
        return x

    def __call__(self, x: Array) -> Tuple[Tuple[None, distrax.Distribution], Array]:
        # Process each path
        def process_path(i):
            base = x[: self.num_base_features]
            path = jax.lax.dynamic_slice(
                x, (self.num_base_features + i * self.num_path_features,), (self.num_path_features,)
            )
            features = jnp.concatenate([base, path])
            actor_hidden = self._forward_layers(features, self.actor_layers)

            if self.discrete:
                return self.actor_output(actor_hidden) / self.temperature
            else:
                alpha = self.min_concentration + jax.nn.softplus(self.alpha_out(actor_hidden)) * (
                    self.max_concentration - self.min_concentration
                )
                beta = self.min_concentration + jax.nn.softplus(self.beta_out(actor_hidden)) * (
                    self.max_concentration - self.min_concentration
                )
                return jnp.concatenate([alpha, beta])

        # Use vmap instead of scan for simpler Equinox pattern
        dist_params = jax.vmap(process_path)(
            jnp.arange(self.k_paths, dtype=dtype_config.INDEX_DTYPE)
        )

        # Critic forward pass
        critic_hidden = self._forward_layers(x, self.critic_layers)  # ty: ignore[unresolved-attribute]
        value = jnp.squeeze(self.critic_output(critic_hidden), axis=-1)  # ty: ignore[unresolved-attribute]

        # Create distribution
        if self.discrete:
            dist = distrax.Categorical(logits=dist_params)
        else:
            alpha = dist_params[:, 0]
            beta = dist_params[:, 1]
            dist = distrax.Beta(alpha, beta)

        return (None, dist), value

    def sample_action(self, seed, dist, log_prob=False, deterministic=False):
        """Sample an action and convert to power level"""
        if self.discrete:
            if deterministic:
                raw_action = dist.mode()
            else:
                raw_action = dist.sample(seed=seed)
            processed_action = self.power_levels[raw_action].reshape((self.k_paths, 1))
        else:
            if deterministic:
                mean = dist.alpha / (dist.alpha + dist.beta)
                raw_action = jnp.clip(mean, self.epsilon, 1.0 - self.epsilon)
            else:
                raw_action = jnp.clip(dist.sample(seed=seed), self.epsilon, 1.0 - self.epsilon)
            processed_action = self.min_power_dbm + raw_action * (
                self.max_power_dbm - self.min_power_dbm
            )
        processed_action = from_dbm(processed_action)
        if log_prob:
            return processed_action, dist.log_prob(jnp.squeeze(raw_action))
        return processed_action

    def get_action_probs(self, dist):
        """Get probabilities for discrete case or pdf for continuous case"""
        if self.discrete:
            return dist.probs()
        else:
            x = jnp.linspace(0, 1, 100)
            return dist.prob(x)
