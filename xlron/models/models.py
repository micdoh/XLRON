import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import distrax
import jraph
import chex
from flax.linen.initializers import constant, orthogonal
from flax import linen as nn
from typing import Sequence, Callable, Sequence
from jraph._src.utils import segment_softmax, segment_sum

from xlron.environments.env_funcs import EnvState, EnvParams, get_path_slots, read_rsa_request, format_vone_slot_request
from xlron.environments.vone import make_vone_env
from xlron.environments.rsa import make_rsa_env


def add_graphs_tuples(
    graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple
) -> jraph.GraphsTuple:
    """Adds the nodes, edges and global features from other_graphs to graphs."""
    return graphs._replace(
        nodes=graphs.nodes + other_graphs.nodes,
        edges=graphs.edges + other_graphs.edges,
        globals=graphs.globals + other_graphs.globals if graphs.globals is not None else None,
    )


def make_dense_layers(x, num_units, num_layers, activation):
    if activation == "relu":
        activation = nn.relu
    else:
        activation = nn.tanh
    layer = nn.Dense(
        num_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
    )(x)
    layer = activation(layer)
    for _ in range(num_layers - 1):
        layer = nn.Dense(
            num_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(layer)
        layer = activation(layer)
    return layer


# TODO - Remove request from observation that is fed to state value function
#  requires separation of actor and critic into separate classes and modification of flax train_state
#  to hold two sets of params (alternatively can just set request array to all zeroes before passing to critic)
#  https://flax.readthedocs.io/en/latest/_modules/flax/training/train_state.html
class ActorCriticMLP(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    num_layers: int = 2
    num_units: int = 64

    @nn.compact
    def __call__(self, x):
        actor_mean = make_dense_layers(x, self.num_units, self.num_layers, self.activation)
        action_dists = []
        for dim in self.action_dim:
            actor_mean_dim = nn.Dense(
                dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
            pi_dim = distrax.Categorical(logits=actor_mean_dim)
            action_dists.append(pi_dim)

        critic = make_dense_layers(x, self.num_units, self.num_layers, self.activation)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return action_dists, jnp.squeeze(critic, axis=-1)


class MLP(nn.Module):
    """A multi-layer perceptron."""

    feature_sizes: Sequence[int]
    dropout_rate: float = 0
    deterministic: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(
                x
            )
        return x


class ActorCriticGNN(nn.Module):
    """Combine the GNN actor and critic networks into a single class"""
    def __init__(self):
        raise NotImplementedError  # Implemented in forthcoming release!
