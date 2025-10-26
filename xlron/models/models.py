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
import collections

from xlron.environments.env_funcs import EnvState, EnvParams, get_path_slots, read_rsa_request
from xlron.environments.gn_model.isrs_gn_model import isrs_gn_model, to_dbm, from_dbm
from xlron.environments.make_env import make
from xlron.models.gnn import GraphNetwork, GraphNetGAT, GAT
from xlron.dtype_config import COMPUTE_DTYPE, PARAMS_DTYPE, LARGE_INT_DTYPE, LARGE_FLOAT_DTYPE, \
    SMALL_INT_DTYPE, SMALL_FLOAT_DTYPE, MED_INT_DTYPE

# Immutable class for storing nested node/edge features containing an embedding and a recurrent state.
StatefulField = collections.namedtuple("StatefulField", ["embedding", "state"])


def crelu(x):
    """Computes the Concatenated ReLU (CReLU) activation function."""
    x = jnp.concatenate([x, -x], axis=-1)
    return nn.relu(x)


def add_graphs_tuples(
    graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple
) -> jraph.GraphsTuple:
    """Adds the nodes, edges and global features from other_graphs to graphs."""
    return graphs._replace(
        nodes=graphs.nodes + other_graphs.nodes,
        edges=graphs.edges + other_graphs.edges,
        globals=graphs.globals + other_graphs.globals if graphs.globals is not None else None,
    )


def make_dense_layers(x, num_units, num_layers, activation, layer_norm=False):
    if activation == "relu":
        activation = nn.relu
    elif activation == "crelu":
        activation = crelu
    else:
        activation = nn.tanh
    layer = nn.Dense(
        num_units,
        kernel_init=orthogonal(np.sqrt(2)),
        bias_init=constant(0.0),
        dtype=COMPUTE_DTYPE,
        param_dtype=PARAMS_DTYPE,
    )(x)
    layer = nn.LayerNorm(
        dtype=COMPUTE_DTYPE,
        param_dtype=PARAMS_DTYPE,
    )(layer) if layer_norm else layer
    layer = activation(layer)
    for _ in range(num_layers - 1):
        layer = nn.Dense(
            num_units,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAMS_DTYPE,
        )(layer)
        layer = nn.LayerNorm(
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAMS_DTYPE,
        )(layer) if layer_norm else layer
        layer = activation(layer)
    return layer


class MLP(nn.Module):
    """A multi-layer perceptron."""

    feature_sizes: Sequence[int]
    dropout_rate: float = 0
    deterministic: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    layer_norm: bool = False

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(features=size,
                         dtype=COMPUTE_DTYPE,
                         param_dtype=PARAMS_DTYPE,)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(
                x
            )
            if self.layer_norm:
                x = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,)(x)
        return x


# TODO - Remove request from observation that is fed to state value function
#  requires separation of actor and critic into separate classes and modification of flax train_state
#  to hold two sets of params (alternatively can just set request array to all zeroes before passing to critic)
#  https://flax.readthedocs.io/en/latest/_modules/flax/training/train_state.html
class ActorCriticMLP(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    num_layers: int = 2
    num_units: int = 64
    layer_norm: bool = False
    temperature: float = 1.0

    @nn.compact
    def __call__(self, x):
        actor_mean = make_dense_layers(x, self.num_units, self.num_layers, self.activation)
        stacked_logits = []
        actor_mean_dim = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAMS_DTYPE,
        )(actor_mean)
        logits = actor_mean_dim / self.temperature
        stacked_logits.append(logits)

        # If there are multiple action dimensions, concatenate the logits
        action_dist = distrax.Categorical(logits=logits)

        critic = make_dense_layers(x, self.num_units, self.num_layers, self.activation, layer_norm=self.layer_norm)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAMS_DTYPE,
        )(
            critic
        )

        return action_dist, jnp.squeeze(critic, axis=-1)

    def sample_action(self, seed,  dist, log_prob=False, deterministic=False):
        """Sample an action from the distribution"""
        action = jnp.argmax(dist.probs()) if deterministic else dist.sample(seed=seed)
        if log_prob:
            return action, dist.log_prob(action)
        return action


class LaunchPowerActorCriticMLP(nn.Module):
    """In this implementation, we take an observation of th current request + statistics on each of the K candidate paths.
    We make K forward passes, one for each path, and output a distribution over power levels for each path.
    In action selection, we then sample from each distribution and use the sampled power levels to mask paths for the
    routing heuristic, which then determines the path taken. The selected path index is then used to select which output
    action, distribution, and value to use for the loss calculation."""
    action_dim: Sequence[int]
    activation: str = "tanh"
    num_layers: int = 2
    num_units: int = 64
    layer_norm: bool = False
    min_power_dbm: float = 0.0
    max_power_dbm: float = 2.0
    step_power_dbm: float = 0.1
    discrete: bool = True
    temperature: float = 1.0
    k_paths: int = 5
    num_base_features: int = 4
    num_path_features: int = 7
    # Beta distribution parameters (used only if discrete=False)
    min_concentration: float = 0.1
    max_concentration: float = 20.0
    epsilon: float = 1e-6

    @property
    def num_power_levels(self):
        """Calculate number of power levels dynamically"""
        return int((self.max_power_dbm - self.min_power_dbm) / self.step_power_dbm) + 1

    @property
    def power_levels(self):
        """Calculate power levels dynamically"""
        return jnp.linspace(self.min_power_dbm, self.max_power_dbm, self.num_power_levels, dtype=SMALL_FLOAT_DTYPE)

    @nn.compact
    def __call__(self, x):
        # Helper function to create MLP layers
        def make_mlp(prefix):
            layers = []
            for i in range(self.num_layers):
                layers.append(nn.Dense(
                    self.num_units,
                    kernel_init=orthogonal(np.sqrt(2)),
                    name=f"{prefix}_dense_{i}",
                    dtype=COMPUTE_DTYPE,
                    param_dtype=PARAMS_DTYPE,
                ))
                if self.layer_norm:
                    layers.append(nn.LayerNorm(name=f"{prefix}_norm_{i}", dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,))
            return layers

        # Initialize actor network layers
        actor_net = make_mlp("actor")
        if self.discrete:
            actor_out = nn.Dense(
                self.num_power_levels,
                kernel_init=orthogonal(0.01),
                name="actor_output",
                dtype=COMPUTE_DTYPE,
                param_dtype=PARAMS_DTYPE,
            )
        else:
            alpha_out = nn.Dense(1, kernel_init=orthogonal(0.01), name="alpha", dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,)
            beta_out = nn.Dense(1, kernel_init=orthogonal(0.01), name="beta", dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,)

        def activate(x):
            if self.activation == "relu": return jax.nn.relu(x)
            if self.activation == "crelu": return crelu(x)
            return jnp.tanh(x)

        def forward(x, layers):
            for layer in layers:
                x = layer(x)
                if isinstance(layer, nn.Dense):
                    x = activate(x)
            return x

        num_base_features = self.num_base_features
        num_path_features = self.num_path_features
        temperature = self.temperature
        discrete = self.discrete
        min_concentration = self.min_concentration
        max_concentration = self.max_concentration

        # Create a class to handle the scan
        class PathProcessor(nn.Module):
            def __call__(self, carry, i):
                base = x[:num_base_features]
                path = jax.lax.dynamic_slice(
                    x,
                    (num_base_features + i *num_path_features,),
                    (num_path_features,)
                )
                features = jnp.concatenate([base, path])
                actor_hidden = forward(features, actor_net)
                if discrete:
                    out = actor_out(actor_hidden) / temperature
                else:
                    alpha = min_concentration + jax.nn.softplus(alpha_out(actor_hidden)) * (
                            max_concentration - min_concentration
                    )
                    beta = min_concentration + jax.nn.softplus(beta_out(actor_hidden)) * (
                            max_concentration - min_concentration
                    )
                    out = (alpha, beta)

                return carry, out

        # Scan over paths
        _, dist_params = nn.scan(
            PathProcessor,
            variable_broadcast="params",
            split_rngs={"params": False},
            length=self.k_paths,
        )()(None, jnp.arange(self.k_paths, dtype=MED_INT_DTYPE))

        # Initialize critic network layers
        critic_net = make_mlp("critic")
        critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), name="critic_output", dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,)
        value = jnp.squeeze(critic_out(forward(x, critic_net)), axis=-1)

        # Create appropriate distribution
        if self.discrete:
            dist = distrax.Categorical(logits=dist_params)
        else:
            dist = distrax.Beta(dist_params[0].reshape((self.k_paths,)), dist_params[1].reshape((self.k_paths,)))
        # N.B. that this is a single distribution object, but it is batched over K paths

        return [None, dist], value

    def sample_action(self, seed, dist, log_prob=False, deterministic=False):
        """Sample an action and convert to power level"""
        if self.discrete:
            if deterministic:
                # Take most probable action
                raw_action = dist.mode()
            else:
                # Sample from distribution
                raw_action = dist.sample(seed=seed)
            processed_action = self.power_levels[raw_action].reshape((self.k_paths, 1))
        else:
            if deterministic:
                # Use mean of Beta distribution for deterministic action
                mean = dist.alpha / (dist.alpha + dist.beta)
                raw_action = jnp.clip(mean, self.epsilon, 1.0 - self.epsilon)
            else:
                # Sample from Beta (clipping to avoid edge values with undefined gradient) and scale to power range
                raw_action = jnp.clip(dist.sample(seed=seed), self.epsilon, 1.0 - self.epsilon)
            processed_action = self.min_power_dbm + raw_action * (self.max_power_dbm - self.min_power_dbm)
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


class GraphNet(nn.Module):
    """A complete Graph Network model defined with Jraph."""

    message_passing_steps: int
    mlp_layers: int = None
    mlp_latent: int = None
    edge_embedding_size: int = 128
    edge_mlp_layers: int = 3
    edge_mlp_latent: int = 128
    edge_output_size: int = 0
    global_embedding_size: int = 8
    global_mlp_layers: int = 0
    global_mlp_latent: int = 0
    global_output_size: int = 0
    node_embedding_size: int = 16
    node_mlp_layers: int = 2
    node_mlp_latent: int = 128
    node_output_size: int = 0
    attn_mlp_layers: int = 2
    attn_mlp_latent: int = 128
    dropout_rate: float = 0
    skip_connections: bool = True
    use_edge_model: bool = True
    gnn_layer_norm: bool = True
    mlp_layer_norm: bool = False
    deterministic: bool = True  # If true, no dropout (better for RL purposes)
    use_attention: bool = True

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Template code from here: https://github.com/google/flax/blob/main/examples/ogbg_molpcba/models.py
        # We will first linearly project the original features as 'embeddings'.
        if self.mlp_latent is not None:
            global_mlp_dims = edge_mlp_dims = node_mlp_dims = attn_mlp_dims = [self.mlp_latent] * self.mlp_layers
        else:
            global_mlp_dims = [self.global_mlp_latent] * self.global_mlp_layers
            edge_mlp_dims = [self.edge_mlp_latent] * self.edge_mlp_layers
            node_mlp_dims = [self.node_mlp_latent] * self.node_mlp_layers
            attn_mlp_dims = [self.attn_mlp_latent] * self.attn_mlp_layers
        if self.skip_connections:
            # If using skip connections, we need to add the input dimensions to the output dimensions,
            # so that the output of the MLP is the same size as the input for summing output/input
            edge_mlp_dims = edge_mlp_dims + [self.edge_embedding_size]
            node_mlp_dims = node_mlp_dims + [self.node_embedding_size]
            global_mlp_dims = global_mlp_dims + [self.global_embedding_size]

        embedder = jraph.GraphMapFeatures(
            embed_edge_fn=nn.Dense(self.edge_embedding_size),
            embed_node_fn=nn.Dense(self.node_embedding_size),
            embed_global_fn=nn.Dense(self.global_embedding_size),
        )
        if graphs.edges.ndim >= 3:
            # Dims are (edges, slots, features e.g. power, source/dest)
            # Keep the leading dimension fixed and combine the remaining dimensions
            edges = graphs.edges.reshape((graphs.edges.shape[0], -1))
            graphs = graphs._replace(edges=edges)
        processed_graphs = embedder(graphs)

        # Now, we will apply a Graph Network once for each message-passing round.
        for _ in range(self.message_passing_steps):
            if self.use_edge_model:
                update_edge_fn = jraph.concatenated_args(
                    MLP(
                        edge_mlp_dims,
                        dropout_rate=self.dropout_rate,
                        deterministic=self.deterministic,
                        layer_norm=self.mlp_layer_norm,
                    )
                )
            else:
                update_edge_fn = None

            update_node_fn = jraph.concatenated_args(
                MLP(
                    node_mlp_dims,
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                    layer_norm=self.mlp_layer_norm,
                )
            )
            if self.global_output_size > 0:
                update_global_fn = jraph.concatenated_args(
                    MLP(
                        global_mlp_dims,
                        dropout_rate=self.dropout_rate,
                        deterministic=self.deterministic,
                        layer_norm=self.mlp_layer_norm,
                    )
                )
            else:
                update_global_fn = None

            if self.use_attention:
                def _attention_logit_fn(edges, sender_attr, receiver_attr, global_edge_attributes):
                    """Calculate attention logits using edges, nodes and global attributes."""
                    x = jnp.concatenate((edges, sender_attr, receiver_attr, global_edge_attributes), axis=1)
                    return MLP(attn_mlp_dims + [1], dropout_rate=self.dropout_rate,
                               deterministic=self.deterministic)(x)

                def _attention_reduce_fn(
                    edges: jnp.ndarray, attention: jnp.ndarray
                ) -> jnp.ndarray:
                    # TODO - try more sophisticated attention reduce function (not sure what it would be)
                    #  (here might help https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial7/GNN_overview.html)
                    return attention * edges

                graph_net = GraphNetGAT(
                    update_node_fn=update_node_fn,
                    update_edge_fn=update_edge_fn,  # Update the edges with MLP prior to attention
                    update_global_fn=update_global_fn,
                    attention_logit_fn=_attention_logit_fn,
                    attention_reduce_fn=_attention_reduce_fn,
                )
            else:
                graph_net = GraphNetwork(
                    update_node_fn=update_node_fn,
                    update_edge_fn=update_edge_fn,
                    update_global_fn=update_global_fn,
                )

            if self.skip_connections:
                processed_graphs = add_graphs_tuples(
                graph_net(processed_graphs), processed_graphs
            )
            else:
                processed_graphs = graph_net(processed_graphs)

            if self.gnn_layer_norm:
                processed_graphs = processed_graphs._replace(
                    nodes=nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,)(processed_graphs.nodes),
                    edges=nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,)(processed_graphs.edges),
                    globals=nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,)(processed_graphs.globals) if processed_graphs.globals is not None else None,
                )

        decoder = jraph.GraphMapFeatures(
            embed_global_fn=nn.Dense(self.global_output_size, dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,) if self.global_output_size > 0 else None,
            embed_node_fn=nn.Dense(self.node_output_size, dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,) if self.node_output_size > 0 else None,
            embed_edge_fn=nn.Dense(self.edge_output_size, dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,),
        )
        processed_graphs = decoder(processed_graphs)

        return processed_graphs


class CriticGNN(nn.Module):
    activation: str = "tanh"
    num_layers: int = 2
    num_units: int = 64
    message_passing_steps: int = 1
    mlp_layers: int = None
    mlp_latent: int = None
    edge_embedding_size: int = 128
    edge_mlp_layers: int = 3
    edge_mlp_latent: int = 128
    edge_output_size: int = 0
    global_embedding_size: int = 8
    global_mlp_layers: int = 0
    global_mlp_latent: int = 0
    global_output_size: int = 0
    node_embedding_size: int = 16
    node_mlp_layers: int = 2
    node_mlp_latent: int = 128
    node_output_size: int = 0
    attn_mlp_layers: int = 2
    attn_mlp_latent: int = 128
    use_attention: bool = True
    normalise_by_link_length: bool = True  # Normalise the processed edge features by the link length
    gnn_layer_norm: bool = True
    mlp_layer_norm: bool = False

    @nn.compact
    def __call__(self, state: EnvState, params: EnvParams):
        # Remove globals from graph s.t. state value does not depend on the current request
        state = state.replace(graph=state.graph._replace(globals=jnp.zeros((1, 1), dtype=LARGE_FLOAT_DTYPE)))
        processed_graph = GraphNet(
            message_passing_steps=self.message_passing_steps,
            mlp_layers=self.mlp_layers,
            mlp_latent=self.mlp_latent,
            edge_embedding_size=self.edge_embedding_size,
            edge_mlp_layers=self.edge_mlp_layers,
            edge_mlp_latent=self.edge_mlp_latent,
            edge_output_size=self.edge_output_size,
            global_embedding_size=self.global_embedding_size,
            global_mlp_layers=self.global_mlp_layers,
            global_mlp_latent=self.global_mlp_latent,
            global_output_size=self.global_output_size,
            node_embedding_size=self.node_embedding_size,
            node_mlp_layers=self.node_mlp_layers,
            node_mlp_latent=self.node_mlp_latent,
            node_output_size=self.node_output_size,
            attn_mlp_layers=self.attn_mlp_layers,
            attn_mlp_latent=self.attn_mlp_latent,
            use_attention=self.use_attention,
            gnn_layer_norm=self.gnn_layer_norm,
            mlp_layer_norm=self.mlp_layer_norm,
        )(state.graph)
        if self.global_output_size > 0:
            critic = processed_graph.globals.reshape((-1,))
        else:
            # Take first half processed_graph.edges as edge features
            edge_features = processed_graph.edges if params.directed_graph else processed_graph.edges[:len(processed_graph.edges) // 2]
            if self.normalise_by_link_length:
                edge_features = edge_features * (params.link_length_array.val/jnp.sum(params.link_length_array.val, promote_integers=False))
            # Index every other row of the edge features to get the link-slot array
            edge_features_flat = jnp.reshape(edge_features, (-1,))
            # pass aggregated features through MLP
            critic = make_dense_layers(edge_features_flat, self.num_units, self.num_layers, self.activation, layer_norm=self.mlp_layer_norm)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), dtype=COMPUTE_DTYPE, param_dtype=PARAMS_DTYPE,)(
            critic
        )
        return jnp.squeeze(critic, axis=-1)


class ActorGNN(nn.Module):
    """
    Actor network for PPO algorithm. Takes the current state and returns a distrax.Categorical distribution
    over actions.
    """
    activation: str = "tanh"
    num_layers: int = 2
    num_units: int = 64
    mlp_layers: int = None
    mlp_latent: int = None
    edge_embedding_size: int = 128
    edge_mlp_layers: int = 3
    edge_mlp_latent: int = 128
    edge_output_size: int = 0
    global_embedding_size: int = 8
    global_mlp_layers: int = 0
    global_mlp_latent: int = 0
    global_output_size: int = 0
    node_embedding_size: int = 16
    node_mlp_layers: int = 2
    node_mlp_latent: int = 128
    node_output_size: int = 0
    attn_mlp_layers: int = 2
    attn_mlp_latent: int = 128
    dropout_rate: float = 0
    deterministic: bool = False
    message_passing_steps: int = 1
    use_attention: bool = True
    normalise_by_link_length: bool = True  # Normalise the processed edge features by the link length
    gnn_layer_norm: bool = True
    mlp_layer_norm: bool = False
    temperature: float = 1.0
    # Launch power specific parameters
    min_power_dbm: float = 0.0
    max_power_dbm: float = 2.0
    step_power_dbm: float = 0.1
    discrete: bool = True
    # Beta distribution parameters (used only if discrete=False)
    min_concentration: float = 0.1
    max_concentration: float = 20.0
    epsilon: float = 1e-6

    @property
    def num_power_levels(self):
        """Calculate number of power levels dynamically"""
        return int((self.max_power_dbm - self.min_power_dbm) / self.step_power_dbm) + 1

    @property
    def power_levels(self):
        """Calculate power levels dynamically"""
        return jnp.linspace(self.min_power_dbm, self.max_power_dbm, self.num_power_levels, dtype=SMALL_FLOAT_DTYPE)

    @nn.compact
    def __call__(self, state: EnvState, params: EnvParams):
        """
        The ActorGNN network takes the current network state in the form of a GraphTuple and returns
        a distrax.Categorical distribution over actions.
        The graph is processed by a GraphNet module, and the resulting graph is indexed to construct a matrix of
        the edge features. The edge features are then normalised by the link length from the environment parameters,
        and the current request is read from the request array.
        The request is used to retrieve the edge features from the edge_features array for the corresponding
        shortest k-paths. The edge features are aggregated for each path according to the "agg_func" e.g. sum,
        and the action distribution array is updated.
        Returns a distrax.Categorical distribution, from which actions can be sampled.

        :param state: EnvState
        :param params: EnvParams

        :return: distrax.Categorical distribution over actions
        """
        processed_graph = GraphNet(
            message_passing_steps=self.message_passing_steps,
            mlp_layers=self.mlp_layers,
            mlp_latent=self.mlp_latent,
            edge_embedding_size=self.edge_embedding_size,
            edge_mlp_layers=self.edge_mlp_layers,
            edge_mlp_latent=self.edge_mlp_latent,
            edge_output_size=self.edge_output_size,
            global_embedding_size=self.global_embedding_size,
            global_mlp_layers=self.global_mlp_layers,
            global_mlp_latent=self.global_mlp_latent,
            global_output_size=self.global_output_size,
            node_embedding_size=self.node_embedding_size,
            node_mlp_layers=self.node_mlp_layers,
            node_mlp_latent=self.node_mlp_latent,
            node_output_size=self.node_output_size,
            attn_mlp_layers=self.attn_mlp_layers,
            attn_mlp_latent=self.attn_mlp_latent,
            use_attention=self.use_attention,
            gnn_layer_norm=self.gnn_layer_norm,
            mlp_layer_norm=self.mlp_layer_norm,
        )(state.graph)

        # Index edge features to resemble the link-slot array
        edge_features = processed_graph.edges if params.directed_graph else processed_graph.edges[:len(processed_graph.edges) // 2]
        # Normalise features by normalised link length from state.link_length_array
        if self.normalise_by_link_length:
            edge_features = edge_features * (params.link_length_array.val/jnp.sum(params.link_length_array.val, promote_integers=False))
        # Get the current request and initialise array of action distributions per path
        nodes_sd, requested_bw = read_rsa_request(state.request_array)
        init_action_array = jnp.zeros(params.k_paths * self.edge_output_size, dtype=SMALL_FLOAT_DTYPE)

        # Define a body func to retrieve path slots and update action array
        def get_path_action_dist(i, action_array):
            # Get the processed graph edge features corresponding to the i-th path
            path_features = get_path_slots(edge_features, params, nodes_sd, i, agg_func="sum")
            # Update the action array with the path features
            action_array = jax.lax.dynamic_update_slice(action_array, path_features, (i * self.edge_output_size,))
            return action_array

        path_action_logits = jax.lax.fori_loop(0, params.k_paths, get_path_action_dist, init_action_array)
        path_action_logits = jnp.reshape(path_action_logits, (-1,)) / self.temperature

        power_action_dist = None
        mlp_features = [self.num_units] * self.num_layers
        output_size = self.num_power_levels if self.discrete else 2
        path_mlp = MLP(
            mlp_features + [output_size],
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
            layer_norm=self.mlp_layer_norm,
        )
        if params.__class__.__name__ == "RSAGNModelEnvParams":
            if self.global_output_size > 0:
                power_logits = processed_graph.globals.reshape((-1,)) / self.temperature
            else:
                init_feature_array = jnp.zeros((params.k_paths, edge_features.shape[1]), dtype=LARGE_FLOAT_DTYPE)
                # Define a body func to retrieve path slots and update action array
                def get_power_action_dist(i, feature_array):
                    # Get the processed graph edge features corresponding to the i-th path
                    path_features = get_path_slots(edge_features, params, nodes_sd, i, agg_func="sum").reshape((1, -1))
                    # Update the array with the path features
                    action_array = jax.lax.dynamic_update_slice(feature_array, path_features,(i, 0))
                    return action_array
                path_feature_batch = jax.lax.fori_loop(0, params.k_paths, get_power_action_dist, init_feature_array)
                power_logits = path_mlp(path_feature_batch)
            if self.discrete:
                power_action_dist = distrax.Categorical(logits=power_logits)
            else:
                alpha = self.min_concentration + jax.nn.softplus(power_logits) * (
                        self.max_concentration - self.min_concentration
                )
                beta = self.min_concentration + jax.nn.softplus(power_logits) * (
                        self.max_concentration - self.min_concentration
                )
                power_action_dist = distrax.Beta(alpha, beta)

        # Return a distrax.Categorical distribution over actions (which can be masked later)
        path_action_dist = distrax.Categorical(logits=path_action_logits)
        return (path_action_dist, power_action_dist)


class ActorCriticGNN(nn.Module):
    """Combine the GNN actor and critic networks into a single class"""
    activation: str = "tanh"
    num_layers: int = 2
    num_units: int = 64
    message_passing_steps: int = 1
    mlp_layers: int = None
    mlp_latent: int = None
    edge_embedding_size: int = 128
    edge_mlp_layers: int = 3
    edge_mlp_latent: int = 128
    edge_output_size_actor: int = 1
    edge_output_size_critic: int = 1
    global_embedding_size: int = 8
    global_mlp_layers: int = 0
    global_mlp_latent: int = 0
    global_output_size_actor: int = 0
    global_output_size_critic: int = 0
    node_embedding_size: int = 16
    node_mlp_layers: int = 2
    node_mlp_latent: int = 128
    node_output_size_actor: int = 0
    node_output_size_critic: int = 0
    attn_mlp_layers: int = 2
    attn_mlp_latent: int = 128
    gnn_mlp_layers: int = 1
    use_attention: bool = True
    normalise_by_link_length: bool = True  # Normalise the processed edge features by the link length
    gnn_layer_norm: bool = True
    mlp_layer_norm: bool = False
    vmap: bool = True
    temperature: float = 1.0
    # Launch power specific parameters
    min_power_dbm: float = 0.0
    max_power_dbm: float = 2.0
    step_power_dbm: float = 0.1
    discrete: bool = True
    # Beta distribution parameters (used only if discrete=False)
    min_concentration: float = 0.1
    max_concentration: float = 20.0
    epsilon: float = 1e-6
    # Bools to determine which actions to output
    output_path: bool = True
    output_power: bool = True
    assert edge_output_size_actor > 0
    assert edge_output_size_critic + global_output_size_critic > 0

    @property
    def num_power_levels(self):
        """Calculate number of power levels dynamically"""
        return int((self.max_power_dbm - self.min_power_dbm) / self.step_power_dbm) + 1

    @property
    def power_levels(self):
        """Calculate power levels dynamically"""
        return jnp.linspace(self.min_power_dbm, self.max_power_dbm, self.num_power_levels, dtype=LARGE_FLOAT_DTYPE)

    @nn.compact
    def __call__(self, state: EnvState, params: EnvParams):
        actor = ActorGNN(
            num_layers=self.num_layers,
            num_units=self.num_units,
            message_passing_steps=self.message_passing_steps,
            mlp_layers=self.mlp_layers,
            mlp_latent=self.mlp_latent,
            edge_embedding_size=self.edge_embedding_size,
            edge_mlp_layers=self.edge_mlp_layers,
            edge_mlp_latent=self.edge_mlp_latent,
            edge_output_size=self.edge_output_size_actor,
            global_embedding_size=self.global_embedding_size,
            global_mlp_layers=self.global_mlp_layers,
            global_mlp_latent=self.global_mlp_latent,
            global_output_size=self.global_output_size_actor,
            node_embedding_size=self.node_embedding_size,
            node_mlp_layers=self.node_mlp_layers,
            node_mlp_latent=self.node_mlp_latent,
            node_output_size=self.node_output_size_actor,
            attn_mlp_layers=self.attn_mlp_layers,
            attn_mlp_latent=self.attn_mlp_latent,
            use_attention=self.use_attention,
            normalise_by_link_length=self.normalise_by_link_length,
            gnn_layer_norm=self.gnn_layer_norm,
            mlp_layer_norm=self.mlp_layer_norm,
            temperature=self.temperature,
            min_power_dbm=self.min_power_dbm,
            max_power_dbm=self.max_power_dbm,
            step_power_dbm=self.step_power_dbm,
            discrete=self.discrete,
            min_concentration=self.min_concentration,
            max_concentration=self.max_concentration,
            epsilon=self.epsilon,
        )
        critic = CriticGNN(
            activation=self.activation,
            num_layers=self.num_layers,
            num_units=self.num_units,
            message_passing_steps=self.message_passing_steps,
            mlp_layers=self.mlp_layers,
            mlp_latent=self.mlp_latent,
            edge_embedding_size=self.edge_embedding_size,
            edge_mlp_layers=self.edge_mlp_layers,
            edge_mlp_latent=self.edge_mlp_latent,
            edge_output_size=self.edge_output_size_critic,
            global_embedding_size=self.global_embedding_size,
            global_mlp_layers=self.global_mlp_layers,
            global_mlp_latent=self.global_mlp_latent,
            global_output_size=self.global_output_size_critic,
            node_embedding_size=self.node_embedding_size,
            node_mlp_layers=self.node_mlp_layers,
            node_mlp_latent=self.node_mlp_latent,
            node_output_size=self.node_output_size_critic,
            attn_mlp_layers=self.attn_mlp_layers,
            attn_mlp_latent=self.attn_mlp_latent,
            use_attention=self.use_attention,
            normalise_by_link_length=self.normalise_by_link_length,
            gnn_layer_norm=self.gnn_layer_norm,
            mlp_layer_norm=self.mlp_layer_norm,
        )

        if self.vmap:
            actor = jax.vmap(actor, in_axes=(0, None))
            critic = jax.vmap(critic, in_axes=(0, None))
        actor_out = actor(state, params)
        critic_out = critic(state, params)
        return actor_out, critic_out

    def sample_action_path(self, seed, dist, log_prob=False, deterministic=False):
        """Sample an action from the distribution."""
        action = jnp.argmax(dist.probs()).astype(MED_INT_DTYPE) if deterministic else dist.sample(seed=seed)
        if log_prob:
            return action, dist.log_prob(action)
        return action

    def sample_action_power(self, seed, dist, log_prob=False, deterministic=False):
        """Sample an action and convert to power level"""
        if self.discrete:
            if deterministic:
                # Take most probable action
                raw_action = dist.mode()
            else:
                # Sample from distribution
                raw_action = dist.sample(seed=seed)
            processed_action = self.power_levels[raw_action]
        else:
            if deterministic:
                # Use mean of Beta distribution for deterministic action
                mean = dist.alpha / (dist.alpha + dist.beta)
                raw_action = jnp.clip(mean, self.epsilon, 1.0 - self.epsilon)
            else:
                # Sample from Beta (clipping to avoid edge values with undefined gradient) and scale to power range
                raw_action = jnp.clip(dist.sample(seed=seed), self.epsilon, 1.0 - self.epsilon)
            processed_action = self.min_power_dbm + raw_action * (self.max_power_dbm - self.min_power_dbm)
        processed_action = from_dbm(processed_action)
        if log_prob:
            return processed_action, dist.log_prob(raw_action)
        return processed_action

    def sample_action_path_power(self, seed, dist, log_prob=False, deterministic=False):
        """Sample an action from the distributions.
        This assumes dist is a tuple of path and power distributions."""
        path_action = self.sample_action_path(seed, dist[0], log_prob=log_prob, deterministic=deterministic)
        power_action  = self.sample_action_power(seed, dist[1], log_prob=log_prob, deterministic=deterministic)
        if log_prob:
            return path_action[0], power_action[0], path_action[1]+power_action[1]
        return path_action, power_action

    def sample_action(self, seed, dist, log_prob=False, deterministic=False):
        """Sample an action from the distributions.
        This assumes dist is a tuple of path and power distributions OR just the appropriate distribution."""
        if self.output_path and self.output_power:
            return self.sample_action_path_power(seed, dist, log_prob=log_prob, deterministic=deterministic)
        elif self.output_path:
            return self.sample_action_path(seed, dist, log_prob=log_prob, deterministic=deterministic)
        elif self.output_power:
            return self.sample_action_power(seed, dist, log_prob=log_prob, deterministic=deterministic)
        else:
            raise ValueError("No action type specified for sampling.")


if __name__ == "__main__":
    from collections import namedtuple
    #env, env_params = make_vone_env({"link_resources": 10})
    env, env_params = make_rsa_env({"link_resources": 10, "env_name": "nsfnet"})
    config = namedtuple(
        "Config",
        [
            "ACTIVATION",
            "NUM_LAYERS",
            "NUM_UNITS",
            "output_edges_size",
            "output_nodes_size",
            "output_globals_size",
            "gnn_latent",
            "message_passing_steps",
            "gnn_mlp_layers"
        ]
    )
    config.ACTIVATION = "tanh"
    config.NUM_LAYERS = 2
    config.NUM_UNITS = 64
    config.output_edges_size = 10
    config.output_nodes_size = 1
    config.output_globals_size = 1
    config.gnn_latent = 128
    config.message_passing_steps = 1
    config.gnn_mlp_layers = 1
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key, env_params)
    graph = state.graph
    action = jnp.array([1,2,3])

    actor_critic = ActorCriticGNN(
        activation=config.ACTIVATION,
        num_layers=config.NUM_LAYERS,
        num_units=config.NUM_UNITS,
        gnn_latent=config.gnn_latent,
        message_passing_steps=config.message_passing_steps,
        output_edges_size=config.output_edges_size,
        output_nodes_size=config.output_nodes_size,
        output_globals_size=config.output_globals_size,
        gnn_mlp_layers=config.gnn_mlp_layers,
    )
    # Reshape all arrays to include batch dimension
    state = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)
    env_params = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), env_params)
    params = actor_critic.init(key, state, env_params)

    out = jax.jit(actor_critic.apply)(params, state, env_params)
    print(out)
