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

from xlron.environments.env_funcs import EnvState, EnvParams, get_path_slots, read_rsa_request, format_vone_slot_request
from xlron.environments.vone import make_vone_env
from xlron.environments.rsa import make_rsa_env
from xlron.models.gnn import GraphNetwork, GraphNetGAT, GAT


# Immutable class for storing nested node/edge features containing an embedding and a recurrent state.
StatefulField = collections.namedtuple("StatefulField", ["embedding", "state"])


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


class GraphNet(nn.Module):
    """A complete Graph Network model defined with Jraph."""

    latent_size: int
    num_mlp_layers: int
    message_passing_steps: int
    output_globals_size: int = 0
    output_edges_size: int = 0
    output_nodes_size: int = 0
    dropout_rate: float = 0
    skip_connections: bool = True
    use_edge_model: bool = True
    layer_norm: bool = True
    deterministic: bool = True  # If true, no dropout (better for RL purposes)
    use_attention: bool = True

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Template code from here: https://github.com/google/flax/blob/main/examples/ogbg_molpcba/models.py
        # We will first linearly project the original features as 'embeddings'.
        embedder = jraph.GraphMapFeatures(
            embed_node_fn=nn.Dense(self.latent_size) if self.output_nodes_size > 0 else None,
            embed_edge_fn=nn.Dense(self.latent_size),
            embed_global_fn=nn.Dense(self.latent_size) if self.output_globals_size > 0 else None,
        )
        processed_graphs = embedder(graphs)

        # Now, we will apply a Graph Network once for each message-passing round.
        for _ in range(self.message_passing_steps):
            mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers

            # TODO - Allow RNN/SSM to be used as update functions
            # https://github.com/luchris429/popjaxrl/blob/main/algorithms/ppo_gru.py
            if self.use_edge_model:
                update_edge_fn = jraph.concatenated_args(
                    MLP(
                        mlp_feature_sizes,
                        dropout_rate=self.dropout_rate,
                        deterministic=self.deterministic,
                    )
                )
            else:
                update_edge_fn = None

            update_node_fn = jraph.concatenated_args(
                MLP(
                    mlp_feature_sizes,
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                )
            )
            update_global_fn = jraph.concatenated_args(
                MLP(
                    mlp_feature_sizes,
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                )
            )

            if self.use_attention:
                def _attention_logit_fn(
                    edges: jnp.ndarray, sender_attr: jnp.ndarray, receiver_attr: jnp.ndarray, global_edge_attributes: jnp.ndarray
                ) -> jnp.ndarray:
                    """Calculate attention logits for each edge using the edges & global edge attributes."""
                    # TODO - would need to change this for VONE to incorporate node information
                    #  e.g. use same as update_global/edge/node_fn
                    # TODO - try using jraph.concatenated_args() here
                    x = jnp.concatenate((edges, global_edge_attributes), axis=1)
                    return MLP(mlp_feature_sizes, dropout_rate=self.dropout_rate,
                                deterministic=self.deterministic,)(x)

                def _attention_reduce_fn(
                    edges: jnp.ndarray, attention: jnp.ndarray
                ) -> jnp.ndarray:
                    # TODO - could try more sophisticated attention reduce function (not sure what it would be)
                    return attention * edges

                graph_net = GraphNetGAT(
                    update_node_fn=update_node_fn if self.output_nodes_size > 0 else None,
                    update_edge_fn=update_edge_fn,  # Update the edges with MLP prior to attention
                    update_global_fn=update_global_fn if self.output_globals_size > 0 else None,
                    attention_logit_fn=_attention_logit_fn,
                    attention_reduce_fn=_attention_reduce_fn,  # TODO - check this attention reduce function
                    # (here might help https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial7/GNN_overview.html)
                )
            else:
                graph_net = GraphNetwork(
                    update_node_fn=update_node_fn if self.output_nodes_size > 0 else None,
                    update_edge_fn=update_edge_fn,
                    update_global_fn=update_global_fn if self.output_globals_size > 0 else None,
                )

            if self.skip_connections:
                processed_graphs = add_graphs_tuples(
                graph_net(processed_graphs), processed_graphs
            )
            else:
                processed_graphs = graph_net(processed_graphs)

            if self.layer_norm:
                processed_graphs = processed_graphs._replace(
                    nodes=nn.LayerNorm()(processed_graphs.nodes),
                    edges=nn.LayerNorm()(processed_graphs.edges),
                    globals=nn.LayerNorm()(processed_graphs.globals) if processed_graphs.globals is not None else None,
                )

        decoder = jraph.GraphMapFeatures(
            embed_global_fn=nn.Dense(self.output_globals_size) if self.output_globals_size > 0 else None,
            embed_node_fn=nn.Dense(self.output_nodes_size) if self.output_nodes_size > 0 else None,
            embed_edge_fn=nn.Dense(self.output_edges_size),
        )
        processed_graphs = decoder(processed_graphs)

        return processed_graphs


class CriticGNN(nn.Module):
    activation: str = "tanh"
    num_layers: int = 2
    num_units: int = 64
    gnn_latent: int = 128
    message_passing_steps: int = 1
    output_edges_size: int = 10
    output_nodes_size: int = 1
    output_globals_size: int = 1
    gnn_mlp_layers: int = 1
    use_attention: bool = True

    @nn.compact
    def __call__(self, state: EnvState, params: EnvParams):
        # Remove globals from graph s.t. state value does not depend on the current request
        state = state.replace(graph=state.graph._replace(globals=jnp.zeros((1, 1))))
        processed_graph = GraphNet(
            latent_size=self.gnn_latent,
            message_passing_steps=self.message_passing_steps,
            output_edges_size=self.output_edges_size,
            output_nodes_size=self.output_nodes_size,
            output_globals_size=self.output_globals_size,
            num_mlp_layers=self.gnn_mlp_layers,
            use_attention=self.use_attention,
        )(state.graph)
        # Take every other edge as edges are bi-driectional and therefore duplicated
        edge_features = processed_graph.edges[::2]
        edge_features = edge_features * (params.link_length_array.val/jnp.sum(params.link_length_array.val))
        # TODO - does processed graph include processed globals in node and edge features?
        #  Or does globals feature the node and edge features, but not the other way round?
        # Index every other row of the edge features to get the link-slot array
        edge_features_flat = jnp.reshape(edge_features, (-1,))
        # pass aggregated features through MLP
        critic = make_dense_layers(edge_features_flat, self.num_units, self.num_layers, self.activation)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return jnp.squeeze(critic, axis=-1)


class ActorGNN(nn.Module):
    """
    Actor network for PPO algorithm. Takes the current state and returns a distrax.Categorical distribution
    over actions.
    """
    gnn_latent: int = 128
    message_passing_steps: int = 1
    output_edges_size: int = 10
    output_nodes_size: int = 1
    output_globals_size: int = 1
    gnn_mlp_layers: int = 1
    use_attention: bool = True
    normalise_by_link_length: bool = True  # Normalise the processed edge features by the link length

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
            latent_size=self.gnn_latent,
            message_passing_steps=self.message_passing_steps,
            output_edges_size=self.output_edges_size,
            output_nodes_size=self.output_nodes_size,
            output_globals_size=self.output_globals_size,
            num_mlp_layers=self.gnn_mlp_layers,
            use_attention=self.use_attention,
        )(state.graph)

        # Index edge features to resemble the link-slot array
        edge_features = processed_graph.edges[::2]
        # Normalise features by normalised link length from state.link_length_array
        if self.normalise_by_link_length:
            edge_features = edge_features * (params.link_length_array.val/jnp.sum(params.link_length_array.val))
        # Get the current request and initialise array of action distributions per path
        nodes_sd, requested_bw = read_rsa_request(state.request_array)
        init_action_array = jnp.zeros(params.k_paths * self.output_edges_size)

        # Define a body func to retrieve path slots and update action array
        def get_path_action_dist(i, action_array):
            # Get the processed graph edge features corresponding to the i-th path
            path_features = get_path_slots(edge_features, params, nodes_sd, i, agg_func="sum")
            # Update the action array with the path features
            action_array = jax.lax.dynamic_update_slice(action_array, path_features, (i * self.output_edges_size,))
            return action_array

        action_dist = jax.lax.fori_loop(0, params.k_paths, get_path_action_dist, init_action_array)
        # Return a distrax.Categorical distribution over actions (which can be masked later)
        return distrax.Categorical(logits=jnp.reshape(action_dist, (-1,)))


class ActorCriticGNN(nn.Module):
    """Combine the GNN actor and critic networks into a single class"""
    activation: str = "tanh"
    num_layers: int = 2
    num_units: int = 64
    gnn_latent: int = 128
    message_passing_steps: int = 1
    output_edges_size: int = 10
    output_nodes_size: int = 1
    output_globals_size: int = 0
    gnn_mlp_layers: int = 1
    use_attention: bool = True
    normalise_by_link_length: bool = True  # Normalise the processed edge features by the link length

    @nn.compact
    def __call__(self, state: EnvState, params: EnvParams):
        actor = jax.vmap(ActorGNN(
            gnn_latent=self.gnn_latent,
            message_passing_steps=self.message_passing_steps,
            output_edges_size=self.output_edges_size,
            output_nodes_size=self.output_nodes_size,
            output_globals_size=self.output_globals_size,
            gnn_mlp_layers=self.gnn_mlp_layers,
            use_attention=self.use_attention,
            normalise_by_link_length=self.normalise_by_link_length,
        ), in_axes=0)(state, params)
        critic = jax.vmap(CriticGNN(
            activation=self.activation,
            num_layers=self.num_layers,
            num_units=self.num_units,
            gnn_latent=self.gnn_latent,
            message_passing_steps=self.message_passing_steps,
            output_edges_size=self.output_edges_size,
            output_nodes_size=self.output_nodes_size,
            output_globals_size=self.output_globals_size,
            gnn_mlp_layers=self.gnn_mlp_layers,
            use_attention=self.use_attention,
        ), in_axes=0)(state, params)
        # Actor is returned as a list for compatibility with MLP VONE option in PPO script
        return [actor], critic


# TODO - adapt to VONE environment
# class ActorGNNVone(nn.Module):
#     """
#     Actor network for PPO algorithm. Takes the current state and returns a distrax.Categorical distribution
#     over actions.
#     """
#
#     @nn.compact
#     def __call__(self, state: EnvState, params: EnvParams, config, action: chex.Array = None):
#         """
#         :param state: EnvState
#         :param params: EnvParams
#         :param config: Config - flags from parent script (e.g. train.py) used to configure the environment
#         :param action: (optional) chex.Array - only used for VONE environment
#         """
#         processed_graph = GraphNet(
#             latent_size=128,
#             message_passing_steps=1,
#             output_edges_size=10,
#             output_nodes_size=1,
#             output_globals_size=0,
#             num_mlp_layers=1
#         )(state.graph)
#         # Index edge features to resemble the link-slot array
#         edge_features = processed_graph.edges[::2]
#         # Get the current request and initialise array of action distributions per path
#         request = format_vone_slot_request(state, action) if config.env_name == "vone" else state.request_array
#         nodes_sd, requested_bw = read_rsa_request(request)
#         init_action_array = jnp.zeros(params.k_paths * config.output_edges_size)
#
#         # Define a budy func to retrieve path slots and update action array
#         def get_path_action_dist(i, action_array):
#             path_features = get_path_slots(edge_features, params, nodes_sd, i)
#             action_array = jax.lax.dynamic_update_slice(action_array, path_features, (i * config.output_edges_size,))
#             return action_array
#
#         action_dist = jax.lax.fori_loop(0, params.k_paths, get_path_action_dist, init_action_array)
#         # Return a distrax.Categorical distribution over actions
#         return distrax.Categorical(logits=jnp.reshape(action_dist, (-1,)))


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
