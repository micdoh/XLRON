import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import distrax
import jraph
import chex
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Callable, Sequence
from functools import partial

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
    deterministic: bool = True

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # We will first linearly project the original features as 'embeddings'.
        embedder = jraph.GraphMapFeatures(
            embed_node_fn=nn.Dense(self.latent_size) if self.output_nodes_size > 0 else None,
            embed_edge_fn=nn.Dense(self.latent_size),
            embed_global_fn=nn.Dense(self.latent_size) if self.output_globals_size > 0 else None,
        )
        processed_graphs = embedder(graphs)

        # Now, we will apply a Graph Network once for each message-passing round.
        mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
        for _ in range(self.message_passing_steps):
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

        graph_net = jraph.GraphNetwork(
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

    @nn.compact
    def __call__(self, state: EnvState):
        # Remove globals from graph s.t. state value does not depend on the current request
        state = state.replace(graph=state.graph._replace(globals=jnp.zeros((1, 1))))
        processed_graph = GraphNet(
            latent_size=self.gnn_latent,
            message_passing_steps=self.message_passing_steps,
            output_edges_size=self.output_edges_size,
            output_nodes_size=self.output_nodes_size,
            output_globals_size=self.output_globals_size,
            num_mlp_layers=self.gnn_mlp_layers,
        )(state.graph)
        # TODO - does processed graph include processed globals in node and edge features?
        #  Or does globals feature the node and edge features, but not the other way round?
        # Index every other row of the edge features to get the link-slot array
        edge_features_flat = jnp.reshape(processed_graph.edges[::2], (-1,))
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

    @partial(jax.jit, static_argnums=(2,))
    @nn.compact
    def __call__(self, state: EnvState, params: EnvParams):
        """
        :param state: EnvState
        :param params: EnvParams
        :param config: Config - flags from parent script (e.g. train.py) used to configure the environment
        """
        processed_graph = GraphNet(
            latent_size=self.gnn_latent,
            message_passing_steps=self.message_passing_steps,
            output_edges_size=self.output_edges_size,
            output_nodes_size=self.output_nodes_size,
            output_globals_size=self.output_globals_size,
            num_mlp_layers=self.gnn_mlp_layers
        )(state.graph)
        # Index edge features to resemble the link-slot array
        edge_features = processed_graph.edges[::2]
        # TODO - add a normalisation step (normalise the edge features by link length)
        # Get the current request and initialise array of action distributions per path
        nodes_sd, requested_bw = read_rsa_request(state.request_array)
        init_action_array = jnp.zeros(params.k_paths * config.output_edges_size)

        # Define a budy func to retrieve path slots and update action array
        def get_path_action_dist(i, action_array):
            # Get the processed graph edge features corresponding to the i-th path
            path_features = get_path_slots(edge_features, params, nodes_sd, i)
            # Update the action array with the path features
            action_array = jax.lax.dynamic_update_slice(action_array, path_features, (i * config.output_edges_size,))
            return action_array

        action_dist = jax.lax.fori_loop(0, params.k_paths, get_path_action_dist, init_action_array)
        # Return a distrax.Categorical distribution over actions
        return distrax.Categorical(logits=jnp.reshape(action_dist, (-1,)))
        # TODO - within PPO, mask invalid actions


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

    @nn.compact
    def __call__(self, state: EnvState, params: EnvParams):
        actor = ActorGNN(
            gnn_latent=self.gnn_latent,
            message_passing_steps=self.message_passing_steps,
            output_edges_size=self.output_edges_size,
            output_nodes_size=self.output_nodes_size,
            output_globals_size=self.output_globals_size,
            gnn_mlp_layers=self.gnn_mlp_layers
        )
        critic = CriticGNN(
            activation=self.activation,
            num_layers=self.num_layers,
            num_units=self.num_units,
            gnn_latent=self.gnn_latent,
            message_passing_steps=self.message_passing_steps,
            output_edges_size=self.output_edges_size,
            output_nodes_size=self.output_nodes_size,
            output_globals_size=self.output_globals_size,
            gnn_mlp_layers=self.gnn_mlp_layers
        )
        action_dist = actor(state, params)
        critic_dist = critic(state)
        return action_dist, critic_dist


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
    # net = GraphNet(
    #     latent_size=128,
    #     message_passing_steps=1,
    #     output_edges_size=10,
    #     output_nodes_size=0,
    #     output_globals_size=0,
    #     num_mlp_layers=1
    # )
    action = jnp.array([1,2,3])
    #params = net.init(key, graph)
    #out = net.apply(params, graph)
    #print(out)
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
    params = actor_critic.init(key, state, env_params)
    out = actor_critic.apply(params, state, env_params)
    print(out)
    # actor = ActorGNN()
    # params = actor.init(key, state, env_params, config)
    # out = actor.apply(params, state, env_params, config)
    # print(out)
    # critic = CriticGNN()
    # params = critic.init(key, state)
    # out = critic.apply(params, state)
    # print(out)
