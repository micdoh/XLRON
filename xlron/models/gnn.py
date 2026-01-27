import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
from jax import Array
from jraph._src import graph as gn_graph
from jraph._src import utils

from xlron import dtype_config
from xlron.environments.dataclasses import EnvParams, EnvState
from xlron.environments.env_funcs import get_path_slots, read_rsa_request
from xlron.environments.gn_model.isrs_gn_model import from_dbm
from xlron.models.mlp import (
    make_linear_with_orthogonal_init,
    select_activation,
)

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet
ArrayTree = Union[jnp.ndarray, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[[EdgeFeatures, jnp.ndarray, int], NodeFeatures]

# Signature:
# (nodes of each graph to be aggregated, segment ids, number of segments) ->
# aggregated nodes
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, jnp.ndarray, int], Globals]

# Signature:
# (edges of each graph to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToGlobalsFn = Callable[[EdgeFeatures, jnp.ndarray, int], Globals]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# attention weights
AttentionLogitFn = Callable[[EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree]

# Signature:
# (edge features, weights) -> edge features for node update
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]

# Signature:
# (edges to be normalized, segment ids, number of segments) ->
# normalized edges
AttentionNormalizeFn = Callable[[EdgeFeatures, jnp.ndarray, int], EdgeFeatures]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# updated edge features
GNUpdateEdgeFn = Callable[[EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[[NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], NodeFeatures]

GNUpdateGlobalFn = Callable[[NodeFeatures, EdgeFeatures, Globals], Globals]

# Signature:
# edge features -> embedded edge features
EmbedEdgeFn = Callable[[EdgeFeatures], EdgeFeatures]

# Signature:
# node features -> embedded node features
EmbedNodeFn = Callable[[NodeFeatures], NodeFeatures]

# Signature:
# globals features -> embedded globals features
EmbedGlobalFn = Callable[[Globals], Globals]


def add_self_edges_fn(
    receivers: jnp.ndarray, senders: jnp.ndarray, total_num_nodes: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Adds self edges. Assumes self edges are not in the graph yet."""
    # TODo - check if self-edges required
    receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)), axis=0)
    senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
    return receivers, senders


def GraphNetwork(
    update_edge_fn: Optional[GNUpdateEdgeFn],
    update_node_fn: Optional[GNUpdateNodeFn],
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    # TODO: allow RNN/SSM to be used in the aggregation function
    #  https://github.com/luchris429/popjaxrl/blob/main/algorithms/ppo_gru.py
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.segment_sum,
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_normalize_fn: Optional[AttentionNormalizeFn] = utils.segment_softmax,
    attention_reduce_fn: Optional[AttentionReduceFn] = None,
):
    """Returns a method that applies a configured GraphNetwork.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    than the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.

    Example usage::

      gn = GraphNetwork(update_edge_function,
      update_node_function, **kwargs)
      # Conduct multiple rounds of message passing with the same parameters:
      for _ in range(num_message_passing_steps):
        graph = gn(graph)

    Args:
      update_edge_fn: function used to update the edges or None to deactivate edge
        updates.
      update_node_fn: function used to update the nodes or None to deactivate node
        updates.
      update_global_fn: function used to update the globals or None to deactivate
        globals updates.
      aggregate_edges_for_nodes_fn: function used to aggregate messages to each
        node.
      aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
        globals.
      aggregate_edges_for_globals_fn: function used to aggregate the edges for the
        globals.
      attention_logit_fn: function used to calculate the attention weights or
        None to deactivate attention mechanism.
      attention_normalize_fn: function used to normalize raw attention logits or
        None if attention mechanism is not active.
      attention_reduce_fn: function used to apply weights to the edge features or
        None if attention mechanism is not active.

    Returns:
      A method that applies the configured GraphNetwork.
    """

    def not_both_supplied(x, y):
        return (x != y) and ((x is None) or (y is None))

    if not_both_supplied(attention_reduce_fn, attention_logit_fn):
        raise ValueError(("attention_logit_fn and attention_reduce_fn must both be supplied."))

    def _ApplyGraphNet(graph):
        """Applies a configured GraphNetwork to a graph.

        This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

        There is one difference. For the nodes update the class aggregates over the
        sender edges and receiver edges separately. This is a bit more general
        the algorithm described in the paper. The original behaviour can be
        recovered by using only the receiver edge aggregations for the update.

        In addition this implementation supports softmax attention over incoming
        edge features.

        Many popular Graph Neural Networks can be implemented as special cases of
        GraphNets, for more information please see the paper.

        Args:
          graph: a `GraphsTuple` containing the graph.

        Returns:
          Updated `GraphsTuple`.
        """
        # pylint: disable=g-long-lambda
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
        # Equivalent to jnp.sum(n_node), but jittable
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
        sum_n_edge = senders.shape[0]
        if not tree.tree_all(tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
            raise ValueError("All node arrays in nest must contain the same number of nodes.")

        sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
        # Here we scatter the global features to the corresponding edges,
        # giving us tensors of shape [num_edges, global_feat].
        global_edge_attributes = tree.tree_map(
            lambda g: jnp.repeat(g, n_edge, axis=0, total_repeat_length=sum_n_edge),
            globals_,
        )

        if update_edge_fn:
            edges = update_edge_fn(
                edges, sent_attributes, received_attributes, global_edge_attributes
            )

        if attention_logit_fn:
            logits = attention_logit_fn(
                edges, sent_attributes, received_attributes, global_edge_attributes
            )
            tree_calculate_weights = functools.partial(
                attention_normalize_fn, segment_ids=receivers, num_segments=sum_n_node
            )
            weights = tree.tree_map(tree_calculate_weights, logits)
            edges = attention_reduce_fn(edges, weights)

        if update_node_fn:
            sent_attributes = tree.tree_map(
                lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges
            )
            received_attributes = tree.tree_map(
                lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node), edges
            )
            # Here we scatter the global features to the corresponding nodes,
            # giving us tensors of shape [num_nodes, global_feat].
            global_attributes = tree.tree_map(
                lambda g: jnp.repeat(g, n_node, axis=0, total_repeat_length=sum_n_node),
                globals_,
            )
            nodes = update_node_fn(nodes, sent_attributes, received_attributes, global_attributes)

        if update_global_fn:
            n_graph = n_node.shape[0]
            graph_idx = jnp.arange(n_graph)
            # To aggregate nodes and edges from each graph to global features,
            # we first construct tensors that map the node to the corresponding graph.
            # For example, if you have `n_node=[1,2]`, we construct the tensor
            # [0, 1, 1]. We then do the same for edges.
            node_gr_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
            edge_gr_idx = jnp.repeat(graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge)
            # We use the aggregation function to pool the nodes/edges per graph.
            node_attributes = tree.tree_map(
                lambda n: aggregate_nodes_for_globals_fn(n, node_gr_idx, n_graph), nodes
            )
            edge_attributes = tree.tree_map(
                lambda e: aggregate_edges_for_globals_fn(e, edge_gr_idx, n_graph), edges
            )
            # These pooled nodes are the inputs to the global update fn.
            globals_ = update_global_fn(node_attributes, edge_attributes, globals_)
        # pylint: enable=g-long-lambda
        return gn_graph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge,
        )

    return _ApplyGraphNet


def GraphNetGAT(
    update_edge_fn: GNUpdateEdgeFn,
    update_node_fn: GNUpdateNodeFn,
    attention_logit_fn: AttentionLogitFn,
    attention_reduce_fn: AttentionReduceFn,
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.segment_sum,
):
    """Returns a method that applies a GraphNet with attention on edge features.

    Args:
      update_edge_fn: function used to update the edges.
      update_node_fn: function used to update the nodes.
      attention_logit_fn: function used to calculate the attention weights.
      attention_reduce_fn: function used to apply attention weights to the edge
        features.
      update_global_fn: function used to update the globals or None to deactivate
        globals updates.
      aggregate_edges_for_nodes_fn: function used to aggregate attention-weighted
        messages to each node.
      aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
        globals.
      aggregate_edges_for_globals_fn: function used to aggregate
        attention-weighted edges for the globals.

    Returns:
      A function that applies a GraphNet Graph Attention layer.
    """
    if (attention_logit_fn is None) or (attention_reduce_fn is None):
        raise ValueError(
            (
                "`None` value not supported for `attention_logit_fn` or "
                "`attention_reduce_fn` in a Graph Attention network."
            )
        )
    return GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn,
        attention_logit_fn=attention_logit_fn,
        attention_reduce_fn=attention_reduce_fn,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
        aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
        aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn,
    )


GATAttentionQueryFn = Callable[[NodeFeatures], NodeFeatures]
GATAttentionLogitFn = Callable[[SenderFeatures, ReceiverFeatures, EdgeFeatures], EdgeFeatures]
GATNodeUpdateFn = Callable[[NodeFeatures], NodeFeatures]


def GAT(
    attention_query_fn: GATAttentionQueryFn,
    attention_logit_fn: GATAttentionLogitFn,
    node_update_fn: Optional[GATNodeUpdateFn] = None,
):
    """Returns a method that applies a Graph Attention Network layer.

    Graph Attention message passing as described in
    https://arxiv.org/abs/1710.10903. This model expects node features as a
    jnp.array, may use edge features for computing attention weights, and
    ignore global features. It does not support nests.

    NOTE: this implementation assumes that the input graph has self edges. To
    recover the behavior of the referenced paper, please add self edges.

    Args:
      attention_query_fn: function that generates attention queries
        from sender node features.
      attention_logit_fn: function that converts attention queries into logits for
        softmax attention.
      node_update_fn: function that updates the aggregated messages. If None,
        will apply leaky relu and concatenate (if using multi-head attention).

    Returns:
      A function that applies a Graph Attention layer.
    """
    # pylint: disable=g-long-lambda
    if node_update_fn is None:
        # By default, apply the leaky relu and then concatenate the heads on the
        # feature axis.
        def node_update_fn(x):
            return jnp.reshape(jax.nn.leaky_relu(x), (x.shape[0], -1))

    def _ApplyGAT(graph):
        """Applies a Graph Attention layer."""
        nodes, edges, receivers, senders, _, _, _ = graph
        # Equivalent to the sum of n_node, but statically known.
        try:
            sum_n_node = nodes.shape[0]
        except IndexError:
            raise IndexError("GAT requires node features")  # pylint: disable=raise-missing-from

        # First pass nodes through the node updater.
        nodes = attention_query_fn(nodes)
        # pylint: disable=g-long-lambda
        # We compute the softmax logits using a function that takes the
        # embedded sender and receiver attributes.
        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]
        softmax_logits = attention_logit_fn(sent_attributes, received_attributes, edges)

        # Compute the softmax weights on the entire tree.
        weights = utils.segment_softmax(
            softmax_logits, segment_ids=receivers, num_segments=sum_n_node
        )
        # Apply weights
        messages = sent_attributes * weights
        # Aggregate messages to nodes.
        nodes = utils.segment_sum(messages, receivers, num_segments=sum_n_node)

        # Apply an update function to the aggregated messages.
        nodes = node_update_fn(nodes)
        return graph._replace(nodes=nodes)

    # pylint: enable=g-long-lambda
    return _ApplyGAT


def add_graphs_tuples(
    graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple
) -> jraph.GraphsTuple:
    """Adds the nodes, edges and global features from other_graphs to graphs."""
    return graphs._replace(
        nodes=graphs.nodes + other_graphs.nodes,
        edges=graphs.edges + other_graphs.edges,
        globals=graphs.globals + other_graphs.globals if graphs.globals is not None else None,
    )


class GraphNet(eqx.Module):
    """A complete Graph Network model defined with Jraph and Equinox."""

    # Embedding layers
    edge_embedder: eqx.nn.Linear
    node_embedder: eqx.nn.Linear
    global_embedder: eqx.nn.Linear

    # MLP layers for each message passing step (list of tuples)
    message_passing_layers: tuple

    # Decoder layers
    edge_decoder: Optional[eqx.nn.Linear]
    node_decoder: Optional[eqx.nn.Linear]
    global_decoder: Optional[eqx.nn.Linear]

    # Layer norms for each step
    layer_norms: Optional[tuple]

    # Static configuration
    message_passing_steps: int = eqx.field(static=True)
    edge_embedding_size: int = eqx.field(static=True)
    node_embedding_size: int = eqx.field(static=True)
    global_embedding_size: int = eqx.field(static=True)
    edge_output_size: int = eqx.field(static=True)
    node_output_size: int = eqx.field(static=True)
    global_output_size: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    skip_connections: bool = eqx.field(static=True)
    use_edge_model: bool = eqx.field(static=True)
    gnn_layer_norm: bool = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)
    use_attention: bool = eqx.field(static=True)

    def __init__(
        self,
        input_edge_features: int,
        input_node_features: int,
        input_global_features: int,
        message_passing_steps: int = 1,
        mlp_layers: int = 0,
        mlp_latent: int = 0,
        edge_embedding_size: int = 128,
        edge_mlp_layers: int = 3,
        edge_mlp_latent: int = 128,
        edge_output_size: int = 0,
        global_embedding_size: int = 8,
        global_mlp_layers: int = 0,
        global_mlp_latent: int = 0,
        global_output_size: int = 0,
        node_embedding_size: int = 16,
        node_mlp_layers: int = 2,
        node_mlp_latent: int = 128,
        node_output_size: int = 0,
        attn_mlp_layers: int = 2,
        attn_mlp_latent: int = 128,
        dropout_rate: float = 0,
        skip_connections: bool = True,
        use_edge_model: bool = True,
        gnn_layer_norm: bool = True,
        mlp_layer_norm: bool = False,
        deterministic: bool = True,
        use_attention: bool = True,
        *,
        key: Array,
    ):
        self.message_passing_steps = message_passing_steps
        self.edge_embedding_size = edge_embedding_size
        self.node_embedding_size = node_embedding_size
        self.global_embedding_size = global_embedding_size
        self.edge_output_size = edge_output_size
        self.node_output_size = node_output_size
        self.global_output_size = global_output_size
        self.dropout_rate = dropout_rate
        self.skip_connections = skip_connections
        self.use_edge_model = use_edge_model
        self.gnn_layer_norm = gnn_layer_norm
        self.deterministic = deterministic
        self.use_attention = use_attention

        # Determine MLP dimensions
        if mlp_latent is not None:
            global_mlp_dims = edge_mlp_dims = node_mlp_dims = attn_mlp_dims = [
                mlp_latent
            ] * mlp_layers
        else:
            global_mlp_dims = [global_mlp_latent] * global_mlp_layers
            edge_mlp_dims = [edge_mlp_latent] * edge_mlp_layers
            node_mlp_dims = [node_mlp_latent] * node_mlp_layers
            attn_mlp_dims = [attn_mlp_latent] * attn_mlp_layers

        if skip_connections:
            edge_mlp_dims = edge_mlp_dims + [edge_embedding_size]
            node_mlp_dims = node_mlp_dims + [node_embedding_size]
            global_mlp_dims = global_mlp_dims + [global_embedding_size]

        # Split keys
        keys = jax.random.split(key, 10 + message_passing_steps * 4)
        key_idx = 0

        # Create embedders
        self.edge_embedder = eqx.nn.Linear(
            input_edge_features, edge_embedding_size, key=keys[key_idx]
        )
        key_idx += 1
        self.node_embedder = eqx.nn.Linear(
            input_node_features, node_embedding_size, key=keys[key_idx]
        )
        key_idx += 1
        self.global_embedder = eqx.nn.Linear(
            input_global_features, global_embedding_size, key=keys[key_idx]
        )
        key_idx += 1

        # Create message passing layers for each step
        mp_layers = []
        layer_norms_list = []

        # Input sizes for MLPs after concatenation
        # Edge MLP: edges + sender_nodes + receiver_nodes + globals
        edge_mlp_input = edge_embedding_size + 2 * node_embedding_size + global_embedding_size
        # Node MLP: nodes + aggregated_sent + aggregated_received + globals
        node_mlp_input = node_embedding_size + 2 * edge_embedding_size + global_embedding_size
        # Global MLP: aggregated_nodes + aggregated_edges + globals
        global_mlp_input = node_embedding_size + edge_embedding_size + global_embedding_size
        # Attention MLP: edges + sender + receiver + globals
        attn_mlp_input = edge_embedding_size + 2 * node_embedding_size + global_embedding_size

        for step in range(message_passing_steps):
            step_layers = {}

            if use_edge_model and edge_mlp_dims:
                step_layers["edge_mlp"] = eqx.nn.MLP(
                    in_size=edge_mlp_input,
                    out_size=edge_mlp_dims[-1] if edge_mlp_dims else edge_embedding_size,
                    width_size=edge_mlp_dims[0] if edge_mlp_dims else edge_embedding_size,
                    depth=len(edge_mlp_dims),
                    activation=jax.nn.relu,
                    key=keys[key_idx],
                )
                key_idx += 1

            if node_mlp_dims:
                step_layers["node_mlp"] = eqx.nn.MLP(
                    in_size=node_mlp_input,
                    out_size=node_mlp_dims[-1] if node_mlp_dims else node_embedding_size,
                    width_size=node_mlp_dims[0] if node_mlp_dims else node_embedding_size,
                    depth=len(node_mlp_dims),
                    activation=jax.nn.relu,
                    key=keys[key_idx],
                )
                key_idx += 1

            if global_output_size > 0 and global_mlp_dims:
                step_layers["global_mlp"] = eqx.nn.MLP(
                    in_size=global_mlp_input,
                    out_size=global_mlp_dims[-1] if global_mlp_dims else global_embedding_size,
                    width_size=global_mlp_dims[0] if global_mlp_dims else global_embedding_size,
                    depth=len(global_mlp_dims),
                    activation=jax.nn.relu,
                    key=keys[key_idx],
                )
                key_idx += 1

            if use_attention and attn_mlp_dims:
                # Ensure at least depth 1 for GATv2-style dynamic attention
                attn_depth = max(len(attn_mlp_dims), 1)
                step_layers["attn_mlp"] = eqx.nn.MLP(
                    in_size=attn_mlp_input,
                    out_size=1,
                    width_size=attn_mlp_dims[0] if attn_mlp_dims else 128,
                    depth=attn_depth,
                    activation=jax.nn.relu,
                    key=keys[key_idx],
                )
                key_idx += 1

            mp_layers.append(step_layers)

            if gnn_layer_norm:
                layer_norms_list.append(
                    {
                        "node": eqx.nn.LayerNorm(node_embedding_size),
                        "edge": eqx.nn.LayerNorm(edge_embedding_size),
                        "global": eqx.nn.LayerNorm(global_embedding_size)
                        if global_output_size > 0
                        else None,
                    }
                )

        self.message_passing_layers = tuple(mp_layers)
        self.layer_norms = tuple(layer_norms_list) if gnn_layer_norm else None

        # Create decoders
        self.edge_decoder = (
            eqx.nn.Linear(edge_embedding_size, edge_output_size, key=keys[key_idx])
            if edge_output_size > 0
            else None
        )
        key_idx += 1
        self.node_decoder = (
            eqx.nn.Linear(node_embedding_size, node_output_size, key=keys[key_idx])
            if node_output_size > 0
            else None
        )
        key_idx += 1
        self.global_decoder = (
            eqx.nn.Linear(global_embedding_size, global_output_size, key=keys[key_idx])
            if global_output_size > 0
            else None
        )

    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Flatten edges if needed
        if graphs.edges.ndim >= 3:
            edges = graphs.edges.reshape((graphs.edges.shape[0], -1))
            graphs = graphs._replace(edges=edges)

        # Embed
        nodes = jax.vmap(self.node_embedder)(graphs.nodes)
        edges = jax.vmap(self.edge_embedder)(graphs.edges)
        globals_ = (
            jax.vmap(self.global_embedder)(graphs.globals)
            if graphs.globals is not None
            else jnp.zeros((1, self.global_embedding_size))
        )

        processed_graphs = graphs._replace(nodes=nodes, edges=edges, globals=globals_)

        # Message passing
        for step, step_layers in enumerate(self.message_passing_layers):
            # Build update functions using the MLPs
            # jraph.concatenated_args wraps a fn(concatenated_input) to accept separate args
            # # Remember to vmap over edges
            def make_update_edge_fn(edge_mlp):
                def update_edge_fn(concatenated_inputs):
                    return jax.vmap(edge_mlp)(concatenated_inputs)

                return jraph.concatenated_args(update_edge_fn)

            def make_update_node_fn(node_mlp):
                def update_node_fn(concatenated_inputs):
                    return jax.vmap(node_mlp)(concatenated_inputs)

                return jraph.concatenated_args(update_node_fn)

            def make_update_global_fn(global_mlp):
                def update_global_fn(concatenated_inputs):
                    return jax.vmap(global_mlp)(concatenated_inputs)

                return jraph.concatenated_args(update_global_fn)

            update_edge_fn = (
                make_update_edge_fn(step_layers["edge_mlp"]) if "edge_mlp" in step_layers else None
            )
            update_node_fn = (
                make_update_node_fn(step_layers["node_mlp"]) if "node_mlp" in step_layers else None
            )
            update_global_fn = (
                make_update_global_fn(step_layers["global_mlp"])
                if "global_mlp" in step_layers
                else None
            )

            if self.use_attention and "attn_mlp" in step_layers:
                attn_mlp = step_layers["attn_mlp"]

                def attention_logit_fn(edges, sender_attr, receiver_attr, global_edge_attributes):
                    x = jnp.concatenate(
                        (edges, sender_attr, receiver_attr, global_edge_attributes), axis=1
                    )
                    return jax.vmap(attn_mlp)(x)

                def attention_reduce_fn(edges, attention):
                    return attention * edges

                graph_net = GraphNetGAT(
                    update_node_fn=update_node_fn,
                    update_edge_fn=update_edge_fn,
                    update_global_fn=update_global_fn,
                    attention_logit_fn=attention_logit_fn,
                    attention_reduce_fn=attention_reduce_fn,
                )
            else:
                graph_net = GraphNetwork(
                    update_node_fn=update_node_fn,
                    update_edge_fn=update_edge_fn,
                    update_global_fn=update_global_fn,
                )

            new_graphs = graph_net(processed_graphs)

            if self.skip_connections:
                processed_graphs = add_graphs_tuples(new_graphs, processed_graphs)
            else:
                processed_graphs = new_graphs

            if self.gnn_layer_norm and self.layer_norms is not None:
                ln = self.layer_norms[step]
                processed_graphs = processed_graphs._replace(
                    nodes=jax.vmap(ln["node"])(processed_graphs.nodes),
                    edges=jax.vmap(ln["edge"])(processed_graphs.edges),
                    globals=jax.vmap(ln["global"])(processed_graphs.globals)
                    if ln["global"] is not None and processed_graphs.globals is not None
                    else processed_graphs.globals,
                )

        # Decode
        if self.edge_decoder is not None:
            edges = jax.vmap(self.edge_decoder)(processed_graphs.edges)
            processed_graphs = processed_graphs._replace(edges=edges)
        if self.node_decoder is not None:
            nodes = jax.vmap(self.node_decoder)(processed_graphs.nodes)
            processed_graphs = processed_graphs._replace(nodes=nodes)
        if self.global_decoder is not None and processed_graphs.globals is not None:
            globals_ = jax.vmap(self.global_decoder)(processed_graphs.globals)
            processed_graphs = processed_graphs._replace(globals=globals_)

        return processed_graphs


class CriticGNN(eqx.Module):
    """Critic network using GNN for processing graph state."""

    graph_net: GraphNet
    critic_mlp: eqx.nn.MLP
    critic_output: eqx.nn.Linear

    # Static configuration
    activation: str = eqx.field(static=True)
    global_output_size: int = eqx.field(static=True)
    normalise_by_link_length: bool = eqx.field(static=True)

    def __init__(
        self,
        input_edge_features: int,
        input_node_features: int,
        input_global_features: int,
        activation: str = "tanh",
        num_layers: int = 2,
        num_units: int = 64,
        message_passing_steps: int = 1,
        mlp_layers: int = None,
        mlp_latent: int = None,
        edge_embedding_size: int = 128,
        edge_mlp_layers: int = 3,
        edge_mlp_latent: int = 128,
        edge_output_size: int = 0,
        global_embedding_size: int = 8,
        global_mlp_layers: int = 0,
        global_mlp_latent: int = 0,
        global_output_size: int = 1,  # Must be 1!
        node_embedding_size: int = 16,
        node_mlp_layers: int = 2,
        node_mlp_latent: int = 128,
        node_output_size: int = 0,
        attn_mlp_layers: int = 2,
        attn_mlp_latent: int = 128,
        use_attention: bool = True,
        normalise_by_link_length: bool = True,
        gnn_layer_norm: bool = True,
        mlp_layer_norm: bool = False,
        *,
        key: Array,
    ):
        assert global_output_size == 1
        self.activation = activation
        self.global_output_size = global_output_size
        self.normalise_by_link_length = normalise_by_link_length

        gnn_key, mlp_key, output_key = jax.random.split(key, 3)

        self.graph_net = GraphNet(
            input_edge_features=input_edge_features,
            input_node_features=input_node_features,
            input_global_features=input_global_features,
            message_passing_steps=message_passing_steps,
            mlp_layers=mlp_layers,
            mlp_latent=mlp_latent,
            edge_embedding_size=edge_embedding_size,
            edge_mlp_layers=edge_mlp_layers,
            edge_mlp_latent=edge_mlp_latent,
            edge_output_size=edge_output_size,
            global_embedding_size=global_embedding_size,
            global_mlp_layers=global_mlp_layers,
            global_mlp_latent=global_mlp_latent,
            global_output_size=global_output_size,
            node_embedding_size=node_embedding_size,
            node_mlp_layers=node_mlp_layers,
            node_mlp_latent=node_mlp_latent,
            node_output_size=node_output_size,
            attn_mlp_layers=attn_mlp_layers,
            attn_mlp_latent=attn_mlp_latent,
            use_attention=use_attention,
            gnn_layer_norm=gnn_layer_norm,
            mlp_layer_norm=mlp_layer_norm,
            key=gnn_key,
        )

        # MLP for processing flattened edge features (only used if global_output_size == 0)
        # We use a placeholder size; actual input size depends on runtime graph
        self.critic_mlp = eqx.nn.MLP(
            in_size=edge_output_size if edge_output_size > 0 else edge_embedding_size,
            out_size=num_units,
            width_size=num_units,
            depth=num_layers,
            activation=select_activation(activation),
            key=mlp_key,
        )

        self.critic_output = make_linear_with_orthogonal_init(num_units, 1, output_key, scale=1.0)

    def __call__(self, state: EnvState, params: EnvParams) -> Array:
        # Remove globals so value does not depend on current request
        graph = state.graph._replace(
            globals=jnp.zeros_like(state.graph.globals)
        )
        state = state.replace(graph=graph)

        processed_graph = self.graph_net(state.graph)

        # Global output is already the scalar value
        # Shape: (1, 1)
        return processed_graph.globals.squeeze()


class ActorGNN(eqx.Module):
    """Actor network using GNN for processing graph state."""

    graph_net: GraphNet
    power_mlp: Optional[eqx.nn.MLP]

    # Static configuration
    activation: str = eqx.field(static=True)
    edge_output_size: int = eqx.field(static=True)
    global_output_size: int = eqx.field(static=True)
    normalise_by_link_length: bool = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    min_power_dbm: float = eqx.field(static=True)
    max_power_dbm: float = eqx.field(static=True)
    step_power_dbm: float = eqx.field(static=True)
    discrete: bool = eqx.field(static=True)
    min_concentration: float = eqx.field(static=True)
    max_concentration: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        input_edge_features: int,
        input_node_features: int,
        input_global_features: int,
        activation: str = "tanh",
        num_layers: int = 2,
        num_units: int = 64,
        mlp_layers: int = None,
        mlp_latent: int = None,
        edge_embedding_size: int = 128,
        edge_mlp_layers: int = 3,
        edge_mlp_latent: int = 128,
        edge_output_size: int = 0,
        global_embedding_size: int = 8,
        global_mlp_layers: int = 0,
        global_mlp_latent: int = 0,
        global_output_size: int = 0,
        node_embedding_size: int = 16,
        node_mlp_layers: int = 2,
        node_mlp_latent: int = 128,
        node_output_size: int = 0,
        attn_mlp_layers: int = 2,
        attn_mlp_latent: int = 128,
        dropout_rate: float = 0,
        deterministic: bool = False,
        message_passing_steps: int = 1,
        use_attention: bool = True,
        normalise_by_link_length: bool = True,
        gnn_layer_norm: bool = True,
        mlp_layer_norm: bool = False,
        temperature: float = 1.0,
        min_power_dbm: float = 0.0,
        max_power_dbm: float = 2.0,
        step_power_dbm: float = 0.1,
        discrete: bool = True,
        min_concentration: float = 0.1,
        max_concentration: float = 20.0,
        epsilon: float = 1e-6,
        *,
        key: Array,
    ):
        self.activation = activation
        self.edge_output_size = edge_output_size
        self.global_output_size = global_output_size
        self.normalise_by_link_length = normalise_by_link_length
        self.temperature = temperature
        self.min_power_dbm = min_power_dbm
        self.max_power_dbm = max_power_dbm
        self.step_power_dbm = step_power_dbm
        self.discrete = discrete
        self.min_concentration = min_concentration
        self.max_concentration = max_concentration
        self.epsilon = epsilon

        gnn_key, mlp_key = jax.random.split(key)

        self.graph_net = GraphNet(
            input_edge_features=input_edge_features,
            input_node_features=input_node_features,
            input_global_features=input_global_features,
            message_passing_steps=message_passing_steps,
            mlp_layers=mlp_layers,
            mlp_latent=mlp_latent,
            edge_embedding_size=edge_embedding_size,
            edge_mlp_layers=edge_mlp_layers,
            edge_mlp_latent=edge_mlp_latent,
            edge_output_size=edge_output_size,
            global_embedding_size=global_embedding_size,
            global_mlp_layers=global_mlp_layers,
            global_mlp_latent=global_mlp_latent,
            global_output_size=global_output_size,
            node_embedding_size=node_embedding_size,
            node_mlp_layers=node_mlp_layers,
            node_mlp_latent=node_mlp_latent,
            node_output_size=node_output_size,
            attn_mlp_layers=attn_mlp_layers,
            attn_mlp_latent=attn_mlp_latent,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
            gnn_layer_norm=gnn_layer_norm,
            mlp_layer_norm=mlp_layer_norm,
            deterministic=deterministic,
            key=gnn_key,
        )

        # Power MLP
        num_power_levels = int((max_power_dbm - min_power_dbm) / step_power_dbm) + 1
        output_size = num_power_levels if discrete else 2
        edge_feat_size = edge_output_size if edge_output_size > 0 else edge_embedding_size

        self.power_mlp = eqx.nn.MLP(
            in_size=edge_feat_size,
            out_size=output_size,
            width_size=num_units,
            depth=num_layers,
            activation=select_activation(activation),
            key=mlp_key,
        )

    @property
    def num_power_levels(self):
        return int((self.max_power_dbm - self.min_power_dbm) / self.step_power_dbm) + 1

    @property
    def power_levels(self):
        return jnp.linspace(
            self.min_power_dbm,
            self.max_power_dbm,
            self.num_power_levels,
            dtype=dtype_config.SMALL_FLOAT_DTYPE,
        )

    def __call__(
        self, state: EnvState, params: EnvParams
    ) -> distrax.Distribution | Tuple[distrax.Distribution, Optional[distrax.Distribution]]:
        processed_graph = self.graph_net(state.graph)

        # Index edge features
        edge_features = (
            processed_graph.edges
            if params.directed_graph
            else processed_graph.edges[: len(processed_graph.edges) // 2]
        )
        if self.normalise_by_link_length:
            edge_features = edge_features * (
                params.link_length_array.val
                / jnp.sum(params.link_length_array.val, promote_integers=False)
            )

        # Get current request
        nodes_sd, requested_bw = read_rsa_request(state.request_array)
        init_action_array = jnp.zeros(
            params.k_paths * self.edge_output_size, dtype=dtype_config.SMALL_FLOAT_DTYPE
        )

        def get_path_action_dist(i, action_array):
            path_features = get_path_slots(edge_features, params, nodes_sd, i, agg_func="sum")
            action_array = jax.lax.dynamic_update_slice(
                action_array, path_features, (i * self.edge_output_size,)
            )
            return action_array

        path_action_logits = jax.lax.fori_loop(
            0, params.k_paths, get_path_action_dist, init_action_array
        )
        path_action_logits = jnp.reshape(path_action_logits, (-1,)) / self.temperature
        path_action_dist = distrax.Categorical(logits=path_action_logits)

        power_action_dist = None
        if params.__class__.__name__ == "RSAGNModelEnvParams":
            if self.global_output_size > 0:
                power_logits = processed_graph.globals.reshape((-1,)) / self.temperature
            else:
                init_feature_array = jnp.zeros(
                    (params.k_paths, edge_features.shape[1]), dtype=dtype_config.LARGE_FLOAT_DTYPE
                )

                def get_power_action_dist(i, feature_array):
                    path_features = get_path_slots(
                        edge_features, params, nodes_sd, i, agg_func="sum"
                    ).reshape((1, -1))
                    feature_array = jax.lax.dynamic_update_slice(
                        feature_array, path_features, (i, 0)
                    )
                    return feature_array

                path_feature_batch = jax.lax.fori_loop(
                    0, params.k_paths, get_power_action_dist, init_feature_array
                )
                power_logits = jax.vmap(self.power_mlp)(path_feature_batch)

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

            return (path_action_dist, power_action_dist)
            
        return path_action_dist



class ActorCriticGNN(eqx.Module):
    """Combined Actor-Critic GNN model."""

    actor: ActorGNN
    critic: CriticGNN

    # Static configuration
    vmap: bool = eqx.field(static=True)
    min_power_dbm: float = eqx.field(static=True)
    max_power_dbm: float = eqx.field(static=True)
    step_power_dbm: float = eqx.field(static=True)
    discrete: bool = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    output_path: bool = eqx.field(static=True)
    output_power: bool = eqx.field(static=True)

    def __init__(
        self,
        input_edge_features: int,
        input_node_features: int,
        input_global_features: int,
        activation: str = "tanh",
        num_layers: int = 2,
        num_units: int = 64,
        message_passing_steps: int = 1,
        mlp_layers: int = None,
        mlp_latent: int = None,
        edge_embedding_size: int = 128,
        edge_mlp_layers: int = 3,
        edge_mlp_latent: int = 128,
        edge_output_size_actor: int = 1,
        edge_output_size_critic: int = 1,
        global_embedding_size: int = 8,
        global_mlp_layers: int = 0,
        global_mlp_latent: int = 0,
        global_output_size_actor: int = 0,
        global_output_size_critic: int = 0,
        node_embedding_size: int = 16,
        node_mlp_layers: int = 2,
        node_mlp_latent: int = 128,
        node_output_size_actor: int = 0,
        node_output_size_critic: int = 0,
        attn_mlp_layers: int = 2,
        attn_mlp_latent: int = 128,
        gnn_mlp_layers: int = 1,
        use_attention: bool = True,
        normalise_by_link_length: bool = True,
        gnn_layer_norm: bool = True,
        mlp_layer_norm: bool = False,
        vmap: bool = True,
        temperature: float = 1.0,
        min_power_dbm: float = 0.0,
        max_power_dbm: float = 2.0,
        step_power_dbm: float = 0.1,
        discrete: bool = True,
        min_concentration: float = 0.1,
        max_concentration: float = 20.0,
        epsilon: float = 1e-6,
        output_path: bool = True,
        output_power: bool = True,
        *,
        key: Array,
    ):
        assert edge_output_size_actor > 0
        assert edge_output_size_critic + global_output_size_critic > 0

        self.vmap = vmap
        self.min_power_dbm = min_power_dbm
        self.max_power_dbm = max_power_dbm
        self.step_power_dbm = step_power_dbm
        self.discrete = discrete
        self.epsilon = epsilon
        self.output_path = output_path
        self.output_power = output_power

        actor_key, critic_key = jax.random.split(key)

        self.actor = ActorGNN(
            input_edge_features=input_edge_features,
            input_node_features=input_node_features,
            input_global_features=input_global_features,
            num_layers=num_layers,
            num_units=num_units,
            message_passing_steps=message_passing_steps,
            mlp_layers=mlp_layers,
            mlp_latent=mlp_latent,
            edge_embedding_size=edge_embedding_size,
            edge_mlp_layers=edge_mlp_layers,
            edge_mlp_latent=edge_mlp_latent,
            edge_output_size=edge_output_size_actor,
            global_embedding_size=global_embedding_size,
            global_mlp_layers=global_mlp_layers,
            global_mlp_latent=global_mlp_latent,
            global_output_size=global_output_size_actor,
            node_embedding_size=node_embedding_size,
            node_mlp_layers=node_mlp_layers,
            node_mlp_latent=node_mlp_latent,
            node_output_size=node_output_size_actor,
            attn_mlp_layers=attn_mlp_layers,
            attn_mlp_latent=attn_mlp_latent,
            use_attention=use_attention,
            normalise_by_link_length=normalise_by_link_length,
            gnn_layer_norm=gnn_layer_norm,
            mlp_layer_norm=mlp_layer_norm,
            temperature=temperature,
            min_power_dbm=min_power_dbm,
            max_power_dbm=max_power_dbm,
            step_power_dbm=step_power_dbm,
            discrete=discrete,
            min_concentration=min_concentration,
            max_concentration=max_concentration,
            epsilon=epsilon,
            key=actor_key,
        )

        self.critic = CriticGNN(
            input_edge_features=input_edge_features,
            input_node_features=input_node_features,
            input_global_features=input_global_features,
            activation=activation,
            num_layers=num_layers,
            num_units=num_units,
            message_passing_steps=message_passing_steps,
            mlp_layers=mlp_layers,
            mlp_latent=mlp_latent,
            edge_embedding_size=edge_embedding_size,
            edge_mlp_layers=edge_mlp_layers,
            edge_mlp_latent=edge_mlp_latent,
            edge_output_size=edge_output_size_critic,
            global_embedding_size=global_embedding_size,
            global_mlp_layers=global_mlp_layers,
            global_mlp_latent=global_mlp_latent,
            global_output_size=global_output_size_critic,
            node_embedding_size=node_embedding_size,
            node_mlp_layers=node_mlp_layers,
            node_mlp_latent=node_mlp_latent,
            node_output_size=node_output_size_critic,
            attn_mlp_layers=attn_mlp_layers,
            attn_mlp_latent=attn_mlp_latent,
            use_attention=use_attention,
            normalise_by_link_length=normalise_by_link_length,
            gnn_layer_norm=gnn_layer_norm,
            mlp_layer_norm=mlp_layer_norm,
            key=critic_key,
        )

    @property
    def num_power_levels(self):
        return int((self.max_power_dbm - self.min_power_dbm) / self.step_power_dbm) + 1

    @property
    def power_levels(self):
        return jnp.linspace(
            self.min_power_dbm,
            self.max_power_dbm,
            self.num_power_levels,
            dtype=dtype_config.LARGE_FLOAT_DTYPE,
        )

    def __call__(self, state: EnvState, params: EnvParams):
        if self.vmap:
            actor_fn = jax.vmap(self.actor, in_axes=(0, None))
            critic_fn = jax.vmap(self.critic, in_axes=(0, None))
        else:
            actor_fn = self.actor
            critic_fn = self.critic

        actor_out = actor_fn(state, params)
        critic_out = critic_fn(state, params)
        return actor_out, critic_out

    def sample_action_path(self, seed, dist, log_prob=False, deterministic=False):
        """Sample an action from the distribution."""
        action = (
            jnp.argmax(dist.probs()).astype(dtype_config.MED_INT_DTYPE)
            if deterministic
            else dist.sample(seed=seed)
        )
        if log_prob:
            return action, dist.log_prob(action)
        return action

    def sample_action_power(self, seed, dist, log_prob=False, deterministic=False):
        """Sample an action and convert to power level"""
        if self.discrete:
            if deterministic:
                raw_action = dist.mode()
            else:
                raw_action = dist.sample(seed=seed)
            processed_action = self.power_levels[raw_action]
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
            return processed_action, dist.log_prob(raw_action)
        return processed_action

    def sample_action_path_power(self, seed, dist, log_prob=False, deterministic=False):
        """Sample an action from the distributions."""
        path_action = self.sample_action_path(
            seed, dist[0], log_prob=log_prob, deterministic=deterministic
        )
        power_action = self.sample_action_power(
            seed, dist[1], log_prob=log_prob, deterministic=deterministic
        )
        if log_prob:
            return path_action[0], power_action[0], path_action[1] + power_action[1]
        return path_action, power_action

    def sample_action(self, seed, dist, log_prob=False, deterministic=False):
        """Sample an action from the distributions."""
        if self.output_path and self.output_power:
            return self.sample_action_path_power(
                seed, dist, log_prob=log_prob, deterministic=deterministic
            )
        elif self.output_path:
            return self.sample_action_path(
                seed, dist, log_prob=log_prob, deterministic=deterministic
            )
        elif self.output_power:
            return self.sample_action_power(
                seed, dist, log_prob=log_prob, deterministic=deterministic
            )
        else:
            raise ValueError("No action type specified for sampling.")
