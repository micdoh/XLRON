import hashlib
import itertools
import json
import math
import pathlib
from collections import defaultdict
from functools import partial
from itertools import combinations, islice
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import box
import chex
import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
from jax._src import core, dtypes, prng
from jax._src.typing import Array, ArrayLike, DTypeLike
from scipy.constants import c, h

from xlron import dtype_config
from xlron.environments.dataclasses import (
    ActionInfo,
    EnvParams,
    EnvState,
    RSAEnvParams,
    RSAEnvState,
    RSAGNModelEnvParams,
    RSAGNModelEnvState,
)
from xlron.environments.diff_utils import (
    differentiable_argmax,
    differentiable_ceil,
    differentiable_compare,
    differentiable_floor,
    differentiable_indexing,
    differentiable_one_hot_index_update,
    differentiable_round,
    differentiable_round_simple,
    differentiable_where,
    straight_through,
)
from xlron.environments.gn_model import isrs_gn_model
from xlron.environments.gn_model.isrs_gn_model import from_db

Shape = Sequence[int]
T = TypeVar("T")  # Declare type variable

one = jnp.array(1.0, dtype=dtype_config.SMALL_INT_DTYPE)
zero = jnp.array(0.0, dtype=dtype_config.SMALL_INT_DTYPE)


@partial(jax.jit, static_argnums=(1,))
def get_spectral_features(laplacian: Array, num_features: int) -> Array:
    """Compute spectral node features from symmetric normalized graph Laplacian.

    Args:
        adj: Adjacency matrix of the graph
        num_features: Number of eigenvector features to extract

    Returns:
        Array of shape (n_nodes, num_features) containing eigenvectors corresponding
        to the smallest non-zero eigenvalues of the graph Laplacian. If the graph has
        fewer nodes than num_features, the result is zero-padded to have num_features columns.

    Notes:
        - Skips trivial eigenvectors (those with near-zero eigenvalues)
        - Eigenvectors are ordered by ascending eigenvalue magnitude
        - Runtime is O(n^3) - use only for small/medium graphs
        - Eigenvector signs are arbitrary (may vary between runs)
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(laplacian)
    n_nodes = laplacian.shape[0]
    # If graph has fewer nodes than requested features, pad with zeros
    if n_nodes < num_features:
        padding = jnp.zeros((n_nodes, num_features - n_nodes), dtype=dtype_config.LARGE_FLOAT_DTYPE)
        return jnp.concatenate([eigenvectors, padding], axis=-1).astype(
            dtype_config.LARGE_FLOAT_DTYPE
        )
    return eigenvectors[:, :num_features].astype(dtype_config.LARGE_FLOAT_DTYPE)


def make_line_graph(graph: nx.Graph) -> nx.Graph:
    """Create the line graph of a NetworkX graph.

    The line graph L(G) has:
    - One node for each edge in the original graph G
    - An edge between two nodes in L(G) if the corresponding edges in G share a node

    This is used for transformer architectures where we treat edges (links) as tokens
    and need positional encodings based on edge relationships.

    Args:
        graph: NetworkX graph (original topology)

    Returns:
        line_graph: NetworkX line graph where nodes correspond to edges in the original
    """
    return nx.line_graph(graph)


def get_line_graph_laplacian(graph: nx.Graph) -> chex.Array:
    """Compute the Laplacian matrix of the line graph.

    Args:
        graph: NetworkX graph (original topology)

    Returns:
        Laplacian matrix of the line graph as a JAX array
    """
    line_graph = make_line_graph(graph)
    if line_graph.is_directed():
        laplacian = nx.directed_laplacian_matrix(line_graph)
    else:
        laplacian = nx.laplacian_matrix(line_graph).todense()
    return jnp.array(laplacian, dtype=dtype_config.LARGE_FLOAT_DTYPE)


def get_line_graph_spectral_features(graph: nx.Graph, num_features: int) -> chex.Array:
    """Compute spectral features for edges using the line graph Laplacian.

    These features are used as positional encodings for transformer architectures
    with WiRE (Wavelet-Induced Rotary Encodings).

    Args:
        graph: NetworkX graph (original topology)
        num_features: Number of spectral features to compute

    Returns:
        Array of shape (num_edges, num_features) containing eigenvectors of the
        line graph Laplacian, ordered by ascending eigenvalue magnitude.
        These serve as positional encodings for edge/link tokens.
    """
    line_laplacian = get_line_graph_laplacian(graph)
    num_edges = line_laplacian.shape[0]
    # Clamp num_features to available dimensions
    actual_features = min(num_features, num_edges)
    eigenvalues, eigenvectors = jnp.linalg.eigh(line_laplacian)
    return eigenvectors[:, :actual_features].astype(dtype_config.LARGE_FLOAT_DTYPE)


@partial(jax.jit, static_argnums=(1, 3))
def init_graph_tuple(
    state: EnvState,
    params: EnvParams,
    adj: Array,
    exclude_source_dest: bool = False,
) -> jraph.GraphsTuple:
    """Initialise graph tuple for use with Jraph GNNs.
    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
        adj (jnp.array): Adjacency matrix of the graph
    Returns:
        jraph.GraphsTuple: Graph tuple
    """
    senders = params.edges.val.T[0].astype(dtype_config.LARGE_INT_DTYPE)
    receivers = params.edges.val.T[1].astype(dtype_config.LARGE_INT_DTYPE)

    # Get source and dest from request array
    # VONE has 2D request_array (2, max_edges*2+1), use first row for node info
    request_array = state.request_array
    if request_array.ndim == 2:
        request_array = request_array[0]
    source_dest, datarate = read_rsa_request(request_array)
    # Global feature is normalised data rate of current request
    globals = jnp.array(
        [datarate / jnp.max(params.values_bw.val)], dtype=dtype_config.LARGE_FLOAT_DTYPE
    )

    if exclude_source_dest:
        source_dest_features = jnp.zeros(
            (params.num_nodes, 2), dtype=dtype_config.LARGE_FLOAT_DTYPE
        )
    else:
        source, dest = source_dest[0], source_dest[2]
        # One-hot encode source and destination (2 additional features)
        source_dest_features = jnp.zeros(
            (params.num_nodes, 2), dtype=dtype_config.LARGE_FLOAT_DTYPE
        )
        source_dest_features = source_dest_features.at[
            source.astype(dtype_config.INDEX_DTYPE), 0
        ].set(1)
        source_dest_features = source_dest_features.at[
            dest.astype(dtype_config.INDEX_DTYPE), 1
        ].set(-1)

    spectral_features = get_spectral_features(adj, num_features=params.num_spectral_features)

    # For dynamic traffic, edge_features are normalised remaining holding time instead of link_slot_array
    holding_time_edge_features = state.link_slot_departure_array / params.mean_service_holding_time

    if params.__class__.__name__ in ["RSAGNModelEnvParams", "RMSAGNModelEnvParams"]:
        # Normalize by max parameters (converted to linear units)
        max_power = isrs_gn_model.from_dbm(params.max_power)
        normalized_power = jnp.round(state.channel_power_array / max_power, 3)
        max_snr = isrs_gn_model.from_db(params.max_snr)
        normalized_snr = jnp.round(state.link_snr_array / max_snr, 3)
        edge_features = jnp.stack([normalized_snr, normalized_power], axis=-1)
        node_features = jnp.concatenate([spectral_features, source_dest_features], axis=-1)
    elif params.__class__.__name__ == "VONEEnvParams":
        edge_features = (
            state.link_slot_array
            if params.mean_service_holding_time > 1e5
            else holding_time_edge_features
        )
        node_features = getattr(
            state,
            "node_capacity_array",
            jnp.zeros(params.num_nodes, dtype=dtype_config.LARGE_FLOAT_DTYPE),
        )
        node_features = node_features.reshape(-1, 1)
        node_features = jnp.concatenate(
            [node_features, spectral_features, source_dest_features], axis=-1
        )
    else:
        edge_features = (
            state.link_slot_array
            if params.mean_service_holding_time > 1e5
            else holding_time_edge_features
        )
        # [n_edges] or [n_edges, ...]
        node_features = jnp.concatenate([spectral_features, source_dest_features], axis=-1)

    if params.disable_node_features:
        node_features = jnp.zeros((1,), dtype=dtype_config.LARGE_FLOAT_DTYPE)

    # Handle undirected graphs (duplicate edges after normalization)
    if not params.directed_graph:
        senders_ = jnp.concatenate([senders, receivers])
        receivers = jnp.concatenate([receivers, senders])
        senders = senders_
        edge_features = jnp.repeat(edge_features, 2, axis=0)

    return jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=jnp.reshape(params.num_nodes, (1,)).astype(dtype_config.LARGE_INT_DTYPE),
        n_edge=jnp.reshape(len(senders), (1,)).astype(dtype_config.LARGE_INT_DTYPE),
        globals=globals,
    )


def update_graph_tuple(state: EnvState, params: EnvParams) -> EnvState:
    """Update graph tuple for use with Jraph GNNs.

    Edge and node features are updated from link_slot_array and node_capacity_array respectively.
    Global features are updated as request_array.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        state (EnvState): Environment state with updated graph tuple
    """
    # Get source and dest from request array
    source_dest, datarate = read_rsa_request(state.request_array)
    source, dest = source_dest[0], source_dest[2]
    # Current request as global feature
    globals = jnp.array(
        [datarate / jnp.max(params.values_bw.val)], dtype=dtype_config.LARGE_FLOAT_DTYPE
    )
    # One-hot encode source and destination
    source_dest_features = jnp.zeros((params.num_nodes, 2), dtype=dtype_config.LARGE_FLOAT_DTYPE)
    # Convert indices to int32 for indexing...
    source_idx = source.astype(dtype_config.LARGE_INT_DTYPE)
    dest_idx = dest.astype(dtype_config.LARGE_INT_DTYPE)
    # ...but maintain grads in value with differentiable index updates
    source_dest_features = differentiable_one_hot_index_update(
        source_dest_features, source_idx, 1.0, params.temperature, params.differentiable
    )
    source_dest_features = differentiable_one_hot_index_update(
        source_dest_features, dest_idx, -1.0, params.temperature, params.differentiable
    )
    spectral_features = state.graph.nodes[..., : params.num_spectral_features]
    holding_time_edge_features = state.link_slot_departure_array / params.mean_service_holding_time

    if params.__class__.__name__ in ["RSAGNModelEnvParams", "RMSAGNModelEnvParams"]:
        # Normalize by max parameters (converted to linear units)
        max_power = isrs_gn_model.from_dbm(params.max_power)
        # Use differentiable rounding
        normalized_power = differentiable_round(
            state.channel_power_array / max_power,
            decimals=3,
            temperature=params.temperature,
            differentiable=params.differentiable,
        )

        max_snr = isrs_gn_model.from_db(params.max_snr)
        # Use differentiable rounding
        normalized_snr = differentiable_round(
            state.link_snr_array / max_snr,
            decimals=3,
            temperature=params.temperature,
            differentiable=params.differentiable,
        )

        edge_features = jnp.stack([normalized_snr, normalized_power], axis=-1)
        node_features = jnp.concatenate([spectral_features, source_dest_features], axis=-1)
    elif params.__class__.__name__ == "VONEEnvParams":
        edge_features = (
            state.link_slot_array
            if params.mean_service_holding_time > 1e5
            else holding_time_edge_features
        )
        node_features = getattr(state, "node_capacity_array", jnp.zeros(params.num_nodes))
        node_features = node_features.reshape(-1, 1)
        node_features = jnp.concatenate(
            [node_features, spectral_features, source_dest_features], axis=-1
        )
    else:
        edge_features = (
            state.link_slot_array
            if params.mean_service_holding_time > 1e5
            else holding_time_edge_features
        )
        node_features = jnp.concatenate([spectral_features, source_dest_features], axis=-1)

    if params.disable_node_features:
        node_features = jnp.zeros((1,), dtype=dtype_config.LARGE_FLOAT_DTYPE)

    edge_features = edge_features if params.directed_graph else jnp.repeat(edge_features, 2, axis=0)
    graph = state.graph._replace(nodes=node_features, edges=edge_features, globals=globals)
    state = state.replace(graph=graph)
    return state


def init_link_length_array(graph: nx.Graph) -> chex.Array:
    """Initialise link length array.
    Args:
        graph (nx.Graph): NetworkX graph
    Returns:

    """
    link_lengths = []
    for edge in sorted(graph.edges):
        link_lengths.append(graph.edges[edge]["distance"])
    return jnp.array(link_lengths, dtype=dtype_config.LARGE_INT_DTYPE)


def init_path_link_array(
    graph: nx.Graph,
    k: int,
    disjoint: bool = False,
    path_sort_criteria: str = "",
    directed: bool = False,
    modulations_array: None | chex.Array = None,
    rwa_lr: bool = False,
    scale_factor: float = 1.0,
    path_snr: bool = False,
) -> chex.Array:
    """Initialise path-link array.
    Each path is defined by a link utilisation array (one row in the path-link array).
    1 indicates link corresponding to index is used, 0 indicates not used.

    Args:
        graph (nx.Graph): NetworkX graph
        k (int): Number of paths
        disjoint (bool, optional): Whether to use edge-disjoint paths. Defaults to False.
        weight (str, optional): Sort paths by edge attribute. Defaults to "".
        directed (bool, optional): Whether graph is directed. Defaults to False.
        modulations_array (chex.Array, optional): Array of maximum spectral efficiency for modulation format on path. Defaults to None.
        rwa_lr (bool, optional): Whether the environment is RWA with lightpath reuse (affects path ordering).
        path_snr (bool, optional): If GN model is used, include extra row of zeroes for unutilised paths
        to ensure correct SNR calculation for empty paths (path index -1).

    Returns:
        chex.Array: Path-link array (N(N-1)*k x E) where N is number of nodes, E is number of edges, k is number of shortest paths
    """
    # Assert that sort_criteria is one of the allowed values
    assert path_sort_criteria in [
        "spectral_resources",
        "hops",
        "distance",
        "hops_distance",
        "capacity",
    ], (
        f"path_sort_criteria must be one of 'spectral_resources', 'hops', 'distance', 'hops_distance', or 'capacity' got '{path_sort_criteria}'"
    )

    # Set weight based on sort_criteria
    weight = (
        ""
        if path_sort_criteria in ["spectral_resources", "hops", "hops_distance", "capacity"]
        else "distance"
    )

    def path_weight(g, path, weight):
        return sum(g[u][v].get(weight, 1) for u, v in zip(path, path[1:]))

    def path_hash(p):
        return int(hashlib.sha256(str(p).encode()).hexdigest(), 16)

    def get_k_shortest_paths(
        g: nx.Graph, source: int, target: int, k: int, weight: str | None
    ) -> List[List[Tuple[int, int]]]:
        paths = list(islice(nx.shortest_simple_paths(g, source, target, weight=weight), k))
        # Ensure deterministic sorting. Sort first by weight (if any), then hops, then hash of path (random)
        paths.sort(
            key=lambda p: (
                path_weight(g, p, weight),
                len(p),
                path_hash(
                    p
                ),  # N.B. that code used for JOCN "Hype or Hope?" did not include this criterion.
            )
        )
        return paths

    def get_k_disjoint_shortest_paths(
        g: nx.Graph, source: int, target: int, k: int, weight: str | None
    ) -> List[List[Tuple[int, int]]]:
        k_paths_disjoint_unsorted = list(nx.edge_disjoint_paths(g, source, target))
        k_paths_shortest = get_k_shortest_paths(g, source, target, k, weight=weight)

        # Keep disjoint paths and add unique shortest paths until k paths reached
        disjoint_ids = [tuple(path) for path in k_paths_disjoint_unsorted]
        k_paths = k_paths_disjoint_unsorted
        for path in k_paths_shortest:
            if tuple(path) not in disjoint_ids:
                k_paths.append(path)
        k_paths = k_paths[:k]
        return k_paths

    paths = []
    edges = sorted(graph.edges)

    # Get the k-shortest paths for each node pair
    k_path_collections = []
    get_paths = get_k_disjoint_shortest_paths if disjoint else get_k_shortest_paths
    for node_pair in combinations(graph.nodes, 2):
        k_paths = get_paths(graph, node_pair[0], node_pair[1], k, weight=weight)
        k_path_collections.append(k_paths)

    if directed:  # Get paths in reverse direction
        for node_pair in combinations(graph.nodes, 2):
            k_paths_rev = get_paths(graph, node_pair[1], node_pair[0], k, weight=weight)
            k_path_collections.append(k_paths_rev)

    # Sort the paths for each node pair
    for k_paths in k_path_collections:
        source, dest = k_paths[0][0], k_paths[0][-1]

        # Get path lengths
        path_distance = [nx.path_weight(graph, path, weight="distance") for path in k_paths]

        # Get path num hops
        path_hops = [len(path) - 1 for path in k_paths]

        # Get spectral efficiency of each path
        if modulations_array is not None:
            path_se = []
            modulations_array = modulations_array[::-1]
            for length in path_distance:
                for modulation in modulations_array:
                    if length <= modulation[0]:
                        path_se.append(modulation[1])
                        break
        else:
            path_se = [1] * len(path_distance)

        if rwa_lr:
            path_capacity = [
                float(calculate_path_capacity(path_length, scale_factor=scale_factor)) + 1e-6
                for path_length in path_distance
            ]
        else:
            path_capacity = [1] * len(path_distance)

        # If less then k unique paths, add dummy paths (just so each node pair still has K rows in the array)
        empty_path = [0] * len(graph.edges)
        num_missing_paths = k - len(k_paths)
        k_paths = k_paths + [empty_path] * num_missing_paths
        path_distance = path_distance + [1e6] * num_missing_paths
        path_hops = path_hops + [1e6] * num_missing_paths
        path_se = path_se + [0] * num_missing_paths
        path_capacity = path_capacity + [0] * num_missing_paths

        # Zip the paths with potential sort criteria
        unsorted_paths = zip(k_paths, path_distance, path_hops, path_se, path_capacity)

        def determine_sort_criteria(x, path_sort_criteria):
            if path_sort_criteria == "spectral_resources":
                # Sort by ratio of hops/se or hops/capacity
                # Use max(..., 1) to avoid division by zero for dummy/padded paths
                return (x[2] / max(x[3], 1)) if not rwa_lr else (x[2] / max(x[4], 1))
            elif path_sort_criteria == "distance":
                return x[1]
            elif path_sort_criteria == "hops":
                return x[2]
            elif path_sort_criteria == "hops_distance":
                return (x[2], x[1])
            elif path_sort_criteria == "capacity":
                return x[4]
            else:
                raise ValueError(f"Path sort criteria: {path_sort_criteria}")

        k_paths_sorted = [
            (source, dest, distance, hops, se, capacity, path)
            for path, distance, hops, se, capacity in sorted(
                unsorted_paths, key=lambda x: determine_sort_criteria(x, path_sort_criteria)
            )
        ]

        # Keep only first k paths
        k_paths_sorted = k_paths_sorted[:k]

        for k_path in k_paths_sorted:
            k_path = k_path[-1]
            link_usage = [0] * len(graph.edges)  # Initialise empty path
            if sum(k_path) == 0:
                link_usage = empty_path
            else:
                for i in range(len(k_path) - 1):
                    s, d = k_path[i], k_path[i + 1]
                    for edge_index, edge in enumerate(edges):
                        condition = (
                            (edge[0] == s and edge[1] == d)
                            if directed
                            else (
                                (edge[0] == s and edge[1] == d) or (edge[0] == d and edge[1] == s)
                            )
                        )
                        if condition:
                            link_usage[edge_index] = 1
            path = link_usage
            paths.append(path)

    # If using GN model, add extra row of zeroes for empty paths for SNR calculation
    if path_snr:
        empty_path = [0] * len(graph.edges)
        paths.append(empty_path)

    return jnp.array(paths, dtype=dtype_config.BINARY_DTYPE)


def get_link_relevance_array(
    paths: Array, paths_se: Array, requested_datarate: Array, params: RSAEnvParams
):
    """Compute 4 link relevance features for the current request.

    Args:
        paths: (k, E) binary path-link indicators
        paths_se: (k, 1) spectral efficiency per path
        requested_datarate: (1,)
        params: environment parameters

    Returns:
        (E, 4) array with columns:
            0: weighted_relevance - combined rank/SE weighted sum across paths
            1: path_count - fraction of k paths using each link
            2: best_rank - 1 - min_rank/k for links on any path, 0 otherwise
            3: best_se - max SE among paths through link, normalized
    """
    k = params.k_paths

    # --- Feature 1: Weighted relevance (existing logic) ---
    ranks = jnp.arange(k)
    rank_weights = 1.0 / (ranks + 1.0)
    num_slots = jax.vmap(
        lambda x: required_slots(
            requested_datarate,
            x,
            params.slot_size,
            guardband=params.guardband,
            temperature=params.temperature,
        )
    )(paths_se.flatten())
    slot_weights = 1.0 / num_slots
    weights = rank_weights * slot_weights.flatten()
    weights = weights / (jnp.sum(weights) + 1e-8)
    weighted_paths = paths * weights[:, None]
    weighted_relevance = jnp.sum(weighted_paths, axis=0)  # (E,)

    # --- Feature 2: Path count - fraction of k paths using each link ---
    path_count = jnp.sum(paths, axis=0) / k  # (E,)

    # --- Feature 3: Best rank - 1 - min_rank/k for links on any path, 0 otherwise ---
    rank_per_path = jnp.arange(k).reshape(k, 1)  # (k, 1)
    # Where path uses link, use rank; otherwise use k (sentinel)
    rank_masked = jnp.where(paths > 0, rank_per_path, k)  # (k, E)
    min_rank = jnp.min(rank_masked, axis=0)  # (E,)
    on_any_path = (path_count > 0).astype(jnp.float32)  # (E,)
    best_rank = on_any_path * (1.0 - min_rank / k)  # (E,)

    # --- Feature 4: Best SE - max SE among paths through link, normalized ---
    se_vals = paths_se.flatten()  # (k,)
    max_se = jnp.max(se_vals) + 1e-8
    se_masked = jnp.where(paths > 0, se_vals[:, None], 0.0)  # (k, E)
    best_se = jnp.max(se_masked, axis=0) / max_se  # (E,)

    return jnp.stack([weighted_relevance, path_count, best_rank, best_se], axis=-1)  # (E, 4)


@partial(jax.jit, static_argnums=(1,))
def get_obs_transformer(state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
    """Retrieves observation for transformer model.

    Creates tokens for each link/edge. Column order:
        [wire_features | edge_features | traffic_marginals | request-specific...]
    where request-specific features are at the end so the critic can strip them.

    Request-specific columns (stripped for critic):
        - holding_time (1 col, departure mode only)
        - request_size (1 col)
        - link_relevance (4 cols)

    Args:
        state: Environment state
        params: Environment parameters

    Returns:
        tokens: Array of shape (num_links, input_size)
    """
    # Get line graph spectral features (WiRE positional encodings)
    wire_features = params.line_graph_spectral_features.val

    # Get edge features based on traffic type (WITHOUT holding time - that's request-specific)
    if params.transformer_obs_type == "occupancy":
        edge_features = state.link_slot_array
    elif params.transformer_obs_type == "capacity":
        edge_features = state.link_capacity_array / 1e6
    else:
        # Dynamic traffic: normalized departure times only
        edge_features = state.link_slot_departure_array / params.mean_service_holding_time

    # --- Traffic matrix node marginal features (NOT request-specific) ---
    send_load = jnp.sum(state.traffic_matrix, axis=1)  # (N,) row marginals
    recv_load = jnp.sum(state.traffic_matrix, axis=0)  # (N,) col marginals
    send_load = send_load / (jnp.sum(send_load) + 1e-8)
    recv_load = recv_load / (jnp.sum(recv_load) + 1e-8)
    link_src = params.edges.val[:, 0].astype(jnp.int32)  # (E,)
    link_dst = params.edges.val[:, 1].astype(jnp.int32)  # (E,)
    endpoint_send = (send_load[link_src] + send_load[link_dst]).reshape(-1, 1)  # (E, 1)
    endpoint_recv = (recv_load[link_src] + recv_load[link_dst]).reshape(-1, 1)  # (E, 1)
    traffic_marginal_features = jnp.concatenate([endpoint_send, endpoint_recv], axis=-1)  # (E, 2)

    # --- Request-specific features (critic should NOT see these) ---
    nodes_sd, requested_datarate = read_rsa_request(state.request_array)

    # Normalized request size
    max_bw = jnp.max(params.values_bw.val)
    request_size_feature = jnp.full(
        (params.num_links, 1),
        requested_datarate / (max_bw + 1e-8),
    )  # (E, 1)

    # Link relevance (4 features)
    paths_se = get_paths_se(params, nodes_sd)
    paths = get_paths(params, nodes_sd)
    link_relevance_features = get_link_relevance_array(
        paths, paths_se, requested_datarate, params
    )  # (E, 4)

    # Concatenation: shared features first, request-specific features last
    # Shared: wire_features, edge_features, traffic_marginals
    # Request-specific: [holding_time (departure only),] request_size, link_relevance
    shared = [wire_features, edge_features, traffic_marginal_features]
    request_specific = [request_size_feature, link_relevance_features]
    if params.transformer_obs_type == "departure":
        holding_time_col = jnp.full(
            (params.num_links, 1),
            state.holding_time / params.mean_service_holding_time,
        )  # (E, 1)
        request_specific = [holding_time_col] + request_specific

    tokens = jnp.concatenate(shared + request_specific, axis=-1)

    return tokens


def init_path_length_array(path_link_array: chex.Array, graph: nx.Graph) -> chex.Array:
    """Initialise path length array.

    Args:
        path_link_array (chex.Array): Path-link array
        graph (nx.Graph): NetworkX graph
    Returns:
        chex.Array: Path length array
    """
    link_length_array = init_link_length_array(graph)
    path_lengths = jnp.dot(path_link_array, link_length_array)
    return path_lengths


def init_modulations_array(modulations_filepath: str | None = None) -> Array:
    """Initialise array of maximum spectral efficiency for modulation format on path.

    Args:
        modulations_filepath (str, optional): Path to CSV file containing modulation formats. Defaults to None.
    Returns:
        jnp.array: Array of maximum spectral efficiency for modulation format on path.
        First two columns are maximum path length and spectral efficiency.
    """
    f = (
        pathlib.Path(modulations_filepath)
        if modulations_filepath
        else (
            pathlib.Path(__file__).parents[1].absolute()
            / "data"
            / "modulations"
            / "modulations.csv"
        )
    )
    modulations = np.genfromtxt(f, delimiter=",")
    # Drop empty first row (headers) and column (name)
    modulations = modulations[1:, 1:]
    return jnp.array(modulations, dtype=dtype_config.LARGE_FLOAT_DTYPE)


def init_path_se_array(path_length_array: Array, modulations_array: Array) -> Array:
    """Initialise array of maximum spectral efficiency for highest-order modulation format on path.

    Args:
        path_length_array (jnp.array): Array of path lengths
        modulations_array (jnp.array): Array of maximum spectral efficiency for modulation format on path

    Returns:
        jnp.array: Array of maximum spectral efficiency for on path
    """
    se_list = []
    # Flip the modulation array so that the shortest path length is first
    modulations_array = modulations_array[::-1]
    for length in path_length_array:
        for modulation in modulations_array:
            if length <= modulation[0]:
                se_list.append(modulation[1])
                break
    return jnp.array(se_list, dtype=dtype_config.SMALL_INT_DTYPE)


def init_list_of_requests(num_requests: int = 1000) -> Array:
    return jnp.zeros([num_requests, 6], dtype=dtype_config.LARGE_INT_DTYPE)


@partial(jax.jit, static_argnums=(1,))
def init_traffic_matrix(key: chex.PRNGKey, params: EnvParams) -> Array:
    """Initialize traffic matrix. Allows for random traffic matrix or uniform traffic matrix.
    Source-dest traffic requests are sampled probabilistically from the resulting traffic matrix.

    Args:
        key (chex.PRNGKey): PRNG key
        params (EnvParams): Environment parameters

    Returns:
        jnp.array: Traffic matrix
    """
    if params.random_traffic:
        traffic_matrix = jax.random.uniform(
            key, shape=(params.num_nodes, params.num_nodes), dtype=dtype_config.SMALL_FLOAT_DTYPE
        )
    else:
        traffic_matrix = jnp.ones(
            (params.num_nodes, params.num_nodes), dtype=dtype_config.SMALL_FLOAT_DTYPE
        )
    diag_elements = jnp.diag_indices_from(traffic_matrix)
    # Set main diagonal to zero so no requests from node to itself
    traffic_matrix = traffic_matrix.at[diag_elements].set(0)
    traffic_matrix = normalise_traffic_matrix(traffic_matrix)
    return traffic_matrix.astype(jnp.float32)


def init_values_nodes(min_value, max_value):
    return jnp.arange(min_value, max_value + 1, dtype=dtype_config.LARGE_INT_DTYPE)


def init_values_slots(min_value, max_value):
    return jnp.arange(min_value, max_value + 1, dtype=dtype_config.LARGE_INT_DTYPE)


# TODO - allow bandwidths to be selected with a specified probability
def init_values_bandwidth(
    min_value: int = 25, max_value: int = 100, step: int = 1, values: int | None = None
) -> chex.Array:
    if values:
        return jnp.array(values, dtype=dtype_config.LARGE_INT_DTYPE)
    else:
        return jnp.arange(min_value, max_value + 1, step, dtype=dtype_config.LARGE_INT_DTYPE)


def get_path_indices(params: EnvParams, s, d, k, N, directed=False):
    # Triangular number formula
    sum_to_s = s * (s - 1) // 2
    sum_to_d = d * (d - 1) // 2

    forward = (N * s + d - sum_to_s - 2 * s - 1) * k
    backward = (N * d + s - sum_to_d - 2 * d - 1) * k

    s_less_d = differentiable_compare(
        s,
        d,
        "<",
        temperature=params.temperature,
        differentiable=params.differentiable,
    )

    # Simple select based on ordering
    base = differentiable_where(
        s_less_d,
        forward,
        backward,
        threshold=0.5,
        temperature=params.temperature,
        differentiable=params.differentiable,
    )

    directed_offset = directed * (1 - s_less_d) * N * (N - 1) * k // 2
    return base + directed_offset


@partial(jax.jit, static_argnums=(0,))
def init_link_slot_array(params: EnvParams):
    """Initialize empty (all zeroes) link-slot array. 0 means slot is free, -1 means occupied.
    Args:
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: Link slot array (E x S) where E is number of edges and S is number of slots"""
    return jnp.zeros(
        (params.num_links, params.link_resources), dtype=dtype_config.LARGE_FLOAT_DTYPE
    )


def init_rsa_request_array():
    """Initialize request array"""
    return jnp.zeros(3, dtype=dtype_config.LARGE_INT_DTYPE)


@partial(jax.jit, static_argnums=(0, 1, 2))
def init_link_slot_mask(params: EnvParams, include_no_op: bool = False, agg: float = 1.0):
    """Initialize link mask"""
    return jnp.ones(
        params.k_paths * math.ceil(params.link_resources / agg) + (1 * include_no_op),
        dtype=dtype_config.LARGE_FLOAT_DTYPE,
    )


@partial(jax.jit, static_argnums=(0,))
def init_mod_format_mask(params: EnvParams):
    """Initialize link mask"""
    return jnp.full(
        (params.k_paths * params.link_resources,), -1.0, dtype=dtype_config.LARGE_FLOAT_DTYPE
    )


def decrease_last_element(array):
    last_value_mask = (
        jnp.arange(array.shape[0], dtype=dtype_config.SMALL_FLOAT_DTYPE) == array.shape[0] - 1
    )
    return jnp.where(last_value_mask, array - 1, array)


@partial(jax.jit, static_argnums=(0,))
def init_link_slot_departure_array(params: EnvParams):
    return jnp.zeros(
        (params.num_links, params.link_resources), dtype=dtype_config.SMALL_FLOAT_DTYPE
    )


def normalise_traffic_matrix(traffic_matrix):
    """Normalise traffic matrix to sum to 1"""
    traffic_matrix /= jnp.sum(traffic_matrix, promote_integers=False)
    return traffic_matrix


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def required_slots(
    bitrate: float,
    se: int,
    channel_width: float,
    guardband: int = 1,
    temperature: float = 1.0,
    differentiable: bool = True,
) -> int:
    """Calculate required slots for a given bitrate and spectral efficiency.

    Args:
        bit_rate (float): Bit rate in Gbps
        se (float): Spectral efficiency in bps/Hz
        channel_width (float): Channel width in GHz
        guardband (int, optional): Guard band. Defaults to 1.
        temperature (float, optional): Temperature for differentiable approximation. Defaults to 1.0.
        differentiable (bool, optional): If False, use non-differentiable operations. Defaults to True.

    Returns:
        int: Required slots
    """
    # Apply differentiable ceiling
    base_calculation = bitrate / (se * channel_width) + guardband
    slots = differentiable_ceil(
        base_calculation, temperature=temperature, differentiable=differentiable
    )
    # Differentiable version of equality comparison bitrate == 0
    is_zero = differentiable_compare(
        bitrate, zero, "==", temperature=temperature, differentiable=differentiable
    )  # High temperature for sharper transition
    # Differentiable version of the conditional zeroing (if bitrate is zero, then required slots should be zero)
    result = slots * (one - is_zero)
    return jnp.squeeze(result).astype(dtype_config.SMALL_INT_DTYPE)


def generate_source_dest_pairs(num_nodes, directed_graph):
    indices = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    # Append reverse if directed graph
    indices = indices + [(j, i) for i, j in indices] if directed_graph else indices
    return jnp.array(indices, dtype=dtype_config.LARGE_INT_DTYPE)


@partial(jax.jit, static_argnums=(2,))
def generate_request_rsa(key: chex.PRNGKey, state: EnvState, params: EnvParams) -> EnvState:
    key_sd, key_slot, key_times = jax.random.split(key, 3)

    if params.deterministic_requests:
        request = differentiable_indexing(
            state.list_of_requests,
            state.total_requests,
            params.temperature,
            params.differentiable,
        )
        source = differentiable_indexing(
            request, 0, params.temperature, params.differentiable
        ).astype(jnp.float32)
        bw = differentiable_indexing(request, 1, params.temperature, params.differentiable).astype(
            jnp.float32
        )
        dest = differentiable_indexing(
            request, 2, params.temperature, params.differentiable
        ).astype(jnp.float32)
        arrival_time = differentiable_indexing(
            request, 3, params.temperature, params.differentiable
        )
        holding_time = differentiable_indexing(
            request, 4, params.temperature, params.differentiable
        )
        current_time = differentiable_indexing(
            request, 5, params.temperature, params.differentiable
        )
    else:
        if params.traffic_array:
            source_dest_index = jax.random.choice(key_sd, state.traffic_matrix.shape[0])
            nodes = differentiable_indexing(
                state.traffic_matrix,
                source_dest_index,
                params.temperature,
                params.differentiable,
            )
        else:
            shape = state.traffic_matrix.shape
            probabilities = state.traffic_matrix.ravel()
            source_dest_index = jax.random.choice(key_sd, probabilities.size, p=probabilities)
            # Faster than unravel_index for 2D
            source = source_dest_index // shape[1]
            dest = source_dest_index % shape[1]
            nodes = jnp.stack((source, dest), dtype=dtype_config.LARGE_INT_DTYPE)

        bw = jax.random.choice(key_slot, params.values_bw.val)
        source, dest = (
            nodes
            if params.directed_graph
            else (jnp.minimum(nodes[0], nodes[1]), jnp.maximum(nodes[0], nodes[1]))
        )

        arrival_time, holding_time = generate_arrival_holding_times(key_times, params)
        current_time = (
            state.current_time + arrival_time
            if not params.relative_arrival_times
            else jnp.array([0.0], dtype=dtype_config.SMALL_FLOAT_DTYPE)
        )

    state = state.replace(
        holding_time=holding_time,
        current_time=current_time,
        arrival_time=arrival_time,
        request_array=jnp.stack((source, bw, dest)),
        total_requests=state.total_requests + 1,
    )

    remove_expired_services = remove_expired_services_rsa
    if params.__class__.__name__ == "RWALightpathReuseEnvParams":
        state = state.replace(
            time_since_last_departure=state.time_since_last_departure + arrival_time
        )
        remove_expired_services = remove_expired_services_rwalr
    elif params.__class__.__name__ == "RMSAGNModelEnvParams":
        remove_expired_services = remove_expired_services_rmsa_gn_model
    elif params.__class__.__name__ == "RSAGNModelEnvParams":
        remove_expired_services = remove_expired_services_rsa_gn_model
    state = remove_expired_services(state, params) if not params.incremental_loading else state
    return state


@partial(jax.jit, static_argnums=(2,))
def generate_request_rwalr(key: chex.PRNGKey, state: EnvState, params: EnvParams) -> EnvState:
    # Flatten the probabilities to a 1D array
    key_sd, key_slot, key_times = jax.random.split(key, 3)
    if params.deterministic_requests:
        request = differentiable_indexing(
            state.list_of_requests,
            state.total_requests.astype(jnp.float32),
            params.temperature,
            params.differentiable,
        )
        source = jax.lax.dynamic_slice(request, (0,), (1,))[0]
        bw = jax.lax.dynamic_slice(request, (1,), (1,))[0]
        dest = jax.lax.dynamic_slice(request, (2,), (1,))[0]
        arrival_time = jax.lax.dynamic_slice(request, (3,), (1,))[0]
        holding_time = jax.lax.dynamic_slice(request, (4,), (1,))[0]
        current_time = jax.lax.dynamic_slice(request, (5,), (1,))[0]
    else:
        shape = state.traffic_matrix.shape
        probabilities = state.traffic_matrix.ravel()
        # Use jax.random.choice to select index based on the probabilities
        source_dest_index = jax.random.choice(
            key_sd, jnp.arange(state.traffic_matrix.size), p=probabilities
        )
        # Convert 1D index back to 2D
        nodes = jnp.unravel_index(source_dest_index, shape)
        # Vectorized conditional replacement using mask
        bw = jax.random.choice(key_slot, params.values_bw.val)
        nodes = jnp.stack(nodes, dtype=dtype_config.LARGE_INT_DTYPE)
        source, dest = nodes if params.directed_graph else jnp.sort(nodes)
        arrival_time, holding_time = generate_arrival_holding_times(key_times, params)
        current_time = (
            state.current_time + arrival_time
            if not params.relative_arrival_times
            else jnp.array([0.0], dtype=dtype_config.SMALL_FLOAT_DTYPE)
        )
    state = state.replace(
        holding_time=holding_time,
        current_time=current_time,
        arrival_time=arrival_time,
        request_array=jnp.stack((source, bw, dest)),
        total_requests=state.total_requests + 1,
        time_since_last_departure=state.time_since_last_departure + arrival_time,
    )
    # Removal of expired services is different for RWA-LR
    remove_expired_services = remove_expired_services_rsa
    if params.__class__.__name__ == "RWALightpathReuseEnvParams":
        state = state.replace(
            time_since_last_departure=state.time_since_last_departure + arrival_time
        )
        remove_expired_services = remove_expired_services_rwalr
    state = remove_expired_services(state, params) if not params.incremental_loading else state
    return state


@partial(jax.jit, static_argnums=(0,))
def get_path_index_array(params: EnvParams, nodes: Array) -> Array:
    """Indices of paths between source and destination from path array"""
    # get source and destination nodes in order (for accurate indexing of path-link array)
    source, dest = nodes.astype(dtype_config.LARGE_INT_DTYPE)
    i = get_path_indices(
        params, source, dest, params.k_paths, params.num_nodes, directed=params.directed_graph
    )
    index_array = differentiable_indexing(
        jnp.arange(0, params.path_link_array.shape[0], dtype=dtype_config.LARGE_INT_DTYPE),
        i + jnp.arange(params.k_paths, dtype=dtype_config.LARGE_FLOAT_DTYPE),
        params.temperature,
        params.differentiable,
    )
    return index_array


@partial(jax.jit, static_argnums=(0,))
def get_path_index(params: EnvParams, nodes: Array, k_path_index: int) -> Array:
    """Get k paths between source and destination"""
    source, dest = nodes
    starting_index = get_path_indices(
        params,
        source,
        dest,
        params.k_paths,
        params.num_nodes,
        directed=params.directed_graph,
    ).astype(jnp.int32)
    path_index = starting_index + k_path_index
    return path_index


@partial(jax.jit, static_argnums=(0,))
def get_path(params: EnvParams, nodes: Array, k_path_index: int) -> Array:
    """Get k paths between source and destination"""
    path_index = get_path_index(params, nodes, k_path_index)
    path = differentiable_indexing(
        params.path_link_array.val,
        path_index,
        params.temperature,
        params.differentiable,
    )
    if params.pack_path_bits:
        path = jnp.unpackbits(path)[: params.num_links]
    return path


@partial(jax.jit, static_argnums=(0,))
def get_paths(params: EnvParams, nodes: Array) -> Array:
    """Get k paths between source and destination"""
    index_array = get_path_index_array(params, nodes)
    paths = differentiable_indexing(
        params.path_link_array.val,
        index_array,
        params.temperature,
        params.differentiable,
    )
    if params.pack_path_bits:
        paths = jnp.unpackbits(paths, axis=1)[:, : params.num_links]
    return paths


@partial(jax.jit, static_argnums=(0,))
def get_path_se(params, nodes, k_path_index):
    """Get k paths between source and destination"""
    path_index = get_path_index(params, nodes, k_path_index)
    return differentiable_indexing(
        params.path_se_array.val, path_index, params.temperature, params.differentiable
    )


@partial(jax.jit, static_argnums=(0,))
def get_paths_se(params, nodes):
    """Get max. spectral efficiency of modulation format on k paths between source and destination"""
    # get source and destination nodes in order (for accurate indexing of path-link array)
    index_array = get_path_index_array(params, nodes)
    return differentiable_indexing(
        params.path_se_array.val, index_array, params.temperature, params.differentiable
    )


def get_path_and_se(params: EnvParams, nodes: Array, k_path_index: int) -> Tuple[Array, Array]:
    """Get k paths and their spectral efficiencies between source and destination"""
    path_index = get_path_index(params, nodes, k_path_index)
    path = differentiable_indexing(
        params.path_link_array.val,
        path_index,
        params.temperature,
        params.differentiable,
    )
    if params.pack_path_bits:
        path = jnp.unpackbits(path)[: params.num_links]
    se = differentiable_indexing(
        params.path_se_array.val, path_index, params.temperature, params.differentiable
    )
    return path, se


@partial(jax.jit, static_argnums=(1, 2, 3))
def poisson(
    key: Union[Array, prng.PRNGKeyArray],
    lam: ArrayLike,
    shape: Shape = (),
    dtype: DTypeLike = dtypes.float_,
) -> Array:
    r"""Sample Exponential random values with given shape and float dtype.

    The values are distributed according to the probability density function:

    .. math::
     f(x) = \lambda e^{-\lambda x}

    on the domain :math:`0 \le x < \infty`.

    Args:
    key: a PRNG key used as the random key.
    lam: a positive float32 or float64 `Tensor` indicating the rate parameter
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

    Returns:
    A random array with the specified shape and dtype.
    """
    key, _ = jax._src.random._check_prng_key(key)
    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(f"dtype argument to `exponential` must be a float dtype, got {dtype}")
    dtype = dtypes.canonicalize_dtype(dtype)
    shape = core.canonicalize_shape(shape)
    return _poisson(key, lam, shape, dtype)


@partial(jax.jit, static_argnums=(1, 2, 3))
def _poisson(key, lam, shape, dtype) -> Array:
    jax._src.random._check_shape("exponential", shape)
    u = jax.random.uniform(key, shape, dtype)
    # taking 1 - u to move the domain of log to (0, 1] instead of [0, 1)
    return jax.lax.div(jax.lax.neg(jax.lax.log1p(jax.lax.neg(u))), lam)


# TODO - consider just making a differentiable version of this whole function
@partial(jax.jit, static_argnums=(1,))
def generate_arrival_holding_times(key, params):
    """
    Generate arrival and holding times based on Poisson distributed events.
    To understand how sampling from e^-x can be transformed to sample from lambda*e^-(x/lambda) see:
    https://en.wikipedia.org/wiki/Inverse_transform_sampling#Examples
    Basically, inverse transform sampling is used to sample from a distribution with CDF F(x).
    The CDF of the exponential distribution (lambda*e^-{lambda*x}) is F(x) = 1 - e^-{lambda*x}.
    Therefore, the inverse CDF is x = -ln(1-u)/lambda, where u is sample from uniform distribution.
    Therefore, we need to divide jax.random.exponential() by lambda in order to scale the standard exponential CDF.
    Experimental histograms of this method compared to random.expovariate() in Python's random library show that
    the two methods are equivalent.
    Also see: https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
    https://jax.readthedocs.io/en/latest/_autosummary/jax.random.exponential.html

    Args:
        key: PRNG key
        params: Environment parameters

    Returns:
        arrival_time: Arrival time
        holding_time: Holding time
    """
    key_arrival, key_holding = jax.random.split(key, 2)
    arrival_time = (
        jax.random.exponential(key_arrival, shape=(1,), dtype=dtype_config.SMALL_FLOAT_DTYPE)
        / params.arrival_rate
    )  # Divide because it is rate (lambda)
    if params.truncate_holding_time:
        # For DeepRMSA, need to generate holding times that are less than 2*mean_service_holding_time
        key_holding = jax.random.split(key, 5)
        holding_times = jax.vmap(
            lambda x: jax.random.exponential(x, shape=(1,)) * params.mean_service_holding_time
        )(key_holding).reshape(-1)
        holding_times = jnp.where(
            holding_times < 2 * params.mean_service_holding_time, holding_times, zero
        )
        # Get first non-zero value in holding_times
        holding_time_indices = differentiable_where(
            holding_times > 0,
            jnp.arange(holding_times.shape[0]),
            0,
            threshold=0.5,
            temperature=params.temperature,
            differentiable=params.differentiable,
        )
        non_zero_index = differentiable_argmax(
            holding_time_indices,
            temperature=params.temperature,
            differentiable=params.differentiable,
        )
        holding_time = differentiable_indexing(
            jnp.squeeze(holding_times),
            (non_zero_index,),
            params.temperature,
            params.differentiable,
        )
    else:
        holding_time = (
            jax.random.exponential(key_holding, shape=(1,), dtype=dtype_config.SMALL_FLOAT_DTYPE)
            * params.mean_service_holding_time
        )  # Multiply because it is mean (1/lambda)
    return arrival_time, holding_time


@partial(jax.jit, donate_argnums=(0,))
def update_path_links(
    link_slot_array: chex.Array,
    action_info: ActionInfo,
    value: int,
) -> chex.Array:
    return link_slot_array + action_info.affected_slots_mask * value


@partial(jax.jit, donate_argnums=(0,))
def set_active_path(link, initial_slot, num_slots, value):
    slot_indices = jnp.arange(link.shape[0], dtype=dtype_config.LARGE_INT_DTYPE)[:, None]
    mask = ((initial_slot <= slot_indices) & (slot_indices < initial_slot + num_slots)).astype(
        dtype_config.LARGE_FLOAT_DTYPE
    )
    return link * (1 - mask) + mask * value


@partial(jax.jit, donate_argnums=(0,))
def set_path_links(
    link_slot_array: Array,
    affected_slots_mask: Array,
    value: int,
) -> Array:
    # Single vectorized update
    return link_slot_array * (1 - affected_slots_mask) + affected_slots_mask * value


@partial(jax.jit, static_argnums=(1,))
def remove_expired_services_rsa(state: EnvState, params: Optional[EnvParams]) -> EnvState:
    """Check for values in link_slot_departure_array that are less than the current time.
    If found, set to zero in link_slot_array and link_slot_departure_array.

    Args:
        state: Environment state
        params: Environment parameters

    Returns:
        Updated environment state
    """
    # Set one where link_slot_departure_array is >= zero and <= current time
    current_time = state.current_time if not params.relative_arrival_times else state.arrival_time
    mask_remove = jnp.where(
        state.link_slot_departure_array <= jnp.squeeze(current_time),
        one,
        zero,
    )
    updated_link_slot_array = jnp.where(mask_remove == one, zero, state.link_slot_array)
    updated_link_slot_departure_array = jnp.where(
        mask_remove == one, zero, state.link_slot_departure_array
    )
    if params.relative_arrival_times:
        mask_subtract = jnp.where(updated_link_slot_departure_array > zero, one, zero)
        updated_link_slot_departure_array = jnp.where(
            mask_subtract == one,
            state.link_slot_departure_array - jnp.squeeze(current_time),
            updated_link_slot_departure_array,
        )
    state = state.replace(
        link_slot_array=updated_link_slot_array,
        link_slot_departure_array=updated_link_slot_departure_array,
    )
    return state


@partial(jax.jit, static_argnums=(1,))
def remove_expired_services_rwalr(state: EnvState, params: Optional[EnvParams]) -> EnvState:
    """

    Args:
        state: Environment state
        params: Environment parameters

    Returns:
        Updated environment state
    """
    # Set one where link_slot_departure_array is >= zero and <= current time
    current_time = state.current_time if not params.relative_arrival_times else state.arrival_time
    mask_remove = jnp.where(
        (zero <= state.link_slot_departure_array)
        & (state.link_slot_departure_array <= jnp.squeeze(current_time)),
        one,
        zero,
    )
    updated_link_slot_departure_array = jnp.where(
        mask_remove == one, zero, state.link_slot_departure_array
    )
    if params.relative_arrival_times:
        mask_subtract = jnp.where(updated_link_slot_departure_array <= zero, zero, one)
        updated_link_slot_departure_array = jnp.where(
            mask_subtract == one,
            state.link_slot_departure_array - jnp.squeeze(current_time),
            updated_link_slot_departure_array,
        )
    state = state.replace(
        link_slot_array=jnp.where(mask_remove == one, zero, state.link_slot_array),
        path_index_array=jnp.where(mask_remove == one, -one, state.path_index_array),
        link_slot_departure_array=updated_link_slot_departure_array,
    )
    return state


@partial(jax.jit, static_argnums=(1,))
def remove_expired_services_rsa_gn_model(state: EnvState, params: Optional[EnvParams]) -> EnvState:
    """

    Args:
        state: Environment state
        params: Environment parameters

    Returns:
        Updated environment state
    """
    # Set one where link_slot_departure_array is >= zero and <= current time
    current_time = state.current_time if not params.relative_arrival_times else state.arrival_time
    mask_remove = jnp.where(
        (zero <= state.link_slot_departure_array)
        & (state.link_slot_departure_array <= jnp.squeeze(current_time)),
        one,
        zero,
    )
    updated_link_slot_departure_array = jnp.where(
        mask_remove == one, zero, state.link_slot_departure_array
    )
    if params.relative_arrival_times:
        mask_subtract = jnp.where(updated_link_slot_departure_array <= zero, zero, one)
        updated_link_slot_departure_array = jnp.where(
            mask_subtract == one,
            state.link_slot_departure_array - jnp.squeeze(current_time),
            updated_link_slot_departure_array,
        )
    state = state.replace(
        link_slot_array=jnp.where(mask_remove == one, zero, state.link_slot_array),
        link_slot_departure_array=updated_link_slot_departure_array,
        link_snr_array=jnp.where(mask_remove == one, zero, state.link_snr_array),
        path_index_array=jnp.where(mask_remove == one, -one, state.path_index_array),
        channel_centre_bw_array=jnp.where(mask_remove == one, zero, state.channel_centre_bw_array),
        channel_power_array=jnp.where(mask_remove == one, zero, state.channel_power_array),
        path_index_array_prev=jnp.where(mask_remove == one, -one, state.path_index_array_prev),
        channel_centre_bw_array_prev=jnp.where(
            mask_remove == one, zero, state.channel_centre_bw_array_prev
        ),
        channel_power_array_prev=jnp.where(
            mask_remove == one, zero, state.channel_power_array_prev
        ),
    )
    if params.monitor_active_lightpaths:
        # The active_lightpaths_array is set to -1 when the lightpath is not active
        # The active_lightpaths_array_departure is set to 0 when the lightpath is not active
        # (active_lightpaths_array is used to calculate the total throughput)
        mask_remove = jnp.where(
            (zero <= state.active_lightpaths_array_departure)
            & (state.active_lightpaths_array_departure <= jnp.squeeze(current_time)),
            one,
            zero,
        )
        state = state.replace(
            active_lightpaths_array=jnp.where(
                mask_remove == one, -one, state.active_lightpaths_array
            ),
            active_lightpaths_array_departure=jnp.where(
                mask_remove == one, zero, state.active_lightpaths_array_departure
            ),
        )
    return state


@partial(jax.jit, static_argnums=(1,))
def remove_expired_services_rmsa_gn_model(state: EnvState, params: Optional[EnvParams]) -> EnvState:
    """

    Args:
        state: Environment state
        params: Environment parameters

    Returns:
        Updated environment state
    """
    # Set one where link_slot_departure_array is >= zero and <= current time
    current_time = state.current_time if not params.relative_arrival_times else state.arrival_time
    mask_remove = jnp.where(
        (zero <= state.link_slot_departure_array)
        & (state.link_slot_departure_array <= jnp.squeeze(current_time)),
        one,
        zero,
    )
    updated_link_slot_departure_array = jnp.where(
        mask_remove == one, zero, state.link_slot_departure_array
    )
    if params.relative_arrival_times:
        mask_subtract = jnp.where(updated_link_slot_departure_array <= zero, zero, one)
        updated_link_slot_departure_array = jnp.where(
            mask_subtract == one,
            state.link_slot_departure_array - jnp.squeeze(current_time),
            updated_link_slot_departure_array,
        )
    state = state.replace(
        link_slot_array=jnp.where(mask_remove == one, zero, state.link_slot_array),
        link_slot_departure_array=updated_link_slot_departure_array,
        link_snr_array=jnp.where(mask_remove == one, zero, state.link_snr_array),
        path_index_array=jnp.where(mask_remove == one, -one, state.path_index_array),
        channel_centre_bw_array=jnp.where(mask_remove == one, zero, state.channel_centre_bw_array),
        channel_power_array=jnp.where(mask_remove == one, zero, state.channel_power_array),
        modulation_format_index_array=jnp.where(
            mask_remove == one, -one, state.modulation_format_index_array
        ),
        path_index_array_prev=jnp.where(mask_remove == one, -one, state.path_index_array_prev),
        channel_centre_bw_array_prev=jnp.where(
            mask_remove == one, zero, state.channel_centre_bw_array_prev
        ),
        channel_power_array_prev=jnp.where(
            mask_remove == one, zero, state.channel_power_array_prev
        ),
        modulation_format_index_array_prev=jnp.where(
            mask_remove == one, -one, state.modulation_format_index_array_prev
        ),
    )
    return state


def complete_step_rsa(
    state: EnvState, action_info: ActionInfo, check: Array, params: EnvParams
) -> EnvState:
    """If the request is unsuccessful i.e. checks fail, then remove the partial (unfinalised) resource allocation.
    Partial resource allocation is indicated by negative time in link slot departure array.
    Check for values in link_slot_departure_array that are less than zero.
    If found, increase link_slot_array by +1 and link_slot_departure_array by current_time + holding_time of current request.

    Args:
        state: Environment state

    Returns:
        Updated environment state
    """
    fail = check
    success = 1 - check
    state = state.replace(
        link_slot_array=state.link_slot_array - (fail * action_info.affected_slots_mask),
        link_slot_departure_array=state.link_slot_departure_array
        - (fail * action_info.affected_slots_mask * (state.current_time + state.holding_time)),
        accepted_services=state.accepted_services + success,
        accepted_bitrate=state.accepted_bitrate + (success * action_info.requested_datarate),
        total_bitrate=state.total_bitrate + action_info.requested_datarate,
        total_timesteps=state.total_timesteps + 1,
    )
    return state


def complete_step_rwalr(
    state: EnvState, action_info: ActionInfo, check: Array, params: EnvParams
) -> EnvState:
    """Complete step for RWA-LR environments.
    Unlike complete_step_rsa, this does not modify link_slot_array on failure,
    because implement_action_rwalr already handles the undo via blending and
    link_slot_array stores a capacity mask (not an occupancy counter).
    """
    fail = check
    success = 1 - check
    state = state.replace(
        link_slot_departure_array=state.link_slot_departure_array
        - (fail * action_info.affected_slots_mask * (state.current_time + state.holding_time)),
        accepted_services=state.accepted_services + success,
        accepted_bitrate=state.accepted_bitrate + (success * action_info.requested_datarate),
        total_bitrate=state.total_bitrate + action_info.requested_datarate,
        total_timesteps=state.total_timesteps + 1,
    )
    return state


@partial(jax.jit, donate_argnums=(0,))
def undo_action_rsa(state: EnvState, action_info: ActionInfo, params: EnvParams) -> EnvState:
    """If the request is unsuccessful i.e. checks fail, then remove the partial (unfinalised) resource allocation.
    Partial resource allocation is indicated by negative time in link slot departure array.
    Check for values in link_slot_departure_array that are less than zero.
    If found, increase link_slot_array by +1 and link_slot_departure_array by current_time + holding_time of current request.

    Args:
        state: Environment state

    Returns:
        Updated environment state
    """
    state = state.replace(
        link_slot_array=state.link_slot_array - action_info.affected_slots_mask,
        link_slot_departure_array=state.link_slot_departure_array
        - (action_info.affected_slots_mask * (state.current_time + state.holding_time)),
        total_bitrate=state.total_bitrate + action_info.requested_datarate,
    )
    return state


@partial(jax.jit, donate_argnums=(0,))
def undo_action_rwalr(state: EnvState, action_info: ActionInfo, params: EnvParams) -> EnvState:
    """If the request is unsuccessful i.e. checks fail, then remove the partial (unfinalised) resource allocation.
    Partial resource allocation is indicated by negative time in link slot departure array.
    Check for values in link_slot_departure_array that are less than zero.
    If found, increase link_slot_array by +1 and link_slot_departure_array by current_time + holding_time of current request.

    Args:
        state: Environment state

    Returns:
        Updated environment state
    """
    state = undo_action_rsa(state, action_info, params)
    return state


def check_no_spectrum_reuse(state: EnvState, action_info: ActionInfo, params: EnvParams) -> bool:
    """slot-=1 when used, should be zero when unoccupied, so check if any < -1 in slot array.

    Args:
        link_slot_array: Link slot array (L x S) where L is number of links and S is number of slots

    Returns:
        bool: True if check failed, False if check passed
    """
    path_mask = action_info.path[:, None]  # (num_links, 1)
    slots = path_mask * state.link_slot_array  # (num_links, link_resources)
    check = differentiable_compare(
        jnp.max(jnp.max(slots, axis=0)), 1, ">", params.temperature, params.differentiable
    )
    return check


def differentiable_check_no_spectrum_reuse(
    state: EnvState, action_info: ActionInfo, params: EnvParams
):
    """
    Differentiable version of check_no_spectrum_reuse with improved gradient properties.

    Args:
        link_slot_array: Link slot array (L x S) where L is number of links and S is number of slots
        temperature: Controls the sharpness of the gradient response
        differentiable: If False, return hard result directly without gradient approximation

    Returns:
        A value that behaves like the original boolean check in forward pass
        but has zero gradient when there are no violations and otherwise
        has gradient pointing toward reducing violations
    """
    # Hard result for forward pass (original behavior)
    hard_result = check_no_spectrum_reuse(state, action_info, params)

    # If not differentiable mode, return hard result directly
    if not params.differentiable:
        return hard_result

    # Measure violations (how much each element exceeds the threshold of -1)
    violations = jnp.maximum(0, -1 - state.link_slot_array)

    # Any violation is considered a violation (alternatively can sum to discourage more egregious violations)
    # TODO - see if sum vs. max makes a difference in solution quality
    total_violation = jnp.max(violations)

    # Scale violations by temperature
    scaled_violation = params.temperature * total_violation

    # Use a function with zero gradient at zero: x²/(1+x²)
    # This function:
    # - Equals 0 when there are no violations
    # - Has gradient 0 when there are no violations
    # - Grows monotonically toward 1 as violations increase
    soft_result = (scaled_violation**2) / (1 + scaled_violation**2)

    # Apply straight-through trick
    return straight_through(hard_result, soft_result)


@partial(jax.jit, static_argnums=(1,))
def process_path_action(
    state: EnvState, params: EnvParams, path_action: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """Process path action to get path index and initial slot index.
    Args:
        state (State): current state
        params (Params): environment parameters
        path_action (int): path action
    Returns:
        int: path index
        int: initial slot index
    """
    num_slot_actions = params.link_resources // params.aggregate_slots
    path_action = differentiable_round_simple(
        path_action, params.temperature, params.differentiable
    )
    path_index = differentiable_floor(
        path_action // num_slot_actions, params.temperature, params.differentiable
    ).astype(dtype_config.LARGE_INT_DTYPE)
    initial_aggregated_slot_index = jnp.mod(path_action, num_slot_actions)
    initial_slot_index = initial_aggregated_slot_index * params.aggregate_slots

    if params.aggregate_slots > 1:
        # Compute flat index into 1D array of shape (k_paths * link_resources,)
        full_mask = state.full_link_slot_mask.reshape(
            (params.k_paths, num_slot_actions, params.aggregate_slots)
        )
        window = jax.lax.dynamic_slice(
            full_mask,
            (path_index, initial_aggregated_slot_index, 0),
            (1, 1, params.aggregate_slots),
        )
        # Use argmax to get index of first 1 in slice of mask
        initial_slot_index = initial_slot_index + differentiable_argmax(
            window, temperature=params.temperature, differentiable=params.differentiable
        ).astype(dtype_config.LARGE_INT_DTYPE)
    return path_index, initial_slot_index


def get_affected_slots_mask(
    initial_slot_index: Array,
    num_slots: Array,
    path: Array,
    params: EnvParams,
) -> Array:
    # Slot mask: shape (num_slots,)
    slot_indices = jnp.arange(params.link_resources)
    slot_mask = differentiable_compare(
        initial_slot_index, slot_indices, "<=", params.temperature, params.differentiable
    ) * differentiable_compare(
        slot_indices, initial_slot_index + num_slots, "<", params.temperature, params.differentiable
    )
    # Combined mask: (num_links, 1) * (1, num_slots) -> (num_links, num_slots)
    combined_mask = path[:, None] * slot_mask[None, :]
    return combined_mask


@partial(jax.jit, static_argnums=(2,), donate_argnums=(0,))
def implement_path_action(
    state: EnvState,
    action_info: ActionInfo,
    params: EnvParams,
) -> EnvState:
    mask = action_info.affected_slots_mask
    departure_delta = state.current_time + state.holding_time
    new_link_slot = state.link_slot_array + mask
    new_departure = state.link_slot_departure_array + mask * departure_delta
    return state.replace(
        link_slot_array=new_link_slot,
        link_slot_departure_array=new_departure,
    )


@partial(jax.jit, static_argnums=(2,), donate_argnums=(0,))
def implement_action_rsa(
    state: RSAEnvState,
    action_info: ActionInfo,
    params: RSAEnvParams,
) -> EnvState:
    """Implement action to assign slots on links.

    Args:
        state: current state
        action: action to implement
        params: environment parameters

    Returns:
        state: updated state
    """
    if params.__class__.__name__ == "RWALightpathReuseEnvParams":
        state = state.replace(
            link_capacity_array=update_path_links(
                state.link_capacity_array,
                action_info,
                action_info.requested_datarate,
            )
        )
        # TODO (Dynamic-RWALR) - to support diverse requested_datarates for RWA-LR, need to update masking
        # TODO (Dynamic-RWALR) - In order to enable dynamic RWA with lightpath reuse (as opposed to just incremental loading),
        #  need to keep track of active requests OR just randomly remove connections
        #  (could do this by using the link_slot_departure array in a novel way... i.e. don't fill it with departure time but current bw)
        capacity_mask = jnp.where(state.link_capacity_array <= 0.0, 1.0, 0.0)
        over_capacity_mask = jnp.where(state.link_capacity_array < 0.0, 1.0, 0.0)
        total_mask = capacity_mask + over_capacity_mask
        state = state.replace(
            link_slot_array=total_mask,
            link_slot_departure_array=update_path_links(
                state.link_slot_departure_array,
                action_info,
                state.current_time + state.holding_time,
            ),
        )
    else:
        state = implement_path_action(state, action_info, params)
    return state


def read_rsa_request(request_array: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Read RSA request from request array. Return source-destination nodes and bandwidth request.
    Args:
        request_array: request array
    Returns:
        Tuple[chex.Array, chex.Array]: source-destination nodes and bandwidth request
    """
    nodes_sd = request_array[jnp.array([0, 2])]
    requested_datarate = request_array[1]
    return nodes_sd, requested_datarate


def make_positive(x):
    return jnp.where(x < 0, -x, x)


@partial(jax.jit, donate_argnums=(0,))
def finalise_action_rsa(state: EnvState, action_info: ActionInfo, params: EnvParams):
    """Turn departure times positive.

    Args:
        state: current state

    Returns:
        state: updated state
    """
    state = state.replace(
        accepted_services=state.accepted_services + 1,
        accepted_bitrate=state.accepted_bitrate + action_info.requested_datarate,
        total_bitrate=state.total_bitrate + action_info.requested_datarate,
    )
    return state


@partial(jax.jit, donate_argnums=(0,))
def finalise_action_rwalr(state: EnvState, params: Optional[EnvParams]):
    """Turn departure times positive.

    Args:
        state: current state

    Returns:
        state: updated state
    """
    _, requested_datarate = read_rsa_request(state.request_array)
    state = state.replace(
        link_slot_departure_array=make_positive(state.link_slot_departure_array),
        accepted_services=state.accepted_services + 1,
        accepted_bitrate=state.accepted_bitrate + requested_datarate,
        total_bitrate=state.total_bitrate + requested_datarate,
    )
    return state


def check_slot_overflow(state: EnvState, action_info: ActionInfo, params: EnvParams):
    """If the action selects slot near the end, then the required slots can
    overflow and start filling from the start of the array, which might be free!
    To prevent this, we check the action index + required slots
    """
    overflow = differentiable_compare(
        action_info.initial_slot_index + action_info.num_slots,
        params.link_resources,
        op_type=">",
        temperature=params.temperature,
        differentiable=params.differentiable,
    )
    return overflow


def check_no_op(state: EnvState, action_info: ActionInfo, params: EnvParams):
    """Check for the "NO OP" action.
    This will be the maximum valid action idex + 1,
    resulting in a path index exceeding K paths."""
    overflow = differentiable_compare(
        action_info.path_index,
        params.k_paths,
        op_type=">=",
        temperature=params.temperature,
        differentiable=params.differentiable,
    )
    return overflow


def check_real_path(state: EnvState, action_info: ActionInfo, params: EnvParams):
    """Check if path is a dummy (all-zeros). A valid path always uses at least one link."""
    is_dummy = differentiable_compare(
        jnp.max(action_info.path),
        0,
        op_type="==",
        temperature=params.temperature,
        differentiable=params.differentiable,
    )
    return is_dummy


def check_action_rsa(state, action_info, params):
    """
    Differentiable version of check_action_rsa.

    Args:
        state: Current environment state
        temperature: Controls sharpness of sigmoid

    Returns:
        Continuous value that behaves like the original boolean check
    """
    # Calculate differentiable version of each check
    spectrum_reuse_check = differentiable_check_no_spectrum_reuse(state, action_info, params)
    overflow_check = check_slot_overflow(state, action_info, params)
    no_action_check = check_no_op(state, action_info, params)
    unique_path_check = check_real_path(state, action_info, params)
    # For multiple checks, use a differentiable version of "any"
    # Instead of jnp.any, use max to combine checks
    combined_check = jnp.max(
        jnp.stack(
            [
                spectrum_reuse_check,
                overflow_check,
                no_action_check,
                unique_path_check,
            ]
        )
    )
    return combined_check


def convert_node_probs_to_traffic_matrix(node_probs: list) -> chex.Array:
    """Convert list of node probabilities to symmetric traffic matrix.

    Args:
        node_probs: node probabilities

    Returns:
        traffic_matrix: traffic matrix
    """
    matrix = jnp.outer(node_probs, node_probs).astype(dtype_config.SMALL_FLOAT_DTYPE)
    # Set lead diagonal to zero
    matrix = jnp.where(jnp.eye(matrix.shape[0]) == 1, 0, matrix)
    matrix = normalise_traffic_matrix(matrix)
    return matrix


def get_edge_disjoint_paths(graph: nx.Graph) -> Dict[int, Dict[int, List[Tuple[int, int]]]]:
    """Get edge disjoint paths between all nodes in graph.

    Args:
        graph: graph

    Returns:
        dict: edge disjoint paths (path is list of edges)
    """
    result = {n: {} for n in graph}
    for n1, n2 in itertools.combinations(graph, 2):
        # Sort by number of links in path
        # TODO - sort by path length
        result[n1][n2] = sorted(list(nx.edge_disjoint_paths(graph, n1, n2)), key=len)
        result[n2][n1] = sorted(list(nx.edge_disjoint_paths(graph, n2, n1)), key=len)
    return result


def make_graph(topology_name: str = "conus", topology_directory: str | None = None):
    """Create graph from topology definition.
    Topologies must be defined in JSON format in the topologies directory and
    named as the topology name with .json extension.

    Args:
        topology_name: topology name
        topology_directory: topology directory

    Returns:
        graph: graph
    """
    topology_path = (
        pathlib.Path(topology_directory)
        if topology_directory
        else (pathlib.Path(__file__).parents[1].absolute() / "data" / "topologies")
    )
    # Create topology
    if topology_name == "4node":
        # 4 node ring
        graph = nx.from_numpy_array(
            np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        )
        # Add edge weights to graph
        nx.set_edge_attributes(graph, {(0, 1): 4, (1, 2): 3, (2, 3): 2, (3, 0): 1}, "distance")
    elif topology_name == "7node":
        # 7 node ring
        graph = nx.from_numpy_array(
            jnp.array(
                [
                    [0, 1, 0, 0, 0, 0, 1],
                    [1, 0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 1, 0],
                ]
            )
        )
        # Add edge weights to graph
        nx.set_edge_attributes(
            graph,
            {
                (0, 1): 4,
                (1, 2): 3,
                (2, 3): 2,
                (3, 4): 1,
                (4, 5): 2,
                (5, 6): 3,
                (6, 0): 4,
            },
            "distance",
        )
    else:
        with open(topology_path / f"{topology_name}.json") as f:
            graph = nx.node_link_graph(json.load(f), edges="links")
    return graph


@partial(jax.jit, static_argnums=(1,))
def get_request_mask(requested_slots, params):
    requested_slots = requested_slots.astype(jnp.int32)
    request_mask = jax.lax.dynamic_update_slice(
        jnp.zeros(params.max_slots * 2, dtype=dtype_config.LARGE_FLOAT_DTYPE),
        jnp.ones(params.max_slots, dtype=dtype_config.LARGE_FLOAT_DTYPE),
        (params.max_slots - requested_slots,),
    )
    # Then cut in half and flip
    request_mask = jnp.flip(jax.lax.dynamic_slice(request_mask, (0,), (params.max_slots,)), axis=0)
    return request_mask


@partial(jax.jit, static_argnums=(1,), donate_argnums=(0,))
def mask_slots(state: EnvState, params: EnvParams) -> Array:
    nodes_sd, requested_datarate = read_rsa_request(state.request_array)

    # Get path indices ONCE, reuse for both paths and SEs
    path_indices = get_path_index_array(params, nodes_sd)  # (k,)

    # Direct take - no recomputing indices
    paths = jnp.take(params.path_link_array.val, path_indices, axis=0)  # (k, num_links)
    if params.pack_path_bits:
        paths = jnp.unpackbits(paths, axis=1)[:, : params.num_links]

    paths_se = jnp.take(params.path_se_array.val, path_indices, axis=0)  # (k,)

    # 2. Compute occupied - this should be fast
    slots_occupied = state.link_slot_array != 0
    occupied = (paths @ slots_occupied) > 0

    # 3. Cumsum approach - fully vectorized
    padded = jnp.concatenate(
        [
            jnp.zeros((params.k_paths, 1)),
            occupied,
            jnp.ones((params.k_paths, params.max_slots - 1)),
        ],
        axis=1,
    )
    cumsum = jnp.cumsum(padded, axis=1)

    # 4. All unique SE values -> req_slots
    all_se_values = params.unique_se_values.val
    all_req_slots = jax.vmap(
        lambda se: required_slots(
            requested_datarate,
            se,
            params.slot_size,
            guardband=params.guardband,
            temperature=params.temperature,
        )
    )(all_se_values)

    # 5. Broadcast window sums - NO LOOPS
    slot_indices = jnp.arange(params.link_resources)
    end_indices = slot_indices[None, :] + all_req_slots[:, None]  # (num_mods, link_resources)

    cumsum_at_end = cumsum[:, end_indices]  # (k, num_mods, link_resources)
    cumsum_at_start = cumsum[:, slot_indices]  # (k, link_resources)

    window_sums = cumsum_at_end - cumsum_at_start[:, None, :]
    all_masks = (window_sums == 0).astype(
        dtype_config.LARGE_FLOAT_DTYPE
    )  # (k, num_mods, link_resources)

    # 6. Select mask per path
    if params.consider_modulation_format:
        num_mods = all_se_values.shape[0]
        mod_indices = jnp.argmax(paths_se[:, None] == all_se_values[None, :], axis=1)
        one_hot = jnp.arange(num_mods)[None, :] == mod_indices[:, None]
        final_masks = jnp.einsum("kmr,km->kr", all_masks, one_hot)
    else:
        final_masks = all_masks[:, 0, :]

    # Identify valid (non-dummy) paths - dummy paths are all-zeros
    # Zero out mask rows for dummy paths so they are unselectable
    path_valid = (jnp.max(paths, axis=1) > 0).astype(dtype_config.LARGE_FLOAT_DTYPE)  # (k,)
    final_masks = final_masks * path_valid[:, None]

    full_link_slot_mask = final_masks.reshape(-1)

    link_slot_mask = (
        aggregate_slots(full_link_slot_mask, params)
        if params.aggregate_slots > 1
        else full_link_slot_mask
    )

    if params.include_no_op:
        link_slot_mask = jnp.concatenate(
            [link_slot_mask, jnp.ones(1, dtype=dtype_config.LARGE_FLOAT_DTYPE)]
        )

    return link_slot_mask, full_link_slot_mask


@partial(jax.jit, static_argnums=(1,))
def aggregate_slots(full_mask: Array, params: EnvParams) -> Array:
    """Aggregate slot mask via max-pooling."""
    num_actions = math.ceil(params.link_resources / params.aggregate_slots)

    # Full mask has shape (k_paths * link_resources,)
    # Pad to make divisible by aggregate_slots
    pad_size = num_actions * params.aggregate_slots - params.link_resources
    if pad_size > 0:
        full_mask = full_mask.reshape((params.k_paths, params.link_resources))
        full_mask = jnp.pad(full_mask, ((0, 0), (0, pad_size)), constant_values=0)

    # Reshape to (k_paths, num_actions, aggregate_slots) and max over windows
    reshaped = full_mask.reshape(params.k_paths, num_actions, params.aggregate_slots)
    agg_mask = jnp.max(reshaped, axis=2).reshape(-1)

    return agg_mask


@partial(jax.jit, static_argnums=(1, 4))
def get_path_slots(
    link_slot_array: chex.Array,
    params: EnvParams,
    nodes_sd: chex.Array,
    i: int,
    agg_func: str = "max",
) -> chex.Array:
    """Get slots on each constitutent link of path from link_slot_array (L x S),
    then aggregate to get (S x 1) representation of slots on path.

    Args:
        link_slot_array: link-slot array
        params: environment parameters
        nodes_sd: source-destination nodes
        i: path index
        agg_func: aggregation function (max or sum).
            If max, result will be available slots on path.
            If sum, result will contain information on edge features.
            if mean, will be mean.

    Returns:
        slots: slots on path
    """
    path = get_path(params, nodes_sd, i)
    slots = path[:, None] * link_slot_array
    # Make any -1s positive then get max for each slot across links
    if agg_func == "max":
        # Use this for getting slots from link_slot_array
        slots = jnp.max(jnp.absolute(slots), axis=0)
    elif agg_func == "sum":
        # TODO - consider using an RNN (or S5) to aggregate edge features
        # Use this (or alternative) for aggregating edge features from GNN
        slots = jnp.sum(slots, axis=0, promote_integers=False)
    elif agg_func == "mean":
        # Use this for getting mean value in slot index along path
        slots = jnp.mean(slots, axis=0)
    elif agg_func == "min":
        # Use this for getting slots from link_slot_array
        slots = jnp.min(slots, axis=0)
    else:
        raise ValueError("agg_func must be 'max' or 'sum' or 'mean' or min")
    return slots


def count_until_next_one(
    array: chex.Array, position: int, temperature: float, differentiable: bool = True
) -> chex.Array:
    """
    Counts positions until the next 1 in the array.
    Made differentiable using straight-through gradient trick.

    Args:
        array: Input array
        position: Starting position for counting
        temperature: Temperature for differentiable approximation
        differentiable: If False, use non-differentiable operations

    Returns:
        Number of positions until the next 1
    """
    # Add 1s to end so that end block is counted and slice shape can be fixed
    shape = array.shape[0]
    array = jnp.concatenate([array, jnp.ones(array.shape[0], dtype=dtype_config.LARGE_INT_DTYPE)])
    # Find the indices of 1 in the array
    one_indices = jax.lax.dynamic_slice(array, (position,), (shape,))
    # Use our differentiable_argmax helper
    next_one_idx = differentiable_argmax(
        one_indices, temperature=temperature, differentiable=differentiable
    )
    return next_one_idx + 1


def count_until_previous_one(
    array: chex.Array, position: int, temperature: float, differentiable: bool = True
) -> int:
    """
    Counts positions until the previous 1 in the array.
    Made differentiable using straight-through gradient trick.

    Args:
        array: Input array
        position: Starting position for counting backwards
        temperature: Temperature for differentiable approximation
        differentiable: If False, use non-differentiable operations

    Returns:
        Number of positions until the previous 1
    """
    # Add 1s to start so that end block is counted and slice shape can be fixed
    shape = array.shape[0]
    array = jnp.concatenate([jnp.ones(array.shape[0], dtype=dtype_config.LARGE_INT_DTYPE), array])
    # Find the indices of 1 in the array
    one_indices = jax.lax.dynamic_slice(array, (-shape - position,), (shape,))
    one_indices = jnp.flip(one_indices)
    # Use our differentiable_argmax helper
    prev_one_idx = differentiable_argmax(
        one_indices, temperature=temperature, differentiable=differentiable
    )
    return prev_one_idx + 1


def find_block_starts(path_slots: Array) -> Array:
    """
    Finds the starting positions of blocks in the path slots.
    Args:
        path_slots: Array of path slots

    Returns:
        Array with 1s at the starting positions of blocks
    """
    # Add a [1] at the beginning to find transitions from 1 to 0
    transitions = jnp.clip(jnp.diff(path_slots, prepend=1), -1, 0)  # Find transitions (1 to 0)
    return jnp.abs(transitions)


def find_block_ends(path_slots: Array) -> Array:
    """
    Finds the end positions of blocks in the path slots.

    Args:
        path_slots: Array of path slots

    Returns:
        Array with 1s at the end positions of blocks
    """
    transitions = jnp.diff(path_slots, append=1)  # Find transition 0 to 1
    return jnp.clip(transitions, 0, 1)


def find_block_sizes(
    path_slots, starts_only=True, reverse=False, temperature=1.0, differentiable=True
):
    n = path_slots.shape[0]
    free = differentiable_compare(
        path_slots, 0, "==", temperature=temperature, differentiable=differentiable
    )

    occupied = 1.0 - free
    cumsum = jnp.cumsum(occupied)
    cumsum_padded = jnp.concatenate([jnp.array([0.0]), cumsum])

    # cum_occ[i, j] = number of occupied slots in [i, j]
    cum_occ = cumsum_padded[1:][None, :] - cumsum_padded[:-1][:, None]  # (n, n)

    upper_tri = jnp.triu(jnp.ones((n, n)))

    # all_free[i, j] = 1 iff every slot from i..j is free (and j >= i)
    all_free = (
        differentiable_compare(
            cum_occ, 0, "==", temperature=temperature, differentiable=differentiable
        )
        * upper_tri
    )

    if reverse:
        # Block size at position j = number of consecutive free slots ending at j
        block_sizes = jnp.sum(all_free, axis=0)
    else:
        # Block size at position i = number of consecutive free slots starting at i
        block_sizes = jnp.sum(all_free, axis=1)

    if starts_only:
        block_starts = find_block_starts(path_slots)
        block_sizes = block_sizes * block_starts
    else:
        block_sizes = block_sizes * free

    return block_sizes


@partial(jax.jit, static_argnums=(1,))
def calculate_path_stats(state, params, request):
    nodes_sd, requested_datarate = read_rsa_request(request)
    link_resources = jnp.array(params.link_resources, dtype=dtype_config.LARGE_FLOAT_DTYPE)
    slot_size = jnp.array(params.slot_size, dtype=dtype_config.LARGE_FLOAT_DTYPE)

    def single_path(i):
        slots = get_path_slots(state.link_slot_array, params, nodes_sd, i)
        se = (
            get_path_se(params, nodes_sd, i)
            if params.consider_modulation_format
            else jnp.array([1], dtype=dtype_config.SMALL_INT_DTYPE)
        )
        req_slots = required_slots(
            requested_datarate,
            se,
            params.slot_size,
            guardband=params.guardband,
            temperature=params.temperature,
        )
        req_slots_norm = req_slots * slot_size / jnp.max(params.values_bw.val)
        free = (slots == 0).astype(dtype_config.LARGE_FLOAT_DTYPE)
        free_slots_norm = jnp.sum(free) / link_resources

        block_sizes = find_block_sizes(
            slots, temperature=params.temperature, differentiable=params.differentiable
        )
        first_block_index = jnp.argmax(block_sizes >= req_slots)
        first_block_index_norm = (
            first_block_index.astype(dtype_config.LARGE_FLOAT_DTYPE) / link_resources
        )
        first_block_size_norm = block_sizes[first_block_index] / req_slots.astype(
            dtype_config.LARGE_FLOAT_DTYPE
        )

        num_blocks = jnp.maximum(
            jnp.sum(find_block_starts(slots)),
            1,
        )
        avg_block_size_norm = jnp.sum(block_sizes) / num_blocks / req_slots

        return jnp.array(
            [
                first_block_size_norm,
                first_block_index_norm,
                req_slots_norm,
                avg_block_size_norm.astype(dtype_config.LARGE_FLOAT_DTYPE),
                free_slots_norm,
            ]
        )

    stats = jax.vmap(single_path)(jnp.arange(params.k_paths))
    return stats


def create_run_name(config: Union[box.Box, dict]) -> str:
    """Create name for run based on config flags"""
    env_type = config["env_type"]
    topology = config["topology_name"]
    slots = config["link_resources"]
    gnn = "_GNN" if config["USE_GNN"] else ""
    incremental = "_INC" if config["incremental_loading"] else ""
    run_name = f"{env_type}_{topology}_{slots}{gnn}{incremental}".upper()
    if config["EVAL_HEURISTIC"]:
        run_name += f"_{config['path_heuristic']}"
        if env_type.lower() == "vone":
            run_name += f"_{config['node_heuristic']}"
    elif config["EVAL_MODEL"]:
        run_name += "_EVAL"
    return run_name


def init_link_capacity_array(params):
    """Initialise link capacity array. Represents available data rate for lightpath on each link.
    Default is high value (1e6) for unoccupied slots. Once lightpath established, capacity is determined by
    corresponding entry in path capacity array."""
    return jnp.full((params.num_links, params.link_resources), 1e6)


def init_path_index_array(params):
    """Initialise path index array. Represents index of lightpath occupying each slot."""
    return jnp.full((params.num_links, params.link_resources), -1)


def calculate_path_capacity(
    path_length,
    min_request=100,
    scale_factor=1.0,
    alpha=0.2e-3,  # dB/m
    NF=4.5,
    B=10e12,
    R_s=100e9,
    beta_2=-21.7e-27,
    gamma=1.2e-3,
    L_s=100e3,
    lambda0=1550e-9,
):
    # Convert alpha from dB/m to Nepers/m
    alpha_np = alpha * jnp.log(10) / 10  # or: alpha / 4.343

    N_spans = jnp.floor(path_length * 1e3 / L_s)
    L_eff = (1 - jnp.exp(-alpha_np * L_s)) / alpha_np

    sigma_2_ase = (jnp.exp(alpha_np * L_s) - 1) * 10 ** (NF / 10) * h * c * R_s / lambda0

    span_NSR = jnp.cbrt(
        2
        * sigma_2_ase**2
        * alpha_np
        * gamma**2
        * L_eff**2
        * jnp.log(jnp.pi**2 * jnp.abs(beta_2) * B**2 / alpha_np)
        / (jnp.pi * jnp.abs(beta_2) * R_s**2)
    )

    path_NSR = jnp.where(N_spans < 1, 1, N_spans) * span_NSR
    path_capacity = 2 * R_s / 1e9 * jnp.log2(1 + 1 / path_NSR)
    path_capacity = jnp.floor(path_capacity * scale_factor / min_request) * min_request
    return path_capacity


def init_path_capacity_array(
    link_length_array: chex.Array,
    path_link_array: chex.Array,
    min_request=1,  # Minimum data rate request size
    scale_factor=1.0,  # Scale factor for link capacity
    alpha=0.2e-3,  # Fibre attenuation coefficient
    NF=4.5,  # Amplifier noise figure
    B=10e12,  # Total modulated bandwidth
    R_s=100e9,  # Symbol rate
    beta_2=-21.7e-27,  # Dispersion parameter
    gamma=1.2e-3,  # Nonlinear coefficient
    L_s=100e3,  # span length
    lambda0=1550e-9,  # Wavelength
) -> chex.Array:
    """Calculated from Nevin paper:
    https://api.repository.cam.ac.uk/server/api/core/bitstreams/b80e7a9c-a86b-4b30-a6d6-05017c60b0c8/content

    Args:
        link_length_array (chex.Array): Array of link lengths
        path_link_array (chex.Array): Array of links on paths
        min_request (int, optional): Minimum data rate request size. Defaults to 100 GBps.
        scale_factor (float, optional): Scale factor for link capacity. Defaults to 1.0.
        alpha (float, optional): Fibre attenuation coefficient. Defaults to 0.2e-3 /m
        NF (float, optional): Amplifier noise figure. Defaults to 4.5 dB.
        B (float, optional): Total modulated bandwidth. Defaults to 10e12 Hz.
        R_s (float, optional): Symbol rate. Defaults to 100e9 Baud.
        beta_2 (float, optional): Dispersion parameter. Defaults to -21.7e-27 s^2/m.
        gamma (float, optional): Nonlinear coefficient. Defaults to 1.2e-3 /W/m.
        L_s (float, optional): Span length. Defaults to 100e3 m.
        lambda0 (float, optional): Wavelength. Defaults to 1550e-9 m.

    Returns:
        chex.Array: Array of link capacities in Gbps
    """
    path_length_array = jnp.dot(path_link_array, link_length_array)
    path_capacity_array = calculate_path_capacity(
        path_length_array,
        min_request=min_request,
        scale_factor=scale_factor,
        alpha=alpha,
        NF=NF,
        B=B,
        R_s=R_s,
        beta_2=beta_2,
        gamma=gamma,
        L_s=L_s,
        lambda0=lambda0,
    )
    return path_capacity_array.astype(dtype_config.LARGE_FLOAT_DTYPE)


@partial(jax.jit, static_argnums=(0,))
def get_lightpath_index(params, nodes, path_index):
    source, dest = nodes
    path_start_index = get_path_indices(
        params,
        source,
        dest,
        params.k_paths,
        params.num_nodes,
        directed=params.directed_graph,
    ).astype(dtype_config.INDEX_DTYPE)
    lightpath_index = path_index + path_start_index
    return lightpath_index


@partial(jax.jit, static_argnums=(2,))
def check_lightpath_available_and_existing(
    state: EnvState, action_info: ActionInfo, params: EnvParams
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    lightpath_index = get_lightpath_index(params, action_info.nodes_sd, action_info.path_index)

    initial_slot_index = action_info.initial_slot_index.astype(dtype_config.INDEX_DTYPE)
    path_index_array = state.path_index_array[:, initial_slot_index].reshape(-1, 1)
    path = action_info.path.reshape(-1, 1)

    # On-path slot indices
    on_path_indices = path * path_index_array + (1 - path) * lightpath_index
    # ^ off-path slots set to lightpath_index so they trivially match

    is_this_lightpath = on_path_indices == lightpath_index
    is_empty = on_path_indices == -1

    # Existing: all on-path slots belong to this lightpath
    lightpath_existing_check = jnp.all(is_this_lightpath > 0)

    # Available: all on-path slots are either this lightpath or empty
    lightpath_available_check = jnp.all((is_this_lightpath + is_empty) > 0)

    curr_lightpath_capacity = jnp.max(
        path * state.link_capacity_array[:, initial_slot_index].reshape(-1, 1)
    )

    return (
        lightpath_available_check,
        lightpath_existing_check,
        curr_lightpath_capacity,
        lightpath_index,
    )


def check_action_rwalr(state: EnvState, action_info: ActionInfo, params: EnvParams) -> bool:
    no_reuse = check_no_spectrum_reuse(state, action_info, params)
    not_available = ~check_lightpath_available_and_existing(state, action_info, params)[0]
    return no_reuse | not_available


@partial(jax.jit, static_argnums=(2,))
def implement_action_rwalr(state: EnvState, action_info: ActionInfo, params: EnvParams) -> EnvState:
    (
        lightpath_available_check,
        lightpath_existing_check,
        curr_lightpath_capacity,
        lightpath_index,
    ) = check_lightpath_available_and_existing(state, action_info, params)

    # Capacity calculation without cond
    initial_capacity = jnp.squeeze(
        jax.lax.dynamic_slice_in_dim(state.path_capacity_array, lightpath_index, 1)
    )
    base_capacity = (
        lightpath_existing_check * curr_lightpath_capacity
        + (1 - lightpath_existing_check) * initial_capacity
    )
    lightpath_capacity = base_capacity - action_info.requested_datarate

    # Conditional update via arithmetic masking
    available = lightpath_available_check

    new_link_capacity = set_path_links(
        state.link_capacity_array,
        action_info.affected_slots_mask,
        lightpath_capacity,
    )
    new_path_index = set_path_links(
        state.path_index_array,
        action_info.affected_slots_mask,
        lightpath_index,
    )

    # Blend: if available, use new; else keep old
    link_capacity_array = (
        state.link_capacity_array * (1 - available) + new_link_capacity * available
    )
    path_index_array = state.path_index_array * (1 - available) + new_path_index * available

    # Undo over-capacity: restore to pre-action capacity
    lightpath_capacity_before = (
        lightpath_existing_check * curr_lightpath_capacity + (1 - lightpath_existing_check) * 1e6
    )
    over_capacity = link_capacity_array < 0.0
    link_capacity_array = (
        link_capacity_array * (1 - over_capacity) + lightpath_capacity_before * over_capacity
    )

    # Capacity mask: 1 where capacity <= 0, plus 1 more where < 0 (over)
    capacity_mask = (link_capacity_array <= 0.0).astype(dtype_config.LARGE_FLOAT_DTYPE)
    total_mask = capacity_mask + over_capacity.astype(dtype_config.LARGE_FLOAT_DTYPE)

    state = state.replace(
        link_capacity_array=link_capacity_array,
        path_index_array=path_index_array,
        link_slot_array=total_mask,
        link_slot_departure_array=update_path_links(
            state.link_slot_departure_array,
            action_info,
            state.current_time + state.holding_time,
        ),
    )
    return state


@partial(jax.jit, static_argnums=(1,))
def mask_slots_rwalr(state: EnvState, params: EnvParams, request: chex.Array) -> EnvState:
    """For use in RWALightpathReuseEnv.
    Each lightpath has a maximum capacity defined in path_capacity_array. This is updated when a lightpath is assigned.
    If remaining path capacity is less than current request, corresponding link-slots are masked out.
    If link-slot is in use by another lightpath for a different source and destination node (even if not full) it is masked out.
    Step 1:
    - Mask out slots that are not valid based on path capacity (check link_capacity_array)
    Step 2:
    - Mask out slots that are not valid based on lightpath reuse (check path_index_array)

    Args:
        state: Environment state
        params: Environment parameters
        request: Request array in format [source_node, data-rate, destination_node]

    Returns:
        state: Updated environment state
    """
    nodes_sd, requested_datarate = read_rsa_request(request)
    source, dest = nodes_sd
    path_start_index = get_path_indices(
        params,
        source,
        dest,
        params.k_paths,
        params.num_nodes,
        directed=params.directed_graph,
    ).astype(dtype_config.INDEX_DTYPE)
    # Step 1 - capacity mask (computed once, shared across paths)
    capacity_mask = (state.link_capacity_array < requested_datarate).astype(
        dtype_config.LARGE_FLOAT_DTYPE
    )

    # Step 2 - lightpath reuse masks (computed once, indexed per path)
    empty_mask = (state.path_index_array != -1).astype(dtype_config.LARGE_FLOAT_DTYPE)

    def single_path(i):
        capacity_slots = get_path_slots(capacity_mask, params, nodes_sd, i)

        lightpath_index = path_start_index + i
        lightpath_mask = (
            1.0 - (state.path_index_array == lightpath_index).astype(dtype_config.LARGE_FLOAT_DTYPE)
        ) * empty_mask
        lightpath_slots = get_path_slots(lightpath_mask, params, nodes_sd, i)

        # Combine: masked if either mask is active, then invert (1 = valid)
        combined = jnp.maximum(capacity_slots, lightpath_slots)
        return 1.0 - jnp.minimum(combined, 1.0)

    link_slot_mask = jax.vmap(single_path)(jnp.arange(params.k_paths)).reshape(-1)

    if params.aggregate_slots > 1:
        state = state.replace(full_link_slot_mask=link_slot_mask)
        link_slot_mask, _ = aggregate_slots(link_slot_mask.reshape(params.k_paths, -1), params)
        link_slot_mask = link_slot_mask.reshape(-1)

    if params.include_no_op:
        link_slot_mask = jnp.concatenate([link_slot_mask, jnp.ones((1,))])

    return link_slot_mask


def pad_array(array, fill_value):
    """
    Pad a ragged multidimensional array to rectangular shape.
    Used for training on multiple topologies.
    Source: https://codereview.stackexchange.com/questions/222623/pad-a-ragged-multidimensional-array-to-rectangular-shape

    Args:
        array: array to pad
        fill_value: value to fill with

    Returns:
        result: padded array
    """

    def get_dimensions(array, level=0):
        yield level, len(array)
        try:
            for row in array:
                yield from get_dimensions(row, level + 1)
        except TypeError:  # not an iterable
            pass

    def get_max_shape(array):
        dimensions = defaultdict(int)
        for level, length in get_dimensions(array):
            dimensions[level] = max(dimensions[level], length)
        return [value for _, value in sorted(dimensions.items())]

    def iterate_nested_array(array, index=()):
        try:
            for idx, row in enumerate(array):
                yield from iterate_nested_array(row, (*index, idx))
        except TypeError:  # final level
            yield (*index, slice(len(array))), array

    dimensions = get_max_shape(array)
    result = np.full(dimensions, fill_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return result


def init_link_length_array_gn_model(
    graph: nx.Graph, max_span_length: int, max_spans: int
) -> chex.Array:
    """Initialise link length array for environements that use GN model of physical layer.
    We assume each link has spans of equal length.

    Args:
        graph (nx.Graph): NetworkX graph
    Returns:
        jnp.array: Link length array (L x max_spans)
    """
    link_lengths = []
    directed = graph.is_directed()
    graph = graph.to_undirected()
    edges = sorted(graph.edges)
    for edge in edges:
        link_lengths.append(graph.edges[edge]["distance"])
    if directed:
        for edge in edges:
            link_lengths.append(graph.edges[edge]["distance"])
    span_length_array = []
    for length in link_lengths:
        num_spans = math.ceil(length / max_span_length)
        avg_span_length = length / num_spans
        span_lengths = [avg_span_length] * num_spans
        span_lengths.extend([0] * (max_spans - num_spans))
        span_length_array.append(span_lengths)
    return jnp.array(span_length_array, dtype=dtype_config.LARGE_INT_DTYPE)


def init_link_snr_array(params: EnvParams):
    """Initialise signal-to-noise ratio (SNR) array.
    Args:
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: SNR array
    """
    # The SNR is kept in linear units to allow summation of 1/SNR across links
    return jnp.full(
        (params.num_links, params.link_resources), -1e5, dtype=dtype_config.LARGE_FLOAT_DTYPE
    )


def init_channel_power_array(params: EnvParams):
    """Initialise channel power array.

    Args:
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: Channel power array
    """
    return jnp.full(
        (params.num_links, params.link_resources), 0.0, dtype=dtype_config.LARGE_FLOAT_DTYPE
    )


def init_channel_centre_bw_array(params: EnvParams):
    """Initialise channel centre array.
    Args:
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: Channel centre array
    """
    return jnp.full(
        (params.num_links, params.link_resources), 0.0, dtype=dtype_config.LARGE_FLOAT_DTYPE
    )


def init_modulation_format_index_array(params: EnvParams):
    """Initialise modulation format index array.
    Args:
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: Modulation format index array
    """
    return jnp.full(
        (params.num_links, params.link_resources), -1, dtype=dtype_config.LARGE_INT_DTYPE
    )  # -1 so that highest order is assumed (closest to Gaussian)


def init_active_path_array(params: EnvParams):
    """Initialise active path array. Stores details of full path utilised by lightpath on each frequency slot.
    Args:
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: Active path array
    """
    return jnp.full(
        (params.num_links, params.link_resources, params.num_links),
        -1,
        dtype=dtype_config.LARGE_INT_DTYPE,
    )


def init_transceiver_amplifier_noise_arrays(
    link_resources: int,
    ref_lambda: float,
    slot_size: float,
    noise_data_filepath: str | None = None,
) -> Tuple[chex.Array, chex.Array]:
    """Initialise transceiver and amplifier noise arrays.
    Args:
        link_resources (int): Number of link resources
        ref_lambda (float): Reference wavelength
        slot_size (float): Slot size
        noise_data_filepath (str, optional): Path to CSV file containing modulation formats. Defaults to None.
    Returns:
        Tuple[chex.Array, chex.Array]: Transceiver noise array, Amplifier noise array
    """
    f = (
        pathlib.Path(noise_data_filepath)
        if noise_data_filepath
        else (
            pathlib.Path(__file__).parents[1].absolute()
            / "data"
            / "gn_model"
            / "transceiver_amplifier_data.csv"
        )
    )
    noise_data = np.genfromtxt(f, delimiter=",")
    # Drop empty first row (headers) and column (name)
    noise_data = noise_data[1:, 1:]
    # Columns are: wavelength_min_nm,wavelength_max_nm,frequency_min_ghz,frequency_max_ghz,NF_ASE_dB,SNR_TRX_dB
    frequency_min_ghz = noise_data[:, 2]
    frequency_max_ghz = noise_data[:, 3]
    amplifier_noise_db = noise_data[:, 4]  # NF_ASE_dB
    transceiver_snr_db = noise_data[:, 5]  # SNR_TRX_dB

    # Define slot centres in GHz relative to central wavelength
    slot_centres = (jnp.arange(link_resources) - (link_resources - 1) / 2) * slot_size

    # Transform relative slot centres to absolute frequencies in GHz
    ref_frequency_ghz = c / ref_lambda / 1e9
    slot_frequencies_ghz = ref_frequency_ghz + slot_centres

    # Initialize output arrays
    transceiver_snr_array = jnp.zeros(link_resources)
    amplifier_noise_figure_array = jnp.zeros(link_resources)

    # For each slot, find which band it belongs to
    for i, freq in enumerate(slot_frequencies_ghz):
        # Find the band this frequency falls into
        for j in range(len(frequency_min_ghz)):
            if frequency_min_ghz[j] <= freq <= frequency_max_ghz[j]:
                transceiver_snr_array = transceiver_snr_array.at[i].set(transceiver_snr_db[j])
                amplifier_noise_figure_array = amplifier_noise_figure_array.at[i].set(
                    amplifier_noise_db[j]
                )
                break
        else:
            # If frequency is outside all bands, could raise error or use default
            raise ValueError(f"Frequency {freq} GHz is outside the defined bands")

    return transceiver_snr_array, amplifier_noise_figure_array


@partial(jax.jit, static_argnums=(1, 2))
def get_required_snr_se_kurtosis_from_mod_format(mod_format_index, col_index, params):
    return params.modulations_array[mod_format_index][
        col_index
    ]  # column 1 is spectral efficiency 2 is required SNR


@partial(jax.jit, static_argnums=(1, 2))
def get_required_snr_se_kurtosis_on_link(mod_format_link, col_index, params):
    return jax.vmap(get_required_snr_se_kurtosis_from_mod_format, in_axes=(0, None, None))(
        mod_format_link, col_index, params
    )


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def get_required_snr_se_kurtosis_array(
    modulation_format_index_array: chex.Array,
    col_index: int,
    params: RSAGNModelEnvParams,
) -> chex.Array:
    """Convert modulation format index to required SNR or spectral efficiency.
    Modulation format index array contains the index of the modulation format used by the channel.
    The modulation index references a row in the modulations array, which contains SNR and SE values.

    Args:
        modulation_format_index_array (chex.Array): Modulation format index array
        col_index (int): Column index for required SNR or spectral efficiency
        params (RSAGNModelEnvParams): Environment parameters

    Returns:
        jnp.array: Required SNR for each channel (min. SNR for empty channel (mod. index 0))
    """
    return jax.vmap(get_required_snr_se_kurtosis_on_link, in_axes=(0, None, None))(
        modulation_format_index_array, col_index, params
    )


@partial(jax.jit, static_argnums=(2,))
def get_centre_frequency(
    initial_slot_index: int, num_slots: int, params: RSAGNModelEnvParams
) -> chex.Array:
    """Get centre frequency for new lightpath

    Args:
        initial_slot_index (chex.Array): Centre frequency of first slot
        num_slots (float): Number of slots
        params (RSAGNModelEnvParams): Environment parameters

    Returns:
        chex.Array: Centre frequency for new lightpath
    """
    slot_centres = (
        jnp.arange(params.link_resources) - (params.link_resources - 1) / 2
    ) * params.slot_size
    return slot_centres[initial_slot_index] + ((params.slot_size * (num_slots - 1)) / 2)


@partial(jax.jit, static_argnums=(2,))
def get_required_slots_on_link(bw_link, se_link, params):
    return jax.vmap(required_slots, in_axes=(0, 0, None, None))(
        bw_link, se_link, params.slot_size, params.guardband
    )


@partial(jax.jit, static_argnums=(2,))
def get_centre_freq_on_link(slot_index, num_slots_link, params):
    return jax.vmap(get_centre_frequency, in_axes=(0, 0, None))(slot_index, num_slots_link, params)


@partial(jax.jit, static_argnums=(1,))
def get_centre_frequencies_array(
    state: RSAGNModelEnvState, params: RSAGNModelEnvParams
) -> chex.Array:
    slot_indices = jnp.arange(params.link_resources)
    se_array = get_required_snr_se_kurtosis_array(state.modulation_format_index_array, 1, params)
    required_slots_array = jax.vmap(get_required_slots_on_link, in_axes=(0, 0, None))(
        state.channel_centre_bw_array, se_array, params
    )
    centre_freq_array = jax.vmap(get_centre_freq_on_link, in_axes=(None, 0, None))(
        slot_indices, required_slots_array, params
    )
    return centre_freq_array


@partial(jax.jit, static_argnums=(1,))
def get_path_from_path_index_array(
    path_index_array: chex.Array, path_link_array: chex.Array
) -> chex.Array:
    """Get path from path index array.
    Args:
        path_index_array (chex.Array): Path index array
        path_link_array (chex.Array): Path link array

    Returns:
        jnp.array: path index values replaced with binary path-link arrays
    """

    # TODO - support unpacking bits (if this function ends up being used)
    def get_index_from_link(link):
        return jax.vmap(lambda x: path_link_array[x], in_axes=(0,))(link)

    return jax.vmap(get_index_from_link, in_axes=(0,))(path_index_array)


def init_active_lightpaths_array(params: RSAGNModelEnvParams):
    """Initialise active lightpath array. Stores path indices of all active paths on the network in a 1 x M array.
    M is MIN(max_requests, num_links * link_resources / min_slots).
    min_slots is the minimum number of slots required for a lightpath i.e. max(values_bw)/ slot_size.

    Args:
        params (RSAGNModelEnvParams): Environment parameters
    Returns:
        jnp.array: Active path array (default value -1, empty path)
    """
    total_slots = params.num_links * params.link_resources  # total slots on networks
    min_slots = (
        jnp.max(params.values_bw.val) / params.slot_size
    )  # minimum number of slots required for lightpath
    return jnp.full((int(total_slots / min_slots), 3), -1, dtype=dtype_config.LARGE_INT_DTYPE)


def init_active_lightpaths_array_departure(params: RSAGNModelEnvParams):
    """Initialise active lightpath array. Stores path indices of all active paths on the network in a 1 x M array.
    M is MIN(max_requests, num_links * link_resources / min_slots).
    min_slots is the minimum number of slots required for a lightpath i.e. max(values_bw)/ slot_size.

    Args:
        params (RSAGNModelEnvParams): Environment parameters
    Returns:
        jnp.array: Active path array (default value -1, empty path)
    """
    total_slots = params.num_links * params.link_resources  # total slots on networks
    min_slots = (
        jnp.max(params.values_bw.val) / params.slot_size
    )  # minimum number of slots required for lightpath
    return jnp.full((int(total_slots / min_slots), 3), 0.0, dtype=dtype_config.SMALL_FLOAT_DTYPE)


def update_active_lightpaths_array(
    state: RSAGNModelEnvState, path_index: int, initial_slot_index: int, num_slots: int
) -> chex.Array:
    """Update active lightpaths array with new path index.
    Find the first index of the array with value -1 and replace with path index.
    Args:
        state (RSAGNModelEnvState): Environment state
        path_index (int): Path index to add to active lightpaths array
    Returns:
        jnp.array: Updated active lightpaths array
    """
    first_empty_index = jnp.argmin(
        state.active_lightpaths_array[:, 0]
    ) # Just look at the first column
    return jax.lax.dynamic_update_slice(
        state.active_lightpaths_array,
        jnp.array([[path_index, initial_slot_index, num_slots]], dtype=state.active_lightpaths_array.dtype),
        (first_empty_index, 0),
    )


def update_active_lightpaths_array_departure(state: RSAGNModelEnvState, time: float) -> chex.Array:
    """Update active lightpaths array with new path index.
    Find the first index of the array with value -1 and replace with path index.
    Args:
        state (RSAGNModelEnvState): Environment state
        time (float): Departure time
    Returns:
        jnp.array: Updated active lightpaths array
    """
    first_empty_index = jnp.argmin(
        state.active_lightpaths_array[:, 0]
    )  # Just look at the first column
    return jax.lax.dynamic_update_slice(
        state.active_lightpaths_array_departure,
        jnp.stack((time, time, time)),
        (first_empty_index, 0),
    )


def get_snr_for_path(path, link_snr_array, params):
    nsr_slots = jnp.where(
        path.reshape((params.num_links, 1)) == 1,
        1 / link_snr_array,
        jnp.zeros(params.link_resources, dtype=dtype_config.LARGE_FLOAT_DTYPE),
    )
    nsr_path_slots = jnp.sum(nsr_slots, axis=0, promote_integers=False)
    return jnp.nan_to_num(
        isrs_gn_model.to_db(1 / nsr_path_slots), nan=-50, neginf=-50, posinf=50
    )  # Link SNR array must be in linear units so that 1/inf = 0


def get_lightpath_snr(state: RSAGNModelEnvParams, params: RSAGNModelEnvParams) -> chex.Array:
    """Get SNR for each link on path.
    N.B. that in most cases it is more efficient to calculate the SNR for every possible path, rather than a slot-by-slot basis.
    But in some cases slot-by-slot is better i.e. when k*N(N-1)/2 > L*S
    Args:
        state (RSAGNModelEnvState): Environment state
        params (RSAGNModelEnvParams): Environment parameters

    Returns:
        chex.array: SNR for each link on path
    """
    # Get the SNR for the channel that the path occupies
    path_snr_array = jax.vmap(get_snr_for_path, in_axes=(0, None, None))(
        params.path_link_array.val, state.link_snr_array, params
    )
    # Where value in path_index_array matches index of path_snr_array, substitute in SNR value
    slot_indices = jnp.arange(params.link_resources)
    lightpath_snr_array = jax.vmap(
        jax.vmap(lambda x, si: path_snr_array[x][si], in_axes=(0, 0)), in_axes=(0, None)
    )(state.path_index_array, slot_indices)
    return lightpath_snr_array


def check_snr_sufficient(state: RSAGNModelEnvState, params: RSAGNModelEnvParams) -> chex.Array:
    """Check if SNR is sufficient for all connections
    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: 1 if SNR is sufficient for connection else 0
    """
    # TODO - this check needs to be faster!
    required_snr_array = get_required_snr_se_kurtosis_array(
        state.modulation_format_index_array, 2, params
    )
    # Transform lightpath index array by getting lightpath value, getting path-link array, and summing inverse link SNRs
    lightpath_snr_array = get_lightpath_snr(state, params)
    check_snr_sufficient = jnp.where(lightpath_snr_array >= required_snr_array, 0, 1)
    # jax.debug.print("check_snr_sufficient {}", check_snr_sufficient, ordered=True)
    # jax.debug.print("required_snr_array {}", required_snr_array, ordered=True)
    # jax.debug.print("lightpath_snr_array {}", lightpath_snr_array, ordered=True)
    # jax.debug.print("state.modulation_format_index_array {}", state.modulation_format_index_array, ordered=True)
    # jax.debug.print("state.channel_centre_bw_array {}", state.channel_centre_bw_array, ordered=True)
    # jax.debug.print("state.channel_power_array {}", state.channel_power_array, ordered=True)
    return jnp.any(check_snr_sufficient)


@partial(jax.jit, static_argnums=(1,))
def get_snr_link_array(state: EnvState, params: EnvParams) -> chex.Array:
    """Get SNR per link
    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: SNR per link
    """

    def get_link_snr(link_index, state, params):
        # Get channel power, channel centre, bandwidth, and noise figure
        link_lengths = params.link_length_array[link_index, :]
        num_spans = jnp.ceil(jnp.sum(link_lengths) * 1e3 / params.max_span_length).astype(
            dtype_config.LARGE_INT_DTYPE
        )
        if params.mod_format_correction:
            mod_format_link = state.modulation_format_index_array[link_index, :]
            kurtosis_link = get_required_snr_se_kurtosis_on_link(mod_format_link, 4, params)
            se_link = get_required_snr_se_kurtosis_on_link(mod_format_link, 1, params)
        else:
            kurtosis_link = jnp.zeros(params.link_resources).astype(jnp.float32)
            se_link = jnp.ones(params.link_resources, dtype=jnp.float32)
        bw_link = state.channel_centre_bw_array[link_index, :]
        ch_power_link = state.channel_power_array[link_index, :]
        required_slots_link = get_required_slots_on_link(bw_link, se_link, params)
        ch_centres_link = get_centre_freq_on_link(
            jnp.arange(params.link_resources), required_slots_link, params
        )

        # Calculate SNR
        P = dict(
            num_channels=params.link_resources,
            num_spans=num_spans,
            max_spans=params.max_spans,
            ref_lambda=params.ref_lambda,
            length=link_lengths,
            attenuation_i=jnp.array(params.attenuation),
            attenuation_bar_i=jnp.array(params.attenuation_bar),
            nonlinear_coeff=jnp.array(params.nonlinear_coeff),
            raman_gain_slope_i=jnp.array(params.raman_gain_slope),
            dispersion_coeff=jnp.array(params.dispersion_coeff),
            dispersion_slope=jnp.array(params.dispersion_slope),
            coherent=params.coherent,
            num_roadms=params.num_roadms,
            roadm_loss=params.roadm_loss,
            amplifier_noise_figure=params.amplifier_noise_figure.val,
            transceiver_snr=params.transceiver_snr.val,
            mod_format_correction=params.mod_format_correction,
            ch_power_w_i=ch_power_link,
            ch_centre_i=ch_centres_link * 1e9,
            ch_bandwidth_i=bw_link * 1e9,
            excess_kurtosis_i=kurtosis_link,
            uniform_spans=params.uniform_spans,
        )
        snr = isrs_gn_model.get_snr(**P)[0]

        return snr

    link_snr_array = jax.vmap(get_link_snr, in_axes=(0, None, None))(
        jnp.arange(params.num_links), state, params
    )
    link_snr_array = jnp.nan_to_num(link_snr_array, nan=1e-5)
    return link_snr_array


@partial(jax.jit, static_argnums=(1,))
def get_snr_link_array_fused(state: EnvState, params: EnvParams) -> chex.Array:
    """Get SNR per link using fused computation (uniform spans, no mod_format_correction).

    Drop-in replacement for get_snr_link_array that uses get_snr_fused to
    reduce XLA op count and kernel launch overhead on GPU.
    """
    # Precompute per-link num_spans and span_length from static link_length_array
    # link_length_array is (L, max_spans) in km
    link_lengths_km = params.link_length_array.val  # (L, max_spans)
    total_link_length_km = jnp.sum(link_lengths_km, axis=1)  # (L,)
    num_spans_per_link = jnp.ceil(total_link_length_km * 1e3 / params.max_span_length).astype(
        dtype_config.LARGE_INT_DTYPE
    )  # (L,)
    # span_length in km to match original get_snr: span_length = sum(length) / num_spans
    # where length is link_length_array values (in km)
    span_length_per_link = total_link_length_km / jnp.maximum(num_spans_per_link, 1).astype(
        jnp.float32
    )  # (L,) in km

    # Per-link channel data from state: (L, N)
    ch_power_all = state.channel_power_array  # (L, N)
    bw_all = state.channel_centre_bw_array  # (L, N) in GHz

    # Compute centre frequencies for all links: (L, N)
    # When mod_format_correction=False, se=1 everywhere
    se_all = jnp.ones_like(bw_all)
    required_slots_all = jax.vmap(get_required_slots_on_link, in_axes=(0, 0, None))(
        bw_all, se_all, params
    )  # (L, N)
    slot_indices = jnp.arange(params.link_resources)
    ch_centres_all = jax.vmap(get_centre_freq_on_link, in_axes=(None, 0, None))(
        slot_indices, required_slots_all, params
    )  # (L, N) in GHz

    # Convert to Hz for the GN model
    ch_centres_hz = ch_centres_all * 1e9  # (L, N)
    bw_hz = bw_all * 1e9  # (L, N)

    # Tile amplifier noise figure and transceiver SNR to (L, N) for vmap
    amp_nf = jnp.broadcast_to(params.amplifier_noise_figure.val, ch_power_all.shape)
    trx_snr = jnp.broadcast_to(params.transceiver_snr.val, ch_power_all.shape)

    def _link_snr_fused(ch_pow, ch_centre, ch_bw, n_spans, s_length, amp_nf_link, trx_snr_link):
        return isrs_gn_model.get_snr_fused(
            ch_power_w_i=ch_pow,
            ch_centre_i=ch_centre,
            ch_bandwidth_i=ch_bw,
            num_spans=n_spans,
            span_length=s_length,
            num_channels=params.link_resources,
            ref_lambda=params.ref_lambda,
            attenuation=params.attenuation,
            attenuation_bar=params.attenuation_bar,
            nonlinear_coeff=params.nonlinear_coeff,
            raman_gain_slope=params.raman_gain_slope,
            dispersion_coeff=params.dispersion_coeff,
            dispersion_slope=params.dispersion_slope,
            amplifier_noise_figure=amp_nf_link,
            transceiver_snr=trx_snr_link,
            roadm_loss=params.roadm_loss,
            num_roadms=params.num_roadms,
            coherent=params.coherent,
        )

    link_snr_array = jax.vmap(_link_snr_fused)(
        ch_power_all,
        ch_centres_hz,
        bw_hz,
        num_spans_per_link,
        span_length_per_link,
        amp_nf,
        trx_snr,
    )
    link_snr_array = jnp.nan_to_num(link_snr_array, nan=1e-5)
    return link_snr_array


# @partial(jax.jit, static_argnums=(3,))
# def get_best_modulation_format(
#     state: EnvState,
#     path: chex.Array,
#     initial_slot_index: int,
#     launch_power: chex.Array,
#     params: EnvParams,
# ) -> chex.Array:
#     """Get best modulation format for lightpath. "Best" is the highest order that has SNR requirements below available.
#     Try each modulation format, calculate SNR for each, then return the highest order possible.
#     Args:
#         state (EnvState): Environment state
#         path (chex.Array): Path array
#         initial_slot_index (int): Initial slot index
#         params (EnvParams): Environment parameters
#     Returns:
#         jnp.array: Acceptable modulation format indices
#     """
#     _, requested_datarate = read_rsa_request(state.request_array)
#     mod_format_count = params.modulations_array.val.shape[0]
#     acceptable_mod_format_indices = jnp.full((mod_format_count,), -2)

#     def acceptable_modulation_format(i, acceptable_format_indices):
#         req_snr = params.modulations_array.val[i][2] + params.snr_margin
#         se = params.modulations_array.val[i][1]
#         req_slots = required_slots(
#             requested_datarate,
#             se,
#             params.slot_size,
#             params.guardband,
#             temperature=params.temperature,
#         )
#         # TODO - need to check we don't overwrite values in already occupied slots
#         # Possible approaches:
#         # Check slot occupancy? Probably would need to iterate through for num_slots, but that's an issue
#         # What about we allocate and then fix up later, e.g. could it be possible to just add the modulation format on top without
#         # check sum of path links prior to assigning?
#         #
#         affected_slots_mask = get_affected_slots_mask(initial_slot_index, req_slots, path, params)
#         new_state = state.replace(
#             channel_power_array=set_path_links(
#                 state.channel_power_array,
#                 affected_slots_mask,
#                 launch_power,
#             ),
#             channel_centre_bw_array=set_path_links(
#                 state.channel_centre_bw_array,
#                 affected_slots_mask,
#                 params.slot_size,
#             ),
#         )
#         snr_value = get_minimum_snr_of_channels_on_path(
#             new_state, path, initial_slot_index, req_slots, params
#         )
#         # jax.debug.print("snr_value {}", snr_value, ordered=True)
#         # jax.debug.print("req_snr {}", req_snr, ordered=True)
#         acceptable_format_index = jnp.where(snr_value >= req_snr, i, -1).reshape((1,))
#         acceptable_format_indices = jax.lax.dynamic_update_slice(
#             acceptable_format_indices, acceptable_format_index, (i,)
#         )
#         # jax.debug.print("acceptable_format_indices {}", acceptable_format_indices, ordered=True)
#         return acceptable_format_indices

#     acceptable_mod_format_indices = jax.lax.fori_loop(
#         0, mod_format_count, acceptable_modulation_format, acceptable_mod_format_indices
#     )
#     return acceptable_mod_format_indices


@partial(jax.jit, static_argnums=(3,))
def get_best_modulation_format(
    state: EnvState,
    path: chex.Array,
    initial_slot_index: int,
    launch_power: chex.Array,
    params: EnvParams,
) -> chex.Array:
    _, requested_datarate = read_rsa_request(state.request_array)

    mod_formats = params.modulations_array.val  # (mod_format_count, ...)

    def check_single_format(mod_format_row, i):
        req_snr = mod_format_row[2] + params.snr_margin
        se = mod_format_row[1]
        req_slots = required_slots(
            requested_datarate,
            se,
            params.slot_size,
            params.guardband,
            temperature=params.temperature,
        )
        affected_slots_mask = get_affected_slots_mask(initial_slot_index, req_slots, path, params)
        new_state = state.replace(
            channel_power_array=set_path_links(
                state.channel_power_array,
                affected_slots_mask,
                launch_power,
            ),
            channel_centre_bw_array=set_path_links(
                state.channel_centre_bw_array,
                affected_slots_mask,
                params.slot_size,
            ),
        )
        snr_value = get_minimum_snr_of_channels_on_path(
            new_state, path, initial_slot_index, req_slots, params
        )
        return jnp.where(snr_value >= req_snr, i, -1)

    indices = jnp.arange(mod_formats.shape[0])
    acceptable_mod_format_indices = jax.vmap(check_single_format)(mod_formats, indices)

    return acceptable_mod_format_indices


@partial(jax.jit, static_argnums=(3,))
def get_best_modulation_format_simple(
    state: RSAGNModelEnvState,
    path: chex.Array,
    initial_slot_index: int,
    params: RSAGNModelEnvParams,
) -> chex.Array:
    """Get modulation format for lightpath.
    Assume worst case (least Gaussian) modulation format when calculating SNR.
    Args:
        state (EnvState): Environment state
        path (chex.Array): Path array
        initial_slot_index (int): Initial slot index
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: Acceptable modulation format indices
    """
    link_snr_array = get_snr_link_array(state, params)
    snr_value = (
        get_snr_for_path(path, link_snr_array, params)[initial_slot_index] - params.snr_margin
    )  # Margin
    mod_format_count = params.modulations_array.val.shape[0]
    acceptable_mod_format_indices = jnp.arange(mod_format_count)
    req_snr = params.modulations_array.val[:, 2] + params.snr_margin
    acceptable_mod_format_indices = jnp.where(
        snr_value >= req_snr,
        acceptable_mod_format_indices,
        jnp.full((mod_format_count,), -2),
    )
    return acceptable_mod_format_indices


@partial(jax.jit, static_argnums=(1, 2))
def set_band_gaps(link_slot_array: chex.Array, params: RSAGNModelEnvParams, val: int) -> chex.Array:
    """Set band gaps in link slot array
    Args:
        link_slot_array (chex.Array): Link slot array
        params (RSAGNModelEnvParams): Environment parameters
        val (int): Value to set
    Returns:
        chex.Array: Link slot array with band gaps
    """
    # Create array that is size of link_slot array with values of column index
    mask = jnp.arange(params.link_resources)
    mask = jnp.tile(mask, (params.num_links, 1))

    def set_band_gap(i, arr):
        gap_start = params.gap_starts.val[i]
        gap_end = gap_start + params.gap_widths.val[i]
        condition = jnp.logical_and(arr >= gap_start, arr < gap_end)
        arr = jnp.where(condition, -one, arr)
        return arr

    mask = jax.lax.fori_loop(0, params.gap_widths.val.shape[0], set_band_gap, mask)
    link_slot_array = jnp.where(mask == -one, val, link_slot_array)
    return link_slot_array


@partial(jax.jit, static_argnums=(1,))
def check_action_rmsa_gn_model(
    state: EnvState, action: Optional[chex.Array], params: EnvParams
) -> bool:
    """Check if action is valid for RSA GN model
    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
        action (chex.Array): Action array
    Returns:
        bool: True if action is invalid, False if action is valid
    """
    # Check if action is valid
    # TODO - log failure reasons in info
    snr_sufficient_check = check_snr_sufficient(state, params)
    spectrum_reuse_check = check_no_spectrum_reuse(state, action, params)
    # jax.debug.print("spectrum_reuse_check {}", spectrum_reuse_check, ordered=True)
    # jax.debug.print("snr_sufficient_check {}", snr_sufficient_check, ordered=True)
    return jnp.any(
        jnp.stack(
            (
                spectrum_reuse_check,
                snr_sufficient_check,
            )
        )
    )


@partial(jax.jit, static_argnums=(2,))
def implement_action_rsa_gn_model(
    state: RSAGNModelEnvState, action_info: ActionInfo, params: RSAGNModelEnvParams
) -> EnvState:
    """Implement action for RSA GN model. Update following arrays:
    - link_slot_array
    - link_slot_departure_array
    - link_snr_array
    - modulation_format_index_array
    - channel_power_array
    - active_path_array
    Args:
        state (EnvState): Environment state
        action (chex.Array): Action tuple (first is path action, second is launch_power)
        params (EnvParams): Environment parameters
    Returns:
        EnvState: Updated environment state
    """
    path_action, power_action = action_info.action
    path_action = path_action.astype(dtype_config.LARGE_INT_DTYPE)
    lightpath_index = get_lightpath_index(params, action_info.nodes_sd, action_info.path_index)
    launch_power = get_launch_power(state, path_action, power_action, params)
    # Update link_slot_array and link_slot_departure_array, then other arrays
    state = implement_path_action(state, action_info, params)
    state = state.replace(
        path_index_array=set_path_links(
            state.path_index_array,
            action_info.affected_slots_mask,
            lightpath_index,
        ),
        channel_power_array=set_path_links(
            state.channel_power_array,
            action_info.affected_slots_mask,
            launch_power,
        ),
        # TODO - update this to use separate arrays to track
        # channel centres and bandwidths and update with bandwidth
        # (that may or may not equal slot size)
        channel_centre_bw_array=set_path_links(
            state.channel_centre_bw_array,
            action_info.affected_slots_mask,
            params.slot_size,
        ),
    )
    if params.monitor_active_lightpaths:
        state = state.replace(
            active_lightpaths_array=update_active_lightpaths_array(
                state,
                lightpath_index,
                action_info.initial_slot_index,
                action_info.num_slots - params.guardband,
            ),
            active_lightpaths_array_departure=update_active_lightpaths_array_departure(
                state, -state.current_time - state.holding_time
            ),
        )
        # No need to check SNR until end of episode
        return state
    # Update link_snr_array
    state = state.replace(link_snr_array=get_snr_link_array(state, params))
    return state


@partial(jax.jit, static_argnums=(2,))
def implement_action_rmsa_gn_model(
    state: RSAGNModelEnvState, action_info: ActionInfo, params: RSAGNModelEnvParams
) -> EnvState:
    """Implement action for RSA GN model. Update following arrays:
    - link_slot_array
    - link_slot_departure_array
    - link_snr_array
    - modulation_format_index_array
    - channel_power_array
    - active_path_array
    Args:
        state (EnvState): Environment state
        action (chex.Array): Action tuple (first is path action, second is launch_power)
        params (EnvParams): Environment parameters
    Returns:
        EnvState: Updated environment state
    """
    path_action, power_action = action_info.action
    path_action = path_action.astype(dtype_config.LARGE_INT_DTYPE)
    lightpath_index = get_lightpath_index(params, action_info.nodes_sd, action_info.path_index)
    launch_power = get_launch_power(state, path_action, power_action, params)
    # TODO(GN MODEL) - get mod. format based on maximum reach
    mod_format_index = jax.lax.dynamic_slice(state.mod_format_mask, (path_action,), (1,)).astype(
        dtype_config.LARGE_INT_DTYPE
    )[0]
    # Update link_slot_array and link_slot_departure_array, then other arrays
    state = implement_path_action(state, action_info, params)
    state = state.replace(
        path_index_array=set_path_links(
            state.path_index_array,
            action_info.affected_slots_mask,
            lightpath_index,
        ),
        channel_power_array=set_path_links(
            state.channel_power_array,
            action_info.affected_slots_mask,
            launch_power,
        ),
        modulation_format_index_array=set_path_links(
            state.modulation_format_index_array,
            action_info.affected_slots_mask,
            mod_format_index,
        ),
        channel_centre_bw_array=set_path_links(
            state.channel_centre_bw_array,
            action_info.affected_slots_mask,
            params.slot_size,
        ),
    )
    # Update link_snr_array
    state = state.replace(link_snr_array=get_snr_link_array(state, params))
    # jax.debug.print("launch_power {}", launch_power, ordered=True)
    # jax.debug.print("mod_format_index {}", mod_format_index, ordered=True)
    # jax.debug.print("initial_slot_index {}", initial_slot_index, ordered=True)
    # jax.debug.print("state.mod_format_mask {}", state.mod_format_mask, ordered=True)
    # jax.debug.print("path_snr {}", get_snr_for_path(path, state.link_snr_array, params), ordered=True)
    # jax.debug.print("required_snr {}", params.modulations_array.val[mod_format_index][2] + params.snr_margin, ordered=True)
    return state


@partial(jax.jit, static_argnums=(1,))
def undo_action_rsa_gn_model(state: RSAGNModelEnvState, params: RSAGNModelEnvParams) -> EnvState:
    """Undo action for RSA GN model
    Args:
        state (EnvState): Environment state
        action (chex.Array): Action array
        params (EnvParams): Environment parameters
    Returns:
        EnvState: Updated environment state
    """
    state = undo_action_rsa(state, params)  # Undo link_slot_array and link_slot_departure_array
    state = state.replace(
        link_slot_array=set_band_gaps(state.link_slot_array, params, -one),  # Set C+L band gap
        channel_centre_bw_array=state.channel_centre_bw_array_prev,
        path_index_array=state.path_index_array_prev,
        channel_power_array=state.channel_power_array_prev,
    )
    if params.monitor_active_lightpaths:
        # If departure array is negative, then undo the action
        mask = jnp.where(state.active_lightpaths_array_departure < zero, one, zero)
        state = state.replace(
            active_lightpaths_array=jnp.where(mask == one, -one, state.active_lightpaths_array),
            active_lightpaths_array_departure=jnp.where(
                mask == one,
                state.active_lightpaths_array_departure + state.current_time + state.holding_time,
                state.active_lightpaths_array_departure,
            ),
        )
    return state


@partial(jax.jit, static_argnums=(1,))
def undo_action_rmsa_gn_model(state: RSAGNModelEnvState, params: RSAGNModelEnvParams) -> EnvState:
    """Undo action for RMSA GN model
    Args:
        state (EnvState): Environment state
        action (chex.Array): Action array
        params (EnvParams): Environment parameters
    Returns:
        EnvState: Updated environment state
    """
    state = undo_action_rsa(state, params)  # Undo link_slot_array and link_slot_departure_array
    state = state.replace(
        link_slot_array=set_band_gaps(state.link_slot_array, params, -one),  # Set C+L band gap
        channel_centre_bw_array=state.channel_centre_bw_array_prev,
        path_index_array=state.path_index_array_prev,
        channel_power_array=state.channel_power_array_prev,
        modulation_format_index_array=state.modulation_format_index_array_prev,
    )
    return state


def finalise_action_rsa_gn_model(
    state: RSAGNModelEnvState, params: Optional[EnvParams]
) -> EnvState:
    state = finalise_action_rsa(state, params)
    state = state.replace(
        link_slot_array=set_band_gaps(state.link_slot_array, params, -one),  # Set C+L band gap
        channel_centre_bw_array_prev=state.channel_centre_bw_array,
        path_index_array_prev=state.path_index_array,
        channel_power_array_prev=state.channel_power_array,
    )
    if params.monitor_active_lightpaths:
        state = state.replace(
            active_lightpaths_array_departure=make_positive(
                state.active_lightpaths_array_departure
            ),
        )
    return state


def finalise_action_rmsa_gn_model(
    state: RSAGNModelEnvState, params: Optional[EnvParams]
) -> EnvState:
    state = finalise_action_rsa(state, params)
    state = state.replace(
        link_slot_array=set_band_gaps(state.link_slot_array, params, -one),  # Set C+L band gap
        channel_centre_bw_array_prev=state.channel_centre_bw_array,
        path_index_array_prev=state.path_index_array,
        channel_power_array_prev=state.channel_power_array,
        modulation_format_index_array_prev=state.modulation_format_index_array,
    )
    return state


def calculate_throughput_from_active_lightpaths(
    state: RSAGNModelEnvState, params: RSAGNModelEnvParams
) -> chex.Array:
    # Update the SNR
    state = state.replace(link_snr_array=get_snr_link_array(state, params))
    slot_indices = jnp.arange(params.link_resources, dtype=dtype_config.LARGE_INT_DTYPE)

    def get_throughput_iter(i, throughput):
        # Get path index from active lightpaths array
        path_index, initial_slot_index, num_slots = state.active_lightpaths_array[i]
        path_packed = params.path_link_array.val[path_index]
        path_snr = get_snr_for_path(path_packed, state.link_snr_array, params)
        path = (
            jnp.unpackbits(path_packed)[: params.num_links]
            if params.pack_path_bits
            else path_packed
        )
        path = path.reshape((params.num_links, 1))
        # Get slots on path that use this path index
        path_indices_on_slots = jnp.where(
            path, state.path_index_array, jnp.full(state.path_index_array.shape, -1)
        )
        path_indices_on_slots = jnp.max(path_indices_on_slots, axis=0)
        mask = jnp.where(path_indices_on_slots == path_index, one, zero)
        # Update mask to be 1 for slots used by this lightpath
        condition = jnp.logical_and(
            slot_indices >= initial_slot_index,
            slot_indices < initial_slot_index + num_slots,
        )
        mask = jnp.where(condition, mask, zero)
        # Mask SNR below minimum required
        path_snr = jnp.where(path_snr < params.min_snr, -50, path_snr)
        # Get SNR for this lightpath (need to convert to linear units for throughput calc)
        snr = from_db(path_snr) * mask
        # Calculate throughput from Shannon-Hartley, with 2 for polarisation, symbol rate = bandwidth, 28% FEC
        datarate_per_channel = jnp.log2(1 + snr) * params.slot_size * 2 * (1 - params.fec_threshold)
        throughput += jnp.sum(datarate_per_channel)
        # jax.debug.print("datarate {} throughput {}", jnp.sum(datarate_per_channel), throughput, ordered=True)
        return throughput

    total_throughput = jax.lax.fori_loop(
        0,
        state.active_lightpaths_array.shape[0],
        get_throughput_iter,
        jnp.zeros((1,), dtype=state.link_snr_array.dtype),
    )
    return total_throughput[0]


@partial(jax.jit, static_argnums=(2,))
def get_minimum_snr_of_channels_on_path(
    state: RSAGNModelEnvState,
    path: chex.Array,
    slot_index: chex.Array,
    req_slots: int,
    params: RSAGNModelEnvParams,
) -> chex.Array:
    """Get the minimum value of the SNR on newly assigned channels.
    N.B. this requires the link_snr_array to have already been calculated and present in state."""
    snr_value_all_channels = get_snr_for_path(path, state.link_snr_array, params)
    min_snr_value_sub_channels = jnp.min(
        jnp.concatenate(
            [
                snr_value_all_channels[slot_index].reshape((1,)),
                snr_value_all_channels[slot_index + req_slots - 1].reshape((1,)),
            ],
            axis=0,
        )
    )
    return min_snr_value_sub_channels


@partial(jax.jit, static_argnums=(1,))
def mask_slots_rmsa_gn_model(
    state: RSAGNModelEnvState, params: RSAGNModelEnvParams, request: chex.Array
) -> EnvState:
    """For use in RSAGNModelEnv.
    1. For each path:
        1.1 Get path slots
        1.2 Get launch power


    Args:
        state: Environment state
        params: Environment parameters
        request: Request array in format [source_node, data-rate, destination_node]

    Returns:
        state: Updated environment state
    """
    nodes_sd, requested_datarate = read_rsa_request(request)
    init_mask = jnp.zeros((params.link_resources * params.k_paths)).astype(jnp.float32)

    def mask_path(i, mask):
        path = get_paths(params, nodes_sd)[i]
        # Get slots for path
        slots = get_path_slots(state.link_slot_array, params, nodes_sd, i)
        # Add padding to slots at end
        # 0 means slot is free, 1 is occupied
        slots = jnp.concatenate((slots, jnp.ones(params.max_slots)), dtype=jnp.float32)
        launch_power = get_launch_power(state, i, state.launch_power_array[i], params)
        lightpath_index = get_lightpath_index(params, nodes_sd, i)

        # This function checks through each available modulation format, checks the first and last available slots,
        # calculates the SNR, checks it meets the requirements, and returns the resulting mask
        def check_modulation_format(mod_format_index, init_path_mask):
            se = params.modulations_array.val[mod_format_index][1]
            req_slots = required_slots(
                requested_datarate,
                se,
                params.slot_size,
                guardband=params.guardband,
                temperature=params.temperature,
            )[0]
            bandwidth_per_subchannel = params.slot_size
            req_snr = params.modulations_array.val[mod_format_index][2] + params.snr_margin
            # Get mask used to check if request will fit slots
            request_mask = get_request_mask(req_slots, params)

            def check_slots_available(j, val):
                # Multiply through by request mask to check if slots available
                slot_sum = (
                    jnp.sum(
                        request_mask * jax.lax.dynamic_slice(val, (j,), (params.max_slots,)),
                        promote_integers=False,
                    )
                    <= 0
                )
                slot_sum = slot_sum.reshape((1,)).astype(dtype_config.LARGE_FLOAT_DTYPE)
                return jax.lax.dynamic_update_slice(val, slot_sum, (j,))

            # Mask out slots that are not valid
            slot_mask = jax.lax.fori_loop(
                0,
                int(params.link_resources + 1),  # No need to check last requested_slots-1 slots
                check_slots_available,
                slots,
            )
            # Cut off padding
            slot_mask = jax.lax.dynamic_slice(slot_mask, (0,), (params.link_resources,))
            # Check first and last available slots for suitability
            ff_path_mask = jnp.concatenate((slot_mask, jnp.ones((1,), jnp.float32)), axis=0)
            lf_path_mask = jnp.concatenate((jnp.ones((1,), jnp.float32), slot_mask), axis=0)
            first_available_slot_index = jnp.argmax(ff_path_mask)
            last_available_slot_index = (
                params.link_resources - jnp.argmax(jnp.flip(lf_path_mask)) - 1
            )
            # Assign "req_slots" subchannels (each with bandwidth = slot width) for the first and last possible slots
            affected_slots_mask = get_affected_slots_mask(
                first_available_slot_index, req_slots, path, params
            )
            ff_temp_state = state.replace(
                channel_centre_bw_array=set_path_links(
                    state.channel_centre_bw_array,
                    affected_slots_mask,
                    bandwidth_per_subchannel,
                ),
                channel_power_array=set_path_links(
                    state.channel_power_array,
                    affected_slots_mask,
                    launch_power,
                ),
                path_index_array=set_path_links(
                    state.path_index_array,
                    affected_slots_mask,
                    lightpath_index,
                ),
                modulation_format_index_array=set_path_links(
                    state.modulation_format_index_array,
                    affected_slots_mask,
                    mod_format_index,
                ),
            )
            lf_temp_state = state.replace(
                channel_centre_bw_array=set_path_links(
                    state.channel_centre_bw_array,
                    affected_slots_mask,
                    bandwidth_per_subchannel,
                ),
                channel_power_array=set_path_links(
                    state.channel_power_array,
                    affected_slots_mask,
                    launch_power,
                ),
                path_index_array=set_path_links(
                    state.path_index_array,
                    affected_slots_mask,
                    lightpath_index,
                ),
                modulation_format_index_array=set_path_links(
                    state.modulation_format_index_array,
                    affected_slots_mask,
                    mod_format_index,
                ),
            )
            ff_temp_state = ff_temp_state.replace(
                link_snr_array=get_snr_link_array(ff_temp_state, params)
            )
            lf_temp_state = lf_temp_state.replace(
                link_snr_array=get_snr_link_array(lf_temp_state, params)
            )
            # Take the minimum value of SNR from all the subchannels
            ff_snr_value = get_minimum_snr_of_channels_on_path(
                ff_temp_state, path, first_available_slot_index, req_slots, params
            )
            lf_snr_value = get_minimum_snr_of_channels_on_path(
                lf_temp_state, path, last_available_slot_index, req_slots, params
            )
            # Check that other paths SNR is still sufficient (True if failure)
            ff_snr_check = 1 - check_action_rmsa_gn_model(ff_temp_state, None, params)
            lf_snr_check = 1 - check_action_rmsa_gn_model(lf_temp_state, None, params)
            ff_check = (ff_snr_value >= req_snr) * ff_snr_check
            lf_check = (lf_snr_value >= req_snr) * lf_snr_check

            slot_indices = jnp.arange(params.link_resources, dtype=dtype_config.LARGE_INT_DTYPE)
            mod_format_mask = jnp.where(slot_indices == first_available_slot_index, ff_check, False)
            mod_format_mask = jnp.where(
                slot_indices == last_available_slot_index, lf_check, mod_format_mask
            )
            path_mask = jnp.where(mod_format_mask, mod_format_index, init_path_mask)
            # jax.debug.print("ff_snr_check {}", ff_snr_check, ordered=True)
            # jax.debug.print("lf_snr_check {}", lf_snr_check, ordered=True)
            # jax.debug.print("ff_snr_value {}", ff_snr_value, ordered=True)
            # jax.debug.print("lf_snr_value {}", lf_snr_value, ordered=True)
            # jax.debug.print("first_available_slot_index {}", first_available_slot_index, ordered=True)
            # jax.debug.print("last_available_slot_index {}", last_available_slot_index, ordered=True)
            # jax.debug.print("req_snr {}", req_snr, ordered=True)
            # jax.debug.print("mod_format_mask {}", mod_format_mask, ordered=True)
            # jax.debug.print("path_mask {}", path_mask, ordered=True)
            return path_mask

        path_mask = jax.lax.fori_loop(
            0,
            params.modulations_array.val.shape[0],
            check_modulation_format,
            jnp.full((params.link_resources,), -1.0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
        )

        # Update total mask with path mask
        mask = jax.lax.dynamic_update_slice(mask, path_mask, (i * params.link_resources,))
        return mask

    # Loop over each path
    mod_format_mask = jax.lax.fori_loop(0, params.k_paths, mask_path, init_mask)
    link_slot_mask = jnp.where(mod_format_mask >= 0, 1.0, 0.0)
    if params.aggregate_slots > 1:
        # Full link slot mask is used in process_path_action to get the correct slot from the aggregated slot action
        state = state.replace(full_link_slot_mask=link_slot_mask)
        link_slot_mask, _ = aggregate_slots(link_slot_mask.reshape(params.k_paths, -1), params)
        link_slot_mask = link_slot_mask.reshape(-1)
    if params.include_no_op:
        # Include extra unmasked action for "no op"
        link_slot_mask = jnp.hstack([link_slot_mask, jnp.ones((1,))])
    state = state.replace(
        link_slot_mask=link_slot_mask,
        mod_format_mask=mod_format_mask,
    )
    return state


@partial(jax.jit, static_argnums=(3,))
def get_launch_power(
    state: EnvState,
    path_action: chex.Array,
    power_action: chex.Array,
    params: EnvParams,
) -> chex.Array:
    """Get launch power for new lightpath. N.B. launch power is specified in dBm but is converted to linear units
    when stored in channel_power_array. This func returns linear units (mW).
    Path action is used to determine the launch power in the case of tabular launch power type.
    Power action is used to determine the launch power in the case of RL launch power type. During masking,
    power action is set as state.launch_power_array[0], which is set by the RL agent.
    Args:
        state (EnvState): Environment state
        path_action (chex.Array): Action specifying path index (0 to k_paths-1)
        power_action (chex.Array): Action specifying launch power in dBm
        params (EnvParams): Environment parameters
    Returns:
        chex.Array: Launch power for new lightpath
    """
    k_path_index, _ = process_path_action(state, params, path_action)
    if params.launch_power_type == 1:  # Fixed
        return state.launch_power_array[0]
    elif params.launch_power_type == 2:  # Tabular (one row per path)
        nodes_sd, requested_datarate = read_rsa_request(state.request_array)
        source, dest = nodes_sd
        i = get_path_indices(
            params,
            source,
            dest,
            params.k_paths,
            params.num_nodes,
            directed=params.directed_graph,
        ).astype(jnp.int32)
        return state.launch_power_array[i + k_path_index]
    elif params.launch_power_type == 3:  # RL
        return power_action
    elif params.launch_power_type == 4:  # Scaled
        nodes_sd, requested_datarate = read_rsa_request(state.request_array)
        source, dest = nodes_sd
        i = get_path_indices(
            params,
            source,
            dest,
            params.k_paths,
            params.num_nodes,
            directed=params.directed_graph,
        )
        # Get path length
        link_length_array = jnp.sum(params.link_length_array.val, axis=1, promote_integers=False)
        path_length = jnp.sum(link_length_array[i + k_path_index], promote_integers=False)
        path_link_array = (
            jnp.unpackbits(params.path_link_array.val)[:, params.num_links]
            if params.pack_path_bits
            else params.path_link_array.val
        )
        maximum_path_length = jnp.max(jnp.dot(path_link_array, params.link_length_array.val))
        return state.launch_power_array[0] * (path_length / maximum_path_length)
    else:
        raise ValueError("Invalid launch power type. Check params.launch_power_type")


@partial(jax.jit, static_argnums=(1,))
def get_paths_obs_gn_model(state: RSAGNModelEnvState, params: RSAGNModelEnvParams) -> chex.Array:
    # TODO - make this just show the stats from just one path at a time
    """Get observation space for launch power optimization (with numerical stability)."""
    request_array = state.request_array.reshape((-1,))
    path_stats = calculate_path_stats(state, params, request_array)
    # Remove first 3 items of path stats for each path
    path_stats = path_stats[:, 3:]
    link_length_array = jnp.sum(params.link_length_array.val, axis=1, promote_integers=False)
    lightpath_snr_array = get_lightpath_snr(state, params)
    nodes_sd, requested_datarate = read_rsa_request(request_array)
    source, dest = nodes_sd

    def calculate_gn_path_stats(k_path_index, init_val):
        # Get path index
        path_index = (
            get_path_indices(params, source, dest, params.k_paths, params.num_nodes) + k_path_index
        )
        path_link_array = (
            jnp.unpackbits(params.path_link_array.val, axis=1)[:, : params.num_links]
            if params.pack_path_bits
            else params.path_link_array.val
        )
        path = params.path_link_array[path_index]
        path_length = jnp.dot(path, link_length_array)
        max_path_length = jnp.max(jnp.dot(path_link_array, link_length_array))
        path_length / max_path_length
        max_path_length_hops = jnp.max(jnp.sum(path_link_array, axis=1, promote_integers=False))
        path_length_hops_norm = (
            jnp.sum(path, promote_integers=False).astype(dtype_config.LARGE_FLOAT_DTYPE)
            / max_path_length_hops
        )
        # Connections on path
        num_connections = jnp.where(
            path == 1,
            jnp.where(state.channel_power_array > 0, one, zero).sum(axis=1),
            zero,
        ).sum()
        num_connections_norm = num_connections / jnp.array(
            params.link_resources, dtype=dtype_config.LARGE_FLOAT_DTYPE
        )
        # Mean power of connections on path
        # make path with row length equal to link_resource (+1 to avoid zero division)
        mean_power_norm = jnp.where(
            path == one, state.channel_power_array.sum(axis=1), zero
        ).sum() / (jnp.where(num_connections > zero, num_connections, one) * params.max_power)
        # Mean SNR of connections on the path links
        max_snr = jnp.array(
            50, dtype=dtype_config.LARGE_FLOAT_DTYPE
        )  # Nominal value for max GSNR in dB
        mean_snr_norm = jnp.where(path == one, lightpath_snr_array.sum(axis=1), zero).sum(
            promote_integers=False
        ) / (jnp.where(num_connections > zero, num_connections, one) * max_snr)
        return jax.lax.dynamic_update_slice(
            init_val,
            jnp.array(
                [
                    [
                        path_length,
                        path_length_hops_norm,
                        num_connections_norm,
                        mean_power_norm,
                        mean_snr_norm,
                    ]
                ]
            ),
            (k_path_index, 0),
        )

    gn_path_stats = jnp.zeros((params.k_paths, 5), dtype=dtype_config.LARGE_FLOAT_DTYPE)
    gn_path_stats = jax.lax.fori_loop(0, params.k_paths, calculate_gn_path_stats, gn_path_stats)
    all_stats = jnp.concatenate([path_stats, gn_path_stats], axis=1)
    return jnp.concatenate(
        (
            jnp.array([source]),
            requested_datarate / 100.0,
            jnp.array([dest]),
            jnp.reshape(state.holding_time, (-1,)),
            jnp.reshape(all_stats, (-1,)),
        ),
        axis=0,
    )

    # TODO - instead of recalculating SNR for each slot, interpolate between SNR values
    # This approach instead tries every slot, calculates the SNR and therefore best mod. format,
    # def check_snr_get_mod_return_slots(initial_slot_index):
    #     # Get acceptable modulation formats
    #     temp_state = state.replace(
    #         channel_centre_bw_array=set_path_links(state.channel_centre_bw_array, path, initial_slot_index, 1, required_bandwidth),
    #         channel_power_array=set_path_links(state.channel_power_array, path, initial_slot_index, 1, launch_power),
    #     )
    #     mod_format_indices = get_best_modulation_format_simple(temp_state, path, initial_slot_index, launch_power, params)
    #     mod_format_index = jnp.max(mod_format_indices)  # -1 if no suitable mod format due to low SNR
    #     se = params.modulations_array.val[mod_format_index][1]
    #     req_slots = required_slots(requested_datarate, se, params.slot_size, guardband=params.guardband)[0]
    #     options = jnp.array([mod_format_index, req_slots])
    #     req_slots = options[jnp.argmin(jnp.array([mod_format_index*1000, req_slots]))]  # will always return -1 if SNR is too low, else required slots
    #     return req_slots.astype(jnp.float32)
    #
    # slot_indices = jnp.arange(params.link_resources)
    # req_slots_path = jax.vmap(check_snr_get_mod_return_slots)(slot_indices)
    # # Add padding
    # req_slots_path = jnp.concatenate((req_slots_path, jnp.full((params.max_slots,), -1)))
    # # Contains +ve integer for required slots, -1 if slot is occupied or SNR too low
    # req_slots_path = jnp.where(
    #     slots == 1,
    #     req_slots_path,
    #     slots
    # )
    #
    # # Make all request masks in advance, then use a scan to apply them to slots
    # xs = jnp.vstack((req_slots_path, jnp.arange(req_slots_path.shape[0]))).T
    # masks = jax.vmap(lambda x: get_request_mask(x, params))(jnp.arange(params.max_slots))
    #
    # def scan_fn(carry, x):
    #     req_slots = x[0].astype(jnp.int32)
    #     index = x[1].astype(jnp.int32)
    #     mask = masks[req_slots, :]
    #     sum = jnp.sum(mask * jax.lax.dynamic_slice(slots, (index,), (mask.shape[0],)))
    #     y = (sum == req_slots) & (sum > 0)
    #     return carry, y.astype(jnp.float32)
    #
    # _, path_mask = jax.lax.scan(scan_fn, slots, xs, length=slots.shape[0])
    #
    # # Cut off padding
    # path_mask = jax.lax.dynamic_slice(path_mask, (0,), (params.link_resources,))

    # Check that existing lightpaths are ok, set 0 if not
    # def apply_action_check_snr(state, params, initial_slot_index):
    #     req_slots_val = req_slots_path[initial_slot_index]  # -1 if None required
    #     mod_format_index = (jnp.floor(requested_datarate / (req_slots_val * params.slot_size)) - 1)
    #     mod_format_index = jnp.max(jnp.concatenate([jnp.zeros(1), mod_format_index])).astype(jnp.int32)  # Ensure mod_format_index is not negative
    #     launch_power = get_launch_power(state, i, state.launch_power_array[0], params)
    #     # TODO - fix this >>>>
    #     required_bandwidth = requested_datarate
    #     temp_state = state.replace(
    #         #path_index_array=set_path_links(state.path_index_array, path, initial_slot_index, 1, lightpath_index),
    #         active_path_array=set_path_links(state.active_path_array, path, initial_slot_index, 1, path),
    #         channel_centre_bw_array=set_path_links(state.channel_centre_bw_array, path, initial_slot_index, 1, required_bandwidth),
    #         channel_power_array=set_path_links(state.channel_power_array, path, initial_slot_index, 1, launch_power),
    #         modulation_format_index_array=set_path_links(state.modulation_format_index_array, path, initial_slot_index, 1, mod_format_index),
    #     )
    #     # check_snr_sufficient returns False if sufficient, so need to invert
    #     check = jnp.logical_not(check_snr_sufficient(temp_state, params)).astype(jnp.float32)
    #     return check
    #
    # Check SNR impact on updating every channel
    # snr_check_mask = jax.vmap(apply_action_check_snr, in_axes=(None, None, 0))(state, params, slot_indices)
    # Combine SNR impact mask only on slots where channels could be assigned
    # path_mask = jnp.where(path_mask == 1, snr_check_mask, path_mask)
