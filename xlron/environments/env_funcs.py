from itertools import combinations, islice
from functools import partial
from typing import Sequence, Union, Optional, Tuple
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper
from absl import flags
import math
import pathlib
import itertools
import networkx as nx
import jax.numpy as jnp
import chex
import jax
import timeit
import json
import numpy as np
import jraph
from flax import struct
from jax._src import core
from jax._src import dtypes
from jax._src import prng
from jax._src.typing import Array, ArrayLike, DTypeLike
from typing import Generic, TypeVar
from collections import defaultdict
from xlron.environments.dataclasses import *


Shape = Sequence[int]
T = TypeVar('T')      # Declare type variable


@partial(jax.jit, static_argnums=(1,))
def init_graph_tuple(state: EnvState, params: EnvParams):
    """Initialise graph tuple for use with Jraph GNNs.
    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
    Returns:
        jraph.GraphsTuple: Graph tuple
    """
    senders = params.edges.val.T[0]  # .val because of HashableArrayWrapper
    receivers = params.edges.val.T[1]
    edge_features = state.link_slot_array
    if not params.directed_graph:
        # TODO - investigate just using senders or receivers to avoid duplication for undirected graphs
        senders_undir = jnp.concatenate((senders, receivers))
        receivers_undir = jnp.concatenate((receivers, senders))
        senders = senders_undir
        receivers = receivers_undir
        # Repeat every row of link_slot_array so that it matches length of senders/receivers
        edge_features = jnp.repeat(edge_features, 2, axis=0)
    # Get node features from node_capacity_array if available (VONE problem)
    node_features = getattr(state, "node_capacity_array", jnp.zeros(params.num_nodes))
    node_features = node_features.reshape(-1, 1)
    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=jnp.reshape(params.num_nodes, (1,)),
        n_edge=jnp.reshape(jnp.array(len(senders)), (1,)),
        # Store current request as global feature and reshape so that downstream processed graph features have same shape
        globals=jnp.reshape(state.request_array, (1, -1)),
    )
    return graph


def update_graph_tuple(state: EnvState, params: EnvParams):
    """Update graph tuple for use with Jraph GNNs.
    Edge and node features are updated from link_slot_array and node_capacity_array respectively.
    Global features are updated as request_array.
    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
    Returns:
        state (EnvState): Environment state with updated graph tuple
    """
    edge_features = state.link_slot_array if params.directed_graph else jnp.repeat(state.link_slot_array, 2, axis=0)
    node_features = getattr(state, "node_capacity_array", jnp.zeros(params.num_nodes))
    node_features = node_features.reshape(-1, 1)
    graph = state.graph._replace(nodes=node_features, edges=edge_features, globals=state.request_array)
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
        link_lengths.append(graph.edges[edge]["weight"])
    return jnp.array(link_lengths)


def init_path_link_array(
        graph: nx.Graph,
        k: int,
        disjoint: bool = False,
        weight: str = "weight",
        directed: bool = False,
        modulations_array: chex.Array = None,
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
        weight (str, optional): Sort paths by edge attribute. Defaults to "weight".
        directed (bool, optional): Whether graph is directed. Defaults to False.
        modulations_array (chex.Array, optional): Array of maximum spectral efficiency for modulation format on path. Defaults to None.
        rwa_lr (bool, optional): Whether the environment is RWA with lightpath reuse (affects path ordering).
        path_snr (bool, optional): If GN model is used, include extra row of zeroes for unutilised paths
        to ensure correct SNR calculation for empty paths (path index -1).

    Returns:
        chex.Array: Path-link array (N(N-1)*k x E) where N is number of nodes, E is number of edges, k is number of shortest paths
    """
    def get_k_shortest_paths(g, source, target, k, weight):
        return list(
            islice(nx.shortest_simple_paths(g, source, target, weight=weight), k)
        )

    def get_k_disjoint_shortest_paths(g, source, target, k, weight):
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

        # Sort the paths by # of hops then by length, or just length
        path_lengths = [nx.path_weight(graph, path, weight='weight') for path in k_paths]
        path_num_links = [len(path) - 1 for path in k_paths]

        # Get maximum spectral efficiency for modulation format on path
        if modulations_array is not None and rwa_lr is not True:
            se_of_path = []
            modulations_array = modulations_array[::-1]
            for length in path_lengths:
                for modulation in modulations_array:
                    if length <= modulation[0]:
                        se_of_path.append(modulation[1])
                        break
            # Sorting by the num_links/se instead of just path length is observed to improve performance
            path_weighting = [num_links/se for se, num_links in zip(se_of_path, path_num_links)]
        elif rwa_lr:
            path_capacity = [float(calculate_path_capacity(path_length, scale_factor=scale_factor)) for path_length in path_lengths]
            path_weighting = [num_links/path_capacity for num_links, path_capacity in zip(path_num_links, path_capacity)]
        elif weight is None:
            path_weighting = path_num_links
        else:
            path_weighting = path_lengths

        # if less then k unique paths, add empty paths
        empty_path = [0] * len(graph.edges)
        k_paths = k_paths + [empty_path] * (k - len(k_paths))
        path_weighting = path_weighting + [1e6] * (k - len(path_weighting))
        path_lengths = path_lengths + [1e6] * (k - len(path_lengths))

        # Sort by number of links then by length (or just by length if weight is specified)
        unsorted_paths = zip(k_paths, path_weighting, path_lengths)
        k_paths_sorted = [(source, dest, weighting, path) for path, weighting, _ in sorted(unsorted_paths, key=lambda x: (x[1], 1/x[2]) if weight is None else x[2])]

        # Keep only first k paths
        k_paths_sorted = k_paths_sorted[:k]

        prev_link_usage = empty_path
        for k_path in k_paths_sorted:
            k_path = k_path[-1]
            link_usage = [0]*len(graph.edges)  # Initialise empty path
            if sum(k_path) == 0:
                link_usage = prev_link_usage
            else:
                for i in range(len(k_path)-1):
                    s, d = k_path[i], k_path[i + 1]
                    for edge_index, edge in enumerate(edges):
                        condition = (edge[0] == s and edge[1] == d) if directed else \
                            ((edge[0] == s and edge[1] == d) or (edge[0] == d and edge[1] == s))
                        if condition:
                            link_usage[edge_index] = 1
            path = link_usage
            prev_link_usage = link_usage
            paths.append(path)

    # If using GN model, add extra row of zeroes for empty paths for SNR calculation
    if path_snr:
        empty_path = [0] * len(graph.edges)
        paths.append(empty_path)

    return jnp.array(paths, dtype=jnp.float32)


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


def init_modulations_array(modulations_filepath: str = None):
    """Initialise array of maximum spectral efficiency for modulation format on path.

    Args:
        modulations_filepath (str, optional): Path to CSV file containing modulation formats. Defaults to None.
    Returns:
        jnp.array: Array of maximum spectral efficiency for modulation format on path.
        First two columns are maximum path length and spectral efficiency.
    """
    f = pathlib.Path(modulations_filepath) if modulations_filepath else (
            pathlib.Path(__file__).parents[2].absolute() / "examples" / "modulations.csv")
    modulations = np.genfromtxt(f, delimiter=',')
    # Drop empty first row (headers) and column (name)
    modulations = modulations[1:, 1:]
    return jnp.array(modulations)


def init_path_se_array(path_length_array, modulations_array):
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
    return jnp.array(se_list)


def init_virtual_topology_patterns(pattern_names: str) -> chex.Array:
    """Initialise virtual topology patterns.
    First 3 digits comprise the "action counter": first index is num unique nodes, second index is total steps,
    final is remaining steps until completion of request.
    Remaining digits define the topology pattern, with 1 to indicate links and other positive integers are node indices.

    Args:
        pattern_names (list): List of virtual topology pattern names

    Returns:
        chex.Array: Array of virtual topology patterns
    """
    patterns = []
    # TODO - Allow 2 node requests in VONE (check if any modifications necessary other than below)
    #if "2_bus" in pattern_names:
    #    patterns.append([2, 1, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    if "3_bus" in pattern_names:
        patterns.append([3, 2, 2, 2, 1, 3, 1, 4])
    if "3_ring" in pattern_names:
        patterns.append([3, 3, 3, 2, 1, 3, 1, 4, 1, 2])
    if "4_bus" in pattern_names:
        patterns.append([4, 3, 3, 2, 1, 3, 1, 4, 1, 5])
    if "4_ring" in pattern_names:
        patterns.append([4, 4, 4, 2, 1, 3, 1, 4, 1, 5, 1, 2])
    if "5_bus" in pattern_names:
        patterns.append([5, 4, 4, 2, 1, 3, 1, 4, 1, 5, 1, 6])
    if "5_ring" in pattern_names:
        patterns.append([5, 5, 5, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 2])
    if "6_bus" in pattern_names:
        patterns.append([6, 5, 5, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7])
    max_length = max([len(pattern) for pattern in patterns])
    # Pad patterns with zeroes to match longest
    for pattern in patterns:
        pattern.extend([0]*(max_length-len(pattern)))
    return jnp.array(patterns, dtype=jnp.int32)


@partial(jax.jit, static_argnums=(1,))
def init_traffic_matrix(key: chex.PRNGKey, params: EnvParams):
    """Initialize traffic matrix. Allows for random traffic matrix or uniform traffic matrix.
    Source-dest traffic requests are sampled probabilistically from the resulting traffic matrix.

    Args:
        key (chex.PRNGKey): PRNG key
        params (EnvParams): Environment parameters

    Returns:
        jnp.array: Traffic matrix
    """
    if params.random_traffic:
        traffic_matrix = jax.random.uniform(key, shape=(params.num_nodes, params.num_nodes))
    else:
        traffic_matrix = jnp.ones((params.num_nodes, params.num_nodes))
    diag_elements = jnp.diag_indices_from(traffic_matrix)
    # Set main diagonal to zero so no requests from node to itself
    traffic_matrix = traffic_matrix.at[diag_elements].set(0)
    traffic_matrix = normalise_traffic_matrix(traffic_matrix)
    return traffic_matrix


def init_values_nodes(min_value, max_value):
    return jnp.arange(min_value, max_value+1)


def init_values_slots(min_value, max_value):
    return jnp.arange(min_value, max_value+1)


# TODO - allow bandwidths to be selected with a specified probability
def init_values_bandwidth(min_value: int = 25, max_value: int = 100, step: int = 1, values: int = None) -> chex.Array:
    if values:
        return jnp.array(values)
    else:
        return jnp.arange(min_value, max_value+1, step)


@partial(jax.jit, static_argnums=(2, 3))
def get_path_indices(s: int, d: int, k: int, N: int, directed: bool = False) -> chex.Array:
    """Get path indices for a given source, destination and number of paths.
    If source > destination and the graph is directed (two fibres per link, one in each direction) then an offset is
    added to the index to get the path in the other direction (the offset is the total number source-dest pairs).

    Args:
        s (int): Source node index
        d (int): Destination node index
        k (int): Number of paths
        N (int): Number of nodes
        directed (bool, optional): Whether graph is directed. Defaults to False.

    Returns:
        jnp.array: Start index on path-link array for candidate paths
    """
    node_indices = jnp.arange(N, dtype=jnp.int32)
    indices_to_s = jnp.where(node_indices < s, node_indices, 0)
    indices_to_d = jnp.where(node_indices < d, node_indices, 0)
    # If two fibres per link, add offset to index to get fibre in other direction if source > destination
    directed_offset = directed * (s > d) * N * (N - 1) * k / 2
    # The following equation is based on the combinations formula
    forward = ((N * s + d - jnp.sum(indices_to_s) - 2 * s - 1) * k)
    backward = ((N * d + s - jnp.sum(indices_to_d) - 2 * d - 1) * k)
    return forward * (s < d) + backward * (s > d) + directed_offset


@partial(jax.jit, static_argnums=(0,))
def init_node_capacity_array(params: EnvParams):
    """Initialize node array with uniform resources.
    Args:
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: Node capacity array (N x 1) where N is number of nodes"""
    return jnp.array([params.node_resources] * params.num_nodes, dtype=jnp.float32)


@partial(jax.jit, static_argnums=(0,))
def init_link_slot_array(params: EnvParams):
    """Initialize empty (all zeroes) link-slot array.
    Args:
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: Link slot array (E x S) where E is number of edges and S is number of slots"""
    return jnp.zeros((params.num_links, params.link_resources))


# TODO(MULTIBAND) - Add function(s) to initialise multi-band array(s)
@partial(jax.jit, static_argnums=(0,))
def init_link_slot_array_multiband(params: EnvParams):
    init_array = jnp.zeros((params.num_links, params.link_resources))
    # Mask column at bandgap
    bandgap_slots = int(params.bandgap / params.slot_size)
    bandgap_start_slots = int(params.bandgap_start / params.slot_size)
    result = init_array.at[:,bandgap_start_slots:bandgap_start_slots+bandgap_slots].set(-1)
    jax.debug.print("init_link_slot_array_multiband {}", result, ordered=True)
    return result


@partial(jax.jit, static_argnums=(0,))
def init_vone_request_array(params: EnvParams):
    """Initialize request array either with uniform resources"""
    return jnp.zeros((2, params.max_edges*2+1, ))


def init_rsa_request_array():
    """Initialize request array"""
    return jnp.zeros(3)


@partial(jax.jit, static_argnums=(0,))
def init_node_mask(params: EnvParams):
    """Initialize node mask"""
    return jnp.ones(params.num_nodes)


@partial(jax.jit, static_argnums=(0, 1))
def init_link_slot_mask(params: EnvParams, agg: int = 1):
    """Initialize link mask"""
    return jnp.ones(params.k_paths*math.ceil(params.link_resources / agg))


def init_action_counter():
    """Initialize action counter.
    First index is num unique nodes, second index is total steps, final is remaining steps until completion of request.
    Only used in VONE environments.
    """
    return jnp.zeros(3, dtype=jnp.int32)


def decrement_action_counter(state):
    """Decrement action counter in-place. Used in VONE environments."""
    state.action_counter.at[-1].add(-1)
    return state


def decrease_last_element(array):
    last_value_mask = jnp.arange(array.shape[0]) == array.shape[0] - 1
    return jnp.where(last_value_mask, array - 1, array)


@partial(jax.jit, static_argnums=(0,))
def init_node_departure_array(params: EnvParams):
    return jnp.full((params.num_nodes, params.node_resources), jnp.inf)


@partial(jax.jit, static_argnums=(0,))
def init_link_slot_departure_array(params: EnvParams):
    return jnp.zeros((params.num_links, params.link_resources))


@partial(jax.jit, static_argnums=(0,))
def init_node_resource_array(params: EnvParams):
    """Array to track node resources occupied by virtual nodes"""
    return jnp.zeros((params.num_nodes, params.node_resources), dtype=jnp.float32)


@partial(jax.jit, static_argnums=(0,))
def init_action_history(params: EnvParams):
    """Initialize action history"""
    return jnp.full(params.max_edges*2+1, -1)


def normalise_traffic_matrix(traffic_matrix):
    """Normalise traffic matrix to sum to 1"""
    traffic_matrix /= jnp.sum(traffic_matrix)
    return traffic_matrix


@partial(jax.jit, static_argnums=(2,3))
def required_slots(bit_rate: float, se: int, channel_width: float, guardband: int = 1) -> int:
    """Calculate required slots for a given bitrate and spectral efficiency.

    Args:
        bit_rate (float): Bit rate in Gbps
        se (float): Spectral efficiency in bps/Hz
        channel_width (float): Channel width in GHz
        guardband (int, optional): Guard band. Defaults to 1.

    Returns:
        int: Required slots
    """
    return jnp.int32(jnp.ceil(bit_rate/(se*channel_width))+guardband)


@partial(jax.jit, static_argnums=(2,))
def generate_vone_request(key: chex.PRNGKey, state: EnvState, params: EnvParams):
    """Generate a new request for the VONE environment.
    The request has two rows. The first row shows the node and slot values.
    The first three elements of the second row show the number of unique nodes, the total number of steps, and the remaining steps.
    These first three elements comprise the action counter.
    The remaining elements of the second row show the virtual topology pattern, i.e. the connectivity of the virtual topology.
    """
    shape = params.max_edges*2+1  # shape of request array
    key_topology, key_node, key_slot, key_times = jax.random.split(key, 4)
    # Randomly select topology, node resources, slot resources
    pattern = jax.random.choice(key_topology, state.virtual_topology_patterns)
    action_counter = jax.lax.dynamic_slice(pattern, (0,), (3,))
    topology_pattern = jax.lax.dynamic_slice(pattern, (3,), (pattern.shape[0]-3,))
    selected_node_values = jax.random.choice(key_node, state.values_nodes, shape=(shape,))
    selected_bw_values = jax.random.choice(key_slot, params.values_bw.val, shape=(shape,))
    # Create a mask for odd and even indices
    mask = jnp.tile(jnp.array([0, 1]), (shape+1) // 2)[:shape]
    # Vectorized conditional replacement using mask
    first_row = jnp.where(mask, selected_bw_values, selected_node_values)
    # Make sure node request values are consistent for same virtual nodes
    first_row = jax.lax.fori_loop(
        2,  # Lowest node index in virtual topology requests is 2
        shape,  # Highest possible node index in virtual topology requests is shape-1
        lambda i, x: jnp.where(topology_pattern == i, selected_node_values[i], x),
        first_row
    )
    # Mask out unused part of request array
    first_row = jnp.where(topology_pattern == 0, 0, first_row)
    # Set times
    arrival_time, holding_time = generate_arrival_holding_times(key, params)
    state = state.replace(
        holding_time=holding_time,
        current_time=state.current_time + arrival_time,
        action_counter=action_counter,
        request_array=jnp.vstack((first_row, topology_pattern)),
        action_history=init_action_history(params),
        total_requests=state.total_requests + 1
    )
    state = remove_expired_node_requests(state) if not params.incremental_loading else state
    state = remove_expired_services_rsa(state) if not params.incremental_loading else state
    return state


@partial(jax.jit, static_argnums=(2,))
def generate_request_rsa(key: chex.PRNGKey, state: EnvState, params: EnvParams) -> EnvState:
    # Flatten the probabilities to a 1D array
    key_sd, key_slot, key_times = jax.random.split(key, 3)
    if params.deterministic_requests:
        request = params.list_of_requests[state.total_requests]
        source = jax.lax.dynamic_slice(request, (0,), (1,))[0]
        bw = jax.lax.dynamic_slice(request, (1,), (1,))[0]
        dest = jax.lax.dynamic_slice(request, (2,), (1,))[0]
    else:
        shape = state.traffic_matrix.shape
        probabilities = state.traffic_matrix.ravel()
        # Use jax.random.choice to select index based on the probabilities
        source_dest_index = jax.random.choice(key_sd, jnp.arange(state.traffic_matrix.size), p=probabilities)
        # Convert 1D index back to 2D
        nodes = jnp.unravel_index(source_dest_index, shape)
        source, dest = jnp.stack(nodes) if params.directed_graph else jnp.sort(jnp.stack(nodes))
        # Vectorized conditional replacement using mask
        bw = jax.random.choice(key_slot, params.values_bw.val)
    arrival_time, holding_time = generate_arrival_holding_times(key_times, params)
    state = state.replace(
        holding_time=holding_time,
        current_time=state.current_time + arrival_time,
        request_array=jnp.stack((source, bw, dest)),
        total_requests=state.total_requests + 1
    )
    # Removal of expired services is different for RWA-LR
    remove_expired_services = remove_expired_services_rsa
    if params.__class__.__name__ == "RWALightpathReuseEnvParams":
        state = state.replace(time_since_last_departure=state.time_since_last_departure + arrival_time)
        remove_expired_services = remove_expired_services_rwalr
    state = remove_expired_services(state) if not params.incremental_loading else state
    return state


@partial(jax.jit, static_argnums=(2,))
def generate_request_rwalr(key: chex.PRNGKey, state: EnvState, params: EnvParams) -> EnvState:
    # Flatten the probabilities to a 1D array
    key_sd, key_slot, key_times = jax.random.split(key, 3)
    if params.deterministic_requests:
        request = params.list_of_requests[state.total_requests]
        source = jax.lax.dynamic_slice(request, (0,), (1,))[0]
        bw = jax.lax.dynamic_slice(request, (1,), (1,))[0]
        dest = jax.lax.dynamic_slice(request, (2,), (1,))[0]
    else:
        shape = state.traffic_matrix.shape
        probabilities = state.traffic_matrix.ravel()
        # Use jax.random.choice to select index based on the probabilities
        source_dest_index = jax.random.choice(key_sd, jnp.arange(state.traffic_matrix.size), p=probabilities)
        # Convert 1D index back to 2D
        nodes = jnp.unravel_index(source_dest_index, shape)
        source, dest = jnp.stack(nodes) if params.directed_graph else jnp.sort(jnp.stack(nodes))
        # Vectorized conditional replacement using mask
        bw = jax.random.choice(key_slot, params.values_bw.val)
    arrival_time, holding_time = generate_arrival_holding_times(key_times, params)
    state = state.replace(
        holding_time=holding_time,
        current_time=state.current_time + arrival_time,
        request_array=jnp.stack((source, bw, dest)),
        total_requests=state.total_requests + 1,
        time_since_last_departure=state.time_since_last_departure + arrival_time
    )
    # Removal of expired services is different for RWA-LR
    remove_expired_services = remove_expired_services_rsa
    if params.__class__.__name__ == "RWALightpathReuseEnvParams":
        state = state.replace(time_since_last_departure=state.time_since_last_departure + arrival_time)
        remove_expired_services = remove_expired_services_rwalr
    state = remove_expired_services(state) if not params.incremental_loading else state
    return state


@partial(jax.jit, static_argnums=(0,))
def get_path_index_array(params, nodes):
    """Indices of paths between source and destination from path array"""
    # get source and destination nodes in order (for accurate indexing of path-link array)
    source, dest = nodes
    i = get_path_indices(source, dest, params.k_paths, params.num_nodes, directed=params.directed_graph).astype(jnp.int32)
    index_array = jax.lax.dynamic_slice(jnp.arange(0, params.path_link_array.shape[0]), (i,), (params.k_paths,))
    return index_array


@partial(jax.jit, static_argnums=(0,))
def get_paths(params, nodes):
    """Get k paths between source and destination"""
    index_array = get_path_index_array(params, nodes)
    return jnp.take(params.path_link_array.val, index_array, axis=0)


@partial(jax.jit, static_argnums=(0,))
def get_paths_se(params, nodes):
    """Get max. spectral efficiency of modulation format on k paths between source and destination"""
    # get source and destination nodes in order (for accurate indexing of path-link array)
    index_array = get_path_index_array(params, nodes)
    return jnp.take(params.path_se_array.val, index_array, axis=0)


@partial(jax.jit, static_argnums=(1, 2, 3))
def poisson(key: Union[Array, prng.PRNGKeyArray],
            lam: ArrayLike,
            shape: Shape = (),
            dtype: DTypeLike = dtypes.float_) -> Array:
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
        raise ValueError(f"dtype argument to `exponential` must be a float "
                       f"dtype, got {dtype}")
    dtype = dtypes.canonicalize_dtype(dtype)
    shape = core.canonicalize_shape(shape)
    return _poisson(key, lam, shape, dtype)


@partial(jax.jit, static_argnums=(1, 2, 3))
def _poisson(key, lam, shape, dtype) -> Array:
    jax._src.random._check_shape("exponential", shape)
    u = jax.random.uniform(key, shape, dtype)
    # taking 1 - u to move the domain of log to (0, 1] instead of [0, 1)
    return jax.lax.div(jax.lax.neg(jax.lax.log1p(jax.lax.neg(u))), lam)


@partial(jax.jit, static_argnums=(1,))
def generate_arrival_holding_times(key, params):
    """
    Generate arrival and holding times based on Poission distirbuted events.
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
    arrival_time = jax.random.exponential(key_arrival, shape=(1,)) \
                   / params.arrival_rate  # Divide because it is rate (lambda)
    if params.truncate_holding_time:
        # For DeepRMSA, need to generate holding times that are less than 2*mean_service_holding_time
        key_holding = jax.random.split(key, 5)
        holding_times = jax.vmap(lambda x: jax.random.exponential(x, shape=(1,)) \
                                * params.mean_service_holding_time)(key_holding)
        holding_times = jnp.where(holding_times < 2*params.mean_service_holding_time, holding_times, 0)
        # Get first non-zero value in holding_times
        non_zero_index = jnp.nonzero(holding_times, size=1)[0][0]
        holding_time = jax.lax.dynamic_slice(jnp.squeeze(holding_times), (non_zero_index,), (1,))
    else:
        holding_time = jax.random.exponential(key_holding, shape=(1,)) \
                       * params.mean_service_holding_time  # Multiply because it is mean (1/lambda)
    return arrival_time, holding_time


def update_action_history(action_history: chex.Array, action_counter: chex.Array, action: chex.Array) -> chex.Array:
    """Update action history by adding action to first available index starting from the end.

    Args:
        action_history: Action history
        action_counter: Action counter
        action: Action to add to history

    Returns:
        Updated action_history
    """
    return jax.lax.dynamic_update_slice(action_history, jnp.flip(action, axis=0).astype(jnp.int32), ((action_counter[-1]-1)*2,))


def update_link(link, initial_slot, num_slots, value):
    slot_indices = jnp.arange(link.shape[0])
    return jnp.where((initial_slot <= slot_indices) & (slot_indices < initial_slot+num_slots), link-value, link)


def update_path(link, link_in_path, initial_slot, num_slots, value):
    return jax.lax.cond(link_in_path == 1, lambda x: update_link(*x), lambda x: x[0], (link, initial_slot, num_slots, value))


@jax.jit
def vmap_update_path_links(link_slot_array: chex.Array, path: chex.Array, initial_slot: int, num_slots: int, value: int) -> chex.Array:
    """Update relevant slots along links in path to current_val - val.

    Args:
        link_slot_array: Link slot array
        path: Path (row from path-link array that indicates links used by path)
        initial_slot: Initial slot
        num_slots: Number of slots
        value: Value to subtract from link slot array

    Returns:
        Updated link slot array
    """
    return jax.vmap(update_path, in_axes=(0, 0, None, None, None))(link_slot_array, path, initial_slot, num_slots, value)


def set_link(link, initial_slot, num_slots, value):
    slot_indices = jnp.arange(link.shape[0])
    return jnp.where((initial_slot <= slot_indices) & (slot_indices < initial_slot+num_slots), value, link)


def set_path(link, link_in_path, initial_slot, num_slots, value):
    return jax.lax.cond(link_in_path == 1, lambda x: set_link(*x), lambda x: x[0], (link, initial_slot, num_slots, value))


@jax.jit
def vmap_set_path_links(link_slot_array: chex.Array, path: chex.Array, initial_slot: int, num_slots: int, value: int) -> chex.Array:
    """Set relevant slots along links in path to val.

    Args:
        link_slot_array: Link slot array
        path: Path (row from path-link array that indicates links used by path)
        initial_slot: Initial slot
        num_slots: Number of slots
        value: Value to set on link slot array

    Returns:
        Updated link slot array
    """
    return jax.vmap(set_path, in_axes=(0, 0, None, None, None))(link_slot_array, path, initial_slot, num_slots, value)


def update_node_departure(node_row, inf_index, value):
    row_indices = jnp.arange(node_row.shape[0])
    return jnp.where(row_indices == inf_index, value, node_row)


def update_selected_node_departure(node_row, node_selected, first_inf_index, value):
    return jax.lax.cond(node_selected != 0, lambda x: update_node_departure(*x), lambda x: node_row, (node_row, first_inf_index, value))


@jax.jit
def vmap_update_node_departure(node_departure_array: chex.Array, selected_nodes: chex.Array, value: int) -> chex.Array:
    """Called when implementing node action.
    Sets request departure time ("value") in place of first "inf" i.e. unoccupied index on node departure array for selected nodes.

    Args:
        node_departure_array: (N x R) Node departure array
        selected_nodes: (N x 1) Selected nodes (non-zero value on selected node indices)
        value: Value to set on node departure array

    Returns:
        Updated node departure array
    """
    first_inf_indices = jnp.argmax(node_departure_array, axis=1)
    return jax.vmap(update_selected_node_departure, in_axes=(0, 0, 0, None))(node_departure_array, selected_nodes, first_inf_indices, value)


def update_node_resources(node_row, zero_index, value):
    row_indices = jnp.arange(node_row.shape[0])
    return jnp.where(row_indices == zero_index, value, node_row)


def update_selected_node_resources(node_row, request, first_zero_index):
    return jax.lax.cond(request != 0, lambda x: update_node_resources(*x), lambda x: node_row, (node_row, first_zero_index, request))


@jax.jit
def vmap_update_node_resources(node_resource_array, selected_nodes):
    """Called when implementing node action.
    Sets requested node resources on selected nodes in place of first "zero" i.e.
    unoccupied index on node resource array for selected nodes.

    Args:
        node_resource_array: (N x R) Node resource array
        selected_nodes: (N x 1) Requested resources on selected nodes

    Returns:
        Updated node resource array
    """
    first_zero_indices = jnp.argmin(node_resource_array, axis=1)
    return jax.vmap(update_selected_node_resources, in_axes=(0, 0, 0))(node_resource_array, selected_nodes, first_zero_indices)


def update_node_array(node_indices, array, node, request):
    """Used to udated selected_nodes array with new requested resources on each node, for use in """
    return jnp.where(node_indices == node, array-request, array)


def remove_expired_services_rsa(state: EnvState) -> EnvState:
    """Check for values in link_slot_departure_array that are less than the current time but greater than zero
    (negative time indicates the request is not yet finalised).
    If found, set to zero in link_slot_array and link_slot_departure_array.

    Args:
        state: Environment state

    Returns:
        Updated environment state
    """
    mask = jnp.where(state.link_slot_departure_array < jnp.squeeze(state.current_time), 1, 0)
    mask = jnp.where(0 <= state.link_slot_departure_array, mask, 0)
    state = state.replace(
        link_slot_array=jnp.where(mask == 1, 0, state.link_slot_array),
        link_slot_departure_array=jnp.where(mask == 1, 0, state.link_slot_departure_array)
    )
    return state


def remove_expired_services_rwalr(state: EnvState) -> EnvState:
    """

    Args:
        state: Environment state

    Returns:
        Updated environment state
    """
    mask = jnp.where(state.link_slot_departure_array < jnp.squeeze(state.current_time), 1, 0)
    mask = jnp.where(0 <= state.link_slot_departure_array, mask, 0)
    state = state.replace(
        link_slot_array=jnp.where(mask == 1, 0, state.link_slot_array),
        link_slot_departure_array=jnp.where(mask == 1, 0, state.link_slot_departure_array)
    )
    return state


def remove_expired_node_requests(state: EnvState) -> EnvState:
    """Check for values in node_departure_array that are less than the current time but greater than zero
    (negative time indicates the request is not yet finalised).
    If found, set to infinity in node_departure_array, set to zero in node_resource_array, and increase
    node_capacity_array by expired resources on each node.

    Args:
        state: Environment state

    Returns:
        Updated environment state
    """
    mask = jnp.where(state.node_departure_array < jnp.squeeze(state.current_time), 1, 0)
    mask = jnp.where(0 < state.node_departure_array, mask, 0)
    expired_resources = jnp.sum(jnp.where(mask == 1, state.node_resource_array, 0), axis=1)
    state = state.replace(
        node_capacity_array=state.node_capacity_array + expired_resources,
        node_resource_array=jnp.where(mask == 1, 0, state.node_resource_array),
        node_departure_array=jnp.where(mask == 1, jnp.inf, state.node_departure_array)
    )
    return state


@partial(jax.jit, donate_argnums=(0,))
def undo_node_action(state: EnvState) -> EnvState:
    """If the request is unsuccessful i.e. checks fail, then remove the partial (unfinalised) resource allocation.
    Partial resource allocation is indicated by negative time in node departure array.
    Check for values in node_departure_array that are less than zero.
    If found, set to infinity in node_departure_array, set to zero in node_resource_array, and increase
    node_capacity_array by expired resources on each node.

    Args:
        state: Environment state

    Returns:
        Updated environment state
    """
    # TODO - Check that node resource clash doesn't happen (so time is always negative after implementation)
    #  and undoing always succeeds with negative time
    mask = jnp.where(state.node_departure_array < 0, 1, 0)
    resources = jnp.sum(jnp.where(mask == 1, state.node_resource_array, 0), axis=1)
    state = state.replace(
        node_capacity_array=state.node_capacity_array + resources,
        node_resource_array=jnp.where(mask == 1, 0, state.node_resource_array),
        node_departure_array=jnp.where(mask == 1, jnp.inf, state.node_departure_array),
    )
    return state


@partial(jax.jit, donate_argnums=(0,))
def undo_action_rsa(state: EnvState) -> EnvState:
    """If the request is unsuccessful i.e. checks fail, then remove the partial (unfinalised) resource allocation.
    Partial resource allocation is indicated by negative time in link slot departure array.
    Check for values in link_slot_departure_array that are less than zero.
    If found, increase link_slot_array by +1 and link_slot_departure_array by current_time + holding_time of current request.

    Args:
        state: Environment state
        
    Returns:
        Updated environment state
    """
    # If departure array is negative, then undo the action
    mask = jnp.where(state.link_slot_departure_array < 0, 1, 0)
    # If link slot array is < -1, then undo the action
    # (departure might be positive because existing service had holding time after current)
    # e.g. (time_in_array = t1 - t2) where t2 < t1 and t2 = current_time + holding_time
    # but link_slot_array = -2 due to double allocation, so undo the action
    mask = jnp.where(state.link_slot_array < -1, 1, mask)
    state = state.replace(
        link_slot_array=jnp.where(mask == 1, state.link_slot_array+1, state.link_slot_array),
        link_slot_departure_array=jnp.where(
            mask == 1,
            state.link_slot_departure_array + state.current_time + state.holding_time,
            state.link_slot_departure_array),
        total_bitrate=state.total_bitrate + read_rsa_request(state.request_array)[1][0]
    )
    return state


@partial(jax.jit, donate_argnums=(0,))
def undo_action_rwalr(state: EnvState) -> EnvState:
    """If the request is unsuccessful i.e. checks fail, then remove the partial (unfinalised) resource allocation.
    Partial resource allocation is indicated by negative time in link slot departure array.
    Check for values in link_slot_departure_array that are less than zero.
    If found, increase link_slot_array by +1 and link_slot_departure_array by current_time + holding_time of current request.

    Args:
        state: Environment state

    Returns:
        Updated environment state
    """
    # If departure array is negative, then undo the action
    mask = jnp.where(state.link_slot_departure_array < 0, 1, 0)
    # If link slot array is < -1, then undo the action
    # (departure might be positive because existing service had holding time after current)
    # e.g. (time_in_array = t1 - t2) where t2 < t1 and t2 = current_time + holding_time
    # but link_slot_array = -2 due to double allocation, so undo the action
    mask = jnp.where(state.link_slot_array < -1, 1, mask)
    state = state.replace(
        link_slot_array=jnp.where(mask == 1, state.link_slot_array + 1, state.link_slot_array),
        link_slot_departure_array=jnp.where(
            mask == 1,
            state.link_slot_departure_array + state.current_time + state.holding_time,
            state.link_slot_departure_array),
        total_bitrate=state.total_bitrate + read_rsa_request(state.request_array)[1][0]
    )
    return state


@jax.jit
def check_unique_nodes(node_departure_array: chex.Array) -> bool:
    """Count negative values on each node (row) in node departure array, must not exceed 1.

    Args:
        node_departure_array: Node departure array (N x R) where N is number of nodes and R is number of resources

    Returns:
        bool: True if check failed, False if check passed
    """
    return jnp.any(jnp.sum(jnp.where(node_departure_array < 0, 1, 0), axis=1) > 1)


def check_all_nodes_assigned(node_departure_array: chex.Array, total_requested_nodes: int) -> bool:
    """Count negative values on each node (row) in node departure array, sum them, must equal total requested_nodes.

    Args:
        node_departure_array: Node departure array (N x R) where N is number of nodes and R is number of resources
        total_requested_nodes: Total requested nodes (int)

    Returns:
        bool: True if check failed, False if check passed
    """
    return jnp.sum(jnp.sum(jnp.where(node_departure_array < 0, 1, 0), axis=1)) != total_requested_nodes


def check_min_two_nodes_assigned(node_departure_array: chex.Array):
    """Count negative values on each node (row) in node departure array, sum them, must be 2 or greater.
    This check is important if e.g. an action contains 2 nodes the same therefore only assigns 1 node.
    Return False if check passed, True if check failed

    Args:
        node_departure_array: Node departure array (N x R) where N is number of nodes and R is number of resources

    Returns:
        bool: True if check failed, False if check passed
    """
    return jnp.sum(jnp.sum(jnp.where(node_departure_array < 0, 1, 0), axis=1)) <= 1


def check_node_capacities(capacity_array: chex.Array) -> bool:
    """Sum selected nodes array and check less than node resources.

    Args:
        capacity_array: Node capacity array (N x 1) where N is number of nodes

    Returns:
        bool: True if check failed, False if check passed
    """
    return jnp.any(capacity_array < 0)


def check_no_spectrum_reuse(link_slot_array):
    """slot-=1 when used, should be zero when unoccupied, so check if any < -1 in slot array.

    Args:
        link_slot_array: Link slot array (L x S) where L is number of links and S is number of slots

    Returns:
        bool: True if check failed, False if check passed
    """
    return jnp.any(link_slot_array < -1)


def check_topology(action_history, topology_pattern):
    """Check that each unique virtual node (as indicated by topology pattern) is assigned to a consistent physical node
    i.e. start and end node of ring is same physical node.
    Method:
    For each node index in topology pattern, mask action history with that index, then find max value in masked array.
    If max value is not the same for all values for that virtual node in action history, then return 1, else 0.
    Array should be all zeroes at the end, so do an any() check on that.
    e.g. virtual topology pattern = [2,1,3,1,4,1,2]  3 node ring
    action history = [0,34,4,0,3,1,0]
    meaning v node "2" goes p node 0, v node "3" goes p node 4, v node "4" goes p node 3
    The numbers in-between relate to the slot action.
    If any value in the array is 1, a virtual node is assigned to multiple different physical nodes.
    Need to check from both perspectives:
    1. For each virtual node, check that all physical nodes are the same
    2. For each physical node, check that all virtual nodes are the same

    Args:
        action_history: Action history
        topology_pattern: Topology pattern

    Returns:
        bool: True if check failed, False if check passed
    """
    def loop_func_virtual(i, val):
        # Get indices of physical node in action history that correspond to virtual node i
        masked_val = jnp.where(i == topology_pattern, val, -1)
        # Get maximum value at those indices (should all be same)
        max_node = jnp.max(masked_val)
        # For relevant indices, if max value then return 0 else 1
        val = jnp.where(masked_val != -1, masked_val != max_node, val)
        return val
    def loop_func_physical(i, val):
        # Get indices of virtual nodes in topology pattern that correspond to physical node i
        masked_val = jnp.where(i == action_history, val, -1)
        # Get maximum value at those indices (should all be same)
        max_node = jnp.max(masked_val)
        # For relevant indices, if max value then return 0 else 1
        val = jnp.where(masked_val != -1, masked_val != max_node, val)
        return val
    topology_pattern = topology_pattern[::2]  # Only look at node indices, not slot actions
    action_history = action_history[::2]
    check_virtual = jax.lax.fori_loop(jnp.min(topology_pattern), jnp.max(topology_pattern)+1, loop_func_virtual, action_history)
    check_physical = jax.lax.fori_loop(jnp.min(action_history), jnp.max(action_history)+1, loop_func_physical, topology_pattern)
    check = jnp.concatenate((check_virtual, check_physical))
    return jnp.any(check)


def implement_node_action(state: EnvState, s_node: chex.Array, d_node: chex.Array, s_request: chex.Array, d_request: chex.Array, n=2) -> EnvState:
    """Update node capacity, node resource and node departure arrays

    Args:
        state (State): current state
        s_node (int): source node
        d_node (int): destination node
        s_request (int): source node request
        d_request (int): destination node request
        n (int, optional): number of nodes to implement. Defaults to 2.
        
    Returns:
        State: updated state
    """
    node_indices = jnp.arange(state.node_capacity_array.shape[0])

    curr_selected_nodes = jnp.zeros(state.node_capacity_array.shape[0])
    # d_request -ve so that selected node is +ve (so that argmin works correctly for node resource array update)
    # curr_selected_nodes is N x 1 array, with requested node resources at index of selected node
    curr_selected_nodes = update_node_array(node_indices, curr_selected_nodes, d_node, -d_request)
    curr_selected_nodes = jax.lax.cond(n == 2, lambda x: update_node_array(*x), lambda x: x[1], (node_indices, curr_selected_nodes, s_node, -s_request))
    # TODO - experiment with jax.lax.fori_loop here to replace cond
    # e.g. curr_selected_nodes = jax.lax.scan(lambda c, x: update_node_array(*x), jnp.concatenate(...requests...), (node_indices, curr_selected_nodes, s_node, -s_request))

    node_capacity_array = state.node_capacity_array - curr_selected_nodes

    node_resource_array = vmap_update_node_resources(state.node_resource_array, curr_selected_nodes)

    node_departure_array = vmap_update_node_departure(state.node_departure_array, curr_selected_nodes, -state.current_time-state.holding_time)

    state = state.replace(
        node_capacity_array=node_capacity_array,
        node_resource_array=node_resource_array,
        node_departure_array=node_departure_array
    )

    return state


@partial(jax.jit, static_argnums=(1,))
def process_path_action(state: EnvState, params: EnvParams, path_action: chex.Array) -> (chex.Array, chex.Array):
    """Process path action to get path index and initial slot index.

    Args:
        state (State): current state
        params (Params): environment parameters
        path_action (int): path action

    Returns:
        int: path index
        int: initial slot index
    """
    num_slot_actions = math.ceil(params.link_resources/params.aggregate_slots)
    path_index = jnp.floor(path_action / num_slot_actions).astype(jnp.int32).reshape(1)
    initial_aggregated_slot_index = jnp.mod(path_action, num_slot_actions).reshape(1)
    initial_slot_index = initial_aggregated_slot_index*params.aggregate_slots
    if params.aggregate_slots > 1:
        # Get the path mask do a dynamic slice and get the index of first unoccupied slot in the slice
        path_mask = jax.lax.dynamic_slice(state.full_link_slot_mask, path_index*params.link_resources, (params.link_resources,))
        path_mask_slice = jax.lax.dynamic_slice(path_mask, initial_slot_index, (params.aggregate_slots,))
        # Use argmax to get index of first 1 in slice of mask
        initial_slot_index = initial_slot_index + jnp.argmax(path_mask_slice)
    return path_index[0], initial_slot_index[0]


def implement_path_action(state: EnvState, path: chex.Array, initial_slot_index: chex.Array, num_slots: chex.Array) -> EnvState:
    """Update link-slot and link-slot departure arrays.
    Times are set to negative until turned positive by finalisation (after checks).

    Args:
        state (State): current state
        path (int): path to implement
        initial_slot_index (int): initial slot index
        num_slots (int): number of slots to implement
    """
    state = state.replace(
        link_slot_array=vmap_update_path_links(state.link_slot_array, path, initial_slot_index, num_slots, 1),
        link_slot_departure_array=vmap_update_path_links(state.link_slot_departure_array, path, initial_slot_index, num_slots, state.current_time+state.holding_time)
    )
    return state


def path_action_only(topology_pattern: chex.Array, action_counter: chex.Array, remaining_actions: chex.Scalar):
    """This is to check if node has already been assigned, therefore just need to assign slots (n=0)

    Args:
        topology_pattern: Topology pattern
        action_counter: Action counter
        remaining_actions: Remaining actions

    Returns:
        bool: True if only path action, False if node action
    """
    # Get topology segment to be assigned e.g. [2,1,4]
    topology_segment = jax.lax.dynamic_slice(topology_pattern, ((remaining_actions-1)*2, ), (3, ))
    topology_indices = jnp.arange(topology_pattern.shape[0])
    # Check if the latest node in the segment is found in "prev_assigned_topology"
    new_node_to_be_assigned = topology_segment[0]
    prev_assigned_topology = jnp.where(topology_indices > (action_counter[-1]-1)*2, topology_pattern, 0)
    nodes_already_assigned_check = jnp.any(jnp.sum(jnp.where(prev_assigned_topology == new_node_to_be_assigned, 1, 0)) > 0)
    return nodes_already_assigned_check


@partial(jax.jit, static_argnums=(4,))
def implement_vone_action(
        state: EnvState,
        action: chex.Array,
        total_actions: chex.Scalar,
        remaining_actions: chex.Scalar,
        params: EnvParams,
):
    """Implement action to assign nodes (1, 2, or 0 nodes assigned per action) and assign slots and links for lightpath.

    Args:
        state: current state
        action: action to implement (node, node, path_slot_action)
        total_actions: total number of actions to implement for current request
        remaining_actions: remaining actions to implement
        k: number of paths to consider
        N: number of nodes to assign

    Returns:
        state: updated state
    """
    request = jax.lax.dynamic_slice(state.request_array[0], ((remaining_actions-1)*2, ), (3, ))
    node_request_s = jax.lax.dynamic_slice(request, (2, ), (1, ))
    bw_request = jax.lax.dynamic_slice(request, (1,), (1,))
    node_request_d = jax.lax.dynamic_slice(request, (0, ), (1, ))
    nodes = action[::2]
    path_index, initial_slot_index = process_path_action(state, params, action[1])
    path = get_paths(params, nodes)[path_index]
    se = get_paths_se(params, nodes)[path_index] if params.consider_modulation_format else jnp.array([1])
    num_slots = required_slots(bw_request, se, params.slot_size, guardband=params.guardband)

    # jax.debug.print("state.request_array {}", state.request_array, ordered=True)
    # jax.debug.print("path {}", path, ordered=True)
    # jax.debug.print("slots {}", jnp.max(jnp.where(path.reshape(-1,1) == 1, state.link_slot_array, jnp.zeros(params.num_links).reshape(-1,1)), axis=0), ordered=True)
    # jax.debug.print("path_index {}", path_index, ordered=True)
    # jax.debug.print("initial_slot_index {}", initial_slot_index, ordered=True)
    # jax.debug.print("bw_request {}", bw_request, ordered=True)
    # jax.debug.print("request {}", request, ordered=True)
    # jax.debug.print("se {}", se, ordered=True)
    # jax.debug.print("num_slots {}", num_slots, ordered=True)

    n_nodes = jax.lax.cond(
        total_actions == remaining_actions,
        lambda x: 2, lambda x: 1,
        (total_actions, remaining_actions))
    path_action_only_check = path_action_only(state.request_array[1], state.action_counter, remaining_actions)

    state = jax.lax.cond(
        path_action_only_check,
        lambda x: x[0],
        lambda x: implement_node_action(x[0], x[1], x[2], x[3], x[4], n=x[5]),
        (state, nodes[0], nodes[1], node_request_s, node_request_d, n_nodes)
    )

    state = implement_path_action(state, path, initial_slot_index, num_slots)

    return state


@partial(jax.jit, static_argnums=(2,))
def implement_action_rsa(
        state: EnvState,
        action: chex.Array,
        params: EnvParams,
) -> EnvState:
    """Implement action to assign slots on links.

    Args:
        state: current state
        action: action to implement
        k: number of slots to assign
        N: number of nodes to assign

    Returns:
        state: updated state
    """
    nodes_sd, bw_request = read_rsa_request(state.request_array)
    path_index, initial_slot_index = process_path_action(state, params, action)
    path = get_paths(params, nodes_sd)[path_index]
    if params.__class__.__name__ == "RWALightpathReuseEnvParams":
        #jax.debug.print("link_capacity_array before {}", state.link_capacity_array, ordered=True)
        state = state.replace(
            link_capacity_array=vmap_update_path_links(
                state.link_capacity_array, path, initial_slot_index, 1, bw_request
            )
        )
        #jax.debug.print("link_capacity_array after {}", state.link_capacity_array, ordered=True)
        #jax.debug.print("link_slot_array before {}", state.link_slot_array, ordered=True)
        # TODO - to support diverse bw_requests for RWA-LR, need to update masking
        # TODO - In order to enable dynamic RWA with lightpath reuse (as opposed to just incremental loading),
        #  need to keep track of active requests OR just randomly remove connections
        #  (could do this by using the link_slot_departure array in a novel way... i.e. don't fill it with departure time but current bw)
        capacity_mask = jnp.where(state.link_capacity_array <= 0., -1., 0.)
        over_capacity_mask = jnp.where(state.link_capacity_array < 0., -1., 0.)
        total_mask = capacity_mask + over_capacity_mask
        #jax.debug.print("total_mask {}", total_mask, ordered=True)
        state = state.replace(
            link_slot_array=total_mask,
            link_slot_departure_array=vmap_update_path_links(state.link_slot_departure_array, path,
                                                                       initial_slot_index, 1,
                                                                       state.current_time + state.holding_time)
        )
        #jax.debug.print("link_slot_array after {}", state.link_slot_array, ordered=True)
    else:
        se = get_paths_se(params, nodes_sd)[path_index] if params.consider_modulation_format else 1
        num_slots = required_slots(bw_request, se, params.slot_size, guardband=params.guardband)
        state = implement_path_action(state, path, initial_slot_index, num_slots)
    return state


def format_vone_slot_request(state: EnvState, action: chex.Array) -> chex.Array:
    """Format slot request for VONE action into format (source-node, slot, destination-node).

    Args:
        state: current state
        action: action to format

    Returns:
        chex.Array: formatted request
    """
    remaining_actions = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.action_counter, 2, 1))
    full_request = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.request_array, 0, 1))
    unformatted_request = jax.lax.dynamic_slice_in_dim(full_request, (remaining_actions - 1) * 2, 3)
    node_s = jax.lax.dynamic_slice_in_dim(action, 0, 1)
    requested_slots = jax.lax.dynamic_slice_in_dim(unformatted_request, 1, 1)
    node_d = jax.lax.dynamic_slice_in_dim(action, 2, 1)
    formatted_request = jnp.concatenate((node_s, requested_slots, node_d))
    return formatted_request


def read_rsa_request(request_array: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Read RSA request from request array. Return source-destination nodes and bandwidth request.

    Args:
        request_array: request array

    Returns:
        Tuple[chex.Array, chex.Array]: source-destination nodes and bandwidth request
    """
    node_s = jax.lax.dynamic_slice(request_array, (0,), (1,))
    bw_request = jax.lax.dynamic_slice(request_array, (1,), (1,))
    node_d = jax.lax.dynamic_slice(request_array, (2,), (1,))
    nodes_sd = jnp.concatenate((node_s, node_d))
    return nodes_sd, bw_request


def make_positive(x):
    return jnp.where(x < 0, -x, x)


@partial(jax.jit, donate_argnums=(0,))
def finalise_vone_action(state):
    """Turn departure times positive.

    Args:
        state: current state

    Returns:
        state: updated state
    """
    state = state.replace(
        node_departure_array=make_positive(state.node_departure_array),
        link_slot_departure_array=make_positive(state.link_slot_departure_array),
        accepted_services=state.accepted_services + 1,
        accepted_bitrate=state.accepted_bitrate  # TODO - get sum of bitrates for requested links
    )
    return state


@partial(jax.jit, donate_argnums=(0,))
def finalise_action_rsa(state):
    """Turn departure times positive.

    Args:
        state: current state

    Returns:
        state: updated state
    """
    _, bw_request = read_rsa_request(state.request_array)
    state = state.replace(
        link_slot_departure_array=make_positive(state.link_slot_departure_array),
        accepted_services=state.accepted_services + 1,
        accepted_bitrate=state.accepted_bitrate + bw_request[0],
        total_bitrate=state.total_bitrate + bw_request[0]
    )
    return state


@partial(jax.jit, donate_argnums=(0,))
def finalise_action_rwalr(state):
    """Turn departure times positive.

    Args:
        state: current state

    Returns:
        state: updated state
    """
    _, bw_request = read_rsa_request(state.request_array)
    state = state.replace(
        link_slot_departure_array=make_positive(state.link_slot_departure_array),
        accepted_services=state.accepted_services + 1,
        accepted_bitrate=state.accepted_bitrate + bw_request[0],
        total_bitrate=state.total_bitrate + bw_request[0]
    )
    return state


def check_vone_action(state, remaining_actions, total_requested_nodes):
    """Check if action is valid.
    Combines checks for:
    - sufficient node capacities
    - unique nodes assigned
    - minimum two nodes assigned
    - all requested nodes assigned
    - correct topology pattern
    - no spectrum reuse

    Args:
        state: current state
        remaining_actions: remaining actions
        total_requested_nodes: total requested nodes

    Returns:
        bool: True if check failed, False if check passed
    """
    checks = jnp.stack((
        check_node_capacities(state.node_capacity_array),
        check_unique_nodes(state.node_departure_array),
        # TODO - Remove two nodes check if impairs performance
        #  (check_all_nodes_assigned is sufficient but fails after last action of request instead of earlier)
        check_min_two_nodes_assigned(state.node_departure_array),
        jax.lax.cond(
            jnp.equal(remaining_actions, jnp.array(1)),
            lambda x: check_all_nodes_assigned(*x),
            lambda x: jnp.array(False),
            (state.node_departure_array, total_requested_nodes)
        ),
        jax.lax.cond(
            jnp.equal(remaining_actions, jnp.array(1)),
            lambda x: check_topology(*x),
            lambda x: jnp.array(False),
            (state.action_history, state.request_array[1])
        ),
        check_no_spectrum_reuse(state.link_slot_array),
    ))
    #jax.debug.print("Checks: {}", checks, ordered=True)
    return jnp.any(checks)


def check_action_rsa(state):
    """Check if action is valid.
    Combines checks for:
    - no spectrum reuse

    Args:
        state: current state

    Returns:
        bool: True if check failed, False if check passed
    """
    return jnp.any(jnp.stack((
        check_no_spectrum_reuse(state.link_slot_array),
    )))


def convert_node_probs_to_traffic_matrix(node_probs: list) -> chex.Array:
    """Convert list of node probabilities to symmetric traffic matrix.

    Args:
        node_probs: node probabilities

    Returns:
        traffic_matrix: traffic matrix
    """
    matrix = jnp.outer(node_probs, node_probs)
    # Set lead diagonal to zero
    matrix = jnp.where(jnp.eye(matrix.shape[0]) == 1, 0, matrix)
    matrix = normalise_traffic_matrix(matrix)
    return matrix


def get_edge_disjoint_paths(graph: nx.Graph) -> dict:
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


def make_graph(topology_name: str = "conus", topology_directory: str = None):
    """Create graph from topology definition.
    Topologies must be defined in JSON format in the topologies directory and
    named as the topology name with .json extension.

    Args:
        topology_name: topology name
        topology_directory: topology directory

    Returns:
        graph: graph
    """
    topology_path = pathlib.Path(topology_directory) if topology_directory else (
            pathlib.Path(__file__).parents[2].absolute() / "topologies")
    # Create topology
    if topology_name == "4node":
        # 4 node ring
        graph = nx.from_numpy_array(np.array([[0, 1, 0, 1],
                                            [1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [1, 0, 1, 0]]))
        # Add edge weights to graph
        nx.set_edge_attributes(graph, {(0, 1): 4, (1, 2): 3, (2, 3): 2, (3, 0): 1}, "weight")
    elif topology_name == "7node":
        # 7 node ring
        graph = nx.from_numpy_array(jnp.array([[0, 1, 0, 0, 0, 0, 1],
                                               [1, 0, 1, 0, 0, 0, 0],
                                               [0, 1, 0, 1, 0, 0, 0],
                                               [0, 0, 1, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0, 1],
                                               [1, 0, 0, 0, 0, 1, 0]]))
        # Add edge weights to graph
        nx.set_edge_attributes(graph, {(0, 1): 4, (1, 2): 3, (2, 3): 2, (3, 4): 1, (4, 5): 2, (5, 6): 3, (6, 0): 4}, "weight")
    else:
        with open(topology_path / f"{topology_name}.json") as f:
            graph = nx.node_link_graph(json.load(f))
    return graph


@partial(jax.jit, static_argnums=(1,))
def mask_slots(state: EnvState, params: EnvParams, request: chex.Array) -> EnvState:
    """Returns binary mask of valid actions. 1 for valid action, 0 for invalid action.

    1. Check request for source and destination nodes
    2. For each path:
        - Get current slots on path (with padding on end to avoid out of bounds)
        - Get mask for required slots on path
        - Multiply through current slots with required slots mask to check if slots available on path
        - Remove padding from mask
        - Return path mask
    3. Update total mask with path mask
    4. If aggregate_slots > 1, aggregate slot mask to reduce action space

    Args:
        state: Environment state
        params: Environment parameters
        request: Request array in format [source_node, data-rate, destination_node]

    Returns:
        state: Updated environment state
    """
    nodes_sd, requested_bw = read_rsa_request(request)
    init_mask = jnp.zeros((params.link_resources * params.k_paths))

    def mask_path(i, mask):
        # Get slots for path
        slots = get_path_slots(state.link_slot_array, params, nodes_sd, i)
        # Add padding to slots at end
        slots = jnp.concatenate((slots, jnp.ones(params.max_slots)))
        # Convert bandwidth to slots for each path
        spectral_efficiency = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        requested_slots = required_slots(requested_bw, spectral_efficiency, params.slot_size, guardband=params.guardband)
        # Get mask used to check if request will fit slots
        request_mask = jax.lax.dynamic_update_slice(
            jnp.zeros(params.max_slots * 2), jnp.ones(params.max_slots), params.max_slots - requested_slots
        )
        # Then cut in half and flip
        request_mask = jnp.flip(jax.lax.dynamic_slice(request_mask, (0,), (params.max_slots,)), axis=0)

        def check_slots_available(j, val):
            # Multiply through by request mask to check if slots available
            slot_sum = jnp.sum(request_mask * jax.lax.dynamic_slice(val, (j,), (params.max_slots,))) <= 0
            slot_sum = slot_sum.reshape((1,)).astype(jnp.float32)
            return jax.lax.dynamic_update_slice(val, slot_sum, (j,))

        # Mask out slots that are not valid
        path_mask = jax.lax.fori_loop(
            0,
            int(params.link_resources+1),  # No need to check last requested_slots-1 slots
            check_slots_available,
            slots,
        )
        # Cut off padding
        path_mask = jax.lax.dynamic_slice(path_mask, (0,), (params.link_resources,))
        # Update total mask with path mask
        mask = jax.lax.dynamic_update_slice(mask, path_mask, (i * params.link_resources,))
        return mask

    # Loop over each path
    link_slot_mask = jax.lax.fori_loop(0, params.k_paths, mask_path, init_mask)
    if params.aggregate_slots > 1:
        # Full link slot mask is used in process_path_action to get the correct slot from the aggregated slot action
        state = state.replace(full_link_slot_mask=link_slot_mask)
        link_slot_mask, _ = aggregate_slots(link_slot_mask.reshape(params.k_paths, -1), params)
        link_slot_mask = link_slot_mask.reshape(-1)
    state = state.replace(link_slot_mask=link_slot_mask)
    return state


@partial(jax.jit, static_argnums=(1,))
def aggregate_slots(full_mask: chex.Array, params: EnvParams) -> chex.Array:
    """Aggregate slot mask to reduce action space. Only used if the --aggregate_slots flag is set to > 1.
    Aggregated action is valid if there is one valid slot action within the aggregated action window.

    Args:
        full_mask: slot mask
        params: environment parameters

    Returns:
        agg_mask: aggregated slot mask
    """

    num_actions = math.ceil(params.link_resources/params.aggregate_slots)
    agg_mask = jnp.zeros((params.k_paths, num_actions), dtype=jnp.float32)

    def get_max(i, mask_val):
        """Get maximum value of array slice of length aggregate_slots."""
        mask_slice = jax.lax.dynamic_slice(
                mask_val,
                (0, i * params.aggregate_slots,),
                (1,  params.aggregate_slots,),
            )
        max_slice = jnp.max(mask_slice).reshape(1, -1)
        return max_slice

    def update_window_max(i, val):
        """Update ith index 'agg_mask' with max of ith slice of length aggregate_slots from 'full_mask'.

        Args:
            i: increments as += aggregate_slots
            val: tuple of (agg_mask, path_mask, path_index).
        Returns:
            new_agg_mask: agg_mask is updated with max of path_mask for window size aggregate_slots
            mask: mask is unchanged
            path_index: path_index is unchanged
        """
        agg_mask = val[0]
        full_mask = val[1]
        path_index = val[2]
        new_agg_mask = jax.lax.dynamic_update_slice(
            agg_mask,
            get_max(i, full_mask),
            (path_index, i),
        )
        return new_agg_mask, full_mask, path_index

    def apply_to_path_mask(i, val):
        """
        Loop through each path for num_actions steps and get_window_max at each step.

        Args:
            i: path index
            val: tuple of (agg_mask, mask) where mask is original link-slot mask and agg_mask is resulting aggregated mask
        Returns:
            new_agg_mask: agg_mask is updated with aggregated path mask
            mask: mask is unchanged
        """
        val = (
            val[0],  # aggregated mask (to be updated)
            val[1][i].reshape(1, -1),  # mask for path i
            i  # path index
        )
        new_agg_mask = jax.lax.fori_loop(
            0,
            num_actions,
            update_window_max,
            val,
        )[0]
        return new_agg_mask, full_mask

    return jax.lax.fori_loop(
            0,
            params.k_paths,
            apply_to_path_mask,
            (agg_mask, full_mask),
        )


@partial(jax.jit, static_argnums=(1,))
def mask_nodes(state: EnvState, num_nodes: chex.Scalar) -> EnvState:
    """Returns mask of valid actions for node selection. 1 for valid action, 0 for invalid action.

    Args:
        state: Environment state
        num_nodes: Number of nodes

    Returns:
        state: Updated environment state
    """
    total_actions = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.action_counter, 1, 1))
    remaining_actions = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.action_counter, 2, 1))
    full_request = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.request_array, 0, 1))
    virtual_topology = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.request_array, 1, 1))
    request = jax.lax.dynamic_slice_in_dim(full_request, (remaining_actions - 1) * 2, 3)
    node_request_s = jax.lax.dynamic_slice_in_dim(request, 2, 1)
    node_request_d = jax.lax.dynamic_slice_in_dim(request, 0, 1)
    prev_action = jax.lax.dynamic_slice_in_dim(state.action_history, (remaining_actions) * 2, 3)
    prev_dest = jax.lax.dynamic_slice_in_dim(prev_action, 0, 1)
    node_indices = jnp.arange(0, num_nodes)
    # Get requested indices from request array virtual topology
    requested_indices = jax.lax.dynamic_slice_in_dim(virtual_topology, (remaining_actions-1)*2, 3)
    requested_index_d = jax.lax.dynamic_slice_in_dim(requested_indices, 0, 1)
    # Get index of previous selected node
    prev_selected_node = jnp.where(virtual_topology == requested_index_d, state.action_history, jnp.full(virtual_topology.shape, -1))
    # will be current index if node only occurs once in virtual topology or will be different index if occurs more than once
    prev_selected_index = jnp.argmax(prev_selected_node)
    prev_selected_node_d = jax.lax.dynamic_slice_in_dim(state.action_history, prev_selected_index, 1)

    # If first action, source and dest both to be assigned -> just mask all nodes based on resources
    # Thereafter, source must be previous dest. Dest can be any node (except previous allocations).
    state = state.replace(
        node_mask_s=jax.lax.cond(
            jnp.equal(remaining_actions, total_actions),
            lambda x: jnp.where(
                state.node_capacity_array >= node_request_s,
                x,
                jnp.zeros(num_nodes)
            ),
            lambda x: jnp.where(
                node_indices == prev_dest,
                x,
                jnp.zeros(num_nodes)
            ),
            jnp.ones(num_nodes),
        )
    )
    state = state.replace(
        node_mask_d=jnp.where(
            state.node_capacity_array >= node_request_d,
            jnp.ones(num_nodes),
            jnp.zeros(num_nodes)
        )
    )
    # If not first move, set node_mask_d to zero wherever node_mask_s is 1
    # to avoid same node selection for s and d
    state = state.replace(
        node_mask_d=jax.lax.cond(
            jnp.equal(remaining_actions, total_actions),
            lambda x: x,
            lambda x: jnp.where(
                state.node_mask_s == 1,
                jnp.zeros(num_nodes),
                x
            ),
            state.node_mask_d,
        )
    )

    def mask_previous_selections(i, val):
        # Disallow previously allocated nodes
        update_slice = lambda j, x: jax.lax.dynamic_update_slice_in_dim(x, jnp.array([0.]), j, axis=0)
        val = jax.lax.cond(
            i % 2 == 0,
            lambda x: update_slice(x[0][i], x[1]),  # i is node request index
            lambda x: update_slice(x[0][i+1], x[1]),  # i is slot request index (so add 1 to get next node)
            (state.action_history, val),
        )
        return val

    state = state.replace(
        node_mask_d=jax.lax.fori_loop(
            remaining_actions*2,
            state.action_history.shape[0]-1,
            mask_previous_selections,
            state.node_mask_d
        )
    )
    # If requested node index is new then disallow previously allocated nodes
    # If not new, then must match previously allocated node for that index
    state = state.replace(
        node_mask_d=jax.lax.cond(
            jnp.squeeze(prev_selected_node_d) >= 0,
            lambda x: jnp.where(
                node_indices == prev_selected_node_d,
                x[1],
                x[0],
            ),
            lambda x: x[2],
            (jnp.zeros(num_nodes), jnp.ones(num_nodes), state.node_mask_d),
        )
    )
    return state


@partial(jax.jit, static_argnums=(1, 4))
def get_path_slots(link_slot_array: chex.Array, params: EnvParams, nodes_sd: chex.Array, i: int, agg_func: str = "max") -> chex.Array:
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

    Returns:
        slots: slots on path
    """
    path = get_paths(params, nodes_sd)[i]
    path = path.reshape((params.num_links, 1))
    # Get links and collapse to single dimension
    num_slots = params.link_resources if agg_func == "max" else math.ceil(params.link_resources/params.aggregate_slots)
    slots = jnp.where(path, link_slot_array, jnp.zeros(num_slots))
    # Make any -1s positive then get max for each slot across links
    if agg_func == "max":
        # Use this for getting slots from link_slot_array
        slots = jnp.max(jnp.absolute(slots), axis=0)
    elif agg_func == "sum":
        # TODO - consider using an RNN (or S5) to aggregate edge features
        # Use this (or alternative) for aggregating edge features from GNN
        slots = jnp.sum(slots, axis=0)
    else:
        raise ValueError("agg_func must be 'max' or 'sum'")
    return slots


def count_until_next_one(array: chex.Array, position: int) -> int:
    # Add 1s to end so that end block is counted and slice shape can be fixed
    shape = array.shape[0]
    array = jnp.concatenate([array, jnp.ones(array.shape[0], dtype=jnp.int32)])
    # Find the indices of 1 in the array
    one_indices = jax.lax.dynamic_slice(array, (position,), (shape,))
    # Find the next 1 after the given position
    next_one_idx = jnp.argmax(one_indices)
    return next_one_idx + 1


def count_until_previous_one(array: chex.Array, position: int) -> int:
    # Add 1s to start so that end block is counted and slice shape can be fixed
    shape = array.shape[0]
    array = jnp.concatenate([jnp.ones(array.shape[0], dtype=jnp.int32), array])
    # Find the indices of 1 in the array
    one_indices = jax.lax.dynamic_slice(array, (-shape-position,), (shape,))
    # Find the next 1 after the given position
    one_indices = jnp.flip(one_indices)
    next_one_idx = jnp.argmax(one_indices)
    return next_one_idx + 1


def find_block_starts(path_slots: chex.Array) -> chex.Array:
    # Add a [1] at the beginning to find transitions from 1 to 0
    path_slots_extended = jnp.concatenate((jnp.array([1]), path_slots), axis=0)
    transitions = jnp.diff(path_slots_extended)  # Find transitions (1 to 0)
    block_starts = jnp.where(transitions == -1, 1, 0)  # transitions=-1 at block starts, 0 elsewhere
    return block_starts


def find_block_ends(path_slots: chex.Array) -> chex.Array:
    # Add a [1] at the end to find transitions from 0 to 1
    path_slots_extended = jnp.concatenate((path_slots, jnp.array([1])), axis=0)
    transitions = jnp.diff(path_slots_extended)  # Find transitions (1 to 0)
    block_ends = jnp.where(transitions == 1, 1, 0)  # transitions=1 at block ends, 0 elsewhere
    return block_ends


@partial(jax.jit, static_argnames=["starts_only", "reverse"])
def find_block_sizes(path_slots: chex.Array, starts_only: bool = True, reverse: bool = False) -> chex.Array:
    def count_forward(i, arrays):
        starts = arrays[0]
        ends = arrays[1]
        new_val = jnp.reshape(count_until_next_one(ends, i), (1,))
        starts = jax.lax.dynamic_update_slice(starts, new_val, (i,))
        return (starts, ends)

    def count_backward(i, arrays):
        starts = arrays[0]
        ends = arrays[1]
        new_val = jnp.reshape(count_until_previous_one(starts, i), (1,))
        ends = jax.lax.dynamic_update_slice(ends, new_val, (-i-1,))
        return (starts, ends)

    block_starts = find_block_starts(path_slots)
    block_ends = find_block_ends(path_slots)
    block_sizes = jax.lax.fori_loop(
        0,
        block_starts.shape[0],
        count_forward if not reverse else count_backward,
        (block_starts, block_ends),
    )[0 if not reverse else 1]
    if starts_only:
        block_sizes = jnp.where(block_starts == 1, block_sizes, 0)
    else:
        block_sizes = jnp.where(path_slots == 0, block_sizes, 0)
    return block_sizes


# TODO - write tests
@partial(jax.jit, static_argnums=(1,))
def calculate_path_stats(state: EnvState, params: EnvParams, request: chex.Array) -> chex.Array:
    """For use in DeepRMSA agent observation space.
    Calculate:
    1. Size of 1st suitable free spectrum block
    2. Index of 1st suitable free spectrum block
    3. Required slots on path
    4. Avg. free block size
    5. Free slots

    Args:
        state: Environment state
        params: Environment parameters
        request: Request array in format [source_node, data-rate, destination_node]

    Returns:
        stats: Array of calculated path statistics
    """
    nodes_sd, requested_bw = read_rsa_request(request)
    init_val = jnp.zeros((params.k_paths, 5))

    def body_fun(i, val):
        slots = get_path_slots(state.link_slot_array, params, nodes_sd, i)
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else jnp.array([1])
        req_slots = jnp.squeeze(required_slots(requested_bw, se, params.slot_size, guardband=params.guardband))
        req_slots_norm = req_slots#*params.slot_size / jnp.max(params.values_bw.val)
        free_slots_norm = jnp.sum(jnp.where(slots == 0, 1, 0)) #/ params.link_resources
        block_sizes = find_block_sizes(slots)
        first_block_index = jnp.argmax(block_sizes >= req_slots)
        first_block_index_norm = first_block_index #/ params.link_resources
        first_block_size_norm = jnp.squeeze(
            jax.lax.dynamic_slice(block_sizes, (first_block_index,), (1,))
        ) / req_slots
        avg_block_size_norm = jnp.sum(block_sizes) / jnp.max(jnp.array([jnp.sum(find_block_starts(slots)), 1])) #/ req_slots
        val = jax.lax.dynamic_update_slice(
            val,
            jnp.array([[first_block_size_norm, first_block_index_norm, req_slots_norm, avg_block_size_norm, free_slots_norm]]),
            (i, 0)
        )  # N.B. that all values are normalised
        return val

    stats = jax.lax.fori_loop(
            0,
            params.k_paths,
            body_fun,
            init_val,
        )

    return stats


def create_run_name(config: flags.FlagValues) -> str:
    """Create name for run based on config flags"""
    config = {k: v.value for k, v in config.__flags.items()}
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
        run_name += f"_EVAL"
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
        min_request=100,  # Minimum data rate request size
        scale_factor=1.0,  # Scale factor for link capacity
        alpha=0.2e-3, # Fibre attenuation coefficient
        NF=4.5,  # Amplifier noise figure
        B=10e12,  # Total modulated bandwidth
        R_s=100e9,  # Symbol rate
        beta_2=-21.7e-27,  # Dispersion parameter
        gamma=1.2e-3,  # Nonlinear coefficient
        L_s=100e3,  # Span length
        lambda0=1550e-9,  # Wavelength
):
    """From Nevin JOCN paper 2022: https://discovery.ucl.ac.uk/id/eprint/10175456/1/RL_JOCN_accepted.pdf"""
    alpha_lin = alpha / 4.343  # Linear attenuation coefficient
    N_spans = jnp.floor(path_length * 1e3 / L_s)  # Number of fibre spans on path
    L_eff = (1 - jnp.exp(-alpha_lin * L_s)) / alpha_lin  # Effective length of span in m
    sigma_2_ase = (jnp.exp(alpha_lin * L_s) - 1) * 10**(NF/10) * 6.626e-34 * 2.998e8 * R_s / lambda0  # ASE noise power
    span_NSR = jnp.cbrt(2 * sigma_2_ase**2 * alpha_lin * gamma**2 * L_eff**2 *
                        jnp.log(jnp.pi**2 * jnp.abs(beta_2) * B**2 / alpha_lin) / (jnp.pi * jnp.abs(beta_2) * R_s**2))  # Noise-to-signal ratio per span
    path_NSR = jnp.where(N_spans < 1, 1, N_spans) * span_NSR  # Noise-to-signal ratio per path
    path_capacity = 2 * R_s/1e9 * jnp.log2(1 + 1/path_NSR)  # Capacity of path in Gbps
    # Round link capacity down to nearest increment of minimum request size and apply scale factor
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
    return path_capacity_array


@partial(jax.jit, static_argnums=(0,))
def get_lightpath_index(params, nodes, path_index):
    source, dest = nodes
    path_start_index = get_path_indices(source, dest, params.k_paths, params.num_nodes, directed=params.directed_graph).astype(jnp.int32)
    lightpath_index = path_index + path_start_index
    return lightpath_index


@partial(jax.jit, static_argnums=(1,))
def check_lightpath_available_and_existing(state: EnvState, params: EnvParams, action: chex.Array) -> (
        Tuple)[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Check if lightpath is available and existing.
    Available means that the lightpath does not use slots occupied by a different lightpath.
    Existing means that the lightpath has already been established.

    Args:
        state: Environment state
        params: Environment parameters

    Returns:
        lightpath_available_check: True if lightpath is available
    """
    nodes_sd, bw_request = read_rsa_request(state.request_array)
    path_index, initial_slot_index = process_path_action(state, params, action)
    path = get_paths(params, nodes_sd)[path_index]
    # Get unique lightpath index
    lightpath_index = get_lightpath_index(params, nodes_sd, path_index)
    # Get mask for slots that lightpath will occupy
    # negative numbers used so as not to conflict with lightpath indices
    new_lightpath_mask = vmap_set_path_links(
        jnp.full((params.num_links, 1), -2), path, 0, 1, -1
    )
    path_index_array = state.path_index_array[:, initial_slot_index].reshape(-1, 1)
    masked_path_index_array = jnp.where(
        new_lightpath_mask == -1, path_index_array, -2
    )
    lightpath_mask = jnp.where(
        path_index_array == lightpath_index, -1, -2
    )  # Allow current lightpath
    lightpath_existing_check = jnp.array_equal(lightpath_mask, new_lightpath_mask)  # True if all slots are same
    lightpath_mask = jnp.where(masked_path_index_array == -1, -1, lightpath_mask)  # Allow empty slots
    # True if all slots are same or empty
    lightpath_available_check = jnp.logical_or(
        jnp.array_equal(lightpath_mask, new_lightpath_mask), lightpath_existing_check
    )
    curr_lightpath_capacity = jnp.max(
        jnp.where(new_lightpath_mask == -1, state.link_capacity_array[:, initial_slot_index].reshape(-1, 1), 0)
    )
    return lightpath_available_check, lightpath_existing_check, curr_lightpath_capacity, lightpath_index


def check_action_rwalr(state: EnvState, action: chex.Array, params: EnvParams) -> bool:
    """Combines checks for:
    - no spectrum reuse
    - lightpath available and existing

    Args:
        state: Environment state

    Returns:
        bool: True if check failed, False if check passed

    """
    return jnp.any(jnp.stack((
        check_no_spectrum_reuse(state.link_slot_array),
        jnp.logical_not(check_lightpath_available_and_existing(state, params, action)[0]),
    )))


@partial(jax.jit, static_argnums=(2,))
def implement_action_rwalr(state: EnvState, action: chex.Array, params: EnvParams) -> EnvState:
    """For use in RWALightpathReuseEnv.
    Update link_slot_array and link_slot_departure_array to reflect new lightpath assignment.
    Update link_capacity_array with new capacity if lightpath is available.
    Undo link_capacity_update if over capacity.

    Args:
        state: Environment state
        action: Action array
        params: Environment parameters

    Returns:
        state: Updated environment state
    """
    nodes_sd, bw_request = read_rsa_request(state.request_array)
    path_index, initial_slot_index = process_path_action(state, params, action)
    path = get_paths(params, nodes_sd)[path_index]
    lightpath_available_check, lightpath_existing_check, curr_lightpath_capacity, lightpath_index = (
        check_lightpath_available_and_existing(state, params, action)
    )
    # Get path capacity - request
    lightpath_capacity = jax.lax.cond(
        lightpath_existing_check,
        lambda x: curr_lightpath_capacity - bw_request,  # Subtract bw_request from current lightpath
        lambda x: jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.path_capacity_array, x, 1)) - bw_request,  # Get initial capacity of lightpath - request
        lightpath_index
    )
    # Update link_capacity_array with new capacity if lightpath is available
    state = jax.lax.cond(
        lightpath_available_check,
        lambda x: x.replace(
            link_capacity_array=vmap_set_path_links(
                state.link_capacity_array, path, initial_slot_index, 1, lightpath_capacity
            ),
            path_index_array=vmap_set_path_links(
                state.path_index_array, path, initial_slot_index, 1, lightpath_index
            ),
        ),
        lambda x: x,
        state
    )
    capacity_mask = jnp.where(state.link_capacity_array <= 0., -1., 0.)
    over_capacity_mask = jnp.where(state.link_capacity_array < 0., -1., 0.)
    # Undo link_capacity_update if over capacity
    # N.B. this will fail if requested capacity is greater than total original capacity of lightpath
    lightpath_capacity_before_action = jax.lax.cond(
        lightpath_existing_check,
        lambda x: curr_lightpath_capacity,  # Subtract bw_request from current lightpath
        lambda x: 1e6,  # Empty slots have high capacity (1e6)
        # Get initial capacity of lightpath - request
        None,
    )
    state = state.replace(
        link_capacity_array=jnp.where(over_capacity_mask == -1, lightpath_capacity_before_action, state.link_capacity_array)
    )
    # Total mask will be 0 if space still available, -1 if capacity is zero or -2 if over capacity
    total_mask = capacity_mask + over_capacity_mask
    # Update link_slot_array and link_slot_departure_array
    state = state.replace(
        link_slot_array=total_mask,
        link_slot_departure_array=vmap_update_path_links(state.link_slot_departure_array, path,
                                                                 initial_slot_index, 1,
                                                                 state.current_time + state.holding_time)
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
    nodes_sd, requested_bw = read_rsa_request(request)
    init_mask = jnp.zeros((params.link_resources * params.k_paths))
    source, dest = nodes_sd
    path_start_index = get_path_indices(source, dest, params.k_paths, params.num_nodes, directed=params.directed_graph).astype(jnp.int32)
    #jax.debug.print("path_start_index {}", path_start_index, ordered=True)
    #jax.debug.print("link_capacity_array {}", state.link_capacity_array, ordered=True)

    def mask_path(i, mask):
        # Step 1 - mask capacity
        capacity_mask = jnp.where(state.link_capacity_array < requested_bw, 1., 0.)
        #jax.debug.print("capacity_mask {}", capacity_mask, ordered=True)
        capacity_slots = get_path_slots(capacity_mask, params, nodes_sd, i)
        #jax.debug.print("capacity_slots {}", capacity_slots, ordered=True)
        # Step 2 - mask lightpath reuse
        lightpath_index = path_start_index + i
        #jax.debug.print("lightpath_index {}", lightpath_index, ordered=True)
        lightpath_mask = jnp.where(state.path_index_array == lightpath_index, 0., 1.)  # Allow current lightpath
        #jax.debug.print("lightpath_mask {}", lightpath_mask, ordered=True)
        lightpath_mask = jnp.where(state.path_index_array == -1, 0., lightpath_mask)  # Allow empty slots
        #jax.debug.print("lightpath_mask {}", lightpath_mask, ordered=True)
        lightpath_slots = get_path_slots(lightpath_mask, params, nodes_sd, i)
        #jax.debug.print("lightpath_slots {}", lightpath_slots, ordered=True)
        # Step 3 combine masks
        path_mask = jnp.max(jnp.stack((capacity_slots, lightpath_slots)), axis=0)
        # Swap zeros for ones
        path_mask = jnp.where(path_mask == 0, 1., 0.)
        #jax.debug.print("path_mask {}", path_mask, ordered=True)
        mask = jax.lax.dynamic_update_slice(mask, path_mask, (i * params.link_resources,))
        return mask

    # Loop over each path
    link_slot_mask = jax.lax.fori_loop(0, params.k_paths, mask_path, init_mask)
    if params.aggregate_slots > 1:
        # Full link slot mask is used in process_path_action to get the correct slot from the aggregated slot action
        state = state.replace(full_link_slot_mask=link_slot_mask)
        link_slot_mask, _ = aggregate_slots(link_slot_mask.reshape(params.k_paths, -1), params)
        link_slot_mask = link_slot_mask.reshape(-1)
    state = state.replace(link_slot_mask=link_slot_mask)
    return state


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
        except TypeError: #not an iterable
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
        except TypeError: # final level
            yield (*index, slice(len(array))), array

    dimensions = get_max_shape(array)
    result = np.full(dimensions, fill_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return result
