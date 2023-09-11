import jax
import jax.numpy as jnp
from xlron.environments.env_funcs import *


@partial(jax.jit, static_argnums=(1,))
def ksp_ff(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the first available slot from all k-shortest paths
    Method: Go through action mask and find the first available slot, starting from shortest path

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Action
    """
    state = mask_slots(state, params, state.request_array)
    mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
    mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
    # Get index of first available slots for each path
    first_slots = jax.vmap(jnp.argmax, in_axes=(0))(mask)
    # Chosen path is the first one with an available slot
    path_index = jnp.argmax(first_slots < params.link_resources)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * state.link_slot_array.shape[0] + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def ff_ksp(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the first available slot from the first k-shortest paths
    Method: Go through action mask and find the first available slot on all paths

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Action
    """
    state = mask_slots(state, params, state.request_array)
    mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
    mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
    # Get index of first available slots for each path
    first_slots = jax.vmap(jnp.argmax, in_axes=(0))(mask)
    # Chosen path is the one with the lowest index of first available slot
    path_index = jnp.argmin(first_slots)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


def find_consecutive_zeros(arr):
    """Find the lengths of consecutive sequences of zeros in an array.

    Args:
        arr: Array of zeros and ones.

    Returns:
        lengths: Array of lengths of consecutive sequences of zeros.
    """
    if not np.any(arr == 0):  # If the input array contains no ones
        return np.array([0])

    # Add a one at the beginning and the end to easily detect consecutive sequences
    padded_arr = np.concatenate(([1], arr, [1]))

    # Find the indices of ones and compute differences between consecutive indices
    zero_indices = np.where(padded_arr == 0)[0]
    diff_indices = np.concatenate((np.diff(zero_indices), np.array([0])))

    # Identify the end of consecutive sequences by finding where the difference is not 1
    end_of_sequences = np.where(diff_indices != 1)[0]

    # Calculate the lengths of consecutive sequences
    lengths = np.diff(np.concatenate(([0], end_of_sequences + 1)))

    return lengths


def calrc_ksp_ff(state: EnvState, params: EnvParams) -> chex.Array:
    """Use CLARC node ranking heuristc followed by k-shortest paths first fit

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        sequence_of_actions: Sequence of actions to take to fulfil complete request
    """
    request = state.request_array

    # Get node ranking

    # Iterate through request
    # Check if node previously assigned, assign to same node if so
    # If not, check if node can meet request


    #


    # Check that substrate nodes can meet virtual node requirements
    selected_nodes = []
    successful_nodes = 0
    for v_node in rank_n_v:
        for i, s_node in enumerate(ranked_nodes):
            substrate_capacity = s_node[2]
            requested_capacity = v_node[2]
            if substrate_capacity >= requested_capacity:
                selected_nodes[v_node[1]] = s_node[1]
                successful_nodes += 1
                ranked_nodes.pop(i)  # Remove selected node from list
                break





    return sequence_of_actions



# def find_consecutive_zeros(arr):
#     # Maybe be use jnp.argmin ?
#
#     # Add a one at the beginning and the end to easily detect consecutive sequences
#     padded_arr = jnp.concatenate((jnp.array([1]), arr, jnp.array([1])))
#
#     # Find the indices of ones and compute differences between consecutive indices
#     arr_indices = jnp.arange(padded_arr.shape[0])
#     zero_indices = jnp.where(padded_arr == 0, arr_indices, jnp.inf)
#     zero_indices = jnp.concatenate(zero_indices, jnp.inf)
#     diff_indices = jnp.concatenate((jnp.diff(zero_indices), jnp.array([0])))
#
#     # Identify the end of consecutive sequences by finding where the difference is not 1
#     end_of_sequences = jnp.where(diff_indices != 1)[0]
#
#     # Calculate the lengths of consecutive sequences
#     lengths = jnp.diff(jnp.concatenate((jnp.array([0]), end_of_sequences + 1)))
#
#     return lengths

def calrc(state: EnvState, params: EnvParams) -> chex.Array:
    """See paper https://ieeexplore.ieee.org/document/6679238"""
    MCSB_sizes = []
    for edge in range(len(params.num_edges)):
        MCSB_sizes.append(find_consecutive_zeros(state.link_slot_array[0]))
    MSBC_links = np.concatenate(MCSB_sizes)

    bw_request_sizes = state.request_array # ...?

    calrc = {i: 0 for i in range(len(params.num_nodes))}
    for i_edge, edge in enumerate(params.edges):
        # MCSB = Maximal Consecutive Spectral Block
        for i_req in bw_request_sizes:
            calrc[edge[0]] += np.sum(MSBC_links[i_edge] - i_req + 1)
            calrc[edge[1]] += np.sum(MSBC_links[i_edge] - i_req + 1)

    for key, val in calrc.items():
        capacity = state.node_capacity_array[key]
        calrc[key] = val * params.capacity[key]
        rank.append((capacity * calrc, node, capacity))
    rank.sort(reverse=True)
    return rank


def rank_nodes_calrc(graph, bw_request):
    """Calculate the Consecutiveness-Aware Local Resource Capacity of a node.

    Args:
        graph: NetworkX graph.
        node: Node to calculate LRC of.
        bw_requests: Bandwidth requests of all virtual links.

    Returns:
        ranking: List of tuples of (node_index, calrc) i.e. ranking of substrate nodes
        based on Consecutiveness-Aware Local Resource Capacity.
    """
    rank = []
    bw_request = set(bw_request)
    for node in graph.nodes:
        calrc = 0
        capacity = graph.nodes[node]["capacity"]
        for _, _, edge_data in graph.edges(node, data=True):
            MSBC_sizes = find_consecutive_ones(edge_data["slots"])
            for i in bw_request:
                calrc += np.sum(MSBC_sizes - i + 1)
        rank.append((capacity * calrc, node, capacity))
    rank.sort(reverse=True)
    return rank