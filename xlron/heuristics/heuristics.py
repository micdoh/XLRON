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
    state = mask_slots_rwa_lightpath_reuse(state, params,state.request_array) \
        if params.__class__.__name__ == "RWALightpathReuseEnvParams" else mask_slots(state, params, state.request_array)
    mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
    mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
    # Get index of first available slots for each path
    first_slots = jax.vmap(jnp.argmax, in_axes=(0))(mask)
    # Chosen path is the first one with an available slot
    path_index = jnp.argmax(first_slots < params.link_resources)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
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
    state = mask_slots_rwa_lightpath_reuse(state, params, state.request_array) \
        if params.__class__.__name__ == "RWALightpathReuseEnvParams" else mask_slots(state, params, state.request_array)
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


@partial(jax.jit, static_argnums=(1,))
def ff_kmc(state: EnvState, params: EnvParams) -> chex.Array:
    """K-Minimum Cut First Fit. Only suitable for RSA/RMSA.
    Method:
    1. Go through action mask and find the first available slot on all paths.
    2. For each path, allocate the first available slot.
    3. Sum number of new consecutive zero regions (cuts) created by assignment (on each link)
    4. Choose path that creates the fewest cuts.
    """
    state = mask_slots_rwa_lightpath_reuse(state, params, state.request_array) \
        if params.__class__.__name__ == "RWALightpathReuseEnvParams" else mask_slots(state, params, state.request_array)
    mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
    mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
    # Get index of first available slots for each path
    first_slots = jax.vmap(jnp.argmax, in_axes=(0))(mask)
    # Initialise array to hold number of cuts on each path
    path_cuts_array = jnp.full((params.k_paths,), 0)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)

    def get_cuts_on_path(i, result):
        path = get_paths(params, nodes_sd)[i]
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        num_slots = required_slots(requested_bw, se, params.slot_size, guardband=params.guardband)
        initial_slot_index = first_slots[i] % params.link_resources
        # Make link-slot_array positive
        slots = jnp.where(state.link_slot_array < 0, 1, 0)
        # Add column of 1s to start and end of slots
        slots = jnp.concatenate((jnp.full((slots.shape[0], 1), 0), slots, jnp.full((slots.shape[0], 1), 0)), axis=1)
        updated_slots = vmap_set_path_links(slots, path, initial_slot_index+1, num_slots, 1)
        block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(slots)
        updated_block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(updated_slots)
        block_sizes_mask = jnp.where(block_sizes > 0, 1, 0)  # Binary array showing initial block starts
        updated_block_sizes_mask = jnp.where(updated_block_sizes > 0, 1, 0)  # Binary array showing updated block starts
        block_count = jnp.sum(block_sizes_mask, axis=0)
        updated_block_count = jnp.sum(updated_block_sizes_mask, axis=0)
        num_cuts = jax.lax.cond(
            params.link_resources == first_slots[i],  # If true, no valid action for path
            lambda x: jnp.full((1,), params.link_resources*params.num_links),  # Return max no. of cuts
            lambda x: jnp.sum(jnp.where(
                updated_block_count - block_count < 0,
                0, # don't allow -ve no. of cuts
                updated_block_count - block_count
            )).reshape((1,)),  # Else, return number of cuts
            1
        )
        result = jax.lax.dynamic_update_slice(result, num_cuts, (i,))
        # jax.debug.print("initial_slot_index {}", initial_slot_index, ordered=True)
        # jax.debug.print("slots {}", state.link_slot_array, ordered=True)
        # jax.debug.print("updated_slots {}", updated_slots, ordered=True)
        # jax.debug.print("block_sizes {}", block_sizes, ordered=True)
        # jax.debug.print("updated_block_sizes {}", updated_block_sizes, ordered=True)
        # jax.debug.print("block_sizes_mask {}", block_sizes_mask, ordered=True)
        # jax.debug.print("updated_block_sizes_mask {}", updated_block_sizes_mask, ordered=True)
        # jax.debug.print("block_count {}", block_count, ordered=True)
        # jax.debug.print("updated_block_count {}", updated_block_count, ordered=True)
        # jax.debug.print("num_cuts {}", num_cuts, ordered=True)
        # jax.debug.print("result {}", result, ordered=True)
        return result

    path_cuts_array = jax.lax.fori_loop(0, params.k_paths, get_cuts_on_path, path_cuts_array)
    path_index = jnp.argmin(path_cuts_array)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def kmc_ff(state: EnvState, params: EnvParams) -> chex.Array:
    """K-Minimum Cut. Only suitable for RSA/RMSA.
    Method:
    1. Go through action mask and find the first available slot on all paths.
    2. For each path, allocate the first available slot.
    3. Sum number of new consecutive zero regions (cuts) created by assignment (on each link)
    4. Choose path that creates the fewest cuts.
    """
    state = mask_slots_rwa_lightpath_reuse(state, params, state.request_array) \
        if params.__class__.__name__ == "RWALightpathReuseEnvParams" else mask_slots(state, params, state.request_array)
    #mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
    #mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
    # Get index of first available slots for each path
    #first_slots = jax.vmap(jnp.argmax, in_axes=(0))(mask)
    # Initialise array to hold number of cuts on each path
    mask = state.link_slot_mask
    path_cuts_array = jnp.full((mask.shape[0],), 0.)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)

    def get_cuts_on_path(i, result):
        path_index, initial_slot_index = process_path_action(state, params, i)
        path = get_paths(params, nodes_sd)[path_index]
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        num_slots = required_slots(requested_bw, se, params.slot_size, guardband=params.guardband)
        # Make link-slot_array positive
        slots = jnp.where(state.link_slot_array < 0, 1, 0)
        # Add column of 1s to start and end of slots
        #slots = jnp.concatenate((jnp.full((slots.shape[0], 1), 0), slots, jnp.full((slots.shape[0], 1), 0)), axis=1)
        # Add +1 to slot_index because we added a column of 1s to the start of slots
        updated_slots = vmap_set_path_links(slots, path, initial_slot_index, num_slots, 1)
        block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(slots)
        updated_block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(updated_slots)
        block_sizes_mask = jnp.where(block_sizes > 0, 1, 0)#[:, 1:-1]  # Binary array showing initial block starts
        updated_block_sizes_mask = jnp.where(updated_block_sizes > 0, 1, 0)#[:, 1:-1]  # Binary array showing updated block starts
        block_count = jnp.sum(block_sizes_mask, axis=1)
        updated_block_count = jnp.sum(updated_block_sizes_mask, axis=1)
        num_cuts = jax.lax.cond(
            mask[i] == 0.,  # If true, no valid action for path
            lambda x: jnp.full((1,), params.link_resources*params.num_links).astype(jnp.float32),  # Return max no. of cuts
            lambda x: jnp.sum(jnp.where(
                updated_block_count - block_count < 0,
                (updated_block_count - block_count)/jnp.sum(path),  # normalise -ve cuts by length of path
                updated_block_count - block_count
            )).reshape((1,)),  # Else, return number of cuts
            1.
        )
        result = jax.lax.dynamic_update_slice(result, num_cuts, (i,))
        # jax.debug.print("valid {}", mask[i], ordered=True)
        # jax.debug.print("mask {}", mask, ordered=True)
        # jax.debug.print("initial_slot_index {}", initial_slot_index, ordered=True)
        # jax.debug.print("slots {}", state.link_slot_array, ordered=True)
        # jax.debug.print("updated_slots {}", updated_slots, ordered=True)
        # jax.debug.print("block_sizes {}", block_sizes, ordered=True)
        # jax.debug.print("updated_block_sizes {}", updated_block_sizes, ordered=True)
        # jax.debug.print("block_sizes_mask {}", block_sizes_mask, ordered=True)
        # jax.debug.print("updated_block_sizes_mask {}", updated_block_sizes_mask, ordered=True)
        # jax.debug.print("block_count {}", block_count, ordered=True)
        # jax.debug.print("updated_block_count {}", updated_block_count, ordered=True)
        # jax.debug.print("num_cuts {}", num_cuts, ordered=True)
        # jax.debug.print("result {}", result, ordered=True)
        return result

    path_cuts_array = jax.lax.fori_loop(0, mask.shape[0], get_cuts_on_path, path_cuts_array)
    action = jnp.argmin(path_cuts_array)
    return action


@partial(jax.jit, static_argnums=(1,))
def ff_kmf(state: EnvState, params: EnvParams) -> chex.Array:
    """K-Minimum Frag-size First Fit.
    Method:
    1. Go through action mask and find the first available slot on all paths.
    2. For each path, allocate the first available slot.
    3. Sum size of new consecutive zero regions (frags) created by assignment (on each link).
    Only consider frag to the "left" i.e. lower slot indices of the assigned slots.
    4. Choose path that creates the smallest total new fragments.
    """
    state = mask_slots_rwa_lightpath_reuse(state, params, state.request_array) \
        if params.__class__.__name__ == "RWALightpathReuseEnvParams" else mask_slots(state, params, state.request_array)
    mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
    mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
    # Get index of first available slots for each path
    first_slots = jax.vmap(jnp.argmax, in_axes=(0))(mask)
    # Initialise array to hold number of cuts on each path
    path_frags_array = jnp.full((params.k_paths,), 0)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)

    def get_frags_on_path(i, result):
        path = get_paths(params, nodes_sd)[i]
        # path = path.reshape((params.num_links, 1))
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        num_slots = required_slots(requested_bw, se, params.slot_size, guardband=params.guardband)
        # slots = jnp.where(path, state.link_slot_array, jnp.zeros(params.link_resources))
        initial_slot_index = first_slots[i] % params.link_resources
        # Make link-slot_array positive
        slots = jnp.where(state.link_slot_array < 0, 1, 0)
        updated_slots = vmap_set_path_links(slots, path, initial_slot_index, num_slots, 1)
        block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(slots)
        updated_block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(updated_slots)
        difference = updated_block_sizes - block_sizes*(1-updated_slots)
        new_frags = jnp.where(difference != 0, block_sizes+difference, 0)
        # Slice new frags up to initial slot index (so as to only consider frags to the left)
        new_frags = jnp.where(jnp.arange(params.link_resources) < initial_slot_index, new_frags, 0)
        new_frag_size = jnp.sum(new_frags)
        num_frags = jax.lax.cond(
            params.link_resources == first_slots[i],  # If true, no valid action for path
            lambda x: jnp.full((1,), params.link_resources*params.num_links),  # Return max frag size
            lambda x: new_frag_size.reshape((1,)),
            # Else, return number of cuts
            1
        )
        result = jax.lax.dynamic_update_slice(result, num_frags, (i,))
        # jax.debug.print("initial_slot_index {}", initial_slot_index, ordered=True)
        # jax.debug.print("slots {}", slots, ordered=True)
        # jax.debug.print("updated_slots {}", updated_slots, ordered=True)
        # jax.debug.print("block_sizes {}", block_sizes, ordered=True)
        # jax.debug.print("updated_block_sizes {}", updated_block_sizes, ordered=True)
        # jax.debug.print("difference {}", difference, ordered=True)
        # jax.debug.print("new_frags {}", new_frags, ordered=True)
        # jax.debug.print("result {}", result, ordered=True)
        return result

    path_frags_array = jax.lax.fori_loop(0, params.k_paths, get_frags_on_path, path_frags_array)
    path_index = jnp.argmin(path_frags_array)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def kmf_ff(state: EnvState, params: EnvParams) -> chex.Array:
    """K-Minimum Cut. Only suitable for RSA/RMSA.
    Method:
    1. Go through action mask and find the first available slot on all paths.
    2. For each path, allocate the first available slot.
    3. Sum number of new consecutive zero regions (cuts) created by assignment (on each link)
    4. Choose path that creates the fewest cuts.
    """
    state = mask_slots_rwa_lightpath_reuse(state, params, state.request_array) \
        if params.__class__.__name__ == "RWALightpathReuseEnvParams" else mask_slots(state, params, state.request_array)
    #mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
    #mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
    # Get index of first available slots for each path
    #first_slots = jax.vmap(jnp.argmax, in_axes=(0))(mask)
    # Initialise array to hold number of cuts on each path
    mask = state.link_slot_mask
    path_cuts_array = jnp.full((mask.shape[0],), 0.)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)

    def get_frags_on_path(i, result):
        path_index, initial_slot_index = process_path_action(state, params, i)
        path = get_paths(params, nodes_sd)[path_index]
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        num_slots = required_slots(requested_bw, se, params.slot_size, guardband=params.guardband)
        # Make link-slot_array positive
        slots = jnp.where(state.link_slot_array < 0, 1, 0)
        # Add column of 1s to start and end of slots
        #slots = jnp.concatenate((jnp.full((slots.shape[0], 1), 0), slots, jnp.full((slots.shape[0], 1), 0)), axis=1)
        # Add +1 to slot_index because we added a column of 1s to the start of slots
        updated_slots = vmap_set_path_links(slots, path, initial_slot_index, num_slots, 1)
        block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(slots)
        updated_block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(updated_slots)
        difference = updated_block_sizes - block_sizes * (1 - updated_slots)
        new_frags = jnp.where(difference != 0, block_sizes + difference, 0)
        # Slice new frags up to initial slot index (so as to only consider frags to the left)
        new_frags = jnp.where(jnp.arange(params.link_resources) < initial_slot_index, new_frags, 0)
        new_frag_size = jnp.sum(new_frags)
        num_frags = jax.lax.cond(
            mask[i] == 0.,  # If true, no valid action for path
            lambda x: jnp.full((1,), params.link_resources * params.num_links),  # Return max frag size
            lambda x: new_frag_size.reshape((1,)),
            # Else, return number of cuts
            1
        )
        result = jax.lax.dynamic_update_slice(result, num_frags, (i,))
        # jax.debug.print("valid {}", mask[i], ordered=True)
        # jax.debug.print("mask {}", mask, ordered=True)
        # jax.debug.print("initial_slot_index {}", initial_slot_index, ordered=True)
        # jax.debug.print("slots {}", state.link_slot_array, ordered=True)
        # jax.debug.print("updated_slots {}", updated_slots, ordered=True)
        # jax.debug.print("block_sizes {}", block_sizes, ordered=True)
        # jax.debug.print("updated_block_sizes {}", updated_block_sizes, ordered=True)
        # jax.debug.print("block_sizes_mask {}", block_sizes_mask, ordered=True)
        # jax.debug.print("updated_block_sizes_mask {}", updated_block_sizes_mask, ordered=True)
        # jax.debug.print("block_count {}", block_count, ordered=True)
        # jax.debug.print("updated_block_count {}", updated_block_count, ordered=True)
        # jax.debug.print("num_cuts {}", num_cuts, ordered=True)
        # jax.debug.print("result {}", result, ordered=True)
        return result

    path_cuts_array = jax.lax.fori_loop(0, mask.shape[0], get_frags_on_path, path_cuts_array)
    action = jnp.argmin(path_cuts_array)
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