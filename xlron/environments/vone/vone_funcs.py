from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax._src.typing import Array

from xlron import dtype_config
from xlron.environments.dataclasses import (
    ActionInfo,
    VONEEnvParams,
    VONEEnvState,
)
from xlron.environments.env_funcs import (
    generate_arrival_holding_times,
    get_affected_slots_mask,
    get_path_and_se,
    implement_path_action,
    make_positive,
    process_path_action,
    remove_expired_services_rsa,
    required_slots,
)

one = jnp.array(1.0, dtype=dtype_config.SMALL_INT_DTYPE)
zero = jnp.array(0.0, dtype=dtype_config.SMALL_INT_DTYPE)


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
    check_virtual = jax.lax.fori_loop(
        jnp.min(topology_pattern),
        jnp.max(topology_pattern) + 1,
        loop_func_virtual,
        action_history,
    )
    check_physical = jax.lax.fori_loop(
        jnp.min(action_history),
        jnp.max(action_history) + 1,
        loop_func_physical,
        topology_pattern,
    )
    check = jnp.concatenate((check_virtual, check_physical))
    return jnp.any(check)


def implement_node_action(
    state: VONEEnvState,
    s_node: chex.Array,
    d_node: chex.Array,
    s_request: chex.Array,
    d_request: chex.Array,
    n=2,
) -> VONEEnvState:
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

    curr_selected_nodes = jnp.zeros(state.node_capacity_array.shape[0]).astype(jnp.float32)
    # d_request -ve so that selected node is +ve (so that argmin works correctly for node resource array update)
    # curr_selected_nodes is N x 1 array, with requested node resources at index of selected node
    curr_selected_nodes = update_node_array(node_indices, curr_selected_nodes, d_node, -d_request)
    curr_selected_nodes = jax.lax.cond(
        n == 2,
        lambda x: update_node_array(*x),
        lambda x: x[1],
        (node_indices, curr_selected_nodes, s_node, -s_request),
    )

    node_capacity_array = state.node_capacity_array - curr_selected_nodes

    node_resource_array = vmap_update_node_resources(state.node_resource_array, curr_selected_nodes)

    node_departure_array = vmap_update_node_departure(
        state.node_departure_array,
        curr_selected_nodes,
        -state.current_time - state.holding_time,
    )

    state = state.replace(
        node_capacity_array=node_capacity_array,
        node_resource_array=node_resource_array,
        node_departure_array=node_departure_array,
    )

    return state


def check_all_nodes_assigned(node_departure_array: chex.Array, total_requested_nodes: int) -> bool:
    """Count negative values on each node (row) in node departure array, sum them, must equal total requested_nodes.

    Args:
        node_departure_array: Node departure array (N x R) where N is number of nodes and R is number of resources
        total_requested_nodes: Total requested nodes (int)

    Returns:
        bool: True if check failed, False if check passed
    """
    return (
        jnp.sum(jnp.sum(jnp.where(node_departure_array < 0, 1, 0), axis=1)) != total_requested_nodes
    )


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


def update_node_departure(node_row, inf_index, value):
    row_indices = jnp.arange(node_row.shape[0])
    return jnp.where(row_indices == inf_index, value, node_row)


def update_selected_node_departure(node_row, node_selected, first_inf_index, value):
    return jax.lax.cond(
        node_selected != 0,
        lambda x: update_node_departure(*x),
        lambda x: node_row,
        (node_row, first_inf_index, value),
    )


@partial(jax.jit, static_argnums=(0,))
def init_action_history(params: VONEEnvParams):
    """Initialize action history"""
    return jnp.full(params.max_edges * 2 + 1, -1, dtype=dtype_config.LARGE_FLOAT_DTYPE)


@jax.jit
def vmap_update_node_departure(
    node_departure_array: chex.Array, selected_nodes: chex.Array, value: int
) -> chex.Array:
    """Called when implementing node action.
    Sets request departure time ("value") in place of first "inf" i.e. unoccupied index on node departure array for selected nodes.

    Args:
        node_departure_array: (N x R) Node departure array
        selected_nodes: (N x 1) Selected nodes (non-zero value on selected node indices)
        value: Value to set on node departure array

    Returns:
        Updated node departure array
    """
    first_inf_indices = jnp.argmax(node_departure_array, axis=1).astype(
        dtype_config.LARGE_INT_DTYPE
    )
    return jax.vmap(update_selected_node_departure, in_axes=(0, 0, 0, None))(
        node_departure_array, selected_nodes, first_inf_indices, value
    )


def update_node_resources(node_row, zero_index, value):
    row_indices = jnp.arange(node_row.shape[0])
    return jnp.where(row_indices == zero_index, value, node_row)


def update_selected_node_resources(node_row, request, first_zero_index):
    return jax.lax.cond(
        request != 0,
        lambda x: update_node_resources(*x),
        lambda x: node_row,
        (node_row, first_zero_index, request),
    )


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
    return jax.vmap(update_selected_node_resources, in_axes=(0, 0, 0))(
        node_resource_array, selected_nodes, first_zero_indices
    )


def update_node_array(node_indices, array, node, request):
    """Used to udated selected_nodes array with new requested resources on each node, for use in"""
    return jnp.where(node_indices == node, array - request, array)


@partial(jax.jit, static_argnums=(0,))
def init_vone_request_array(params: VONEEnvParams):
    """Initialize request array either with uniform resources"""
    return jnp.zeros(
        (
            2,
            params.max_edges * 2 + 1,
        ),
        dtype=dtype_config.LARGE_INT_DTYPE,
    )


@partial(jax.jit, static_argnums=(0,))
def init_node_capacity_array(params: VONEEnvParams):
    """Initialize node array with uniform resources.
    Args:
        params (EnvParams): Environment parameters
    Returns:
        jnp.array: Node capacity array (N x 1) where N is number of nodes"""
    return jnp.array([params.node_resources] * params.num_nodes, dtype=dtype_config.LARGE_INT_DTYPE)


@partial(jax.jit, static_argnums=(0,))
def init_node_departure_array(params: VONEEnvParams):
    return jnp.full((params.num_nodes, params.node_resources), jnp.inf)


@partial(jax.jit, static_argnums=(0,))
def init_node_resource_array(params: VONEEnvParams):
    """Array to track node resources occupied by virtual nodes"""
    return jnp.zeros(
        (params.num_nodes, params.node_resources), dtype=dtype_config.LARGE_FLOAT_DTYPE
    )


@jax.jit
def check_unique_nodes(node_departure_array: chex.Array) -> bool:
    """Count negative values on each node (row) in node departure array, must not exceed 1.

    Args:
        node_departure_array: Node departure array (N x R) where N is number of nodes and R is number of resources

    Returns:
        bool: True if check failed, False if check passed
    """
    return jnp.any(
        jnp.sum(
            jnp.where(node_departure_array < zero, one, zero),
            axis=1,
            promote_integers=False,
        )
        > one
    )


@partial(jax.jit, static_argnums=(2,))
def generate_vone_request(key: chex.PRNGKey, state: VONEEnvState, params: VONEEnvParams):
    """Generate a new request for the VONE environment.
    The request has two rows. The first row shows the node and slot values.
    The first three elements of the second row show the number of unique nodes, the total number of steps, and the remaining steps.
    These first three elements comprise the action counter.
    The remaining elements of the second row show the virtual topology pattern, i.e. the connectivity of the virtual topology.
    """
    shape = params.max_edges * 2 + 1  # shape of request array
    key_topology, key_node, key_slot, key_times = jax.random.split(key, 4)
    # Randomly select topology, node resources, slot resources
    pattern = jax.random.choice(key_topology, state.virtual_topology_patterns)
    action_counter = jax.lax.dynamic_slice(pattern, (0,), (3,))
    topology_pattern = jax.lax.dynamic_slice(pattern, (3,), (pattern.shape[0] - 3,))
    selected_node_values = jax.random.choice(key_node, state.values_nodes, shape=(shape,))
    selected_bw_values = jax.random.choice(key_slot, params.values_bw.val, shape=(shape,))
    # Create a mask for odd and even indices
    mask = jnp.tile(jnp.array([0, 1]), (shape + 1) // 2)[:shape]
    # Vectorized conditional replacement using mask
    first_row = jnp.where(mask, selected_bw_values, selected_node_values)
    # Make sure node request values are consistent for same virtual nodes
    first_row = jax.lax.fori_loop(
        2,  # Lowest node index in virtual topology requests is 2
        shape,  # Highest possible node index in virtual topology requests is shape-1
        lambda i, x: jnp.where(topology_pattern == i, selected_node_values[i], x),
        first_row,
    )
    # Mask out unused part of request array
    first_row = jnp.where(topology_pattern == 0, 0, first_row)
    # Set times
    arrival_time, holding_time = generate_arrival_holding_times(
        key, params, state.arrival_rate, state.mean_service_holding_time
    )
    state = state.replace(
        holding_time=holding_time,
        current_time=state.current_time + arrival_time,
        action_counter=action_counter,
        request_array=jnp.vstack((first_row, topology_pattern)),
        action_history=init_action_history(params),
        total_requests=state.total_requests + 1,
    )
    state = remove_expired_node_requests(state, params) if not params.incremental_loading else state
    state = remove_expired_services_rsa(state, params) if not params.incremental_loading else state
    return state


def undo_link_action_vone(state: VONEEnvState) -> VONEEnvState:
    """Undo tentative link slot assignments for VONE.
    Tentative assignments are indicated by negative values in link_slot_array and
    link_slot_departure_array. Reset these to zero.

    Args:
        state: Environment state

    Returns:
        Updated environment state
    """
    mask = jnp.where(state.link_slot_departure_array < zero, one, zero)
    mask = jnp.where(state.link_slot_array < -one, one, mask)
    state = state.replace(
        link_slot_array=jnp.where(mask == one, state.link_slot_array + one, state.link_slot_array),
        link_slot_departure_array=jnp.where(mask == one, zero, state.link_slot_departure_array),
    )
    return state


@partial(jax.jit, static_argnums=(4,))
def implement_vone_action(
    state: VONEEnvState,
    action: chex.Array,
    total_actions: chex.Scalar,
    remaining_actions: chex.Scalar,
    params: VONEEnvParams,
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
    request = jax.lax.dynamic_slice(state.request_array[0], ((remaining_actions - 1) * 2,), (3,))
    node_request_s = jax.lax.dynamic_slice(request, (2,), (1,))
    requested_datarate = jax.lax.dynamic_slice(request, (1,), (1,))
    node_request_d = jax.lax.dynamic_slice(request, (0,), (1,))
    nodes = action[::2]
    path_index, initial_slot_index = process_path_action(state, params, action[1])
    path, se = get_path_and_se(params, nodes, path_index)
    se = se if params.consider_modulation_format else jnp.array([1])
    num_slots = required_slots(
        requested_datarate,
        se,
        params.slot_size,
        guardband=params.guardband,
        temperature=params.temperature,
    )

    # jax.debug.print("state.request_array {}", state.request_array, ordered=True)
    # jax.debug.print("path {}", path, ordered=True)
    # jax.debug.print("slots {}", jnp.max(jnp.where(path.reshape(-1,1) == 1, state.link_slot_array, jnp.zeros(params.num_links).reshape(-1,1)), axis=0), ordered=True)
    # jax.debug.print("path_index {}", path_index, ordered=True)
    # jax.debug.print("initial_slot_index {}", initial_slot_index, ordered=True)
    # jax.debug.print("requested_datarate {}", requested_datarate, ordered=True)
    # jax.debug.print("request {}", request, ordered=True)
    # jax.debug.print("se {}", se, ordered=True)
    # jax.debug.print("num_slots {}", num_slots, ordered=True)

    n_nodes = jax.lax.cond(
        total_actions == remaining_actions,
        lambda x: 2,
        lambda x: 1,
        (total_actions, remaining_actions),
    )
    path_action_only_check = path_action_only(
        state.request_array[1], state.action_counter, remaining_actions
    )

    state = jax.lax.cond(
        path_action_only_check,
        lambda x: x[0],
        lambda x: implement_node_action(x[0], x[1], x[2], x[3], x[4], n=x[5]),
        (state, nodes[0], nodes[1], node_request_s, node_request_d, n_nodes),
    )

    affected_slots_mask = get_affected_slots_mask(initial_slot_index, num_slots, path, params)
    action_info = ActionInfo(
        action=action,
        path_index=path_index,
        initial_slot_index=initial_slot_index,
        num_slots=num_slots,
        path=path,
        se=se,
        requested_datarate=requested_datarate,
        nodes_sd=nodes,
        affected_slots_mask=-affected_slots_mask,
        power_action=jnp.float32(0.0),
    )
    state = implement_path_action(state, action_info, params)

    return state


def path_action_only(
    topology_pattern: chex.Array,
    action_counter: chex.Array,
    remaining_actions: chex.Scalar,
) -> bool:
    """This is to check if node has already been assigned, therefore just need to assign slots (n=0)

    Args:
        topology_pattern: Topology pattern
        action_counter: Action counter
        remaining_actions: Remaining actions

    Returns:
        bool: True if only path action, False if node action
    """
    # Get topology segment to be assigned e.g. [2,1,4]
    topology_segment = jax.lax.dynamic_slice(topology_pattern, ((remaining_actions - 1) * 2,), (3,))
    topology_indices = jnp.arange(topology_pattern.shape[0])
    # Check if the latest node in the segment is found in "prev_assigned_topology"
    new_node_to_be_assigned = topology_segment[0]
    prev_assigned_topology = jnp.where(
        topology_indices > (action_counter[-1] - 1) * 2, topology_pattern, 0
    )
    nodes_already_assigned_check = jnp.any(
        jnp.sum(jnp.where(prev_assigned_topology == new_node_to_be_assigned, 1, 0)) > 0
    )
    return nodes_already_assigned_check


def format_vone_slot_request(state: VONEEnvState, action: chex.Array) -> chex.Array:
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
        accepted_bitrate=state.accepted_bitrate,  # TODO - get sum of bitrates for requested links
    )
    return state


def check_vone_action(state, action, remaining_actions, total_requested_nodes, params):
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
    checks = jnp.stack(
        (
            check_node_capacities(state.node_capacity_array),
            check_unique_nodes(state.node_departure_array),
            # TODO (VONE) - Remove two nodes check if impairs performance
            #  (check_all_nodes_assigned is sufficient but fails after last action of request instead of earlier)
            check_min_two_nodes_assigned(state.node_departure_array),
            jax.lax.cond(
                jnp.equal(remaining_actions, jnp.array(1)),
                lambda x: check_all_nodes_assigned(*x),
                lambda x: jnp.array(False),
                (state.node_departure_array, total_requested_nodes),
            ),
            jax.lax.cond(
                jnp.equal(remaining_actions, jnp.array(1)),
                lambda x: check_topology(*x),
                lambda x: jnp.array(False),
                (state.action_history, state.request_array[1]),
            ),
            jnp.any(state.link_slot_array < -1),
        )
    )
    return jnp.any(checks)


def init_virtual_topology_patterns(pattern_names: str) -> Array:
    """Initialise virtual topology patterns.
    First 3 digits comprise the "action counter": first index is num unique nodes, second index is total steps,
    final is remaining steps until completion of request.
    Remaining digits define the topology pattern, with 1 to indicate links and other positive integers are node indices.

    Args:
        pattern_names (list): List of virtual topology pattern names

    Returns:
        Array: Array of virtual topology patterns
    """
    patterns = []
    # TODO - Allow 2 node requests in VONE (check if any modifications necessary other than below)
    # if "2_bus" in pattern_names:
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
        pattern.extend([0] * (max_length - len(pattern)))
    return jnp.array(patterns, dtype=dtype_config.SMALL_INT_DTYPE)


def init_action_counter():
    """Initialize action counter.
    First index is num unique nodes, second index is total steps, final is remaining steps until completion of request.
    Only used in VONE environments.
    """
    return jnp.zeros(3, dtype=dtype_config.LARGE_INT_DTYPE)


def decrement_action_counter(state):
    """Decrement action counter in-place. Used in VONE environments."""
    state.action_counter.at[-1].add(-1)
    return state


@partial(jax.jit, static_argnums=(0,))
def init_node_mask(params: VONEEnvParams):
    """Initialize node mask"""
    return jnp.ones(params.num_nodes, dtype=dtype_config.LARGE_FLOAT_DTYPE)


@partial(jax.jit, static_argnums=(1,))
def mask_nodes(state: VONEEnvState, num_nodes: chex.Scalar) -> VONEEnvState:
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
    requested_indices = jax.lax.dynamic_slice_in_dim(
        virtual_topology, (remaining_actions - 1) * 2, 3
    )
    requested_index_d = jax.lax.dynamic_slice_in_dim(requested_indices, 0, 1)
    # Get index of previous selected node
    prev_selected_node = jnp.where(
        virtual_topology == requested_index_d,
        state.action_history,
        jnp.full(virtual_topology.shape, -1),
    )
    # will be current index if node only occurs once in virtual topology or will be different index if occurs more than once
    prev_selected_index = jnp.argmax(prev_selected_node).astype(dtype_config.LARGE_INT_DTYPE)
    prev_selected_node_d = jax.lax.dynamic_slice_in_dim(
        state.action_history, prev_selected_index, 1
    )

    # If first action, source and dest both to be assigned -> just mask all nodes based on resources
    # Thereafter, source must be previous dest. Dest can be any node (except previous allocations).
    state = state.replace(
        node_mask_s=jax.lax.cond(
            jnp.equal(remaining_actions, total_actions),
            lambda x: jnp.where(
                state.node_capacity_array >= node_request_s,
                x,
                jnp.zeros(num_nodes).astype(jnp.float32),
            ),
            lambda x: jnp.where(
                node_indices == prev_dest, x, jnp.zeros(num_nodes).astype(jnp.float32)
            ),
            jnp.ones(num_nodes).astype(jnp.float32),
        )
    )
    state = state.replace(
        node_mask_d=jnp.where(
            state.node_capacity_array >= node_request_d,
            jnp.ones(num_nodes).astype(jnp.float32),
            jnp.zeros(num_nodes).astype(jnp.float32),
        )
    )
    # If not first move, set node_mask_d to zero wherever node_mask_s is 1
    # to avoid same node selection for s and d
    state = state.replace(
        node_mask_d=jax.lax.cond(
            jnp.equal(remaining_actions, total_actions),
            lambda x: x,
            lambda x: jnp.where(
                state.node_mask_s == 1, jnp.zeros(num_nodes).astype(jnp.float32), x
            ),
            state.node_mask_d,
        )
    )

    def mask_previous_selections(i, val):
        # Disallow previously allocated nodes
        def update_slice(j, x):
            return jax.lax.dynamic_update_slice_in_dim(x, jnp.array([0.0]), j, axis=0)

        val = jax.lax.cond(
            i % 2 == 0,
            lambda x: update_slice(x[0][i], x[1]),  # i is node request index
            lambda x: update_slice(
                x[0][i + 1], x[1]
            ),  # i is slot request index (so add 1 to get next node)
            (state.action_history, val),
        )
        return val

    state = state.replace(
        node_mask_d=jax.lax.fori_loop(
            remaining_actions * 2,
            state.action_history.shape[0] - 1,
            mask_previous_selections,
            state.node_mask_d,
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
            (
                jnp.zeros(num_nodes).astype(jnp.float32),
                jnp.ones(num_nodes).astype(jnp.float32),
                state.node_mask_d,
            ),
        )
    )
    return state


@partial(jax.jit, donate_argnums=(0,))
def undo_node_action(state: VONEEnvState) -> VONEEnvState:
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
    resources = jnp.sum(
        jnp.where(mask == 1, state.node_resource_array, 0),
        axis=1,
        promote_integers=False,
    )
    state = state.replace(
        node_capacity_array=state.node_capacity_array + resources,
        node_resource_array=jnp.where(mask == 1, 0, state.node_resource_array),
        node_departure_array=jnp.where(mask == 1, jnp.inf, state.node_departure_array),
    )
    return state


@partial(jax.jit, static_argnums=(1,))
def remove_expired_node_requests(state: VONEEnvState, params: VONEEnvParams) -> VONEEnvState:
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
    expired_resources = jnp.sum(
        jnp.where(mask == 1, state.node_resource_array, 0),
        axis=1,
        promote_integers=False,
    )
    state = state.replace(
        node_capacity_array=state.node_capacity_array + expired_resources,
        node_resource_array=jnp.where(mask == 1, 0, state.node_resource_array),
        node_departure_array=jnp.where(mask == 1, jnp.inf, state.node_departure_array),
    )
    return state


def update_action_history(
    action_history: chex.Array, action_counter: chex.Array, action: chex.Array
) -> chex.Array:
    """Update action history by adding action to first available index starting from the end.

    Args:
        action_history: Action history
        action_counter: Action counter
        action: Action to add to history

    Returns:
        Updated action_history
    """
    return jax.lax.dynamic_update_slice(
        action_history,
        jnp.flip(action, axis=0).astype(action_history.dtype),
        ((action_counter[-1] - 1) * 2,),
    )
