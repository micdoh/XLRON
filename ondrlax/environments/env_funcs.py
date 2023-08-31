from itertools import combinations, islice
from functools import partial
from typing import Tuple, Sequence, Any, Dict, Union, Optional, Generic, TypeVar
from gymnax.environments import environment, spaces
import pathlib
import networkx as nx
import jax.numpy as jnp
import chex
import jax
import timeit
import json
import numpy as np
from flax import struct
from jax._src import core
from jax._src import dtypes
from jax._src import prng
from jax._src.typing import Array, ArrayLike, DTypeLike
from typing import Generic, TypeVar


Shape = Sequence[int]
T = TypeVar('T')      # Declare type variable


class HashableArrayWrapper(Generic[T]):
    def __init__(self, val: T):
        self.val = val

    def __getattribute__(self, prop):
        if prop == 'val' or prop == "__hash__" or prop == "__eq__":
            return super(HashableArrayWrapper, self).__getattribute__(prop)
        return getattr(self.val, prop)

    def __getitem__(self, key):
        return self.val[key]

    def __setitem__(self, key, val):
        self.val[key] = val

    def __hash__(self):
        return hash(self.val.tobytes())

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return self.__hash__() == other.__hash__()

        f = getattr(self.val, "__eq__")
        return f(self, other)


@struct.dataclass
class EnvState:
    current_time: chex.Scalar
    holding_time: chex.Scalar
    total_timesteps: chex.Scalar
    total_requests: chex.Scalar


@struct.dataclass
class EnvParams:
    max_requests: chex.Scalar = struct.field(pytree_node=False)
    max_timesteps: chex.Scalar = struct.field(pytree_node=False)


class RolloutWrapper(object):
    def __init__(
        self,
        model_forward=None,
        env: environment.Environment = None,
        num_env_steps: Optional[int] = None,
        env_params: EnvParams = None,
    ):
        """Wrapper to define batch evaluation for generation parameters."""
        self.env = env
        # Define the RL environment & network forward function
        self.env_params = env_params
        self.model_forward = model_forward

        if num_env_steps is None:
            self.num_env_steps = self.env_params.max_requests
        else:
            self.num_env_steps = num_env_steps

    @partial(jax.jit, static_argnums=(0,))
    def population_rollout(self, rng_eval, policy_params):
        """Reshape parameter vector and evaluate the generation."""
        # Evaluate population of nets on gymnax task - vmap over rng & params
        pop_rollout = jax.vmap(self.batch_rollout, in_axes=(None, 0))
        return pop_rollout(rng_eval, policy_params)

    @partial(jax.jit, static_argnums=(0,))
    def batch_rollout(self, rng_eval, policy_params):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None))
        return batch_rollout(rng_eval, policy_params)

    @partial(jax.jit, static_argnums=(0,))
    def single_rollout(self, rng_input, policy_params):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            if self.model_forward is not None:
                action = self.model_forward(policy_params, obs, rng_net)
            else:
                action = self.env.action_space(self.env_params).sample(rng_net)
            next_obs, next_state, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                policy_params,
                rng,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_obs, done]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                policy_params,
                rng_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.num_env_steps,
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        return obs, action, reward, next_obs, done, cum_return

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape


class TimeIt:
    def __init__(self, tag, frames=None):
        self.tag = tag
        self.frames = frames

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (': Elapsed time=%.2fs' % self.elapsed_secs)
        if self.frames:
            msg += ', FPS=%.2e' % (self.frames / self.elapsed_secs)
        print(msg)


def init_path_link_array(graph, k):
    """Initialise path-link array
    Each path is defined by a link utilisation array. 1 indicates link corrresponding to index is used, 0 indicates not used."""
    def get_k_shortest_paths(g, source, target, k, weight=None):
        return list(
            islice(nx.shortest_simple_paths(g, source, target, weight=weight), k)
        )

    paths = []
    edges = graph.edges
    for node_pair in combinations(graph.nodes, 2):
        k_paths = get_k_shortest_paths(
            graph, node_pair[0], node_pair[1], k
        )
        for k_path in k_paths:
            link_usage = [0]*len(graph.edges)  # Initialise empty path
            for i in range(len(k_path)-1):
                s, d = k_path[i], k_path[i+1]
                for edge_index, edge in enumerate(edges):
                    if edge[0] == s and edge[1] == d or edge[0] == d and edge[1] == s:
                        link_usage[edge_index] = 1
            paths.append(link_usage)
    return jnp.array(paths)


def init_virtual_topology_patterns(pattern_names):
    """Initialise virtual topology patterns"""
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


def init_traffic_matrix(key: chex.PRNGKey, params: EnvParams):
    traffic_matrix = jax.random.uniform(key, shape=(params.num_nodes, params.num_nodes))
    diag_elements = jnp.diag_indices_from(traffic_matrix)
    # Set main diagonal to zero so no requests from node to itself
    traffic_matrix = traffic_matrix.at[diag_elements].set(0)
    traffic_matrix = normalise_traffic_matrix(traffic_matrix)
    return traffic_matrix


def init_values_nodes(min_value, max_value):
    return jnp.arange(min_value, max_value+1)


def init_values_slots(min_value, max_value):
    return jnp.arange(min_value, max_value+1)


@partial(jax.jit, static_argnums=(2, 3))
def get_path_indices(s, d, k, N):
    """Get path indices for a given source, destination and number of paths"""
    node_indices = jnp.arange(N, dtype=jnp.int32)
    indices_to_s = jnp.where(node_indices < s, node_indices, 0)
    # The following equation is based on the combinations formula
    return (N*s + d - jnp.sum(indices_to_s) - 2*s - 1) * k


def init_node_capacity_array(params: EnvParams):
    """Initialize node array either with uniform resources"""
    return jnp.array([params.node_resources] * params.num_nodes)


def init_link_slot_array(params: EnvParams):
    """Initialize link array either with uniform resources"""
    return jnp.zeros((params.num_links, params.link_resources))


def init_vone_request_array(params: EnvParams):
    """Initialize request array either with uniform resources"""
    return jnp.zeros((2, params.max_edges*2+1, ))


def init_rsa_request_array():
    """Initialize request array"""
    return jnp.zeros(3)


def init_node_mask(params: EnvParams):
    """Initialize node mask"""
    return jnp.ones(params.num_nodes)


def init_link_slot_mask(params: EnvParams):
    """Initialize link mask"""
    return jnp.ones(params.k_paths*params.link_resources)


def init_action_counter():
    """Initialize action counter.
    First index is num unique nodes, second index is total steps, final is remaining steps until completion of request."""
    return jnp.zeros(3, dtype=jnp.int32)


@jax.jit
def decrement_action_counter(state):
    """Decrement action counter in-place"""
    state.action_counter.at[-1].add(-1)
    return state


def decrease_last_element(array):
    last_value_mask = jnp.arange(array.shape[0]) == array.shape[0] - 1
    return jnp.where(last_value_mask, array - 1, array)


def init_node_departure_array(params: EnvParams):
    return jnp.full((params.num_nodes, params.node_resources), jnp.inf)


def init_link_slot_departure_array(params: EnvParams):
    return jnp.full((params.num_links, params.link_resources), jnp.inf)


def init_node_resource_array(params: EnvParams):
    """Array to track node resources occupied by virtual nodes"""
    return jnp.zeros((params.num_nodes, params.node_resources))


def init_action_history(params: EnvParams):
    """Initialize action history"""
    return jnp.full(params.max_edges*2+1, -1)


def normalise_traffic_matrix(traffic_matrix):
    """Normalise traffic matrix to sum to 1"""
    traffic_matrix /= jnp.sum(traffic_matrix)
    return traffic_matrix


def generate_vone_request(key: chex.PRNGKey, state: EnvState, params: EnvParams):
    """Generate a new request for the VONE environment.
    The request has two rows. The first row shows the node and slot values. The second row shows the virtual topology.
    The first three elements of the pattern show the number of unique nodes, the total number of steps, and the remaining steps.
    These first three elements comprise the action counter.
    The remaining elements of the  pattern show the virtual topology pattern, i.e. the connectivity of the virtual topology.
    """
    # TODO - update this to be bitrate requests rather than slots
    # Define the four possible patterns for the first row
    shape = state.request_array.shape[1]
    key_topology, key_node, key_slot, key_times = jax.random.split(key, 4)
    # Randomly select topology, node resources, slot resources
    pattern = jax.random.choice(key_topology, state.virtual_topology_patterns)
    action_counter = jax.lax.dynamic_slice(pattern, (0,), (3,))
    topology_pattern = jax.lax.dynamic_slice(pattern, (3,), (pattern.shape[0]-3,))
    selected_node_values = jax.random.choice(key_node, state.values_nodes, shape=(shape,))
    selected_slot_values = jax.random.choice(key_slot, state.values_slots, shape=(shape,))
    # Create a mask for odd and even indices
    mask = jnp.tile(jnp.array([0, 1]), (shape+1) // 2)[:shape]
    # Vectorized conditional replacement using mask
    first_row = jnp.where(mask, selected_slot_values, selected_node_values)
    first_row = jnp.where(topology_pattern == 0, 0, first_row)
    arrival_time, holding_time = generate_arrival_holding_times(key, params)
    state = state.replace(
        holding_time=holding_time,
        current_time=state.current_time + arrival_time,
        action_counter=action_counter,
        request_array=jnp.vstack((first_row, topology_pattern)),
        action_history=init_action_history(params),
        total_requests=state.total_requests + 1
    )
    state = remove_expired_node_requests(state)
    state = remove_expired_slot_requests(state)
    return state


def generate_rsa_request(key: chex.PRNGKey, state: EnvState, params: EnvParams) -> EnvState:
    # TODO - update this to be bitrate requests rather than slots
    # Flatten the probabilities to a 1D array
    shape = state.traffic_matrix.shape
    probabilities = state.traffic_matrix.ravel()
    key_sd, key_slot, key_times = jax.random.split(key, 3)
    # Use jax.random.choice to select index based on the probabilities
    source_dest_index = jax.random.choice(key_sd, jnp.arange(state.traffic_matrix.size), p=probabilities)
    # Convert 1D index back to 2D
    nodes = jnp.unravel_index(source_dest_index, shape)
    source, dest = jnp.sort(jnp.stack(nodes))
    # Vectorized conditional replacement using mask
    slots = jax.random.choice(key_slot, state.values_slots)
    arrival_time, holding_time = generate_arrival_holding_times(key_times, params)
    state = state.replace(
        holding_time=holding_time,
        current_time=state.current_time + arrival_time,
        request_array=jnp.stack((source, slots, dest)),
        total_requests=state.total_requests + 1
    )
    state = remove_expired_slot_requests(state)
    return state


@partial(jax.jit, static_argnums=(0,))
def get_paths(params, nodes):
    """Get k paths between source and destination"""
    # get source and destination nodes in order (for accurate indexing of path-link array)
    source, dest = jnp.sort(nodes)
    i = get_path_indices(source, dest, params.k_paths, params.num_nodes).astype(jnp.int32)
    index_array = jax.lax.dynamic_slice(jnp.arange(0, params.path_link_array.shape[0]), (i,), (params.k_paths,))
    return jnp.take(params.path_link_array.val, index_array, axis=0)


def generate_arrival_holding_times(key, params):
    """
    To understand how sampling from e^-x can be transformed to sample from lambda*e^-(x/lambda) see:
    https://en.wikipedia.org/wiki/Inverse_transform_sampling#Examples
    Basically, inverse transform sampling is used to sample from a distribution with CDF F(x).
    The CDF of the exponential distribution (lambda*e^-{lambda*x}) is F(x) = 1 - e^-{lambda*x}.
    Therefore, the inverse CDF is x = -ln(1-u)/lambda, where u is sample from uniform distribution.
    Also see: https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
    https://jax.readthedocs.io/en/latest/_autosummary/jax.random.exponential.html
    """
    key_arrival, key_holding = jax.random.split(key, 2)
    arrival_time = jax.random.exponential(key_arrival, shape=(1,)) \
                   / params.arrival_rate  # Divide because it is rate (lambda)
    holding_time = jax.random.exponential(key_holding, shape=(1,)) \
                   * params.mean_service_holding_time  # Multiply because it is mean (1/lambda)
    return arrival_time, holding_time


def update_action_history(action_history, action_counter, action):
    """Update action history"""
    return jax.lax.dynamic_update_slice(action_history, jnp.flip(action, axis=0).astype(jnp.int32), ((action_counter[-1]-1)*2,))


def update_link(link, initial_slot, num_slots, value):
    slot_indices = jnp.arange(link.shape[0])
    return jnp.where((initial_slot <= slot_indices) & (slot_indices < initial_slot+num_slots), link-value, link)


def update_path(link, link_in_path, initial_slot, num_slots, value):
    return jax.lax.cond(link_in_path == 1, lambda x: update_link(*x), lambda x: x[0], (link, initial_slot, num_slots, value))


@jax.jit
def vmap_update_path_links(link_array, path, initial_slot, num_slots, value):
    """Set relevant slots along links in path to current_val - val"""
    return jax.vmap(update_path, in_axes=(0, 0, None, None, None))(link_array, path, initial_slot, num_slots, value)


def update_link_departure(link, initial_slot, num_slots, value):
    slot_indices = jnp.arange(link.shape[0])
    return jnp.where((initial_slot <= slot_indices) & (slot_indices < initial_slot+num_slots), value, link)


def update_path_departure(link, link_in_path, initial_slot, num_slots, value):
    return jax.lax.cond(link_in_path == 1, lambda x: update_link_departure(*x), lambda x: x[0], (link, initial_slot, num_slots, value))


@jax.jit
def vmap_update_path_links_departure(link_array, path, initial_slot, num_slots, value):
    """Set relevant slots along links in path to current_val - val"""
    return jax.vmap(update_path_departure, in_axes=(0, 0, None, None, None))(link_array, path, initial_slot, num_slots, value)



def update_node_departure(node_row, inf_index, value):
    row_indices = jnp.arange(node_row.shape[0])
    return jnp.where(row_indices == inf_index, value, node_row)


def update_selected_node_departure(node_row, node_selected, first_inf_index, value):
    return jax.lax.cond(node_selected != 0, lambda x: update_node_departure(*x), lambda x: node_row, (node_row, first_inf_index, value))


@jax.jit
def vmap_update_node_departure(node_departure_array, selected_nodes, value):
    """Called when implementing node action.
    Adds request departure time ("value") to first "inf" i.e. unoccupied index on node departure array for selected nodes"""
    first_inf_indices = jnp.argmax(node_departure_array, axis=1)
    return jax.vmap(update_selected_node_departure, in_axes=(0, 0, 0, None))(node_departure_array, selected_nodes, first_inf_indices, value)


def update_node_resources(node_row, zero_index, value):
    row_indices = jnp.arange(node_row.shape[0])
    return jnp.where(row_indices == zero_index, value, node_row)


def update_selected_node_resources(node_row, request, first_zero_index):
    return jax.lax.cond(request != 0, lambda x: update_node_resources(*x), lambda x: node_row, (node_row, first_zero_index, request))


@jax.jit
def vmap_update_node_resources(node_resource_array, selected_nodes):
    first_zero_indices = jnp.argmin(node_resource_array, axis=1)
    return jax.vmap(update_selected_node_resources, in_axes=(0, 0, 0))(node_resource_array, selected_nodes, first_zero_indices)


def remove_expired_slot_requests(state: EnvState) -> EnvState:
    mask = jnp.where(state.link_slot_departure_array < jnp.squeeze(state.current_time), 1, 0)
    state = state.replace(
        link_slot_array=jnp.where(mask == 1, 0, state.link_slot_array),
        link_slot_departure_array=jnp.where(mask == 1, jnp.inf, state.link_slot_departure_array)
    )
    return state


def remove_expired_node_requests(state):
    mask = jnp.where(state.node_departure_array < jnp.squeeze(state.current_time), 1, 0)
    expired_resources = jnp.sum(jnp.where(mask == 1, state.node_resource_array, 0), axis=1)
    state = state.replace(
        node_capacity_array=state.node_capacity_array + expired_resources,
        node_resource_array=jnp.where(mask == 1, 0, state.node_resource_array),
        node_departure_array=jnp.where(mask == 1, jnp.inf, state.node_departure_array)
    )
    return state


def update_node_array(node_indices, array, node, request):
    return jnp.where(node_indices == node, array-request, array)


def undo_node_action(state):
    """If the request is unsuccessful i.e.e checks fail, then remove the partial resource allocation"""
    mask = jnp.where(state.node_departure_array < 0, 1, 0)
    resources = jnp.sum(jnp.where(mask == 1, state.node_resource_array, 0), axis=1)
    state = state.replace(
        node_capacity_array=state.node_capacity_array + resources,
        node_resource_array=jnp.where(mask == 1, 0, state.node_resource_array),
        node_departure_array=jnp.where(mask == 1, jnp.inf, state.node_departure_array),
    )
    return state


def undo_link_slot_action(state):
    mask = jnp.where(state.link_slot_departure_array < 0, 1, 0)
    state = state.replace(
        link_slot_array=jnp.where(mask == 1, 0, state.link_slot_array),
        link_slot_departure_array=jnp.where(mask == 1, jnp.inf, state.link_slot_departure_array)
    )
    return state


@jax.jit
def check_unique_nodes(node_departure_array):
    """Count negative values on each node (row) in node departure array, must not exceed 1
    Return False if check passed, True if check failed"""
    return jnp.any(jnp.sum(jnp.where(node_departure_array < 0, 1, 0), axis=1) > 1)


def check_all_nodes_assigned(node_departure_array, total_requested_nodes):
    """Count negative values on each node (row) in node departure array, sum them, must equal total requested_nodes
    Return False if check passed, True if check failed"""
    return jnp.sum(jnp.sum(jnp.where(node_departure_array < 0, 1, 0), axis=1)) != total_requested_nodes


def check_min_two_nodes_assigned(node_departure_array):
    """Count negative values on each node (row) in node departure array, sum them, must be 2 or greater.
    This check is important if e.g. an action contains 2 nodes the same therefore only assigns 1 node.
    Return False if check passed, True if check failed"""
    return jnp.sum(jnp.sum(jnp.where(node_departure_array < 0, 1, 0), axis=1)) <= 1


def check_node_capacities(capacity_array):
    """Sum selected nodes array and check less than node resources
    Return False if check passed, True if check failed"""
    return jnp.any(capacity_array < 0)


def check_no_spectrum_reuse(link_slot_array):
    """slot-=1 when used, should be zero when occupied, so check if any < -1 in slot array
    Return False if check passed, True if check failed"""
    #jax.debug.print("check_no_spectrum_reuse {}", link_slot_array, ordered=True)
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
    Return False if check passed, True if check failed
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


def implement_node_action(state: EnvState, s_node: chex.Array, d_node: chex.Array, s_request: chex.Array, d_request: chex.Array, n=2):
    """Update node capacity, node resource and node departure arrays

    Args:
        state (State): current state
        s_node (int): source node
        d_node (int): destination node
        s_request (int): source node request
        d_request (int): destination node request
        n (int, optional): number of nodes to implement. Defaults to 2.
    """
    node_indices = jnp.arange(state.node_capacity_array.shape[0])

    curr_selected_nodes = jnp.zeros(state.node_capacity_array.shape[0])
    curr_selected_nodes = update_node_array(node_indices, curr_selected_nodes, d_node, -d_request)  # -ve so that selected node is +ve (so that argmin works correctly for node resource array update)
    curr_selected_nodes = jax.lax.cond(n == 2, lambda x: update_node_array(*x), lambda x: x[1], (node_indices, curr_selected_nodes, s_node, -s_request))
    # TODO - experiment with jax.lax.fori_loop here to replace cond

    node_capacity_array = state.node_capacity_array - curr_selected_nodes

    node_resource_array = vmap_update_node_resources(state.node_resource_array, curr_selected_nodes)

    node_departure_array = vmap_update_node_departure(state.node_departure_array, curr_selected_nodes, -state.current_time-state.holding_time)

    state = state.replace(
        node_capacity_array=node_capacity_array,
        node_resource_array=node_resource_array,
        node_departure_array=node_departure_array
    )

    return state


def implement_path_action(state: EnvState, path: chex.Array, initial_slot_index: chex.Array, num_slots: chex.Array):
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
        link_slot_departure_array=vmap_update_path_links_departure(state.link_slot_departure_array, path, initial_slot_index, num_slots, -state.current_time-state.holding_time)
    )
    return state


def path_action_only(topology_pattern: chex.Array, action_counter: chex.Array, remaining_actions: chex.Scalar):
    """This is to check if node has already been assigned, therefore just need to assign slots (n=0)

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
    """Implement action to assign nodes (1, 2, or 0) and connecting slots on links.
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
    num_slots = jax.lax.dynamic_slice(request, (1,), (1,))
    node_request_d = jax.lax.dynamic_slice(request, (0, ), (1, ))
    nodes = action[::2]
    path_index = jnp.floor(action[1] / params.link_resources).astype(jnp.int32)
    initial_slot_index = jnp.mod(action[1], params.link_resources)
    path = get_paths(params, nodes)[path_index]
    #jax.debug.print("path {}", path, ordered=True)
    #jax.debug.print("slots {}", jnp.max(jnp.where(path.reshape(-1,1) == 1, state.link_slot_array, jnp.zeros(params.num_links).reshape(-1,1)), axis=0), ordered=True)
    #jax.debug.print("path_index {}", path_index, ordered=True)
    #jax.debug.print("initial_slot_index {}", initial_slot_index, ordered=True)
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
def implement_rsa_action(
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
    node_s = jax.lax.dynamic_slice(state.request_array, (0, ), (1, ))
    num_slots = jax.lax.dynamic_slice(state.request_array, (1, ), (1, ))
    node_d = jax.lax.dynamic_slice(state.request_array, (2, ), (1, ))
    nodes_sd = jnp.concatenate((node_s, node_d))
    path_index = jnp.floor(action[0] / state.link_slot_array.shape[0]).astype(jnp.int32)
    initial_slot_index = jnp.mod(action, state.link_slot_array.shape[0])
    path = get_paths(params, nodes_sd)[path_index]
    state = implement_path_action(state, path, initial_slot_index, num_slots)
    return state


def make_positive(x):
    return jnp.where(x < 0, -x, x)


def finalise_vone_action(state):
    """Turn departure times positive"""
    state = state.replace(
        node_departure_array=make_positive(state.node_departure_array),
        link_slot_departure_array=make_positive(state.link_slot_departure_array)
    )
    return state


def finalise_rsa_action(state):
    """Turn departure times positive"""
    state = state.replace(
        link_slot_departure_array=make_positive(state.link_slot_departure_array)
    )
    return state


def check_vone_action(state, remaining_actions, total_requested_nodes):
    """Check if action is valid.
    Return True if invalid, False if valid."""
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


def check_rsa_action(state):
    """Check if action is valid"""
    return jnp.any(jnp.stack((
        check_no_spectrum_reuse(state.link_slot_array),
    )))


def make_graph(topology_name: str = "conus"):
    """Create graph from topology"""

    topology_path = pathlib.Path(__file__).parents[2].absolute() / "topologies"
    # Create topology
    if topology_name == "conus":
        with open(topology_path/"conus.json") as f:
            graph = nx.node_link_graph(json.load(f))
    elif topology_name == "nsfnet":
        with open(topology_path/"nsfnet.json") as f:
            graph = nx.node_link_graph(json.load(f))
    elif topology_name == "4node":
        # 4 node ring
        graph = nx.from_numpy_array(jnp.array([[0, 1, 0, 1],
                                               [1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [1, 0, 1, 0]]))
    else:
        # 7 node ring
        graph = nx.from_numpy_array(jnp.array([[0, 1, 0, 0, 0, 0, 1],
                                               [1, 0, 1, 0, 0, 0, 0],
                                               [0, 1, 0, 1, 0, 0, 0],
                                               [0, 0, 1, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0, 1],
                                               [1, 0, 0, 0, 0, 1, 0]]))
    return graph


@partial(jax.jit, static_argnums=(1,))
def mask_slots(state: EnvState, params: EnvParams, request: chex.Array) -> EnvState:
    """Returns mask of valid actions.

    1. Check request for source and destination nodes

    2. For each path, mask out (0) initial slots that are not valid
    """
    node_s = jax.lax.dynamic_slice(request, (0,), (1,))
    requested_slots = jax.lax.dynamic_slice(request, (1,), (1,))
    node_d = jax.lax.dynamic_slice(request, (2,), (1,))
    nodes_sd = jnp.concatenate((node_s, node_d), axis=0)
    init_mask = jnp.zeros((params.link_resources * params.k_paths))
    # Get mask used to check if request will fit slots
    request_mask = jax.lax.dynamic_update_slice(
        jnp.zeros(params.max_slots*2), jnp.ones(params.max_slots), params.max_slots-requested_slots
    )
    # Then cut in half and flip
    request_mask = jnp.flip(jax.lax.dynamic_slice(request_mask, (0,), (params.max_slots,)), axis=0)

    def mask_path(i, mask):
        # Path consists of binary array indicating link indices used in path
        path = get_paths(params, nodes_sd)[i]
        path = path.reshape((params.num_links, 1))
        # Get links and collapse to single dimension
        slots = jnp.where(path, state.link_slot_array, jnp.zeros(params.link_resources))
        # Make any -1s positive then get max for each slot across links
        slots = jnp.max(jnp.absolute(slots), axis=0)
        # Add padding to slots at end
        slots = jnp.concatenate((slots, jnp.ones(params.max_slots)))

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
    state = state.replace(link_slot_mask=jax.lax.fori_loop(0, params.k_paths, mask_path, init_mask))
    return state


@partial(jax.jit, static_argnums=(1,))
def mask_nodes(state: EnvState, num_nodes: chex.Scalar) -> EnvState:
    """Returns mask of valid actions.
    """
    total_actions = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.action_counter, 1, 1))
    remaining_actions = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.action_counter, 2, 1))
    full_request = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.request_array, 0, 1))
    virtual_topology = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.request_array, 1, 1))
    request = jax.lax.dynamic_slice_in_dim(full_request, (remaining_actions - 1) * 2, 3)
    node_request_s = jax.lax.dynamic_slice_in_dim(request, 2, 1)
    node_request_d = jax.lax.dynamic_slice_in_dim(request, 0, 1)
    prev_action = jax.lax.dynamic_slice_in_dim(state.action_history, (remaining_actions - 1) * 2, 3)
    prev_source = jax.lax.dynamic_slice_in_dim(prev_action, 2, 1)
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
                node_indices == prev_source,
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
            lambda x: update_slice(*x),  # i is node request index
            lambda x: update_slice(x[0]+1, x[1]),  # i is slot request index (so add 1 to get next node)
            (i, val),
        )
        return val

    state = state.replace(
        node_mask_d=jax.lax.fori_loop(
            (remaining_actions)*2,
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

if __name__ == "__main__":
    make_graph("conus")