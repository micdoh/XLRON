from typing import Tuple, Union, Optional
from flax import struct
from functools import partial
import pathlib
import math
import chex
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import jraph
from gymnax.environments import environment, spaces
from xlron.environments.env_funcs import (
    HashableArrayWrapper, EnvState, EnvParams, init_rsa_request_array, init_link_slot_array, init_path_link_array,
    convert_node_probs_to_traffic_matrix, init_link_slot_mask, init_link_slot_departure_array, init_traffic_matrix,
    implement_rsa_action, check_rsa_action, undo_link_slot_action, finalise_rsa_action, generate_rsa_request,
    mask_slots, make_graph, init_path_length_array, init_modulations_array, init_path_se_array, required_slots,
    init_values_bandwidth, calculate_path_stats, normalise_traffic_matrix, init_graph_tuple, init_link_length_array,
    init_link_capacity_array, init_path_capacity_array, init_path_index_array, mask_slots_rwa_lightpath_reuse,
    implement_rwa_lightpath_reuse_action, check_rwa_lightpath_reuse_action, pad_array
)


@struct.dataclass
class RSAEnvState(EnvState):
    link_slot_array: chex.Array
    request_array: chex.Array
    link_slot_departure_array: chex.Array
    link_slot_mask: chex.Array
    traffic_matrix: chex.Array
    values_bw: chex.Array


@struct.dataclass
class RSAEnvParams(EnvParams):
    num_nodes: chex.Scalar = struct.field(pytree_node=False)
    num_links: chex.Scalar = struct.field(pytree_node=False)
    link_resources: chex.Scalar = struct.field(pytree_node=False)
    k_paths: chex.Scalar = struct.field(pytree_node=False)
    mean_service_holding_time: chex.Scalar = struct.field(pytree_node=False)
    load: chex.Scalar = struct.field(pytree_node=False)
    arrival_rate: chex.Scalar = struct.field(pytree_node=False)
    path_link_array: chex.Array = struct.field(pytree_node=False)
    random_traffic: bool = struct.field(pytree_node=False)
    max_slots: chex.Scalar = struct.field(pytree_node=False)
    path_se_array: chex.Array = struct.field(pytree_node=False)
    deterministic_requests: bool = struct.field(pytree_node=False)
    list_of_requests: chex.Array = struct.field(pytree_node=False)
    multiple_topologies: bool = struct.field(pytree_node=False)


@struct.dataclass
class DeepRMSAEnvState(RSAEnvState):
    path_stats: chex.Array


@struct.dataclass
class DeepRMSAEnvParams(RSAEnvParams):
    pass


@struct.dataclass
class RWALightpathReuseEnvState(RSAEnvState):
    path_index_array: chex.Array  # Contains indices of lightpaths in use on slots
    path_capacity_array: chex.Array  # Contains remaining capacity of each lightpath
    link_capacity_array: chex.Array  # Contains remaining capacity of lightpath on each link-slot


@struct.dataclass
class RWALightpathReuseEnvParams(RSAEnvParams):
    pass


class RSAEnv(environment.Environment):
    """Jittable abstract base class for all gymnax Environments.
    This environment simulates the Routing Modulation and Spectrum Assignment (RMSA) problem.
    It can model RSA by setting consider_modulation_format=False in params.
    It can model RWA by setting min_bw=0, max_bw=0, and consider_modulation_format=False in params.
    """
    def __init__(
            self,
            key: chex.PRNGKey,
            params: RSAEnvParams,
            values_bw: chex.Array = jnp.array([0]),
            traffic_matrix: chex.Array = None
    ):
        super().__init__()
        state = RSAEnvState(
            current_time=0,
            holding_time=0,
            total_timesteps=0,
            total_requests=-1,
            link_slot_array=init_link_slot_array(params),
            link_slot_departure_array=init_link_slot_departure_array(params),
            request_array=init_rsa_request_array(),
            link_slot_mask=init_link_slot_mask(params, agg=params.aggregate_slots),
            traffic_matrix=traffic_matrix if traffic_matrix is not None else init_traffic_matrix(key, params),
            values_bw=values_bw,
            graph=None,
            full_link_slot_mask=init_link_slot_mask(params),
            accepted_services=0,
            accepted_bitrate=0.,
        )
        self.initial_state = state.replace(graph=init_graph_tuple(state, params))

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: RSAEnvState,
        action: Union[int, float],
        params: Optional[RSAEnvParams] = None,
    ) -> Tuple[chex.Array, RSAEnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jnp.where(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info
        )

    @partial(jax.jit, static_argnums=(0, 2,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[RSAEnvParams] = None
    ) -> Tuple[chex.Array, RSAEnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: RSAEnvState,
        action: Union[int, float],
        params: RSAEnvParams,
    ) -> Tuple[chex.Array, RSAEnvState, chex.Array, chex.Array, chex.Array]:
        """Environment-specific step transition."""
        # Do action
        implement_action = implement_rwa_lightpath_reuse_action \
            if params.__class__.__name__ == "RWALightpathReuseEnvParams" else implement_rsa_action
        state = implement_action(state, action, params)
        # Check if action was valid, calculate reward
        check = check_rwa_lightpath_reuse_action(state, params, action) \
            if params.__class__.__name__ == "RWALightpathReuseEnvParams" else check_rsa_action(state)
        state, reward = jax.lax.cond(
            check,  # Fail if true
            lambda x: (undo_link_slot_action(x), self.get_reward_failure(x)),
            lambda x: (finalise_rsa_action(x), self.get_reward_success(x)),  # Finalise actions if complete
            state
        )
        # TODO - write separate functions for deterministic transition (above) and stochastic transition (below)
        # Generate new request
        state = generate_rsa_request(key, state, params)
        state = state.replace(total_timesteps=state.total_timesteps + 1)
        # Terminate if max_timesteps or max_requests exceeded or, if consecutive loading,
        # then terminate if reward is failure but not before min number of timesteps before update
        if params.continuous_operation:
            done = jnp.array(False)
        elif params.end_first_blocking:
            done = jnp.array(reward == self.get_reward_failure())
        else:
            done = self.is_terminal(state, params)
        info = {}
        # Calculate path stats if DeepRMSAEnv
        if params.__class__.__name__ == "DeepRMSAEnvParams":
            path_stats = calculate_path_stats(state, params, state.request_array)
            state = state.replace(path_stats=path_stats)
        else:
            # Update graph tuple
            state = state.replace(graph=init_graph_tuple(state, params))
        return self.get_obs(state, params), state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2,))
    def reset_env(
        self, key: chex.PRNGKey, params: RSAEnvParams
    ) -> Tuple[chex.Array, RSAEnvState]:
        """Environment-specific reset."""
        #if params.multiple_topologies:
            # TODO - implement this (shuffle through topologies and use the top of the stack)
            # Question - do i need to rewrite every function to take in a params argument and index params[0]?
            # maybe in make_rsa_env function can have a params field that holds all the params (one for each topology),
            # and then cycle select from them randomly and replace the top-level params with the selected one.
            # Then need to init() the env again in order to update the state using the params
        #    raise NotImplementedError
        if params.random_traffic:
            key, key_traffic = jax.random.split(key)
            state = self.initial_state.replace(
                traffic_matrix=init_traffic_matrix(key_traffic, params)
            )
        else:
            state = self.initial_state
        state = generate_rsa_request(key, state, params)
        return self.get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0, 2,))
    def action_mask(self, state: RSAEnvState, params: RSAEnvParams) -> RSAEnvState:
        """Returns mask of valid actions.

        1. Check request for source and destination nodes

        2. For each path, mask out (0) initial slots that are not valid
        """
        state = mask_slots(state, params, state.request_array)
        return state

    @partial(jax.jit, static_argnums=(0, 2,))
    def get_obs_unflat(self, state: RSAEnvState, params: RSAEnvParams) -> Tuple[chex.Array, chex.Array]:
        """Applies observation function to state."""
        return (
            state.request_array,
            state.link_slot_array,
        )

    @partial(jax.jit, static_argnums=(0, 2,))
    def get_obs(self, state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
        """Applies observation function to state."""
        return jnp.concatenate(
            (
                jnp.reshape(state.request_array, (-1,)),
                jnp.reshape(state.link_slot_array, (-1,)),
            ),
            axis=0,
        )

    def is_terminal(self, state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
        """Check whether state transition is terminal."""
        return jnp.logical_or(
            jnp.array(state.total_requests >= params.max_requests),
            jnp.array(state.total_timesteps >= params.max_timesteps)
        )

    def discount(self, state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    def get_reward_failure(self, state: Optional[EnvState] = None) -> chex.Array:
        """Return reward for current state."""
        return jnp.array(-1.0)

    def get_reward_success(self, state: Optional[EnvState] = None) -> chex.Array:
        """Return reward for current state."""
        return jnp.array(1.0)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @staticmethod
    def num_actions(params: RSAEnvParams) -> int:
        """Number of actions possible in environment."""
        return math.ceil(params.link_resources/params.aggregate_slots) * params.k_paths

    def action_space(self, params: RSAEnvParams):
        """Action space of the environment."""
        return spaces.Discrete(params.link_resources * params.k_paths)

    def observation_space(self, params: RSAEnvParams):
        """Observation space of the environment."""
        return spaces.Discrete(
            3 +  # Request array
            params.num_links * params.link_resources  # Link slot array
        )

    def state_space(self, params: RSAEnvParams):
        """State space of the environment."""
        return spaces.Dict(
            {
                "current_time": spaces.Discrete(1),
                "request_array": spaces.Discrete(3),
                "link_slot_array": spaces.Discrete(params.num_links * params.link_resources),
                "link_slot_departure_array": spaces.Discrete(params.num_links * params.link_resources),
            }
        )

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters."""
        return make_rsa_env()[1]


class DeepRMSAEnv(RSAEnv):

    def __init__(
            self,
            key: chex.PRNGKey,
            params: RSAEnvParams,
            values_bw: chex.Array = jnp.array([0]),
            traffic_matrix: chex.Array = None
    ):
        super().__init__(key, params, values_bw=values_bw, traffic_matrix=traffic_matrix)
        self.initial_state = DeepRMSAEnvState(
            current_time=0,
            holding_time=0,
            total_timesteps=0,
            total_requests=-1,
            link_slot_array=init_link_slot_array(params),
            link_slot_departure_array=init_link_slot_departure_array(params),
            request_array=init_rsa_request_array(),
            link_slot_mask=jnp.ones(params.k_paths),
            traffic_matrix=traffic_matrix if traffic_matrix is not None else init_traffic_matrix(key, params),
            values_bw=values_bw,
            path_stats=calculate_path_stats(self.initial_state, params, self.initial_state.request_array),
            graph=None,
            accepted_services=0,
            accepted_bitrate=0.,
        )

    def step_env(
            self,
            key: chex.PRNGKey,
            state: DeepRMSAEnvState,
            action: Union[int, float],
            params: RSAEnvParams,
    ) -> Tuple[chex.Array, RSAEnvState, chex.Array, chex.Array, chex.Array]:
        """Environment-specific step transition."""
        # Do action
        # state = mask_slots(state, params, state.request_array)
        # mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
        # # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
        # mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
        # # Get index of first available slots for each path
        # first_slots = jax.vmap(jnp.argmax, in_axes=(0))(mask)
        # slot_index = first_slots[action] % params.link_resources
        # # Convert indices to action
        # action = action * params.link_resources + slot_index
        # TODO - alter this if allowing J>1
        slot_index = jnp.squeeze(jax.lax.dynamic_slice(state.path_stats, (action, 2), (1, 1)))
        action = action * params.link_resources + slot_index * params.link_resources  # Undo normalisation
        return super().step_env(key, state, action, params)

    @partial(jax.jit, static_argnums=(0, 2,))
    def action_mask(self, state: RSAEnvState, params: RSAEnvParams) -> RSAEnvState:
        mask = jnp.where(state.path_stats[:, 3] >= 1, 1., 0.)
        # If mask is all zeros, make all ones
        mask = jnp.where(jnp.sum(mask) == 0, 1., mask)
        state = state.replace(link_slot_mask=mask)
        return state

    def action_space(self, params: RSAEnvParams):
        """Action space of the environment."""
        return spaces.Discrete(params.k_paths)

    @staticmethod
    def num_actions(params: RSAEnvParams) -> int:
        """Number of actions possible in environment."""
        return params.k_paths

    @partial(jax.jit, static_argnums=(0, 2,))
    def get_obs(self, state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
        """Applies observation function to state."""
        request = state.request_array
        s = jax.lax.dynamic_slice(request, (0,), (1,))
        s = jax.nn.one_hot(s, params.num_nodes)
        d = jax.lax.dynamic_slice(request, (2,), (1,))
        d = jax.nn.one_hot(d, params.num_nodes)
        request_encoding = s + d
        return jnp.concatenate(
            (
                #jnp.reshape(request_encoding, (-1,)),
                jnp.reshape(s, (-1,)),
                jnp.reshape(d, (-1,)),
                jnp.reshape(state.holding_time, (-1,)),
                jnp.reshape(state.path_stats, (-1,)),
            ),
            axis=0,
        )

    def observation_space(self, params: RSAEnvParams):
        """Observation space of the environment."""
        return spaces.Discrete(
            params.num_nodes  # Request encoding
            + params.num_nodes
            + 1  # Holding time
            + params.k_paths * 5  # Path stats
        )


class RWALightpathReuseEnv(RSAEnv):
    # TODO - need to keep track of active requests to enable
    #  dynamic RWA with lightpath reuse (as opposed to just incremental loading) OR just randomly remove connections

    def __init__(
            self,
            key: chex.PRNGKey,
            params: RSAEnvParams,
            values_bw: chex.Array = jnp.array([0]),
            traffic_matrix: chex.Array = None,
            path_capacity_array: chex.Array = None,
    ):
        super().__init__(key, params, values_bw=values_bw, traffic_matrix=traffic_matrix)
        state = RWALightpathReuseEnvState(
            current_time=0,
            holding_time=0,
            total_timesteps=0,
            total_requests=-1,
            link_slot_array=init_link_slot_array(params),
            link_slot_departure_array=init_link_slot_departure_array(params),
            request_array=init_rsa_request_array(),
            link_slot_mask=init_link_slot_mask(params, agg=params.aggregate_slots),
            traffic_matrix=traffic_matrix if traffic_matrix is not None else init_traffic_matrix(key, params),
            values_bw=values_bw,
            graph=None,
            full_link_slot_mask=init_link_slot_mask(params),
            path_index_array=init_path_index_array(params),
            path_capacity_array=path_capacity_array,
            link_capacity_array=init_link_capacity_array(params),
            accepted_services=0,
            accepted_bitrate=0.,
        )
        self.initial_state = state.replace(graph=init_graph_tuple(state, params))

    @partial(jax.jit, static_argnums=(0, 2,))
    def action_mask(self, state: RSAEnvState, params: RSAEnvParams) -> RSAEnvState:
        """Returns mask of valid actions."""
        state = mask_slots_rwa_lightpath_reuse(state, params, state.request_array)
        return state


def make_rsa_env(config):
    """Create RSA environment.
    Args:
        config: Configuration dictionary
    Returns:
        env: RSA environment
        params: RSA environment parameters
    """
    seed = config.get("seed", 0)
    topology_name = config.get("topology_name", "conus")
    load = config.get("load", 100)
    k = config.get("k", 5)
    mean_service_holding_time = config.get("mean_service_holding_time", 10)
    incremental_loading = config.get("incremental_loading", False)
    end_first_blocking = config.get("end_first_blocking", False)
    random_traffic = config.get("random_traffic", False)
    max_requests = config.get("max_requests", 1e4)
    max_timesteps = config.get("max_timesteps", 1e4)
    link_resources = config.get("link_resources", 100)
    values_bw = config.get("values_bw", None)
    node_probabilities = config.get("node_probabilities", None)
    if values_bw:
        values_bw = [int(val) for val in values_bw]
    slot_size = config.get("slot_size", 12.5)
    min_bw = config.get("min_bw", 25)
    max_bw = config.get("max_bw", 100)
    step_bw = config.get("step_bw", 1)
    env_type = config.get("env_type", "").lower()
    continuous_operation = config.get("continuous_operation", False)
    custom_traffic_matrix_csv_filepath = config.get("custom_traffic_matrix_csv_filepath", None)
    traffic_requests_csv_filepath = config.get("traffic_requests_csv_filepath", None)
    multiple_topologies_directory = config.get("multiple_topologies_directory", None)
    aggregate_slots = config.get("aggregate_slots", 1)
    disjoint_paths = config.get("disjoint_paths", False)
    guardband = config.get("guardband", 1)
    symbol_rate = config.get("symbol_rate", 100)
    weight = config.get("weight", None)
    remove_array_wrappers = config.get("remove_array_wrappers", False)

    rng = jax.random.PRNGKey(seed)
    rng, _, _, _, _ = jax.random.split(rng, 5)
    graph = make_graph(topology_name, topology_directory=config.get("topology_directory", None))
    arrival_rate = load / mean_service_holding_time
    num_nodes = len(graph.nodes)
    num_links = len(graph.edges)
    path_link_array = init_path_link_array(graph, k, disjoint=disjoint_paths, weight=weight, directed=graph.is_directed())
    if custom_traffic_matrix_csv_filepath:
        random_traffic = False  # Set this False so that traffic matrix isn't replaced on reset
        traffic_matrix = jnp.array(np.loadtxt(custom_traffic_matrix_csv_filepath, delimiter=","))
        traffic_matrix = normalise_traffic_matrix(traffic_matrix)
    elif node_probabilities:
        random_traffic = False  # Set this False so that traffic matrix isn't replaced on reset
        node_probabilities = [float(prob) for prob in config.get("node_probs")]
        traffic_matrix = convert_node_probs_to_traffic_matrix(node_probabilities)
    else:
        traffic_matrix = None

    if traffic_requests_csv_filepath:
        deterministic_requests = True
        # Remove headers from array
        list_of_requests = np.loadtxt(traffic_requests_csv_filepath, delimiter=",")[1:, :]
        list_of_requests = jnp.array(list_of_requests)
        max_requests = len(list_of_requests)
    else:
        deterministic_requests = False
        list_of_requests = jnp.array([])

    values_bw = init_values_bandwidth(min_bw, max_bw, step_bw, values_bw)

    if env_type[:3] == "rsa":
        consider_modulation_format = False
    elif env_type == "rwa":
        values_bw = jnp.array([0])
        consider_modulation_format = False
    elif env_type == "rwa_lightpath_reuse":
        consider_modulation_format = False
        # Set guardband to 0 and slot size to max bandwidth to ensure that requested slots is always 1 but
        # that the bandwidth request is still considered when updating link_capacity_array
        guardband = 0
        slot_size = int(max(values_bw))
    else:
        consider_modulation_format = True

    max_bw = max(values_bw)

    # Automated calculation of max slots requested
    if consider_modulation_format:
        link_length_array = init_link_length_array(graph).reshape((num_links, 1))
        path_length_array = init_path_length_array(path_link_array, graph)
        modulations_array = init_modulations_array(config.get("modulations_csv_filepath", None))
        path_se_array = init_path_se_array(path_length_array, modulations_array)
        min_se = min(path_se_array)  # if consider_modulation_format
        max_slots = required_slots(max_bw, min_se, slot_size, guardband=guardband)
    else:
        link_length_array = jnp.ones((num_links, 1))
        path_se_array = jnp.array([1])
        if env_type == "rwa_lightpath_reuse":
            scale_factor = config.get("scale_factor", 1.0)
            link_length_array = init_link_length_array(graph).reshape((num_links, 1))
            path_capacity_array = init_path_capacity_array(
                link_length_array, path_link_array, symbol_rate=symbol_rate, scale_factor=scale_factor
            )
            max_requests = int(scale_factor * max_requests)
        max_slots = required_slots(max_bw, 1, slot_size, guardband=guardband)

    if incremental_loading:
        mean_service_holding_time = load = 1e6

    # Define edges for use with heuristics and GNNs
    edges = jnp.array(sorted(graph.edges))

    if env_type == "deeprmsa":
        env_params = DeepRMSAEnvParams
    elif env_type == "rwa_lightpath_reuse":
        env_params = RWALightpathReuseEnvParams
    else:
        env_params = RSAEnvParams

    params = env_params(
        max_requests=max_requests,
        max_timesteps=max_timesteps,
        mean_service_holding_time=mean_service_holding_time,
        k_paths=k,
        link_resources=link_resources,
        num_nodes=num_nodes,
        num_links=num_links,
        load=load,
        arrival_rate=arrival_rate,
        path_link_array=HashableArrayWrapper(path_link_array),
        incremental_loading=incremental_loading,
        end_first_blocking=end_first_blocking,
        edges=HashableArrayWrapper(edges),
        random_traffic=random_traffic,
        path_se_array=HashableArrayWrapper(path_se_array),
        link_length_array=HashableArrayWrapper(link_length_array),
        max_slots=int(max_slots),
        consider_modulation_format=consider_modulation_format,
        slot_size=slot_size,
        continuous_operation=continuous_operation,
        aggregate_slots=aggregate_slots,
        guardband=guardband,
        deterministic_requests=deterministic_requests,
        list_of_requests=HashableArrayWrapper(list_of_requests),
        multiple_topologies=False,
        directed_graph=graph.is_directed(),
    ) if not remove_array_wrappers else env_params(
        max_requests=max_requests,
        max_timesteps=max_timesteps,
        mean_service_holding_time=mean_service_holding_time,
        k_paths=k,
        link_resources=link_resources,
        num_nodes=num_nodes,
        num_links=num_links,
        load=load,
        arrival_rate=arrival_rate,
        path_link_array=path_link_array,
        incremental_loading=incremental_loading,
        end_first_blocking=end_first_blocking,
        edges=edges,
        random_traffic=random_traffic,
        path_se_array=path_se_array,
        link_length_array=link_length_array,
        max_slots=int(max_slots),
        consider_modulation_format=consider_modulation_format,
        slot_size=slot_size,
        continuous_operation=continuous_operation,
        aggregate_slots=aggregate_slots,
        guardband=guardband,
        deterministic_requests=deterministic_requests,
        list_of_requests=list_of_requests,
        multiple_topologies=False,
        directed_graph=graph.is_directed(),
    )

    # If training single model on multiple topologies, must store params for each topology within top-level params
    if multiple_topologies_directory:
        # iterate through files in directory
        params_list = []
        p = pathlib.Path(multiple_topologies_directory).glob('**/*')
        files = [x for x in p if x.is_file()]
        config.update(multiple_topologies_directory=None, remove_array_wrappers=True)
        for file in files:
            # Get filename without extension
            config.update(topology_name=file.stem, topology_directory=file.parent)
            env, params = make_rsa_env(config)
            params = params.replace(multiple_topologies=True)
            params_list.append(params)
        # for params in params_list, concatenate the field from each params into one array per field
        # from https://stackoverflow.com/questions/73765064/jax-vmap-over-batch-of-dataclasses
        cls = type(params_list[0])
        fields = params_list[0].__dict__.keys()
        field_dict = {}
        for k in fields:
            values = [getattr(v, k) for v in params_list]
            #values = [list(v) if isinstance(v, chex.Array) else v for v in values]
            # Pad arrays to same shape
            padded_values = HashableArrayWrapper(jnp.array(pad_array(values, fill_value=0)))
            field_dict[k] = padded_values
        params = cls(**field_dict)

    if remove_array_wrappers:
        # Only remove array wrappers if multiple_topologies=True for the inner files loop above
        env = None
    else:
        if env_type == "deeprmsa":
            env = DeepRMSAEnv(rng, params, values_bw=values_bw, traffic_matrix=traffic_matrix)
        elif env_type == "rwa_lightpath_reuse":
            env = RWALightpathReuseEnv(
                rng, params, values_bw=values_bw, traffic_matrix=traffic_matrix, path_capacity_array=path_capacity_array)
        else:
            env = RSAEnv(rng, params, values_bw=values_bw, traffic_matrix=traffic_matrix)

    return env, params
