from typing import Tuple, Union, Optional
from flax import struct
from functools import partial
import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from xlron.environments.env_funcs import (
    HashableArrayWrapper, EnvState, EnvParams, init_rsa_request_array, init_link_slot_array, init_path_link_array,
    init_values_slots, init_link_slot_mask, init_link_slot_departure_array, init_traffic_matrix, implement_rsa_action,
    check_rsa_action, undo_link_slot_action, finalise_rsa_action, generate_rsa_request, mask_slots, make_graph,
    init_path_length_array, init_modulations_array, init_path_se_array, required_slots, init_values_bandwidth
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
    uniform_traffic: bool = struct.field(pytree_node=False)
    max_slots: chex.Scalar = struct.field(pytree_node=False)
    path_se_array: chex.Array = struct.field(pytree_node=False)


class RSAEnv(environment.Environment):
    """Jittable abstract base class for all gymnax Environments.
    This environment simulates the Routing Modulation and Spectrum Assignment (RMSA) problem.
    It can model RSA by setting consider_modulation_format=False in params.
    It can model RWA by setting min_bw=0, max_bw=0, and consider_modulation_format=False in params.
    """
    def __init__(self, key: chex.PRNGKey, params: RSAEnvParams, values_bw: chex.Array = jnp.array([0])):
        super().__init__()
        self.initial_state = RSAEnvState(
            current_time=0,
            holding_time=0,
            total_timesteps=0,
            total_requests=-1,
            link_slot_array=init_link_slot_array(params),
            link_slot_departure_array=init_link_slot_departure_array(params),
            request_array=init_rsa_request_array(),
            link_slot_mask=init_link_slot_mask(params),
            traffic_matrix=init_traffic_matrix(key, params),
            values_bw=values_bw,
        )

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
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
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
        state = implement_rsa_action(state, action, params)
        # Check if action was valid, calculate reward
        check = check_rsa_action(state)
        state, reward = jax.lax.cond(
            check,  # Fail if true
            lambda x: (undo_link_slot_action(x), self.get_reward_failure(x)),
            lambda x: (finalise_rsa_action(x), self.get_reward_success(x)),  # Finalise actions if complete
            state
        )
        # Generate new request
        state = generate_rsa_request(key, state, params)
        state = state.replace(total_timesteps=state.total_timesteps + 1)
        # Terminate if max_timesteps or max_requests exceeded or, if consecutive loading,
        # then terminate if reward is failure but not before min number of timesteps before update
        done = self.is_terminal(state, params) \
            if not params.consecutive_loading else jnp.array(reward == self.get_reward_failure())
        info = {}
        return self.get_obs(state), state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2,))
    def reset_env(
        self, key: chex.PRNGKey, params: RSAEnvParams
    ) -> Tuple[chex.Array, RSAEnvState]:
        """Environment-specific reset."""
        if params.uniform_traffic:
            state = self.initial_state
        else:
            key, key_traffic = jax.random.split(key)
            state = self.initial_state.replace(
                traffic_matrix=init_traffic_matrix(key_traffic, params)
            )
        state = generate_rsa_request(key, state, params)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0, 2,))
    def action_mask(self, state: RSAEnvState, params: RSAEnvParams) -> RSAEnvState:
        """Returns mask of valid actions.

        1. Check request for source and destination nodes

        2. For each path, mask out (0) initial slots that are not valid
        """
        state = mask_slots(state, params, state.request_array)
        return state

    def get_obs_unflat(self, state: RSAEnvState) -> Tuple[chex.Array]:
        """Applies observation function to state."""
        return (
            state.request_array,
            state.node_capacity_array,
            state.link_slot_array,
        )

    def get_obs(self, state: RSAEnvState) -> chex.Array:
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
        return params.link_resources * params.k_paths

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


# TODO - make this like make_vone_env
def make_rsa_env(config):
    """Create RSA environment.
    Args:
        k: number of paths to consider
        load: load in Erlangs
        topology_name: name of topology to use
        mean_service_holding_time: mean service holding time
        link_resources: number of resources per link
        max_requests: maximum number of requests
        min_slots: minimum number of slots per link
        max_slots: maximum number of slots per link
        consecutive_loading: whether to use consecutive loading
    Returns:
        env: RSA environment
        params: RSA environment parameters
    """
    seed = config.get("seed", 0)
    topology_name = config.get("topology_name", "conus")
    load = config.get("load", 100)
    k = config.get("k", 5)
    mean_service_holding_time = config.get("mean_service_holding_time", 10)
    consecutive_loading = config.get("consecutive_loading", False)
    uniform_traffic = config.get("uniform_traffic", True)
    max_requests = config.get("max_requests", 1e4)
    max_timesteps = config.get("max_timesteps", 1e4)
    link_resources = config.get("link_resources", 100)
    values_bw = config.get("values_bw", None)
    slot_size = config.get("slot_size", 12.5)
    min_bw = config.get("min_bw", 25)
    max_bw = config.get("max_bw", 100)
    step_bw = config.get("step_bw", 25)

    rng = jax.random.PRNGKey(seed)
    rng, _, _, _, _ = jax.random.split(rng, 5)
    graph = make_graph(topology_name)
    arrival_rate = load / mean_service_holding_time
    num_nodes = len(graph.nodes)
    num_links = len(graph.edges)
    path_link_array = init_path_link_array(graph, k)

    values_bw = init_values_bandwidth(min_bw, max_bw, step_bw, values_bw)

    if config.get("env_type", "").lower()[:3] == "rsa":
        consider_modulation_format = False
    elif config.get("env_type", "").lower()[:3] == "rwa":
        values_bw = jnp.array([0])
        consider_modulation_format = False
    else:
        consider_modulation_format = True

    max_bw = max(values_bw)

    # Automated calculation of max slots requested
    if consider_modulation_format:
        path_length_array = init_path_length_array(path_link_array, graph)
        modulations_array = init_modulations_array(config.get("modulations_file", "modulations.csv"))
        path_se_array = init_path_se_array(path_length_array, modulations_array)
        min_se = min(path_se_array)  # if consider_modulation_format
        max_slots = required_slots(max_bw, min_se, slot_size)
    else:
        path_se_array = jnp.array([1])
        max_slots = required_slots(max_bw, 1, slot_size)

    if consecutive_loading:
        mean_service_holding_time = load = 1e6

    # Define edges for use with heuristics and GNNs
    edges = jnp.array(sorted(graph.edges))

    params = RSAEnvParams(
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
        consecutive_loading=consecutive_loading,
        edges=HashableArrayWrapper(edges),
        uniform_traffic=uniform_traffic,
        path_se_array=HashableArrayWrapper(path_se_array),
        max_slots=int(max_slots),
        consider_modulation_format=consider_modulation_format,
        slot_size=slot_size,
    )

    env = RSAEnv(rng, params, values_bw=values_bw)

    return env, params
