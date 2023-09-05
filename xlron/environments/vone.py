from typing import Tuple, Union, Optional
from flax import struct
from functools import partial
import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from xlron.environments.env_funcs import (
    HashableArrayWrapper, EnvState, EnvParams, init_vone_request_array, init_link_slot_array, init_path_link_array,
    init_values_slots, init_link_slot_mask, init_link_slot_departure_array, implement_vone_action,
    check_vone_action, undo_link_slot_action, finalise_vone_action, generate_vone_request, mask_slots, make_graph,
    init_node_capacity_array, init_node_mask, init_node_resource_array, init_node_departure_array, init_values_nodes,
    init_action_counter, init_action_history, update_action_history, decrease_last_element, undo_node_action,
    init_virtual_topology_patterns, mask_nodes
)


@struct.dataclass
class VONEEnvState(EnvState):
    link_slot_array: chex.Array
    node_capacity_array: chex.Array
    node_resource_array: chex.Array
    node_departure_array: chex.Array
    link_slot_departure_array: chex.Array
    request_array: chex.Array
    action_counter: chex.Array
    action_history: chex.Array
    node_mask_s: chex.Array
    link_slot_mask: chex.Array
    node_mask_d: chex.Array
    virtual_topology_patterns: chex.Array
    values_nodes: chex.Array
    values_slots: chex.Array


@struct.dataclass
class VONEEnvParams(EnvParams):
    num_nodes: chex.Scalar = struct.field(pytree_node=False)
    num_links: chex.Scalar = struct.field(pytree_node=False)
    node_resources: chex.Scalar = struct.field(pytree_node=False)
    link_resources: chex.Scalar = struct.field(pytree_node=False)
    k_paths: chex.Scalar = struct.field(pytree_node=False)
    load: chex.Scalar = struct.field(pytree_node=False)
    mean_service_holding_time: chex.Scalar = struct.field(pytree_node=False)
    arrival_rate: chex.Scalar = struct.field(pytree_node=False)
    max_edges: chex.Scalar = struct.field(pytree_node=False)
    min_slots: chex.Scalar = struct.field(pytree_node=False)
    max_slots: chex.Scalar = struct.field(pytree_node=False)
    min_node_resources: chex.Scalar = struct.field(pytree_node=False)
    max_node_resources: chex.Scalar = struct.field(pytree_node=False)
    path_link_array: chex.Array = struct.field(pytree_node=False)
    # TODO - Add Laplacian matrix (for node heuristics and might be useful for GNNs)


class VONEEnv(environment.Environment):
    """Jittable abstract base class for all gymnax Environments."""
    def __init__(self, params: VONEEnvParams, virtual_topologies=["3_ring"]):
        super().__init__()
        self.initial_state = VONEEnvState(
            current_time=0,
            holding_time=0,
            total_timesteps=0,
            total_requests=-1,
            link_slot_array=init_link_slot_array(params),
            link_slot_departure_array=init_link_slot_departure_array(params),
            node_capacity_array=init_node_capacity_array(params),
            node_resource_array=init_node_resource_array(params),
            node_departure_array=init_node_departure_array(params),
            request_array=init_vone_request_array(params),
            action_counter=init_action_counter(),
            action_history=init_action_history(params),
            node_mask_s=init_node_mask(params),
            link_slot_mask=init_link_slot_mask(params),
            node_mask_d=init_node_mask(params),
            virtual_topology_patterns=init_virtual_topology_patterns(virtual_topologies),
            values_nodes=init_values_nodes(params.min_node_resources, params.max_node_resources),
            values_slots=init_values_slots(params.min_slots, params.max_slots),
        )

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
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
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, chex.Array]:
        """Environment-specific step transition."""
        action = jnp.stack(action)  # NN outputs single array but env sample action is tuple of 3 single-value arrays
        # Find actions taken and remaining until end of request
        total_requested_nodes = jnp.squeeze(jax.lax.dynamic_slice(state.action_counter, (0, ), (1, )))
        total_actions = jnp.squeeze(jax.lax.dynamic_slice(state.action_counter, (1, ), (1, )))
        remaining_actions = jnp.squeeze(jax.lax.dynamic_slice(state.action_counter, (2, ), (1, )))
        # Do action
        state = implement_vone_action(
            state, action, total_actions, remaining_actions, params,
        )
        # Update history and counter
        state = state.replace(
            action_history=update_action_history(state.action_history, state.action_counter, action),
            action_counter=decrease_last_element(state.action_counter),
            total_timesteps=state.total_timesteps + 1,
        )
        # Check if action was valid, calculate reward
        check = check_vone_action(state, remaining_actions, total_requested_nodes)
        state, reward = jax.lax.cond(
            check,  # Fail if true
            lambda x: (undo_link_slot_action(undo_node_action(x)), self.get_reward_failure(x)),
            lambda x: jax.lax.cond(
                jnp.any(remaining_actions <= 1),  # Final action
                lambda xx: (finalise_vone_action(xx), self.get_reward_success(xx)),  # Finalise actions if complete
                lambda xx: (xx, self.get_reward_neutral(xx)),
                x
            ),
            state
        )
        # Generate new request if all actions have been taken or if action was invalid
        state = jax.lax.cond(
            jnp.any(remaining_actions <= 1) | check,
            lambda x: generate_vone_request(*x),
            lambda x: x[1],
            (key, state, params)
        )
        done = self.is_terminal(state, params)
        info = {}
        return self.get_obs(state), state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2,))
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset."""
        state = self.initial_state
        state = generate_vone_request(key, state, params)
        return self.get_obs(state), state

    def action_mask_nodes(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Returns action mask for state."""
        return mask_nodes(state, params.num_nodes)

    def action_mask_slots(self, state: EnvState, params: EnvParams, action: chex.Array) -> chex.Array:
        """Returns action mask for state."""
        remaining_actions = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.action_counter, 2, 1))
        full_request = jnp.squeeze(jax.lax.dynamic_slice_in_dim(state.request_array, 0, 1))
        unformatted_request = jax.lax.dynamic_slice_in_dim(full_request, (remaining_actions - 1) * 2, 3)
        node_s = jax.lax.dynamic_slice_in_dim(action, 0, 1)
        requested_slots = jax.lax.dynamic_slice_in_dim(unformatted_request, 1, 1)
        node_d = jax.lax.dynamic_slice_in_dim(action, 2, 1)
        formatted_request = jnp.concatenate((node_s, requested_slots, node_d))
        return mask_slots(state, params, formatted_request)

    def get_obs_unflat(self, state: EnvState) -> Tuple[chex.Array]:
        """Applies observation function to state."""
        return (
            state.request_array,
            state.node_capacity_array,
            state.link_slot_array,
        )

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.concatenate(
            (
                jnp.reshape(state.request_array, (-1,)),
                jnp.reshape(state.node_capacity_array, (-1,)),
                jnp.reshape(state.link_slot_array, (-1,)),
            ),
            axis=0,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Check whether state transition is terminal."""
        return jnp.logical_or(
            jnp.array(state.total_requests >= params.max_requests),
            jnp.array(state.total_timesteps >= params.max_timesteps)
        )

    def discount(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    # TODO - Allow configurable rewards and write tests
    def get_reward_success(self, state: EnvState) -> chex.Array:
        """Return reward for current state."""
        return jnp.array(10.0) #jnp.mean(state.request_array[0]) * state.request_array.shape[1] // 2

    def get_reward_failure(self, state: Optional[EnvState] = None) -> chex.Array:
        """Return reward for current state."""
        return jnp.array(-10.0)

    def get_reward_neutral(self, state: Optional[EnvState] = None) -> chex.Array:
        """Return reward for current state."""
        return jnp.array(0.0, dtype=jnp.float32)

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @staticmethod
    def num_actions(params: EnvParams) -> int:
        """Number of actions possible in environment."""
        return params.num_nodes + params.num_nodes + params.link_resources * params.k_paths

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return spaces.Tuple(
            [
                spaces.Discrete(params.num_nodes),  # Source node
                spaces.Discrete(params.link_resources * params.k_paths),  # Path
                spaces.Discrete(params.num_nodes),  # Destination node
            ]
        )

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Discrete(2*(2*params.max_edges + 1) + params.num_nodes + params.num_links * params.link_resources)

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return spaces.Dict(
            {
                "node_capacity_array": spaces.Discrete(params.num_nodes),
                "current_time": spaces.Discrete(1),
                "request_array": spaces.Discrete(2*(2*params.max_edges + 1)),
                "link_slot_array": spaces.Discrete(params.num_links * params.link_resources),
                "node_resource_array": spaces.Discrete(params.num_nodes * params.node_resources),
                "action_history": spaces.Discrete(params.num_nodes * params.num_nodes * params.link_resources * params.k_paths),
                "action_counter": spaces.Discrete(params.num_nodes * params.num_nodes * params.link_resources * params.k_paths),
                "node_departure_array": spaces.Discrete(params.num_nodes * params.node_resources),
                "link_slot_departure_array": spaces.Discrete(params.num_links * params.link_resources),
            }
        )

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters."""
        return make_vone_env()[1]


def make_vone_env(
        k: int = 5,
        load: float = 100.0,
        topology_name: str = "conus",
        mean_service_holding_time: float = 10.0,
        node_resources: int = 30,
        link_resources: int = 100,
        max_requests: int = 1e4,
        max_timesteps: int = 3e4,
        virtual_topologies: [str] = ["3_ring"],  # virtual topologies to use
        min_slots: int = 1,
        max_slots: int = 2,
        min_node_resources: int = 1,
        max_node_resources: int = 2,
):
    """Create VONE environment.
    Args:
        k: number of paths to consider
        load: load in Erlangs
        topology_name: name of topology to use
        mean_service_holding_time: mean service holding time
        node_resources: number of resources per node
        link_resources: number of resources per link
        max_requests: maximum number of requests
        virtual_topologies: virtual topologies to use
        min_slots: minimum number of slots per link
        max_slots: maximum number of slots per link
        min_node_resources: minimum number of resources per node
        max_node_resources: maximum number of resources per node
    Returns:
        env: VONE environment
        params: VONE environment parameters
    """
    graph = make_graph(topology_name)
    arrival_rate = load / mean_service_holding_time
    num_nodes = len(graph.nodes)
    num_links = len(graph.edges)

    # Automated calculation of max edges in virtual topologies
    max_edges = 0
    for topology in virtual_topologies:
        num, shape = topology.split("_")
        max_edges = max(max_edges, int(num) - (0 if shape == "ring" else 1))

    params = VONEEnvParams(
        max_requests=max_requests,
        max_timesteps=max_timesteps,
        mean_service_holding_time=mean_service_holding_time,
        k_paths=k,
        node_resources=node_resources,
        link_resources=link_resources,
        num_nodes=num_nodes,
        num_links=num_links,
        load=load,
        arrival_rate=arrival_rate,
        max_edges=max_edges,
        min_slots=min_slots,
        max_slots=max_slots,
        min_node_resources=min_node_resources,
        max_node_resources=max_node_resources,
        path_link_array=HashableArrayWrapper(init_path_link_array(graph, k)),
    )

    env = VONEEnv(params, virtual_topologies=virtual_topologies)

    return env, params