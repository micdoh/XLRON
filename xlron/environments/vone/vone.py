from typing import Tuple, Union, Optional
from flax import struct
from functools import partial
import math
import chex
import jax
import jax.numpy as jnp
import jraph
from gymnax.environments import environment, spaces
from xlron.environments.env_funcs import (
    init_vone_request_array, init_link_slot_array, init_link_slot_mask, init_link_slot_departure_array,
    check_vone_action, undo_action_rsa, finalise_vone_action, generate_vone_request, mask_slots, init_traffic_matrix,
    init_node_capacity_array, init_node_mask, init_node_resource_array, init_node_departure_array, init_values_nodes,
    init_action_counter, init_action_history, update_action_history, decrease_last_element, undo_node_action,
    init_virtual_topology_patterns, mask_nodes, init_graph_tuple, format_vone_slot_request, implement_vone_action,
)
from xlron.environments.dataclasses import *
from xlron.environments.wrappers import *


class VONEEnv(environment.Environment):
    """This environment simulates the Virtual Optical Network Embedding (VONE) problem.

    """
    def __init__(self,
                 key: chex.PRNGKey,
                 params: VONEEnvParams,
                 virtual_topologies=["3_ring"],
                 traffic_matrix: chex.Array = None,
                 list_of_requests: chex.Array = None,
                 laplacian_matrix: chex.Array = None,):
        super().__init__()
        state = VONEEnvState(
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
            link_slot_mask=init_link_slot_mask(params, agg=params.aggregate_slots),
            node_mask_d=init_node_mask(params),
            virtual_topology_patterns=init_virtual_topology_patterns(virtual_topologies),
            values_nodes=init_values_nodes(params.min_node_resources, params.max_node_resources),
            graph=None,
            full_link_slot_mask=init_link_slot_mask(params),
            accepted_services=0,
            accepted_bitrate=0.,
            total_bitrate=0.,
            list_of_requests=list_of_requests,
            traffic_matrix=traffic_matrix if traffic_matrix is not None else init_traffic_matrix(key, params),
        )
        self.initial_state = state.replace(graph=init_graph_tuple(state, params, laplacian_matrix, exclude_source_dest=True))

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
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
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
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
            lambda x: (undo_action_rsa(undo_node_action(x), params), self.get_reward_failure(x)),
            lambda x: jax.lax.cond(
                jnp.any(remaining_actions <= 1),  # Final action
                lambda xx: (finalise_vone_action(xx), self.get_reward_success(xx)),  # Finalise actions if complete
                lambda xx: (xx, self.get_reward_neutral(xx)),
                x
            ),
            state
        )
        # TODO - write separate functions for deterministic transition (above) and stochastic transition (below)
        # Generate new request if all actions have been taken or if action was invalid
        state = jax.lax.cond(
            jnp.any(remaining_actions <= 1) | check,
            lambda x: generate_vone_request(*x),
            lambda x: x[1],
            (key, state, params)
        )
        # Terminate if max_requests exceeded or, if consecutive loading,
        # then terminate if reward is failure but not before min number of timesteps before update
        if params.continuous_operation:
            done = jnp.array(False)
        elif params.incremental_loading:
            done = jnp.array(reward == self.get_reward_failure())
        else:
            done = self.is_terminal(state, params)
        # Update graph tuple
        # TODO - get the update graph tuple working with VONE
        state = state.replace(graph=init_graph_tuple(state, params))
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

    def action_mask_dest_node(self, state: EnvState, params: EnvParams, source_action: chex.Array) -> chex.Array:
        """Returns action mask for state."""
        empty_mask = jnp.ones(params.num_nodes)
        mask_source_node = jax.lax.dynamic_update_slice(empty_mask, jnp.array([0.]), (source_action, ))
        node_mask_d = jnp.min(jnp.stack((state.node_mask_d, mask_source_node)), axis=0)
        state = state.replace(node_mask_d=node_mask_d)
        return state

    def action_mask_slots(self, state: EnvState, params: EnvParams, action: chex.Array) -> chex.Array:
        """Returns action mask for state."""
        formatted_request = format_vone_slot_request(state, action)
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
        return jnp.array(state.total_requests >= params.max_requests)

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
        return (params.num_nodes +
                params.num_nodes +
                math.ceil(params.link_resources/params.aggregate_slots) * params.k_paths)

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
