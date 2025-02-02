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
    init_rsa_request_array, init_link_slot_array, init_path_link_array,
    convert_node_probs_to_traffic_matrix, init_link_slot_mask, init_link_slot_departure_array, init_traffic_matrix,
    implement_action_rsa, check_action_rsa, undo_action_rsa, finalise_action_rsa, generate_request_rsa,
    mask_slots, make_graph, init_path_length_array, init_modulations_array, init_path_se_array, required_slots,
    init_values_bandwidth, calculate_path_stats, normalise_traffic_matrix, init_graph_tuple, init_link_length_array,
    init_link_capacity_array, init_path_capacity_array, init_path_index_array, mask_slots_rwalr,
    implement_action_rwalr, check_action_rwalr, pad_array, undo_action_rwalr, update_graph_tuple,
    finalise_action_rwalr, generate_request_rwalr, init_link_snr_array,
    init_channel_centre_bw_array, check_action_rsa_gn_model, read_rsa_request, implement_action_rsa_gn_model,
    undo_action_rsa_gn_model, finalise_action_rsa_gn_model, init_modulation_format_index_array,
    init_channel_power_array, mask_slots_rsa_gn_model, init_link_length_array_gn_model, get_snr_for_path,
    get_lightpath_snr,
    generate_source_dest_pairs, init_list_of_requests
)
from xlron.environments.dataclasses import *
from xlron.environments.wrappers import *
from xlron.environments.isrs_gn_model import from_dbm, to_dbm


class RSAEnv(environment.Environment):
    """This environment simulates the Routing Modulation and Spectrum Assignment (RMSA) problem.
    It can model RSA by setting consider_modulation_format=False in params.
    It can model RWA by setting min_bw=0, max_bw=0, and consider_modulation_format=False in params.
    """
    def __init__(
            self,
            key: chex.PRNGKey,
            params: RSAEnvParams,
            traffic_matrix: chex.Array = None,
            list_of_requests: chex.Array = None,
            laplacian_matrix: chex.Array = None,
    ):
        """Initialise the environment state and set as initial state.

        Args:
            key: PRNG key
            params: Environment parameters
            traffic_matrix (optional): Traffic matrix

        Returns:
            None
        """
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
            list_of_requests=list_of_requests,
            graph=None,
            full_link_slot_mask=init_link_slot_mask(params),
            accepted_services=0,
            accepted_bitrate=0.,
            total_bitrate=0.,
        )
        if not params.__class__.__name__ == "RSAGNModelEnvParams":
            self.initial_state = state.replace(graph=init_graph_tuple(state, params, laplacian_matrix))

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: RSAEnvState,
        action: Union[int, float],
        params: Optional[RSAEnvParams] = None,
    ) -> Tuple[chex.Array, RSAEnvState, float, bool, dict]:
        """Performs step transitions in the environment.

        Args:
            key: PRNG key
            state: Environment state
            action: Action to take (single value array)
            params: Environment parameters

        Returns:
            obs: Observation
            state: New environment state
            reward: Reward
            done: Termination flag
            info: Additional information
        """
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
        """Performs resetting of environment.

        Args:
            key: PRNG key
            params: Environment parameters

        Returns:
            obs: Observation
            state: Reset environment state
        """
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
        """Environment-specific step transition.
        1. Implement action
        2. Check if action was valid
            - If valid, calculate reward and finalise action
            - If invalid, calculate reward and undo action
        3. Generate new request, update current time, remove expired requests
        4. Update timesteps
        5. Terminate if max_timesteps or max_requests exceeded
        6. if DeepRMSAEnv, calculate path stats
        7. if using GNN policy, update graph tuple

        Args:
            key: PRNG key
            state: Environment state
            action: Action to take (single value array)
            params: Environment parameters

        Returns:
            obs: Observation
            state: New environment state
            reward: Reward
            done: Termination flag
            info: Additional information
        """
        # Do action
        undo = undo_action_rsa
        finalise = finalise_action_rsa
        generate_request = generate_request_rsa
        input_state = [state, action, params]
        check_state = [state]
        undo_finalise_state = [state, params]
        if params.__class__.__name__ == "RWALightpathReuseEnvParams":
            implement_action = implement_action_rwalr
            check_action = check_action_rwalr
            input_state = check_state = [state, action, params]
            if not params.incremental_loading:
                # These are relevant to dynamic RWA-LR (upcoming)
                undo = undo_action_rwalr
                finalise = finalise_action_rwalr
                generate_request = generate_request_rwalr
        elif params.__class__.__name__ == "RSAGNModelEnvParams":
            implement_action = implement_action_rsa_gn_model
            check_action = check_action_rsa_gn_model
            undo = undo_action_rsa_gn_model
            finalise = finalise_action_rsa_gn_model
            input_state = check_state = [state, action, params]
            undo_finalise_state = [state, params, action]
        else:
            implement_action = implement_action_rsa
            check_action = check_action_rsa

        # Check if action was valid, calculate reward
        check_state[0] = undo_finalise_state[0] = state = implement_action(*input_state)
        check = check_action(*check_state)
        state, reward = jax.lax.cond(
            check,  # Fail if true
            lambda x: (undo(*x[:2]), self.get_reward_failure(*x)),
            lambda x: (finalise(*x[:2]), self.get_reward_success(*x)),  # Finalise actions if complete
            undo_finalise_state
        )
        # TODO (DYNAMIC-RWALR) - calculate allocated bandwidth
        # TODO (DYNAMIC-RWALR) - generate new request if allocated DR equals requested DR, else update requested DR do not advance time do not replace source-dest
        # TODO (AFTERSTATE) - write separate functions for deterministic transition (above) and stochastic transition (below)
        # Generate new request
        state = generate_request(key, state, params)
        jax.debug.print("a_time {} h_time {} request {}", state.current_time, state.holding_time, state.request_array, ordered=True)
        state = state.replace(total_timesteps=state.total_timesteps + 1)
        # Terminate if max_timesteps or max_requests exceeded or, if consecutive loading,
        # then terminate if reward is failure but not before min number of timesteps before update
        if params.continuous_operation:
            done = jnp.array(False)
        elif params.end_first_blocking:
            done = jnp.array(reward == self.get_reward_failure(state, params))
        else:
            done = self.is_terminal(state, params)
        info = {}
        # Calculate path stats if DeepRMSAEnv
        if params.__class__.__name__ == "DeepRMSAEnvParams":
            path_stats = calculate_path_stats(state, params, state.request_array)
            state = state.replace(path_stats=path_stats)
        else:
            # Update graph tuple
            state = update_graph_tuple(state, params)
        return self.get_obs(state, params), state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2,))
    def reset_env(
        self, key: chex.PRNGKey, params: RSAEnvParams
    ) -> Tuple[chex.Array, RSAEnvState]:
        """Environment-specific reset.
        Generates new random traffic matrix if random_traffic is True, otherwise uses the provided traffic matrix.
        Generates new request.

        Args:
            key: PRNG key
            params: Environment parameters

        Returns:
            obs: Observation
            state: Reset environment state
        """
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
        state = generate_request_rsa(key, state, params)
        return self.get_obs(state, params), state

    @partial(jax.jit, static_argnums=(0, 2,))
    def action_mask(self, state: RSAEnvState, params: RSAEnvParams) -> RSAEnvState:
        """Returns mask of valid actions.
        1. Check request for source and destination nodes
        2. For each path, mask out (0) initial slots that are not valid
        See mask_slots function for more details.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            state: Environment state with action mask
        """
        state = mask_slots(state, params, state.request_array)
        return state

    @partial(jax.jit, static_argnums=(0, 2,))
    def get_obs_unflat(self, state: RSAEnvState, params: RSAEnvParams) -> Tuple[chex.Array, chex.Array]:
        """Retrieves observation from state.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            obs: Observation (request array, link slot array)
        """
        return (
            state.request_array,
            state.link_slot_array,
        )

    @partial(jax.jit, static_argnums=(0, 2,))
    def get_obs(self, state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
        """Retrieves observation from state and reshapes into single array.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            obs: Observation (flattened request array and link slot array)
        """
        return jnp.concatenate(
            (
                jnp.reshape(state.request_array, (-1,)),
                jnp.reshape(state.link_slot_array, (-1,)),
            ),
            axis=0,
        )

    def is_terminal(self, state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
        """Check whether state transition is terminal.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            done: Boolean termination flag
        """
        return jnp.logical_or(
            jnp.array(state.total_requests >= params.max_requests),
            jnp.array(state.total_timesteps >= params.max_timesteps)
        )

    def discount(self, state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
        """Return a discount of zero if the episode has terminated.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            discount: Binary discount factor
        """
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    def get_reward_failure(
            self,
            state: Optional[EnvState] = None,
            params: Optional[EnvParams] = None,
            action: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Return reward for current state.

        Args:
            state (optional): Environment state

        Returns:
            reward: Reward for failure
        """
        if params.reward_type == "service":
            reward = jnp.array(-1.0)
        elif params.reward_type == "bitrate":
            reward = state.request_array[1] * -1.0 / jnp.max(params.values_bw.val)
        else:
            reward = -1.0 * read_rsa_request(state.request_array)[1] / jnp.max(params.values_bw) if params.maximise_throughput else jnp.array(-1.0)
        return reward

    def get_reward_success(
            self,
            state: Optional[EnvState] = None,
            params: Optional[EnvParams] = None,
            action: Optional[chex.Array] = None
    ) -> chex.Array:
        """Return reward for current state.

        Args:
            state: (optional) Environment state

        Returns:
            reward: Reward for success
        """
        if params.reward_type == "service":
            reward = jnp.array(1.0)
        elif params.reward_type == "bitrate":
            reward = state.request_array[1] * 1.0 / jnp.max(params.values_bw.val)
            if params.__class__.__name__ == "RSAGNModelEnvParams":
                # Need to get the SNR of the path
                path_action, power_action = action
                path_index, slot_index = process_path_action(state, params, path_action)
                path = params.path_link_array[path_index]
                path_snr = get_snr_for_path(path, state.link_snr_array, params)[slot_index.astype(jnp.int32)]
                # set to 0 if negative and divide by large SNR (50. dB) to scale below 1
                # N.B. negative SNR in dB would be a fail anyway since min. required is 10dB
                path_snr_norm = jnp.where(path_snr < 0, 0, path_snr) / 50.
                reward = reward * path_snr_norm
        else:
            reward = read_rsa_request(state.request_array)[1] / jnp.max(params.values_bw) if params.maximise_throughput else jnp.array(1.0)
        return reward

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
        return spaces.Discrete(self.num_actions(params))

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
    """This environment simulates the Routing Modulation and Spectrum Assignment (RMSA) problem,
    with action and observation spaces that match those defined in the DeepRMSA paper:

    https://ieeexplore.ieee.org/document/8738827/

    See above paper and path_stats function for more details of the observation space.
    """
    def __init__(
            self,
            key: chex.PRNGKey,
            params: RSAEnvParams,
            traffic_matrix: chex.Array = None
    ):
        super().__init__(key, params, traffic_matrix=traffic_matrix)
        self.initial_state = DeepRMSAEnvState(
            current_time=0,
            holding_time=0,
            total_timesteps=0,
            total_requests=-1,
            link_slot_array=init_link_slot_array(params),
            link_slot_departure_array=init_link_slot_departure_array(params),
            request_array=init_rsa_request_array(),
            link_slot_mask=jnp.ones(params.k_paths),
            full_link_slot_mask=jnp.ones(params.k_paths),
            traffic_matrix=traffic_matrix if traffic_matrix is not None else init_traffic_matrix(key, params),
            path_stats=calculate_path_stats(self.initial_state, params, self.initial_state.request_array),
            graph=None,
            accepted_services=0,
            accepted_bitrate=0.,
            total_bitrate=0.,
        )

    def step_env(
            self,
            key: chex.PRNGKey,
            state: DeepRMSAEnvState,
            action: Union[int, float],
            params: RSAEnvParams,
    ) -> Tuple[chex.Array, RSAEnvState, chex.Array, chex.Array, chex.Array]:
        """Environment-specific step transition.

        Args:
            key: PRNG key
            state: Environment state
            action: Action to take (single value array indicating which of k paths to use)
            params: Environment parameters

        Returns:
            obs: Observation
            state: New environment state
            reward: Reward
            done: Termination flag
            info: Additional information
        """
        # TODO - alter this if allowing J>1
        slot_index = jnp.squeeze(jax.lax.dynamic_slice(state.path_stats, (action, 1), (1, 1)))
        action = jnp.array(action * params.link_resources + slot_index).astype(jnp.int32)
        return super().step_env(key, state, action, params)

    @partial(jax.jit, static_argnums=(0, 2,))
    def action_mask(self, state: RSAEnvState, params: RSAEnvParams) -> RSAEnvState:
        """Returns mask of valid actions.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            state: Environment state with action mask
        """
        mask = jnp.where(state.path_stats[:, 0] >= 1, 1., 0.)
        # If mask is all zeros, make all ones
        mask = jnp.where(jnp.sum(mask) == 0, 1., mask)
        state = state.replace(link_slot_mask=mask)
        return state

    def action_space(self, params: RSAEnvParams) -> spaces.Discrete:
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
        return jnp.concatenate(
            (
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
    """This environment simulates the Routing and Wavelength Assignment (RWA) problem with lightpath reuse.
    Lightpath reuse means the transponders in the network are assumed to be "flex-rate", therefore multiple traffic
    requests with same source-destination nodes can be packed onto the same lightpath, so long as it has sufficient
    remaining capacity. See the following paper for more details of the problem, which this environment recreates
    exactly:

    https://discovery.ucl.ac.uk/id/eprint/10175456/
    """
    def __init__(
            self,
            key: chex.PRNGKey,
            params: RSAEnvParams,
            traffic_matrix: chex.Array = None,
            path_capacity_array: chex.Array = None,
            laplacian_matrix: chex.Array = None,
    ):
        """Initialise the environment state and set as initial state.

        Args:
            key: PRNG key
            params: Environment parameters
            traffic_matrix (optional): Traffic matrix
            path_capacity_array: Array of path capacities

        Returns:
            None
        """
        super().__init__(key, params, traffic_matrix=traffic_matrix)
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
            graph=None,
            full_link_slot_mask=init_link_slot_mask(params),
            path_index_array=init_path_index_array(params),
            path_capacity_array=path_capacity_array,
            link_capacity_array=init_link_capacity_array(params),
            accepted_services=0,
            accepted_bitrate=0.,
            total_bitrate=0.,
            time_since_last_departure=0.,
        )
        self.initial_state = state.replace(graph=init_graph_tuple(state, params, laplacian_matrix))

    @partial(jax.jit, static_argnums=(0, 2,))
    def action_mask(self, state: RSAEnvState, params: RSAEnvParams) -> RSAEnvState:
        """Returns mask of valid actions.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            state: Environment state with action mask
        """
        state = mask_slots_rwalr(state, params, state.request_array)
        return state


class RSAGNModelEnv(RSAEnv):
    """RMSA + GNN model environment."""
    def __init__(
            self,
            key: chex.PRNGKey,
            params: RSAGNModelEnvParams,
            traffic_matrix: chex.Array = None,
            launch_power_array: chex.Array = None,
            list_of_requests: chex.Array = None,
            laplacian_matrix: chex.Array = None,
    ):
        """Initialise the environment state and set as initial state.

        Args:
            key: PRNG key
            params: Environment parameters
            traffic_matrix (optional): Traffic matrix
            launch_power_array (optional): Launch power array

        Returns:
            None
        """
        super().__init__(key, params, traffic_matrix=traffic_matrix, list_of_requests=list_of_requests, laplacian_matrix=laplacian_matrix)
        state = RSAGNModelEnvState(
            current_time=0,
            holding_time=0,
            total_timesteps=0,
            total_requests=-1,
            link_slot_array=init_link_slot_array(params),
            link_slot_departure_array=init_link_slot_departure_array(params),
            request_array=init_rsa_request_array(),
            link_slot_mask=init_link_slot_mask(params, agg=params.aggregate_slots),
            traffic_matrix=traffic_matrix if traffic_matrix is not None else init_traffic_matrix(key, params),
            graph=None,
            full_link_slot_mask=init_link_slot_mask(params),
            accepted_services=0,
            accepted_bitrate=0.,
            total_bitrate=0.,
            list_of_requests=list_of_requests,
            link_snr_array=init_link_snr_array(params),
            path_index_array=init_path_index_array(params),
            path_index_array_prev=init_path_index_array(params),
            channel_centre_bw_array=init_channel_centre_bw_array(params),
            channel_power_array=init_channel_power_array(params),
            modulation_format_index_array=init_modulation_format_index_array(params),
            channel_centre_bw_array_prev=init_channel_centre_bw_array(params),
            channel_power_array_prev=init_channel_power_array(params),
            modulation_format_index_array_prev=init_modulation_format_index_array(params),
            #active_path_array=init_active_path_array(params),
            #active_path_array_prev=init_active_path_array(params),
            launch_power_array=launch_power_array,
        )
        self.initial_state = state.replace(graph=init_graph_tuple(state, params, laplacian_matrix))

    @partial(jax.jit, static_argnums=(0, 2,))
    def action_mask(self, state: RSAEnvState, params: RSAEnvParams) -> RSAEnvState:
        """Returns mask of valid actions.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            state: Environment state with action mask
        """
        state = mask_slots_rsa_gn_model(state, params, state.request_array)
        return state

    # def get_obs(self, state: RSAGNModelEnvState, params: RSAGNModelEnvParams) -> chex.Array:
    #     """Get observation space for launch power optimization with numerical stability.
    #
    #     Args:
    #         state: Current environment state
    #         params: Environment parameters
    #
    #     Returns:
    #         chex.Array: Concatenated observation vector
    #     """
    #     eps = 1e-10  # Small constant for numerical stability
    #
    #     # Safely reshape request array
    #     request_array = jnp.reshape(state.request_array, (-1,))
    #
    #     # Calculate base path stats
    #     path_stats = calculate_path_stats(state, params, request_array)
    #     # Ensure we have enough columns before slicing
    #     min_cols = 4  # Minimum columns needed (3 to remove + 1 to keep)
    #     path_stats = jnp.where(
    #         path_stats.shape[1] >= min_cols,
    #         path_stats[:, 3:],
    #         jnp.zeros((path_stats.shape[0], max(0, path_stats.shape[1] - 3)))
    #     )
    #
    #     # Safely calculate link lengths
    #     link_length_array = jnp.nan_to_num(
    #         jnp.sum(params.link_length_array.val, axis=1),
    #         nan=0.0, posinf=0.0, neginf=0.0
    #     )
    #
    #     # Get SNR array with error handling
    #     lightpath_snr_array = jnp.nan_to_num(
    #         get_lightpath_snr(state, params),
    #         nan=-50.0, posinf=100.0, neginf=-50.0
    #     )
    #
    #     # Safely get source-destination nodes
    #     nodes_sd, requested_datarate = read_rsa_request(request_array)
    #     source, dest = nodes_sd
    #
    #     def calculate_gn_path_stats(k_path_index, init_val):
    #         """Calculate path statistics with numerical stability."""
    #         # Get path index safely
    #         path_index = jnp.clip(
    #             get_path_indices(source, dest, params.k_paths, params.num_nodes,
    #                              directed=params.directed_graph).astype(jnp.int32) + k_path_index,
    #             0, params.path_link_array.shape[0] - 1
    #         )
    #
    #         # Get path safely
    #         path = params.path_link_array[path_index]
    #
    #         # Calculate path length with unit conversion
    #         path_length = jnp.clip(jnp.dot(path, link_length_array) / 100, 0.0, 1e6)
    #
    #         # Calculate path hops
    #         path_length_hops = jnp.sum(jnp.abs(path))
    #
    #         # Calculate number of connections safely
    #         path_mask = (path == 1).astype(jnp.float32)
    #         mod_mask = (state.modulation_format_index_array > -1).astype(jnp.float32)
    #         num_connections = jnp.sum(
    #             jnp.where(path_mask.reshape(-1, 1), mod_mask, 0.0)
    #         )
    #
    #         # Calculate mean power safely
    #         power_sum = jnp.sum(
    #             jnp.where(path_mask.reshape(-1, 1),
    #                       state.channel_power_array,
    #                       0.0)
    #         )
    #         # Avoid division by zero
    #         mean_power = jnp.where(
    #             num_connections > eps,
    #             power_sum / (num_connections + eps),
    #             -50.0  # Default value for no connections
    #         )
    #
    #         # Calculate mean SNR safely
    #         snr_sum = jnp.sum(
    #             jnp.where(path_mask.reshape(-1, 1),
    #                       lightpath_snr_array,
    #                       0.0)
    #         )
    #         mean_snr = jnp.where(
    #             num_connections > eps,
    #             snr_sum / (num_connections + eps),
    #             -50.0  # Default value for no connections
    #         )
    #
    #         # Clip values to reasonable ranges
    #         mean_power = jnp.clip(mean_power, -50.0, 10.0)  # dBm range
    #         mean_snr = jnp.clip(mean_snr, -50.0, 100.0)  # dB range
    #
    #         # Create stats array with safe values
    #         path_stats = jnp.array([
    #             path_length,
    #             path_length_hops,
    #             num_connections,
    #             mean_power,
    #             mean_snr
    #         ])
    #
    #         # Handle NaN/Inf values
    #         path_stats = jnp.nan_to_num(
    #             path_stats,
    #             nan=-50.0, posinf=100.0, neginf=-50.0
    #         )
    #
    #         return jax.lax.dynamic_update_slice(
    #             init_val,
    #             path_stats.reshape((1, 5)),
    #             (k_path_index, 0),
    #         )
    #
    #     # Initialize path stats array
    #     gn_path_stats = jnp.zeros((params.k_paths, 5))
    #
    #     # Calculate path stats for each path
    #     gn_path_stats = jax.lax.fori_loop(
    #         0, params.k_paths, calculate_gn_path_stats, gn_path_stats
    #     )
    #
    #     # Ensure all values are finite
    #     gn_path_stats = jnp.nan_to_num(
    #         gn_path_stats,
    #         nan=-50.0, posinf=100.0, neginf=-50.0
    #     )
    #
    #     # Safely concatenate stats
    #     try:
    #         all_stats = jnp.concatenate([path_stats, gn_path_stats], axis=1)
    #     except:
    #         # Fallback if shapes don't match
    #         all_stats = gn_path_stats
    #
    #     # Final array construction with shape checking
    #     holding_time = jnp.reshape(state.holding_time, (-1,))
    #     all_stats_flat = jnp.reshape(all_stats, (-1,))
    #
    #     # Ensure all components have expected shapes before concatenating
    #     expected_length = 3 + holding_time.shape[0] + all_stats_flat.shape[0]
    #     result = jnp.concatenate(
    #         (
    #             request_array,
    #             holding_time,
    #             all_stats_flat,
    #         ),
    #         axis=0,
    #     )
    #
    #     # Final safeguard against any remaining NaN/Inf values
    #     result = jnp.nan_to_num(
    #         result,
    #         nan=-50.0, posinf=100.0, neginf=-50.0
    #     )
    #
    #     # Clip final values to reasonable ranges
    #     result = jnp.clip(result, -1e6, 1e6)
    #     jax.debug.print("result {}", result, ordered=True)
    #
    #     return result

    @partial(jax.jit, static_argnums=(0, 2,))
    def get_obs(self, state: RSAGNModelEnvState, params: RSAGNModelEnvParams) -> chex.Array:
        # TODO - make this just show the stats from just one path at a time
        """Get observation space for launch power optimization (with numerical stability)."""
        request_array = state.request_array.reshape((-1,))
        path_stats = calculate_path_stats(state, params, request_array)
        # Remove first 3 items of path stats for each path
        path_stats = path_stats[:, 3:]
        link_length_array = jnp.sum(params.link_length_array.val, axis=1)
        lightpath_snr_array = get_lightpath_snr(state, params)
        nodes_sd, requested_datarate = read_rsa_request(request_array)
        source, dest = nodes_sd

        def calculate_gn_path_stats(k_path_index, init_val):
            # Get path index
            path_index = get_path_indices(source, dest, params.k_paths, params.num_nodes, directed=params.directed_graph).astype(
                jnp.int32) + k_path_index
            path = params.path_link_array[path_index]
            path_length = jnp.dot(path, link_length_array)
            max_path_length = jnp.max(jnp.dot(params.path_link_array.val, link_length_array))
            path_length_norm = path_length / max_path_length
            path_length_hops_norm = jnp.sum(path) / jnp.max(jnp.sum(params.path_link_array.val, axis=1))
            # Connections on path
            num_connections = jnp.where(path == 1, jnp.where(state.modulation_format_index_array > -1, 1, 0).sum(axis=1), 0).sum()
            num_connections_norm = num_connections / params.link_resources
            # Mean power of connections on path
            # make path with row length equal to link_resource (+1 to avoid zero division)
            mean_power_norm = (jnp.where(path == 1, state.channel_power_array.sum(axis=1), 0).sum() /
                               (jnp.where(num_connections > 0., num_connections, 1.) * params.max_power))
            # Mean SNR of connections on the path links
            max_snr = 50. # Nominal value for max GSNR in dB
            mean_snr_norm = (jnp.where(path == 1, lightpath_snr_array.sum(axis=1), 0).sum() /
                             (jnp.where(num_connections > 0., num_connections, 1.) * max_snr))
            return jax.lax.dynamic_update_slice(
                init_val,
                jnp.array([[
                    path_length,
                    path_length_hops_norm,
                    num_connections_norm,
                    mean_power_norm,
                    mean_snr_norm
                ]]),
                (k_path_index, 0),
            )
        gn_path_stats = jnp.zeros((params.k_paths, 5))
        gn_path_stats = jax.lax.fori_loop(
            0, params.k_paths, calculate_gn_path_stats, gn_path_stats
        )
        all_stats = jnp.concatenate([path_stats, gn_path_stats], axis=1)
        return jnp.concatenate(
            (
                jnp.array([source]),
                requested_datarate / 100.,
                jnp.array([dest]),
                jnp.reshape(state.holding_time, (-1,)),
                jnp.reshape(all_stats, (-1,)),
            ),
            axis=0,
        )

    @staticmethod
    def num_actions(params: RSAEnvParams) -> int:
        """Number of actions possible in environment."""
        return 1

    def observation_space(self, params: RSAEnvParams):
        """Observation space of the environment."""
        return spaces.Discrete(
            3 +  # Request array
            1 +  # Holding time
            7 * params.k_paths
            # Path stats:
            # Mean free block size
            # Free slots
            # Path length (100 km)
            # Path length (hops)
            # Number of connections on path
            # Mean power of connection on path
            # Mean SNR of connection on path
        )


def make_rsa_env(config: dict, launch_power_array: Optional[chex.Array] = None):
    """Create RSA environment. This function is the entry point to setting up any RSA-type environment.
    This function takes a dictionary of the commandline flag parameters and configures the
    RSA environment and parameters accordingly.

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
    incremental_loading = config.get("incremental_loading", False)
    end_first_blocking = config.get("end_first_blocking", False)
    random_traffic = config.get("random_traffic", False)
    continuous_operation = config.get("continuous_operation", False)
    total_timesteps = config.get("TOTAL_TIMESTEPS", 1e4)
    max_requests = total_timesteps if continuous_operation else config.get("max_requests", total_timesteps)
    max_timesteps = total_timesteps if continuous_operation else config.get("max_requests", total_timesteps)
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
    custom_traffic_matrix_csv_filepath = config.get("custom_traffic_matrix_csv_filepath", None)
    traffic_requests_csv_filepath = config.get("traffic_requests_csv_filepath", None)
    multiple_topologies_directory = config.get("multiple_topologies_directory", None)
    aggregate_slots = config.get("aggregate_slots", 1)
    disjoint_paths = config.get("disjoint_paths", False)
    log_actions = config.get("log_actions", False)
    guardband = config.get("guardband", 1)
    weight = config.get("weight", None)
    remove_array_wrappers = config.get("remove_array_wrappers", False)
    maximise_throughput = config.get("maximise_throughput", False)
    reward_type = config.get("reward_type", "bitrate")
    truncate_holding_time = config.get("truncate_holding_time", False)
    alpha = config.get("alpha", 0.2) * 1e-3
    amplifier_noise_figure = config.get("amplifier_noise_figure", 4.5)
    beta_2 = config.get("beta_2", -21.7) * 1e-27
    gamma = config.get("gamma", 1.2) * 1e-3
    span_length = config.get("span_length", 100) * 1e3
    lambda0 = config.get("lambda0", 1550) * 1e-9
    B = slot_size * link_resources  # Total modulated bandwidth

    # GN model parameters
    max_span_length = config.get("max_span_length", 100e3)
    ref_lambda = config.get("ref_lambda", 1577.5e-9)  # centre of C+L bands (1530-1625nm)
    nonlinear_coeff = config.get("nonlinear_coeff", 1.2 / 1e3)
    raman_gain_slope = config.get("raman_gain_slope", 0.028 / 1e3 / 1e12)
    attenuation = config.get("attenuation", 0.2 / 4.343 / 1e3)
    attenuation_bar = config.get("attenuation_bar", 0.2 / 4.343 / 1e3)
    dispersion_coeff = config.get("dispersion_coeff", 17 * 1e-12 / 1e-9 / 1e3)
    dispersion_slope = config.get("dispersion_slope", 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9)
    coherent = config.get("coherent", False)
    noise_figure = config.get("noise_figure", 4)
    interband_gap = config.get("interband_gap", 100)
    gap_width = int(math.ceil(interband_gap / slot_size))
    gap_start = config.get("gap_start", link_resources//2)
    mod_format_correction = config.get("mod_format_correction", True)
    num_roadms = config.get("num_roadms", 1)
    roadm_loss = config.get("roadm_loss", 18)
    snr_margin = config.get("snr_margin", 1)
    path_snr = True if env_type == "rsa_gn_model" else False
    max_snr = config.get("max_snr", 50.)
    max_power = config.get("max_power", 9)
    min_power = config.get("min_power", -5)
    step_power = config.get("step_power", 1)
    default_launch_power = float(from_dbm(config.get("launch_power", 0.5)))
    optimise_launch_power = config.get("optimise_launch_power", False)
    traffic_array = config.get("traffic_array", False)

    # optimize_launch_power.py parameters
    num_spans = config.get("num_spans", 10)

    rng = jax.random.PRNGKey(seed)
    rng, _, _, _, _ = jax.random.split(rng, 5)
    graph = make_graph(topology_name, topology_directory=config.get("topology_directory", None))
    traffic_intensity = config.get("traffic_intensity", 0)
    mean_service_holding_time = config.get("mean_service_holding_time", 10)

    # Set traffic intensity / load
    if traffic_intensity:
        arrival_rate = traffic_intensity / mean_service_holding_time
    else:
        arrival_rate = load / mean_service_holding_time
    num_nodes = len(graph.nodes)
    num_links = len(graph.edges)
    scale_factor = config.get("scale_factor", 1.0)
    path_link_array = init_path_link_array(
        graph, k, disjoint=disjoint_paths, weight=weight, directed=graph.is_directed(),
        rwa_lr=True if env_type == "rwa_lightpath_reuse" else False, scale_factor=scale_factor, path_snr=path_snr)

    launch_power_type = config.get("launch_power_type", "fixed")
    # The launch power type determines whether to use:
    # 1. Fixed power for all channels.
    # 2. Tabulated values of power for each path.
    # 3. RL to determine power for each channel.
    # 4. Fixed power scaled by path length.
    if env_type == "rsa_gn_model":
        default_launch_power_array = jnp.array([default_launch_power,])
        if launch_power_type == "fixed":
            # Same power for all channels
            launch_power_array = default_launch_power_array if launch_power_array is None else launch_power_array
            launch_power_type = 1
        elif launch_power_type == "tabular":
            # The power of a channel is determined by the path it takes
            launch_power_array = jnp.zeros(path_link_array.shape[0]) \
                if launch_power_array is None else launch_power_array
            launch_power_type = 2
        elif launch_power_type == "rl":
            # RL sets power per channel
            launch_power_array = default_launch_power_array
            launch_power_type = 3
        elif launch_power_type == "scaled":
            # Power scaled by path length
            launch_power_array = default_launch_power_array if launch_power_array is None else launch_power_array
            launch_power_type = 4
        else:
            pass

    if custom_traffic_matrix_csv_filepath:
        random_traffic = False  # Set this False so that traffic matrix isn't replaced on reset
        traffic_matrix = jnp.array(np.loadtxt(custom_traffic_matrix_csv_filepath, delimiter=","))
        traffic_matrix = normalise_traffic_matrix(traffic_matrix)
    elif node_probabilities:
        random_traffic = False  # Set this False so that traffic matrix isn't replaced on reset
        node_probabilities = [float(prob) for prob in config.get("node_probs")]
        traffic_matrix = convert_node_probs_to_traffic_matrix(node_probabilities)
    elif traffic_array:
        traffic_matrix = generate_source_dest_pairs(num_nodes, graph.is_directed())
    else:
        traffic_matrix = None

    if config.get("deterministic_requests"):
        deterministic_requests = True
        # Remove headers from array
        if traffic_requests_csv_filepath:
            list_of_requests = np.loadtxt(traffic_requests_csv_filepath, delimiter=",")[1:, :]
            list_of_requests = jnp.array(list_of_requests)
        else:
            list_of_requests = init_list_of_requests(int(max_requests))
        max_requests = len(list_of_requests)
    elif optimise_launch_power:
        deterministic_requests = True
        list_of_requests = jnp.array(config.get("list_of_requests", [0]))
    else:
        deterministic_requests = False
        list_of_requests = jnp.array([0])

    values_bw = init_values_bandwidth(min_bw, max_bw, step_bw, values_bw)

    if env_type == "rsa":
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

    link_length_array = init_link_length_array(graph).reshape((num_links, 1))

    # Automated calculation of max slots requested
    if consider_modulation_format:
        modulations_array = init_modulations_array(config.get("modulations_csv_filepath", None))
        if weight is None:  # If paths aren't to be sorted by length alone
            path_link_array = init_path_link_array(graph, k, disjoint=disjoint_paths, directed=graph.is_directed(),
                                                   weight=weight, modulations_array=modulations_array,
                                                   path_snr=path_snr)
        path_length_array = init_path_length_array(path_link_array, graph)
        path_se_array = init_path_se_array(path_length_array, modulations_array)
        min_se = min(path_se_array)  # if consider_modulation_format
        max_slots = required_slots(max_bw, min_se, slot_size, guardband=guardband)
        max_spans = int(jnp.ceil(max(link_length_array) / max_span_length)[0])
        if env_type == "rsa_gn_model":
            link_length_array = init_link_length_array_gn_model(graph, max_span_length, max_spans)
    else:
        path_se_array = jnp.array([1])
        if env_type == "rwa_lightpath_reuse":
            path_capacity_array = init_path_capacity_array(
                link_length_array, path_link_array, R_s=100, scale_factor=scale_factor, alpha=alpha,
                NF=amplifier_noise_figure, beta_2=beta_2, gamma=gamma, L_s=span_length, lambda0=lambda0, B=B
            )
            max_requests = int(scale_factor * max_requests)
        else:
            # If considering just RSA without physical layer considerations
            slot_size = 1
            link_length_array = jnp.ones((num_links, 1))
        max_slots = required_slots(max_bw, 1, slot_size, guardband=guardband)

    if incremental_loading:
        mean_service_holding_time = load = 1e6

    # Define edges for use with heuristics and GNNs
    edges = jnp.array(sorted(graph.edges))

    laplacian_matrix = jnp.array(nx.directed_laplacian_matrix(graph)) if graph.is_directed() \
        else jnp.array(nx.laplacian_matrix(graph).todense())

    max_timesteps = max_requests

    params_dict = dict(
        max_requests=max_requests,
        max_timesteps=max_timesteps,
        mean_service_holding_time=mean_service_holding_time,
        k_paths=k,
        link_resources=link_resources,
        num_nodes=num_nodes,
        num_links=num_links,
        load=load,
        arrival_rate=arrival_rate,
        path_link_array=HashableArrayWrapper(path_link_array) if not remove_array_wrappers else path_link_array,
        incremental_loading=incremental_loading,
        end_first_blocking=end_first_blocking,
        edges=HashableArrayWrapper(edges) if not remove_array_wrappers else edges,
        random_traffic=random_traffic,
        path_se_array=HashableArrayWrapper(path_se_array) if not remove_array_wrappers else path_se_array,
        link_length_array=HashableArrayWrapper(link_length_array) if not remove_array_wrappers else link_length_array,
        max_slots=int(max_slots),
        consider_modulation_format=consider_modulation_format,
        slot_size=slot_size,
        continuous_operation=continuous_operation,
        aggregate_slots=aggregate_slots,
        guardband=guardband,
        deterministic_requests=deterministic_requests,
        multiple_topologies=False,
        directed_graph=graph.is_directed(),
        maximise_throughput=maximise_throughput,
        values_bw=HashableArrayWrapper(values_bw) if not remove_array_wrappers else values_bw,
        reward_type=reward_type,
        truncate_holding_time=truncate_holding_time,
        log_actions=log_actions,
        traffic_array=traffic_array,
    )

    if env_type == "deeprmsa":
        env_params = DeepRMSAEnvParams
    elif env_type == "rwa_lightpath_reuse":
        env_params = RWALightpathReuseEnvParams
    elif env_type == "rsa_gn_model":
        env_params = RSAGNModelEnvParams
        params_dict.update(ref_lambda=ref_lambda, max_spans=max_spans, max_span_length=max_span_length,
                           default_launch_power=default_launch_power,
                           nonlinear_coeff=nonlinear_coeff, raman_gain_slope=raman_gain_slope, attenuation=attenuation,
                           attenuation_bar=attenuation_bar, dispersion_coeff=dispersion_coeff,
                           dispersion_slope=dispersion_slope, coherent=coherent,
                           modulations_array=HashableArrayWrapper(modulations_array) if not remove_array_wrappers else modulations_array,
                           noise_figure=noise_figure, interband_gap=interband_gap, mod_format_correction=mod_format_correction,
                           gap_start=gap_start, gap_width=gap_width, roadm_loss=roadm_loss, num_roadms=num_roadms,
                           num_spans=num_spans, launch_power_type=launch_power_type, snr_margin=snr_margin,
                           last_fit=config.get("last_fit", False), max_power=max_power, min_power=min_power,
                           step_power=step_power, max_snr=max_snr)
        # TODO - In order to do masking based on maximum reach of mod. format (which avoids extra calculation)
        #  calculate maximum reach here and update modulations_array.
        #  Write a function that takes params_dict as input, does the launch power optimisation, returns the maximum reach
    else:
        env_params = RSAEnvParams

    params = env_params(**params_dict)

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
            env = DeepRMSAEnv(rng, params, traffic_matrix=traffic_matrix, laplacian_matrix=laplacian_matrix)
        elif env_type == "rwa_lightpath_reuse":
            env = RWALightpathReuseEnv(
                rng, params, traffic_matrix=traffic_matrix, path_capacity_array=path_capacity_array,
                laplacian_matrix=laplacian_matrix)
        elif env_type == "rsa_gn_model":
            env = RSAGNModelEnv(rng, params, traffic_matrix=traffic_matrix, launch_power_array=launch_power_array,
                                list_of_requests=list_of_requests, laplacian_matrix=laplacian_matrix)
        else:
            env = RSAEnv(rng, params, traffic_matrix=traffic_matrix, list_of_requests=list_of_requests,
                         laplacian_matrix=laplacian_matrix)

    return env, params
