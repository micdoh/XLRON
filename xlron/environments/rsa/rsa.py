import math
from typing import Tuple

from gymnax.environments import environment, spaces

from xlron import dtype_config
from xlron.environments.dataclasses import (
    ActionInfo,
    EnvParams,
    EnvState,
    RSAEnvParams,
    RSAEnvState,
    RSAMultibandEnvState,
)
from xlron.environments.diff_utils import *
from xlron.environments.env_funcs import (
    calculate_path_stats,
    check_action_rmsa_gn_model,
    check_action_rsa,
    check_action_rwalr,
    complete_step_rsa,
    generate_request_rsa,
    generate_request_rwalr,
    get_affected_slots_mask,
    get_path_and_se,
    get_path_slots,
    implement_action_rmsa_gn_model,
    implement_action_rsa,
    implement_action_rsa_gn_model,
    implement_action_rwalr,
    init_graph_tuple,
    init_link_slot_array,
    init_link_slot_departure_array,
    init_link_slot_mask,
    init_rsa_request_array,
    init_traffic_matrix,
    mask_slots,
    required_slots,
    set_band_gaps,
    undo_action_rmsa_gn_model,
    undo_action_rsa_gn_model,
    undo_action_rwalr,
    update_graph_tuple,
)
from xlron.environments.wrappers import *

one = jnp.array(1, dtype=dtype_config.LARGE_FLOAT_DTYPE)
zero = jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE)


class RSAEnv(environment.Environment):
    """This environment simulates the Routing Modulation and Spectrum Assignment (RMSA) problem.
    It can model RSA by setting consider_modulation_format=False in params.
    It can model RWA by setting min_bw=0, max_bw=0, and consider_modulation_format=False in params.
    """

    def __init__(
        self,
        key: chex.PRNGKey,
        params: RSAEnvParams,
        traffic_matrix: chex.Array | None = None,
        list_of_requests: chex.Array | None = None,
        laplacian_matrix: chex.Array | None = None,
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
            current_time=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            holding_time=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            arrival_time=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            total_timesteps=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            total_requests=jnp.array(-1, dtype=dtype_config.LARGE_INT_DTYPE),
            link_slot_array=init_link_slot_array(params),
            link_slot_departure_array=init_link_slot_departure_array(params),
            request_array=init_rsa_request_array(),
            link_slot_mask=init_link_slot_mask(
                params, include_no_op=params.include_no_op, agg=params.aggregate_slots
            ),
            traffic_matrix=traffic_matrix
            if traffic_matrix is not None
            else init_traffic_matrix(key, params),
            list_of_requests=list_of_requests,
            graph=None,
            full_link_slot_mask=init_link_slot_mask(params),
            accepted_services=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            accepted_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            total_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            valid_mass=jnp.array(1.0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
        )
        if params.__class__.__name__ not in ["RSAGNModelEnvParams", "RMSAGNModelEnvParams"]:
            self.initial_state = state.replace(
                graph=init_graph_tuple(state, params, laplacian_matrix)
            )

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: RSAEnvState,
        action: Union[int, float],
        params: Optional[RSAEnvParams] = None,
    ) -> Tuple[chex.Array, RSAEnvState, float, bool, bool, dict]:
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
            terminal: True if terminal condition met
            truncated: True if max_requests reached
            info: Additional information
        """
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminal, truncated, info = self.step_env(
            key, state, action, params
        )

        def reset_fn(args):
            key_reset, state_st, params, state = args
            obs_re, state_re = self.reset(key_reset, params, state)
            return obs_re, state_re

        def continue_fn(args):
            _, state_st, _, _ = args
            return obs_st, state_st

        done = jnp.logical_or(terminal, truncated)

        obs, new_state = jax.lax.cond(
            done, reset_fn, continue_fn, (key_reset, state_st, params, state)
        )
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(new_state),
            reward,
            terminal,
            truncated,
            info,
        )

    @partial(
        jax.jit,
        static_argnums=(
            0,
            2,
        ),
    )
    def reset(
        self,
        key: chex.PRNGKey,
        params: Optional[RSAEnvParams] = None,
        state: Optional[RSAEnvState] = None,
    ) -> Tuple[chex.Array, RSAEnvState]:
        """Performs resetting of environment.

        Args:
            key: PRNG key
            params: Environment parameters

        Returns:
            obs: Observation
            state: Reset environment state
        """
        obs, state = self.reset_env(key, params, state)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: RSAEnvState,
        action: Union[int, float],
        params: RSAEnvParams,
    ) -> Tuple[chex.Array, RSAEnvState, chex.Array, chex.Array, chex.Array, dict]:
        """Environment-specific step transition.
        1. Implement action
        2. Check if action was valid
            - If valid, calculate reward and finalise action
            - If invalid, calculate reward and undo action
        3. Generate new request, update current time, remove expired requests
        4. Update timesteps
        5. Check for terminal/truncated conditions
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
            terminal: True if terminal condition met (e.g., end_first_blocking)
            truncated: True if max_requests reached
            info: Additional information
        """
        # Compute relevant info from action
        action_info = jit_profiler.call(params.profile, self.process_action, state, action, params)

        # Define env-type specific functions
        implement_action = implement_action_rsa
        check_action = check_action_rsa
        complete_step = complete_step_rsa
        generate_request = generate_request_rsa

        if params.__class__.__name__ == "RWALightpathReuseEnvParams":
            implement_action = implement_action_rwalr
            check_action = check_action_rwalr
            # input_state = check_state = [state, action_info, params]
            if not params.incremental_loading:
                # These are relevant to dynamic RWA-LR (upcoming)
                complete_step = undo_action_rwalr
                generate_request = generate_request_rwalr
        elif params.__class__.__name__ == "RSAGNModelEnvParams":
            implement_action = implement_action_rsa_gn_model
            check_action = check_action_rsa
            complete_step = undo_action_rsa_gn_model
        elif params.__class__.__name__ == "RMSAGNModelEnvParams":
            implement_action = implement_action_rmsa_gn_model
            check_action = check_action_rmsa_gn_model
            complete_step = undo_action_rmsa_gn_model

        # Implement action
        state = jit_profiler.call(params.profile, implement_action, state, action_info, params)

        # Check action
        check = jit_profiler.call(params.profile, check_action, state, action_info, params)

        # Calculate reward
        reward = jit_profiler.call(params.profile, self.calculate_reward, state, action_info, check, params)

        # Complete step
        state = jit_profiler.call(params.profile, complete_step, state, action_info, check, params)

        # TODO (DYNAMIC-RWALR) - calculate allocated bandwidth
        # TODO (DYNAMIC-RWALR) - generate new request if allocated DR equals requested DR, else update requested DR do not advance time do not replace source-dest
        # TODO (AFTERSTATE) - write separate functions for deterministic transition (above) and stochastic transition (below)
        state = jit_profiler.call(params.profile, generate_request, key, state, params)

        # Terminate if max_requests exceeded or, if consecutive loading,
        # then terminate if reward is failure but not before min number of timesteps before update
        terminal = self.is_terminal(state, params, reward)
        truncated = self.is_truncated(state, params)

        info = {}

        # Calculate path stats if DeepRMSAEnv
        if params.__class__.__name__ == "DeepRMSAEnvParams":
            path_stats = calculate_path_stats(state, params, state.request_array)
            state = state.replace(path_stats=path_stats)
        # Update graph tuple
        elif params.use_gnn:
            state = update_graph_tuple(state, params)

        # Get observation
        obs = jit_profiler.call(params.profile, self.get_obs, state, params)

        return obs, state, reward, terminal, truncated, info

    @partial(jax.jit, static_argnums=(0, 2))
    def reset_env(
        self,
        key: chex.PRNGKey,
        params: RSAEnvParams,
        state: Optional[RSAEnvState] = None,
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
        # if params.multiple_topologies:
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

    def process_action(
        self,
        state: EnvState,
        action: Array,
        params: EnvParams,
    ) -> ActionInfo:
        """Processes action into relevant information."""
        nodes_sd, requested_datarate = jit_profiler.call(
            params.profile, read_rsa_request, state.request_array
        )
        path_index, initial_slot_index = jit_profiler.call(
            params.profile, process_path_action, state, params, action
        )
        path, path_se = jit_profiler.call(
            params.profile, get_path_and_se, params, nodes_sd, path_index
        )
        path_se = path_se if params.consider_modulation_format else one
        num_slots = jit_profiler.call(
            params.profile,
            required_slots,
            requested_datarate,
            path_se,
            params.slot_size,
            guardband=params.guardband,
            temperature=params.temperature,
        )
        affected_slots_mask = jit_profiler.call(
            params.profile,
            get_affected_slots_mask,
            state,
            initial_slot_index,
            num_slots,
            path,
            params,
        )
        action_info = ActionInfo(
            action=action,
            path_index=path_index,
            initial_slot_index=initial_slot_index,
            num_slots=num_slots,
            path=path,
            se=path_se,
            requested_datarate=requested_datarate,
            nodes_sd=nodes_sd,
            affected_slots_mask=affected_slots_mask,
        )
        return action_info

    @partial(jax.jit, static_argnums=(0, 2))
    def action_mask(self, state: RSAEnvState, params: RSAEnvParams) -> Array:
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
        action_mask = jit_profiler.call(params.profile, mask_slots, state, params)
        return action_mask

    @partial(jax.jit, static_argnums=(0, 2))
    def get_obs_unflat(
        self, state: RSAEnvState, params: RSAEnvParams
    ) -> Tuple[chex.Array, chex.Array]:
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

    @partial(jax.jit, static_argnums=(0, 2))
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

    def is_terminal(
        self, state: RSAEnvState, params: RSAEnvParams, reward: chex.Array | None = None
    ) -> chex.Array:
        """Check whether state transition is terminal.

        Args:
            state: Environment state
            params: Environment parameters
            reward: Reward from current step (needed for end_first_blocking check)

        Returns:
            done: Boolean termination flag
        """
        if params.end_first_blocking:
            return jnp.array(reward == self.get_reward_failure(state, params))
        elif params.terminate_on_episode_end:
            return self.is_truncated(state, params)
        else:
            return jnp.array(False)

    def is_truncated(self, state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
        """Check whether state transition is truncated i.e. max steps reached.

        Args:
            state: Environment state
            params: Environment parameters

        Returns:
            done: Boolean termination flag
        """
        if params.continuous_operation:
            return jnp.array(False)
        else:
            return jnp.array(state.total_requests >= params.max_requests)

    @staticmethod
    def add_integer_bonus(action, scale=0.0):
        """
        Adds a small reward bonus that peaks at integer values.

        Args:
            action: The action(s) to evaluate
            scale: How strongly to encourage integer values

        Returns:
            A small reward bonus that peaks at integer values
        """
        return scale * jnp.cos(2 * jnp.pi * action)

    @staticmethod
    def penalise_non_integer_action(action, params, scale=0.0):
        """
        Penalises non-integer actions.

        Args:
            action: The action(s) to evaluate
            scale: How strongly to penalise non-integer values

        Returns:
            A penalty for non-integer actions
        """
        return -scale * jnp.abs(
            action
            - differentiable_round_simple(
                action,
                temperature=params.temperature,
                differentiable=params.differentiable,
            )
        )
        
    def calculate_reward(self, state: RSAEnvState, action_info: ActionInfo, check: Array, params: RSAEnvParams) -> chex.Array:
        """Calculate reward for current state and action.

        Args:
            state: Environment state
            action_info: Processed action information
            check: Result of action validity check
            params: Environment parameters
            
        Returns:
            reward: Calculated reward
        """
        reward = self.get_reward_failure(state, action_info, params) * check + (
            (1 - check) * self.get_reward_success(state, action_info, params)
        )
        return reward  

    def get_reward_failure(
        self,
        state: Optional[EnvState] = None,
        action_info: Optional[ActionInfo] = None,
        params: Optional[EnvParams] = None,
    ) -> chex.Array:
        """Return reward for current state.

        Args:
            state (optional): Environment state

        Returns:
            reward: Reward for failure
        """
        reward = -one
        if params.reward_type == "service":
            pass
        elif params.reward_type == "bitrate":
            reward = (
                differentiable_index(
                    state.request_array,
                    1,
                    temperature=params.temperature,
                    differentiable=params.differentiable,
                )
                * reward
                / jnp.max(params.values_bw.val)
            )
        else:
            reward = (
                reward
                * differentiable_index(
                    read_rsa_request(state.request_array),
                    1,
                    temperature=params.temperature,
                    differentiable=params.differentiable,
                )
                / jnp.max(params.values_bw.val)
                if params.maximise_throughput
                else reward
            )
        return reward

    def get_reward_success(
        self,
        state: Optional[EnvState] = None,
        action_info: Optional[ActionInfo] = None,
        params: Optional[EnvParams] = None,
    ) -> chex.Array:
        """Return reward for current state.

        Args:
            state: (optional) Environment state

        Returns:
            reward: Reward for success
        """
        reward = zero
        if params.__class__.__name__ in ["RSAGNModelEnvParams", "RMSAGNModelEnvParams"]:
            path_action, _ = action_info.action

        if params.reward_type != "service":
            reward = state.request_array[1] * reward / jnp.max(params.values_bw.val)
            if params.reward_type == "bitrate":
                pass  # No additional calculation needed
            elif params.reward_type == "snr":
                # SNR calculation...
                assert params.__class__.__name__ == "RSAGNModelEnvParams"
                path_start_index = get_path_indices(
                    params,
                    action_info.nodes_sd[0],
                    action_info.nodes_sd[1],
                    params.k_paths,
                    params.num_nodes,
                ).astype(dtype_config.LARGE_INT_DTYPE)
                path = params.path_link_array[path_start_index + action_info.path_index]
                path_snr = get_snr_for_path(path, state.link_snr_array, params)[
                    action_info.initial_slot_index.astype(dtype_config.LAREG_INT_DTYPE)
                ]
                # set to 0 if negative and divide by large SNR (e.g. 50. dB) to scale below 1
                # N.B. negative SNR in dB would be a fail anyway since min. required is 10dB
                path_snr_norm = jnp.where(path_snr < zero, zero, path_snr) / params.max_snr
                return reward + path_snr_norm
            elif params.reward_type == "mod_format":
                # Modulation format calculation...
                assert params.__class__.__name__ == "RSAGNModelEnvParams"
                mod_format_index = get_path_slots(
                    state.modulation_format_index_array,
                    params,
                    action_info.nodes_sd,
                    action_info.path_index,
                    agg_func="max",
                )[action_info.initial_slot_index.astype(dtype_config.LARGE_INT_DTYPE)]
                return reward + 0.05 * (one + mod_format_index)
            else:
                return reward
        else:
            reward = reward + self.penalise_non_integer_action(
                action_info.action, params
            )  # + self.add_integer_bonus(action)

        return reward

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @staticmethod
    def num_actions(params: EnvParams) -> int:
        """Number of actions possible in environment."""
        return math.ceil(params.link_resources / params.aggregate_slots) * params.k_paths

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions(params))

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Discrete(
            3  # Request array
            + params.num_links * params.link_resources  # Link slot array
        )

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return spaces.Dict(
            {
                "current_time": spaces.Discrete(1),
                "request_array": spaces.Discrete(3),
                "link_slot_array": spaces.Discrete(params.num_links * params.link_resources),
                "link_slot_departure_array": spaces.Discrete(
                    params.num_links * params.link_resources
                ),
            }
        )


class RSAMultibandEnv(RSAEnv):
    def __init__(
        self,
        key: chex.PRNGKey,
        params: RSAEnvParams,
        traffic_matrix: chex.Array | None = None,
        list_of_requests: chex.Array | None = None,
        laplacian_matrix: chex.Array | None = None,
    ):
        super().__init__(
            key,
            params,
            traffic_matrix=traffic_matrix,
            list_of_requests=list_of_requests,
            laplacian_matrix=laplacian_matrix,
        )
        state = RSAMultibandEnvState(
            current_time=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            holding_time=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            arrival_time=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            total_timesteps=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            total_requests=jnp.array(-1, dtype=dtype_config.LARGE_INT_DTYPE),
            link_slot_array=set_band_gaps(init_link_slot_array(params), params, -1.0),
            link_slot_departure_array=init_link_slot_departure_array(params),
            request_array=init_rsa_request_array(),
            link_slot_mask=init_link_slot_mask(
                params, include_no_op=params.include_no_op, agg=params.aggregate_slots
            ),
            traffic_matrix=traffic_matrix
            if traffic_matrix is not None
            else init_traffic_matrix(key, params),
            graph=None,
            full_link_slot_mask=init_link_slot_mask(params),
            accepted_services=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            accepted_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            total_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            list_of_requests=list_of_requests,
            valid_mass=jnp.array(1.0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
        )
        self.initial_state = state.replace(graph=init_graph_tuple(state, params, laplacian_matrix))
