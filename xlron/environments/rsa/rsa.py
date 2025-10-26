from gymnax.environments import environment, spaces
from xlron.environments.env_funcs import (
    init_rsa_request_array, init_link_slot_array,
    init_link_slot_mask, init_link_slot_departure_array, init_traffic_matrix, update_graph_tuple, implement_action_rsa,
    check_action_rsa, undo_action_rsa, finalise_action_rsa, generate_request_rsa, mask_slots, calculate_path_stats,
    init_graph_tuple, implement_action_rwalr, check_action_rwalr, undo_action_rwalr, finalise_action_rwalr,
    generate_request_rwalr, check_action_rmsa_gn_model, implement_action_rmsa_gn_model, implement_action_rsa_gn_model,
    undo_action_rsa_gn_model, finalise_action_rsa_gn_model, undo_action_rmsa_gn_model, finalise_action_rmsa_gn_model,
    set_band_gaps
)
from xlron.environments.dataclasses import *
from xlron.environments.wrappers import *
from xlron.dtype_config import COMPUTE_DTYPE, PARAMS_DTYPE, LARGE_INT_DTYPE, LARGE_FLOAT_DTYPE, \
    SMALL_INT_DTYPE, SMALL_FLOAT_DTYPE, MED_INT_DTYPE

one = jnp.array(1, dtype=LARGE_FLOAT_DTYPE)
zero = jnp.array(0, dtype=LARGE_FLOAT_DTYPE)


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
            current_time=0.,
            holding_time=0.,
            arrival_time=0.,
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
        if not params.__class__.__name__ in ["RSAGNModelEnvParams", "RMSAGNModelEnvParams"]:
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
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, state, action, params
        )
        def reset_fn(args):
            key_reset, state_st, params, state = args
            obs_re, state_re = self.reset(key_reset, params, state)
            return obs_re, state_re

        def continue_fn(args):
            _, state_st, _, _ = args
            return obs_st, state_st

        obs, new_state = jax.lax.cond(
            done,
            reset_fn,
            continue_fn,
            (key_reset, state_st, params, state)
        )
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(new_state),
            reward,
            done,
            info
        )

    @partial(jax.jit, static_argnums=(0, 2,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[RSAEnvParams] = None, state: Optional[RSAEnvState] = None
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
    ) -> Tuple[chex.Array, RSAEnvState, chex.Array, chex.Array, chex.Array]:
        """Environment-specific step transition.
        1. Implement action
        2. Check if action was valid
            - If valid, calculate reward and finalise action
            - If invalid, calculate reward and undo action
        3. Generate new request, update current time, remove expired requests
        4. Update timesteps
        5. Terminate if max_requests exceeded
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
        undo_finalise_state = [state, params, action]
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
            check_action = check_action_rsa
            undo = undo_action_rsa_gn_model
            finalise = finalise_action_rsa_gn_model
            check_state = [state]
            input_state = [state, action, params]
        elif params.__class__.__name__ == "RMSAGNModelEnvParams":
            implement_action = implement_action_rmsa_gn_model
            check_action = check_action_rmsa_gn_model
            undo = undo_action_rmsa_gn_model
            finalise = finalise_action_rmsa_gn_model
            input_state = check_state = [state, action, params]
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
        state = state.replace(total_timesteps=state.total_timesteps + 1)
        # Terminate if max_requests exceeded or, if consecutive loading,
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
        self, key: chex.PRNGKey, params: RSAEnvParams, state: Optional[RSAEnvState] = None
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
        return jnp.array(state.total_requests >= params.max_requests)

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
        reward = -one
        if params.reward_type == "service":
            pass
        elif params.reward_type == "bitrate":
            reward = state.request_array[1] * reward / jnp.max(params.values_bw.val)
        else:
            reward = reward * read_rsa_request(state.request_array)[1] / jnp.max(params.values_bw.val) if params.maximise_throughput else reward
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
        reward = zero
        if params.__class__.__name__ in ["RSAGNModelEnvParams", "RMSAGNModelEnvParams"]:
            path_action, _ = action
        else:
            path_action = action
        if params.reward_type != "service":
            nodes_sd, requested_datarate = read_rsa_request(state.request_array)
            k_index, slot_index = process_path_action(state, params, path_action)
            reward = state.request_array[1] * reward / jnp.max(params.values_bw.val)
            if params.reward_type == "bitrate":
                return reward
            elif params.reward_type == "snr":
                assert params.__class__.__name__ == "RSAGNModelEnvParams"
                path_start_index = get_path_indices(nodes_sd[0], nodes_sd[1], params.k_paths, params.num_nodes,
                                                    directed=params.directed_graph).astype(LARGE_INT_DTYPE)
                path = params.path_link_array[path_start_index + k_index]
                path_snr = get_snr_for_path(path, state.link_snr_array, params)[slot_index.astype(MED_INT_DTYPE)]
                # set to 0 if negative and divide by large SNR (e.g. 50. dB) to scale below 1
                # N.B. negative SNR in dB would be a fail anyway since min. required is 10dB
                path_snr_norm = jnp.where(path_snr < zero, zero, path_snr) / params.max_snr
                return reward + path_snr_norm
            elif params.reward_type == "mod_format":
                assert params.__class__.__name__ == "RSAGNModelEnvParams"
                mod_format_index = get_path_slots(
                    state.modulation_format_index_array, params, nodes_sd, k_index, agg_func='max'
                )[slot_index.astype(MED_INT_DTYPE)]
                return reward + 0.05 * (one + mod_format_index)
            else:
                return reward
        else:
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


class RSAMultibandEnv(RSAEnv):

    def __init__(
            self,
            key: chex.PRNGKey,
            params: RSAEnvParams,
            traffic_matrix: chex.Array = None,
            list_of_requests: chex.Array = None,
            laplacian_matrix: chex.Array = None,
    ):
        super().__init__(key, params, traffic_matrix=traffic_matrix, list_of_requests=list_of_requests, laplacian_matrix=laplacian_matrix)
        state = RSAMultibandEnvState(
            current_time=0,
            holding_time=0,
            arrival_time=0,
            total_timesteps=0,
            total_requests=-1,
            link_slot_array=set_band_gaps(init_link_slot_array(params), params, -1.),
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
        )
        self.initial_state = state.replace(graph=init_graph_tuple(state, params, laplacian_matrix))
