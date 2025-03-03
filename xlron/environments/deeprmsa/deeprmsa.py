from gymnax.environments import spaces
from xlron.environments.env_funcs import (
    init_rsa_request_array, init_link_slot_array, init_link_slot_departure_array, init_traffic_matrix,
    calculate_path_stats,
)
from xlron.environments.dataclasses import *
from xlron.environments.wrappers import *
from xlron.environments import RSAEnv, RSAEnvParams, RSAEnvState



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
