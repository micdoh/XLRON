from gymnax.environments import spaces
from xlron.environments.env_funcs import (
    init_rsa_request_array, init_link_slot_array, init_link_slot_mask, init_link_slot_departure_array, init_traffic_matrix,
    init_graph_tuple, init_path_index_array, init_link_snr_array,
    init_channel_centre_bw_array, init_channel_power_array,
    init_active_lightpaths_array, init_active_lightpaths_array_departure, get_paths_obs_gn_model
)
from xlron.environments.dataclasses import *
from xlron.environments.wrappers import *
from xlron.environments import RSAEnv, RSAEnvParams


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
            channel_centre_bw_array_prev=init_channel_centre_bw_array(params),
            channel_power_array_prev=init_channel_power_array(params),
            launch_power_array=launch_power_array,
            active_lightpaths_array=init_active_lightpaths_array(params),
            active_lightpaths_array_departure=init_active_lightpaths_array_departure(params),
            current_throughput=0.,
        )
        self.initial_state = state.replace(graph=init_graph_tuple(state, params, laplacian_matrix))

    @partial(jax.jit, static_argnums=(0, 2,))
    def get_obs(self, state: RSAGNModelEnvState, params: RSAGNModelEnvParams) -> chex.Array:
        return get_paths_obs_gn_model(state, params)

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
