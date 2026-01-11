from xlron.environments.env_funcs import (
    init_rsa_request_array, init_link_slot_array, init_link_slot_mask, init_link_slot_departure_array,
    init_traffic_matrix, init_graph_tuple, init_link_capacity_array, init_path_index_array, mask_slots_rwalr,
)
from xlron.environments.dataclasses import *
from xlron.environments.wrappers import *
from xlron.environments import RSAEnv, RSAEnvState, RSAEnvParams


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
            list_of_requests: chex.Array = None,
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
        super().__init__(key, params, traffic_matrix=traffic_matrix, list_of_requests=list_of_requests, laplacian_matrix=laplacian_matrix)
        state = RWALightpathReuseEnvState(
            current_time=0,
            arrival_time=0,
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
            list_of_requests=list_of_requests,
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
