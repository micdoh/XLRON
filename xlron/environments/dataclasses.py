import chex
import jraph
from flax import struct
from typing import NamedTuple, Callable


@struct.dataclass
class VONETransition:
    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: chex.Array
    action_mask_s: chex.Array
    action_mask_p: chex.Array
    action_mask_d: chex.Array


@struct.dataclass
class RSATransition:
    done: chex.Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: chex.Array
    action_mask: chex.Array


@struct.dataclass
class Transition:
    done: chex.Array
    action: chex.Array
    reward: chex.Array
    obs: chex.Array
    info: chex.Array


@struct.dataclass
class EvalState:
    apply_fn: Callable
    sample_fn: Callable
    params: chex.Array



@struct.dataclass
class EnvState:
    """Dataclass to hold environment state. State is mutable and arrays are traced on JIT compilation.

    Args:
        current_time (chex.Scalar): Current time in environment
        holding_time (chex.Scalar): Holding time of current request
        total_timesteps (chex.Scalar): Total timesteps in environment
        total_requests (chex.Scalar): Total requests in environment
        graph (jraph.GraphsTuple): Graph tuple representing network state
        full_link_slot_mask (chex.Array): Action mask for link slot action (including if slot actions are aggregated)
        accepted_services (chex.Array): Number of accepted services
        accepted_bitrate (chex.Array): Accepted bitrate
        """
    current_time: chex.Scalar
    holding_time: chex.Scalar
    total_timesteps: chex.Scalar
    total_requests: chex.Scalar
    graph: jraph.GraphsTuple
    full_link_slot_mask: chex.Array
    accepted_services: chex.Array
    accepted_bitrate: chex.Array
    total_bitrate: chex.Array
    list_of_requests: chex.Array


@struct.dataclass
class EnvParams:
    """Dataclass to hold environment parameters. Parameters are immutable.

    Args:
        max_requests (chex.Scalar): Maximum number of requests in an episode
        incremental_loading (chex.Scalar): Incremental increase in traffic load (non-expiring requests)
        end_first_blocking (chex.Scalar): End episode on first blocking event
        continuous_operation (chex.Scalar): If True, do not reset the environment at the end of an episode
        edges (chex.Array): Two column array defining source-dest node-pair edges of the graph
        slot_size (chex.Scalar): Spectral width of frequency slot in GHz
        consider_modulation_format (chex.Scalar): If True, consider modulation format to determine required slots
        link_length_array (chex.Array): Array of link lengths
        aggregate_slots (chex.Scalar): Number of slots to aggregate into a single action (First-Fit with aggregation)
        guardband (chex.Scalar): Guard band in slots
        directed_graph (bool): Whether graph is directed (one fibre per link per transmission direction)
    """
    max_requests: chex.Scalar = struct.field(pytree_node=False)
    incremental_loading: chex.Scalar = struct.field(pytree_node=False)
    end_first_blocking: chex.Scalar = struct.field(pytree_node=False)
    continuous_operation: chex.Scalar = struct.field(pytree_node=False)
    edges: chex.Array = struct.field(pytree_node=False)
    slot_size: chex.Scalar = struct.field(pytree_node=False)
    consider_modulation_format: chex.Scalar = struct.field(pytree_node=False)
    link_length_array: chex.Array = struct.field(pytree_node=False)
    aggregate_slots: chex.Scalar = struct.field(pytree_node=False)
    guardband: chex.Scalar = struct.field(pytree_node=False)
    directed_graph: bool = struct.field(pytree_node=False)
    maximise_throughput: bool = struct.field(pytree_node=False)
    reward_type: str = struct.field(pytree_node=False)
    values_bw: chex.Array = struct.field(pytree_node=False)
    truncate_holding_time: bool = struct.field(pytree_node=False)
    traffic_array: bool = struct.field(pytree_node=False)


@struct.dataclass
class LogEnvState:
    """Dataclass to hold environment state for logging.

    Args:
        env_state (EnvState): Environment state
        lengths (chex.Scalar): Lengths
        returns (chex.Scalar): Returns
        cum_returns (chex.Scalar): Cumulative returns
        episode_lengths (chex.Scalar): Episode lengths
        episode_returns (chex.Scalar): Episode returns
        accepted_services (chex.Scalar): Accepted services
        accepted_bitrate (chex.Scalar): Accepted bitrate
        done (chex.Scalar): Done
    """
    env_state: EnvState
    lengths: float
    returns: float
    cum_returns: float
    accepted_services: int
    accepted_bitrate: float
    total_bitrate: float
    utilisation: float
    done: bool


@struct.dataclass
class RSAEnvState(EnvState):
    """Dataclass to hold environment state for RSA.

    Args:
        link_slot_array (chex.Array): Link slot array
        request_array (chex.Array): Request array
        link_slot_departure_array (chex.Array): Link slot departure array
        link_slot_mask (chex.Array): Link slot mask
        traffic_matrix (chex.Array): Traffic matrix
    """
    link_slot_array: chex.Array
    request_array: chex.Array
    link_slot_departure_array: chex.Array
    link_slot_mask: chex.Array
    traffic_matrix: chex.Array


@struct.dataclass
class RSAEnvParams(EnvParams):
    """Dataclass to hold environment parameters for RSA.

    Args:
        num_nodes (chex.Scalar): Number of nodes
        num_links (chex.Scalar): Number of links
        link_resources (chex.Scalar): Number of link resources
        k_paths (chex.Scalar): Number of paths
        mean_service_holding_time (chex.Scalar): Mean service holding time
        load (chex.Scalar): Load
        arrival_rate (chex.Scalar): Arrival rate
        path_link_array (chex.Array): Path link array
        random_traffic (bool): Random traffic matrix for RSA on each reset (else uniform or custom)
        max_slots (chex.Scalar): Maximum number of slots
        path_se_array (chex.Array): Path spectral efficiency array
        deterministic_requests (bool): If True, use deterministic requests
        multiple_topologies (bool): If True, use multiple topologies
    """
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
    multiple_topologies: bool = struct.field(pytree_node=False)
    log_actions: bool = struct.field(pytree_node=False)
    disable_node_features: bool = struct.field(pytree_node=False)


@struct.dataclass
class DeepRMSAEnvState(RSAEnvState):
    """Dataclass to hold environment state for DeepRMSA.

    Args:
        path_stats (chex.Array): Path stats array containing
        1. Required slots on path
        2. Total available slots on path
        3. Size of 1st free spectrum block
        4. Avg. free block size
    """
    path_stats: chex.Array


@struct.dataclass
class DeepRMSAEnvParams(RSAEnvParams):
    pass


@struct.dataclass
class RWALightpathReuseEnvState(RSAEnvState):
    """Dataclass to hold environment state for RWA with lightpath reuse.

    Args:
        path_index_array (chex.Array): Contains indices of lightpaths in use on slots
        path_capacity_array (chex.Array): Contains remaining capacity of each lightpath
        link_capacity_array (chex.Array): Contains remaining capacity of lightpath on each link-slot
    """
    path_index_array: chex.Array  # Contains indices of lightpaths in use on slots
    path_capacity_array: chex.Array  # Contains remaining capacity of each lightpath
    link_capacity_array: chex.Array  # Contains remaining capacity of lightpath on each link-slot
    time_since_last_departure: chex.Array  # Time since last departure


@struct.dataclass
class RWALightpathReuseEnvParams(RSAEnvParams):
    pass


@struct.dataclass
class GNModelEnvParams(RSAEnvParams):
    """Dataclass to hold environment state for GN model environments.
    """
    ref_lambda: chex.Scalar = struct.field(pytree_node=False)
    max_spans: chex.Scalar = struct.field(pytree_node=False)
    max_span_length: chex.Scalar = struct.field(pytree_node=False)
    nonlinear_coeff: chex.Scalar = struct.field(pytree_node=False)
    raman_gain_slope: chex.Scalar = struct.field(pytree_node=False)
    attenuation: chex.Scalar = struct.field(pytree_node=False)
    attenuation_bar: chex.Scalar = struct.field(pytree_node=False)
    dispersion_coeff: chex.Scalar = struct.field(pytree_node=False)
    dispersion_slope: chex.Scalar = struct.field(pytree_node=False)
    noise_figure: chex.Scalar = struct.field(pytree_node=False)
    coherent: bool = struct.field(pytree_node=False)
    gap_width: chex.Scalar = struct.field(pytree_node=False)
    gap_start: chex.Scalar = struct.field(pytree_node=False)
    num_roadms: chex.Scalar = struct.field(pytree_node=False)
    roadm_loss: chex.Scalar = struct.field(pytree_node=False)
    num_spans: chex.Scalar = struct.field(pytree_node=False)
    launch_power_type: chex.Scalar = struct.field(pytree_node=False)
    snr_margin: chex.Scalar = struct.field(pytree_node=False)
    max_snr: chex.Scalar = struct.field(pytree_node=False)
    max_power: chex.Scalar = struct.field(pytree_node=False)
    min_power: chex.Scalar = struct.field(pytree_node=False)
    step_power: chex.Scalar = struct.field(pytree_node=False)
    last_fit: bool = struct.field(pytree_node=False)
    default_launch_power: chex.Scalar = struct.field(pytree_node=False)
    mod_format_correction: bool = struct.field(pytree_node=False)


@struct.dataclass
class GNModelEnvState(RSAEnvState):
    """Dataclass to hold environment state for RSA with GN model.
    """
    link_snr_array: chex.Array  # Available SNR on each link
    channel_centre_bw_array: chex.Array  # Channel centre bandwidth for each active connection
    path_index_array: chex.Array  # Contains indices of lightpaths in use on slots (used for lightpath SNR calculation)
    channel_power_array: chex.Array  # Channel power for each active connection
    channel_centre_bw_array_prev: chex.Array  # Channel centre bandwidth for each active connection in previous timestep
    path_index_array_prev: chex.Array  # Contains indices of lightpaths in use on slots in previous timestep
    channel_power_array_prev: chex.Array  # Channel power for each active connection in previous timestep
    launch_power_array: chex.Array  # Launch power array


@struct.dataclass
class RSAGNModelEnvParams(GNModelEnvParams):
    """Dataclass to hold environment params for RSA with GN model.
    """
    pass


@struct.dataclass
class RSAGNModelEnvState(GNModelEnvState):
    """Dataclass to hold environment state for RSA with GN model.
    """
    active_lightpaths_array: chex.Array  # Active lightpath array. 1 x M array. Each value is a lightpath index. Used to calculate total throughput.
    active_lightpaths_array_departure: chex.Array  # Active lightpath array departure time.
    current_throughput: chex.Array  # Current throughput


@struct.dataclass
class RMSAGNModelEnvParams(GNModelEnvParams):
    """Dataclass to hold environment params for RMSA with GN model.

    Args:
        link_snr_array (chex.Array): Link SNR array
    """
    modulations_array: chex.Array = struct.field(pytree_node=False)


@struct.dataclass
class RMSAGNModelEnvState(GNModelEnvState):
    """Dataclass to hold environment state for RMSA with GN model.

    Args:
        link_snr_array (chex.Array): Link SNR array
    """
    modulation_format_index_array: chex.Array  # Modulation format index for each active connection
    modulation_format_index_array_prev: chex.Array  # Modulation format index for each active connection in previous timestep
    mod_format_mask: chex.Array  # Modulation format mask


@struct.dataclass
class VONEEnvState(EnvState):
    """Dataclass to hold environment state for VONE.

    Args:
        link_slot_array (chex.Array): Link slot array
        node_capacity_array (chex.Array): Node capacity array
        node_resource_array (chex.Array): Node resource array
        node_departure_array (chex.Array): Node departure array
        link_slot_departure_array (chex.Array): Link slot departure array
        request_array (chex.Array): Request array
        action_counter (chex.Array): Action counter
        action_history (chex.Array): Action history
        node_mask_s (chex.Array): Node mask for source node
        link_slot_mask (chex.Array): Link slot mask
        node_mask_d (chex.Array): Node mask for destination node
        virtual_topology_patterns (chex.Array): Virtual topology patterns
        values_nodes (chex.Array): Values for nodes
    """
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


@struct.dataclass
class VONEEnvParams(EnvParams):
    """Dataclass to hold environment parameters for VONE.

    Args:
        num_nodes (chex.Scalar): Number of nodes
        num_links (chex.Scalar): Number of links
        node_resources (chex.Scalar): Number of node resources
        k_paths (chex.Scalar): Number of paths
        load (chex.Scalar): Load
        mean_service_holding_time (chex.Scalar): Mean service holding time
        arrival_rate (chex.Scalar): Arrival rate
        max_edges (chex.Scalar): Maximum number of edges
        min_node_resources (chex.Scalar): Minimum number of node resources
        max_node_resources (chex.Scalar): Maximum number of node resources
        path_link_array (chex.Array): Path link array
        max_slots (chex.Scalar): Maximum number of slots
        path_se_array (chex.Array): Path spectral efficiency array
    """
    num_nodes: chex.Scalar = struct.field(pytree_node=False)
    num_links: chex.Scalar = struct.field(pytree_node=False)
    node_resources: chex.Scalar = struct.field(pytree_node=False)
    link_resources: chex.Scalar = struct.field(pytree_node=False)
    k_paths: chex.Scalar = struct.field(pytree_node=False)
    load: chex.Scalar = struct.field(pytree_node=False)
    mean_service_holding_time: chex.Scalar = struct.field(pytree_node=False)
    arrival_rate: chex.Scalar = struct.field(pytree_node=False)
    max_edges: chex.Scalar = struct.field(pytree_node=False)
    min_node_resources: chex.Scalar = struct.field(pytree_node=False)
    max_node_resources: chex.Scalar = struct.field(pytree_node=False)
    path_link_array: chex.Array = struct.field(pytree_node=False)
    max_slots: chex.Scalar = struct.field(pytree_node=False)
    path_se_array: chex.Array = struct.field(pytree_node=False)
    # TODO - Add Laplacian matrix (for node heuristics)
