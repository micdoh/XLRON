from typing import Callable, Dict, Generic, Sequence, Tuple, TypeVar

import chex
import jraph
from flax import struct
from jax import Array

Shape = Sequence[int]
T = TypeVar("T")  # Declare type variable


class HashableArrayWrapper(Generic[T]):
    """Wrapper for making arrays hashable.
    In order to access pre-computed data, such as shortest paths between node-pairs or the constituent links of a path,
    within a jitted function, we need to make the arrays containing this data hashable. By defining this wrapper, we can
    define a __hash__ method that returns a hash of the array's bytes, thus making the array hashable.
    From: https://github.com/google/jax/issues/4572#issuecomment-709677518
    """

    def __init__(self, val: Array):
        self.val = val

    def __getattribute__(self, prop):
        if prop == "val" or prop == "__hash__" or prop == "__eq__":
            return super(HashableArrayWrapper, self).__getattribute__(prop)
        return getattr(self.val, prop)

    def __getitem__(self, key):
        return self.val[key]

    def __setitem__(self, key, val):
        self.val[key] = val

    def __hash__(self):
        return hash(self.val.tobytes())

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return self.__hash__() == other.__hash__()

        f = getattr(self.val, "__eq__")
        return f(self, other)


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

    current_time: chex.Array
    holding_time: chex.Array
    arrival_time: chex.Array
    total_timesteps: chex.Array
    total_requests: chex.Array
    graph: jraph.GraphsTuple
    full_link_slot_mask: chex.Array
    accepted_services: chex.Array
    accepted_bitrate: chex.Array
    total_bitrate: chex.Array
    list_of_requests: chex.Array
    link_slot_array: chex.Array
    request_array: chex.Array
    link_slot_departure_array: chex.Array
    link_slot_mask: chex.Array
    traffic_matrix: chex.Array


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
        temperature (chex.Scalar): Temp. used for softmax differentiable approximation
        window_size (chex.Scalar): Window size for weighted average of neighbouring cells in differentiable indexing
    """

    num_nodes: int = struct.field(pytree_node=False)
    num_links: int = struct.field(pytree_node=False)
    max_requests: int = struct.field(pytree_node=False)
    incremental_loading: bool = struct.field(pytree_node=False)
    end_first_blocking: bool = struct.field(pytree_node=False)
    terminate_on_episode_end: bool = struct.field(pytree_node=False)
    continuous_operation: bool = struct.field(pytree_node=False)
    edges: HashableArrayWrapper = struct.field(pytree_node=False)
    slot_size: int = struct.field(pytree_node=False)
    consider_modulation_format: bool = struct.field(pytree_node=False)
    link_length_array: HashableArrayWrapper = struct.field(pytree_node=False)
    aggregate_slots: int = struct.field(pytree_node=False)
    guardband: int = struct.field(pytree_node=False)
    directed_graph: bool = struct.field(pytree_node=False)
    maximise_throughput: bool = struct.field(pytree_node=False)
    reward_type: str = struct.field(pytree_node=False)
    values_bw: HashableArrayWrapper = struct.field(pytree_node=False)
    truncate_holding_time: bool = struct.field(pytree_node=False)
    traffic_array: bool = struct.field(pytree_node=False)
    pack_path_bits: bool = struct.field(pytree_node=False)
    relative_arrival_times: bool = struct.field(pytree_node=False)
    temperature: float = struct.field(pytree_node=False)
    differentiable: bool = struct.field(pytree_node=False)
    num_spectral_features: int = struct.field(pytree_node=False)
    line_graph_spectral_features: HashableArrayWrapper | None = struct.field(pytree_node=False)
    path_link_array: HashableArrayWrapper = struct.field(pytree_node=False)
    path_se_array: HashableArrayWrapper = struct.field(pytree_node=False)
    k_paths: int = struct.field(pytree_node=False)
    link_resources: int = struct.field(pytree_node=False)
    k_paths: int = struct.field(pytree_node=False)
    mean_service_holding_time: float = struct.field(pytree_node=False)
    load: float = struct.field(pytree_node=False)
    arrival_rate: float = struct.field(pytree_node=False)
    random_traffic: bool = struct.field(pytree_node=False)
    include_no_op: bool = struct.field(pytree_node=False) # Include a "no op" action
    transformer_obs_type: str = struct.field(pytree_node=False)


@struct.dataclass
class LogEnvState:
    """Dataclass to hold environment state for logging.

    Args:
        env_state (EnvState): Environment state
        lengths (chex.Scalar): Lengths
        returns (chex.Scalar): Returns
        cum_returns (chex.Scalar): Cumulative returns
        accepted_services (chex.Scalar): Accepted services
        accepted_bitrate (chex.Scalar): Accepted bitrate
        total_bitrate (chex.Scalar): Total bitrate requested
        utilisation (chex.Scalar): Network utilisation
        terminal (chex.Scalar): Terminal flag (true termination condition met)
        truncated (chex.Scalar): Truncated flag (max steps reached)
    """

    env_state: EnvState
    lengths: chex.Array
    returns: chex.Array
    cum_returns: chex.Array
    accepted_services: chex.Array
    accepted_bitrate: chex.Array
    total_bitrate: chex.Array
    utilisation: chex.Array
    terminal: chex.Array
    truncated: chex.Array


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

    pass


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

    max_slots: chex.Scalar = struct.field(pytree_node=False)
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
class MultiBandRSAEnvState(RSAEnvState):
    """Dataclass to hold environment state for MultiBandRSA (RBSA)."""

    pass


@struct.dataclass
class MultiBandRSAEnvParams(RSAEnvParams):
    """Dataclass to hold environment parameters for MultiBandRSA (RBSA)."""

    gap_start: chex.Scalar = struct.field(pytree_node=False)
    gap_width: chex.Scalar = struct.field(pytree_node=False)


@struct.dataclass
class GNModelEnvParams(RSAEnvParams):
    """Dataclass to hold environment state for GN model environments."""

    ref_lambda: chex.Scalar = struct.field(pytree_node=False)
    max_spans: chex.Scalar = struct.field(pytree_node=False)
    max_span_length: chex.Scalar = struct.field(pytree_node=False)
    nonlinear_coeff: chex.Scalar = struct.field(pytree_node=False)
    raman_gain_slope: chex.Scalar = struct.field(pytree_node=False)
    attenuation: chex.Scalar = struct.field(pytree_node=False)
    attenuation_bar: chex.Scalar = struct.field(pytree_node=False)
    dispersion_coeff: chex.Scalar = struct.field(pytree_node=False)
    dispersion_slope: chex.Scalar = struct.field(pytree_node=False)
    transceiver_snr: HashableArrayWrapper = struct.field(pytree_node=False)
    amplifier_noise_figure: HashableArrayWrapper = struct.field(pytree_node=False)
    coherent: bool = struct.field(pytree_node=False)
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
    max_power_per_fibre: chex.Scalar = struct.field(pytree_node=False)
    default_launch_power: chex.Scalar = struct.field(pytree_node=False)
    mod_format_correction: bool = struct.field(pytree_node=False)
    monitor_active_lightpaths: bool = struct.field(
        pytree_node=False
    )  # Monitor active lightpaths for throughput calculation
    gap_starts: HashableArrayWrapper = struct.field(pytree_node=False)
    gap_widths: HashableArrayWrapper = struct.field(pytree_node=False)
    uniform_spans: bool = struct.field(pytree_node=False)


@struct.dataclass
class GNModelEnvState(RSAEnvState):
    """Dataclass to hold environment state for RSA with GN model."""

    link_snr_array: chex.Array  # Available SNR on each link
    channel_centre_bw_array: chex.Array  # Channel centre bandwidth for each active connection
    path_index_array: (
        chex.Array
    )  # Contains indices of lightpaths in use on slots (used for lightpath SNR calculation)
    channel_power_array: chex.Array  # Channel power for each active connection
    channel_centre_bw_array_prev: (
        chex.Array
    )  # Channel centre bandwidth for each active connection in previous timestep
    path_index_array_prev: (
        chex.Array
    )  # Contains indices of lightpaths in use on slots in previous timestep
    channel_power_array_prev: (
        chex.Array
    )  # Channel power for each active connection in previous timestep
    launch_power_array: chex.Array  # Launch power array


@struct.dataclass
class RSAGNModelEnvParams(GNModelEnvParams):
    """Dataclass to hold environment params for RSA with GN model."""

    min_snr: chex.Scalar = struct.field(pytree_node=False)
    fec_threshold: chex.Scalar = struct.field(pytree_node=False)


@struct.dataclass
class RSAGNModelEnvState(GNModelEnvState):
    """Dataclass to hold environment state for RSA with GN model."""

    active_lightpaths_array: chex.Array  # Active lightpath array. 1 x M array. Each value is a lightpath index. Used to calculate total throughput.
    active_lightpaths_array_departure: chex.Array  # Active lightpath array departure time.
    throughput: chex.Array  # Current network throughput


@struct.dataclass
class RMSAGNModelEnvParams(GNModelEnvParams):
    """Dataclass to hold environment params for RMSA with GN model.

    Args:
        link_snr_array (chex.Array): Link SNR array
    """

    modulations_array: HashableArrayWrapper = struct.field(pytree_node=False)


@struct.dataclass
class RMSAGNModelEnvState(GNModelEnvState):
    """Dataclass to hold environment state for RMSA with GN model.

    Args:
        link_snr_array (chex.Array): Link SNR array
    """

    modulation_format_index_array: chex.Array  # Modulation format index for each active connection
    modulation_format_index_array_prev: (
        chex.Array
    )  # Modulation format index for each active connection in previous timestep
    mod_format_mask: chex.Array  # Modulation format mask


@struct.dataclass
class RSAMultibandEnvState(RSAEnvState):
    """Dataclass to hold environment state for MultiBandRSA (RBSA)."""

    pass


@struct.dataclass
class RSAMultibandEnvParams(RSAEnvParams):
    """Dataclass to hold environment parameters for MultiBandRSA (RBSA)."""

    gap_starts: HashableArrayWrapper = struct.field(pytree_node=False)
    gap_widths: HashableArrayWrapper = struct.field(pytree_node=False)


@struct.dataclass
class VONEEnvState(RSAEnvState):
    """Dataclass to hold environment state for VONE.

    Args:
        node_capacity_array (chex.Array): Node capacity array
        node_resource_array (chex.Array): Node resource array
        node_departure_array (chex.Array): Node departure array
        action_counter (chex.Array): Action counter
        action_history (chex.Array): Action history
        node_mask_s (chex.Array): Node mask for source node
        node_mask_d (chex.Array): Node mask for destination node
        virtual_topology_patterns (chex.Array): Virtual topology patterns
        values_nodes (chex.Array): Values for nodes
    """

    node_capacity_array: chex.Array
    node_resource_array: chex.Array
    node_departure_array: chex.Array
    action_counter: chex.Array
    action_history: chex.Array
    node_mask_s: chex.Array
    node_mask_d: chex.Array
    virtual_topology_patterns: chex.Array
    values_nodes: chex.Array


@struct.dataclass
class VONEEnvParams(RSAEnvParams):
    """Dataclass to hold environment parameters for VONE.

    Args:
        node_resources (chex.Scalar): Number of node resources
        max_edges (chex.Scalar): Maximum number of edges
        min_node_resources (chex.Scalar): Minimum number of node resources
        max_node_resources (chex.Scalar): Maximum number of node resources
    """

    node_resources: chex.Scalar = struct.field(pytree_node=False)
    max_edges: chex.Scalar = struct.field(pytree_node=False)
    min_node_resources: chex.Scalar = struct.field(pytree_node=False)
    max_node_resources: chex.Scalar = struct.field(pytree_node=False)
    # TODO - Add Laplacian matrix (for node heuristics)


Obsv = Tuple[EnvState, EnvParams] | Tuple[Array]
SelectActionState = Tuple[chex.PRNGKey, LogEnvState, Obsv]


@struct.dataclass
class Transition:
    terminal: Array
    truncated: Array
    action: Array
    reward: Array
    obs: Obsv
    info: Dict[str, Array]


@struct.dataclass
class VONETransition:
    terminal: Array
    truncated: Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: Obsv
    info: Dict[str, Array]
    action_mask_s: chex.Array
    action_mask_p: chex.Array
    action_mask_d: chex.Array


@struct.dataclass
class RSATransition:
    terminal: Array
    truncated: Array
    action: chex.Array
    value: chex.Array
    reward: chex.Array
    log_prob: chex.Array
    obs: Obsv
    info: Dict[str, Array]
    action_mask: chex.Array
