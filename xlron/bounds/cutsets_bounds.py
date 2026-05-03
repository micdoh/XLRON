"""
Example command:
    uv run python -m xlron.bounds.cutsets_bounds --topology_name=nsfnet_deeprmsa_directed --env_type=rmsa --truncate_holding_time --load=250 --link_resources=100 --min_bw=25 --max_bw=100 --step_bw=1 --slot_size=12.5 --continuous_operation --max_requests=100000 --num_trials=10 --CUTSET_EXHAUSTIVE --CUTSET_BATCH_SIZE=512 --CUTSET_ITERATIONS=32 --CUTSET_TOP_K=256
"""

import math
import sys
from functools import partial

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from absl import app, flags

from xlron.environments.dataclasses import HashableArrayWrapper
from xlron.environments.env_funcs import (
    generate_request_rsa,
    make_graph,
    normalise_traffic_matrix,
    read_rsa_request,
    required_slots,
)
from xlron.environments.make_env import make
from xlron.environments.wrappers import Profiler, TimeIt
from xlron.parameter_flags import *  # noqa: F403,F401
from xlron.train.train_utils import build_run_summary, get_user_flags, write_run_summary

FLAGS = flags.FLAGS

# =====================================================================
#  Cut-set discovery helpers (unchanged from original)
# =====================================================================


def get_weighted_traffic_matrix(graph, params):
    n_nodes = len(graph.nodes())

    # path_se_array is flat with one SE per path, grouped in blocks of k.
    # We want the SE of the first (shortest) path for each (s,d) pair.
    se_array = np.asarray(params.path_se_array.val)
    k = params.k_paths

    # Upper-triangular (s < d) pairs in row-major order
    src_upper, dst_upper = np.triu_indices(n_nodes, k=1)
    num_upper_pairs = len(src_upper)

    # For env_type=rsa (no modulation), path_se_array is [1] (uniform SE=1).
    # In that case, all pairs have equal weight.
    if se_array.size < num_upper_pairs * k:
        first_se_per_pair_upper = np.full(num_upper_pairs, se_array.flat[0], dtype=np.float64)
        traffic_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        traffic_matrix[src_upper, dst_upper] = 1.0 / first_se_per_pair_upper
        traffic_matrix[dst_upper, src_upper] = 1.0 / first_se_per_pair_upper
        return jnp.array(traffic_matrix)

    num_pairs = se_array.shape[0] // k
    first_se_per_pair = se_array.reshape(num_pairs, k)[:, 0]  # (num_pairs,)

    traffic_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if params.directed_graph:
        half = num_pairs // 2
        traffic_matrix[src_upper, dst_upper] = 1.0 / first_se_per_pair[:half]
        traffic_matrix[dst_upper, src_upper] = 1.0 / first_se_per_pair[half:]
    else:
        traffic_matrix[src_upper, dst_upper] = 1.0 / first_se_per_pair
        traffic_matrix[dst_upper, src_upper] = 1.0 / first_se_per_pair

    return jnp.array(traffic_matrix)


@partial(jax.jit, static_argnums=(1,))
def make_complete_subgraph(path_adj_matrix, adj_matrix):
    active = (path_adj_matrix.sum(axis=0) + path_adj_matrix.sum(axis=1)) > 0
    result = jnp.outer(active, active) * adj_matrix.val
    return result


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def edges_to_adjacency(path_array, source_nodes, dest_nodes, num_nodes, directed=False):
    adjacency = jnp.zeros((num_nodes, num_nodes))

    def update_adj(idx, adj):
        s, d = source_nodes.val[idx], dest_nodes.val[idx]
        update_val = jnp.array([[1.0]]) * path_array[idx]
        adj = jax.lax.dynamic_update_slice(adj, update_val, (s, d))
        if not directed:
            adj = jax.lax.dynamic_update_slice(adj, update_val, (d, s))
        return adj

    adjacency = jax.lax.fori_loop(0, len(path_array), update_adj, adjacency)
    return adjacency


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def get_cutset_from_path(path_array, adjacency, source_nodes, dest_nodes, directed=False):
    num_nodes = adjacency.val.shape[0]
    path_adjacency = edges_to_adjacency(path_array, source_nodes, dest_nodes, num_nodes, directed)
    subgraph_adjacency = make_complete_subgraph(path_adjacency, adjacency)
    cutset_adjacency = find_cutset_adj(subgraph_adjacency, adjacency)
    remaining_graph = adjacency.val * (1 - cutset_adjacency)

    reachable = jnp.zeros(num_nodes)
    reachable = reachable.at[0].set(1)

    def update_reachable(reachable, _):
        new_reachable = jnp.matmul(reachable, remaining_graph)
        return jnp.where(new_reachable > 0, 1, reachable), None

    partition1, _ = jax.lax.scan(update_reachable, reachable, jnp.arange(num_nodes - 1))
    partition1 = partition1.astype(jnp.int32)
    partition2 = 1 - partition1
    return partition1, partition2


@partial(jax.jit, static_argnums=(2, 3))
def find_cutset_edges(p1, p2, source_nodes, dest_nodes):
    offset = -jnp.min(jnp.concatenate([source_nodes.val, dest_nodes.val]))

    # Edge crosses cut if its endpoints are in different partitions
    def is_cut_edge(edge):
        u, v = edge
        return jnp.logical_or(jnp.logical_and(p1[u], p2[v]), jnp.logical_and(p1[v], p2[u]))

    cut_set = jax.vmap(is_cut_edge)(
        jnp.stack([source_nodes.val + offset, dest_nodes.val + offset], axis=1)
    )
    return cut_set.astype(jnp.int32)


@partial(jax.jit, static_argnums=(1,))
def find_cutset_adj(subgraph_matrix, full_matrix):
    in_subgraph = (subgraph_matrix.sum(axis=0) + subgraph_matrix.sum(axis=1)) > 0
    inside = in_subgraph.reshape(-1, 1)
    outside = ~in_subgraph.reshape(1, -1)
    cutset = jnp.logical_and(
        full_matrix.val,
        jnp.logical_or(jnp.logical_and(inside, outside), jnp.logical_and(outside.T, inside.T)),
    )
    return cutset


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def calculate_congestion(
    partition_mask, adjacency_matrix, traffic_matrix, source_nodes, dest_nodes, directed=False
):
    num_nodes = adjacency_matrix.shape[0]
    partition1 = partition_mask
    partition2 = 1 - partition_mask

    edges = find_cutset_edges(partition1, partition2, source_nodes, dest_nodes)
    cut_matrix = edges_to_adjacency(edges, source_nodes, dest_nodes, num_nodes, directed)
    cut_size = jnp.sum(cut_matrix)

    # For directed graphs, traffic crosses the cut in both directions
    cross_mask = partition1[:, None] * partition2[None, :]
    if directed:
        cross_mask = cross_mask + partition2[:, None] * partition1[None, :]
    traffic_across_cut = jnp.sum(traffic_matrix.val * cross_mask)
    congestion = jnp.where(cut_size > 0, traffic_across_cut / cut_size, 0.0)

    partitioned_adj = adjacency_matrix.val - cut_matrix
    masked1 = jnp.where(jnp.outer(partition1, partition1) > 0, partitioned_adj, 0)
    masked2 = jnp.where(jnp.outer(partition2, partition2) > 0, partitioned_adj, 0)
    check1 = jnp.where(partition1 > 0, jnp.sum(masked1, axis=0) > 0, True)
    check2 = jnp.where(partition2 > 0, jnp.sum(masked2, axis=0) > 0, True)
    # A singleton partition (1 node) is trivially connected but has no
    # internal edges, so the per-node neighbour check above incorrectly
    # rejects it.  Override: any partition with <= 1 node is connected.
    size1 = jnp.sum(partition1)
    size2 = jnp.sum(partition2)
    connected1 = jnp.where(size1 <= 1, True, jnp.all(check1))
    connected2 = jnp.where(size2 <= 1, True, jnp.all(check2))
    both_connected = (connected1 & connected2).astype(jnp.float32)
    return congestion * both_connected


@partial(jax.jit, static_argnums=(0, 1))
def generate_gray_code_masks(n_nodes: int, max_batch_size: int, start: int):
    base_sequence = jnp.arange(max_batch_size) + start
    gray_numbers = base_sequence ^ (base_sequence >> 1)
    powers_of_two = jnp.power(2, jnp.arange(n_nodes - 1, -1, -1))
    numbers = gray_numbers[:, jnp.newaxis]
    masks = (numbers // powers_of_two) % 2
    return masks


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5), donate_argnums=(0,))
def calculate_congestion_batch(
    partition_masks_batch,
    adjacency_matrix,
    traffic_matrix,
    source_nodes,
    dest_nodes,
    directed=False,
):
    return jax.vmap(calculate_congestion, in_axes=(0, None, None, None, None, None))(
        partition_masks_batch, adjacency_matrix, traffic_matrix, source_nodes, dest_nodes, directed
    )


def find_congested_cuts_exhaustive(
    start,
    num_iterations,
    num_batches_per_iteration,
    adj_matrix,
    traf_matrix,
    num_nodes,
    top_k,
    max_batch_size,
    source_nodes,
    dest_nodes,
    directed=False,
):
    def find_congested_cuts_iter(_, i):
        batch_indices = jnp.arange(num_batches_per_iteration) + i

        def batch_eval_sort(_, j):
            j = j * max_batch_size
            new_masks = generate_gray_code_masks(num_nodes, max_batch_size, j)
            new_congestions = calculate_congestion_batch(
                new_masks, adj_matrix, traf_matrix, source_nodes, dest_nodes, directed
            )
            top_indices = jnp.argsort(new_congestions)[-top_k:]
            new_congestions = new_congestions[top_indices]
            new_masks = new_masks[top_indices]
            return None, (new_congestions, new_masks)

        _, (result_congestions, result_masks) = jax.lax.scan(batch_eval_sort, None, batch_indices)

        result_congestions = result_congestions.reshape((-1,))
        sorted_indices = jnp.argsort(result_congestions)[-top_k:]
        congestions_iter = result_congestions[sorted_indices]
        masks_iter = result_masks.reshape((-1, num_nodes))[sorted_indices]
        return None, (congestions_iter, masks_iter)

    _, (congestions, masks) = jax.lax.scan(
        find_congested_cuts_iter,
        None,
        (jnp.arange(num_iterations) + start) * num_batches_per_iteration,
    )

    congestions = congestions.reshape((-1,))
    masks = masks.reshape((-1, num_nodes))
    partition1 = masks
    partition2 = 1 - masks
    return congestions, partition1, partition2


def _has_gpu():
    """Check if any GPU backend is available."""
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def find_congested_cuts_simple(
    path_link_array,
    source_nodes,
    dest_nodes,
    adjacency_matrix,
    traffic_matrix,
    directed=False,
):
    def compute_single(i):
        path = path_link_array.val[i]
        p1, p2 = get_cutset_from_path(path, adjacency_matrix, source_nodes, dest_nodes, directed)
        congestion = calculate_congestion(
            p1, adjacency_matrix, traffic_matrix, source_nodes, dest_nodes, directed
        )
        return congestion, p1, p2

    path_indices = jnp.arange(path_link_array.shape[0])
    num_paths = path_link_array.shape[0]
    num_nodes = adjacency_matrix.val.shape[0]

    # Estimate peak vmap memory: num_paths * num_nodes^2 * 4 bytes (int32)
    _GPU_VMAP_MEMORY_LIMIT = 2 * 1024**3  # 2 GiB
    estimated_bytes = num_paths * num_nodes * num_nodes * 4

    if _has_gpu() and estimated_bytes <= _GPU_VMAP_MEMORY_LIMIT:
        # On GPU, vmap parallelises independent iterations across cores
        congestions, partition1, partition2 = jax.vmap(compute_single)(path_indices)
    elif _has_gpu():
        # On GPU with large tensors, batch the vmap to stay within memory
        batch_size = max(1, _GPU_VMAP_MEMORY_LIMIT // (num_nodes * num_nodes * 4))
        # Pad to make divisible by batch_size
        num_batches = (num_paths + batch_size - 1) // batch_size
        padded_size = num_batches * batch_size
        padded_indices = jnp.zeros(padded_size, dtype=jnp.int32)
        padded_indices = padded_indices.at[:num_paths].set(path_indices)

        def batch_body(_, batch_indices):
            return None, jax.vmap(compute_single)(batch_indices)

        _, (congestions, partition1, partition2) = jax.lax.scan(
            batch_body, None, padded_indices.reshape(num_batches, batch_size)
        )
        congestions = congestions.reshape(-1)[:num_paths]
        partition1 = partition1.reshape(-1, num_nodes)[:num_paths]
        partition2 = partition2.reshape(-1, num_nodes)[:num_paths]
    else:
        # On CPU, scan avoids materialising all intermediate results at once
        def scan_body(_, i):
            return None, compute_single(i)

        _, (congestions, partition1, partition2) = jax.lax.scan(scan_body, None, path_indices)
    return congestions, partition1, partition2


# =====================================================================
#  Capacity Bound Simulation
# =====================================================================


def precompute_cutset_data(heavy_cut_sets, num_links, source_nodes=None, dest_nodes=None):
    """Precompute data structures for the capacity-bound simulation.

    From the heavy_cut_sets dict, build:
      - unique_link_indices: 1-D array of link indices that appear in any cut-set.
      - cutset_link_mask: (num_cutsets, num_unique_links) binary array mapping each
        cut-set to its member links *within the compressed index space*.
      - partition1, partition2: (num_cutsets, num_nodes) binary partition arrays.

    When source_nodes/dest_nodes are provided (directed graphs), also builds
    directional masks:
      - cutset_link_mask_1to2: links going from partition1 to partition2
      - cutset_link_mask_2to1: links going from partition2 to partition1

    Returns a dict with keys:
        unique_link_indices, cutset_link_mask, partition1, partition2, num_cutsets,
        num_unique_links, and optionally cutset_link_mask_1to2, cutset_link_mask_2to1.
    """
    cutset_edges = heavy_cut_sets["cutset_edges"]  # (num_cutsets, num_links)
    partition1 = heavy_cut_sets["partition1"]  # (num_cutsets, num_nodes)
    partition2 = heavy_cut_sets["partition2"]  # (num_cutsets, num_nodes)

    # Union of all links in any cut-set
    any_cutset = jnp.any(cutset_edges > 0, axis=0)  # (num_links,)
    unique_link_indices = jnp.where(any_cutset, size=num_links, fill_value=-1)[0]
    # Count actual unique links
    num_unique_links = int(jnp.sum(any_cutset))
    unique_link_indices = unique_link_indices[:num_unique_links]

    # Build a mapping from global link index -> compressed index
    # We create a lookup table of size num_links, default -1
    link_to_compressed = jnp.full(num_links, -1, dtype=jnp.int32)
    link_to_compressed = link_to_compressed.at[unique_link_indices].set(
        jnp.arange(num_unique_links)
    )

    # Build cutset_link_mask in compressed space: (num_cutsets, num_unique_links)
    # For each cut-set, select which of the unique links belong to it
    # cutset_edges[i, j] == 1 means global link j is in cut-set i
    # We need to compress: for each cut-set, for each unique link, check if it's in the cut-set
    cutset_link_mask = cutset_edges[:, unique_link_indices]  # (num_cutsets, num_unique_links)

    num_cutsets = cutset_edges.shape[0]

    # Compute cut-set sizes: number of links in each cut-set (= initial capacity per slot)
    cutset_sizes = jnp.sum(cutset_edges, axis=1).astype(jnp.int32)  # (num_cutsets,)

    result = {
        "unique_link_indices": unique_link_indices,  # (num_unique_links,)
        "cutset_link_mask": cutset_link_mask.astype(jnp.int32),  # (num_cutsets, num_unique_links)
        "partition1": partition1.astype(jnp.int32),  # (num_cutsets, num_nodes)
        "partition2": partition2.astype(jnp.int32),  # (num_cutsets, num_nodes)
        "num_cutsets": num_cutsets,
        "num_unique_links": num_unique_links,
        "link_to_compressed": link_to_compressed,  # (num_links,)
        "cutset_sizes": cutset_sizes,  # (num_cutsets,) — num links per cut-set
    }

    # Build directional masks for directed graphs
    if source_nodes is not None and dest_nodes is not None:
        src = np.asarray(source_nodes.val)
        dst = np.asarray(dest_nodes.val)
        p1 = np.asarray(partition1)  # (num_cutsets, num_nodes)
        p2 = np.asarray(partition2)

        # For each unique link, get its source and destination node
        uli = np.asarray(unique_link_indices)
        link_src = src[uli]  # (num_unique_links,)
        link_dst = dst[uli]  # (num_unique_links,)

        # For each cutset c and unique link l:
        #   1to2: source in partition1[c] AND dest in partition2[c]
        #   2to1: source in partition2[c] AND dest in partition1[c]
        # p1[:, link_src] -> (num_cutsets, num_unique_links): is link's src in partition1?
        src_in_p1 = p1[:, link_src]  # (C, L)
        dst_in_p2 = p2[:, link_dst]  # (C, L)
        src_in_p2 = p2[:, link_src]  # (C, L)
        dst_in_p1 = p1[:, link_dst]  # (C, L)

        mask_np = np.asarray(cutset_link_mask)
        mask_1to2 = mask_np * (src_in_p1 * dst_in_p2)  # (C, L)
        mask_2to1 = mask_np * (src_in_p2 * dst_in_p1)  # (C, L)

        result["cutset_link_mask_1to2"] = jnp.array(mask_1to2, dtype=jnp.int32)
        result["cutset_link_mask_2to1"] = jnp.array(mask_2to1, dtype=jnp.int32)
        # Directional sizes: number of links per cutset in each direction
        result["cutset_sizes_1to2"] = jnp.array(mask_1to2.sum(axis=1), dtype=jnp.int32)  # (C,)
        result["cutset_sizes_2to1"] = jnp.array(mask_2to1.sum(axis=1), dtype=jnp.int32)  # (C,)

    return result


# -------------------------------------------------------------------
# Core simulation step (designed for JAX tracing / jit)
# Supports RMSA: multi-slot contiguous block assignment with
# modulation-format-dependent spectral efficiency.
# -------------------------------------------------------------------


def build_best_se_matrix(params):
    """Precompute best (highest) spectral efficiency for each (s,d) pair.

    For the capacity upper bound we give routing freedom, so for each (s,d)
    we use the best modulation format achievable on any of the k-shortest paths.
    If consider_modulation_format is False, returns all-ones.

    Returns:
        best_se: (num_nodes, num_nodes) int array of SE values.
    """
    num_nodes = params.num_nodes

    if not params.consider_modulation_format:
        return jnp.ones((num_nodes, num_nodes), dtype=jnp.int32)

    # path_se_array is a flat 1D array with one SE value per path.
    # For undirected: N*(N-1)/2 * k paths; for directed: N*(N-1) * k paths.
    # Reshape to (num_pairs, k) and take max over k to get best SE per pair.
    # IMPORTANT: Only consider paths that actually have links (non-empty).
    # Empty/padded paths can have bogus SE values that would inflate the bound.
    se_array = np.asarray(params.path_se_array.val)
    link_array = np.asarray(params.path_link_array.val)
    k = params.k_paths
    num_pairs = se_array.shape[0] // k
    se_reshaped = se_array.reshape(num_pairs, k)
    # A path is valid if it uses at least one link
    path_has_links = link_array.reshape(num_pairs, k, -1).any(axis=2)  # (num_pairs, k)
    # Mask out empty paths by setting their SE to 0
    se_masked = np.where(path_has_links, se_reshaped, 0)
    se_per_pair = se_masked.max(axis=1)  # (num_pairs,)

    # Build (src, dst) index arrays matching get_path_indices triangular ordering
    # Upper-triangular pairs (s < d) in row-major order
    src_upper, dst_upper = np.triu_indices(num_nodes, k=1)

    best_se = np.ones((num_nodes, num_nodes), dtype=np.int32)
    if params.directed_graph:
        # First half: s<d pairs (forward), second half: s>d pairs (backward)
        half = num_pairs // 2
        best_se[src_upper, dst_upper] = se_per_pair[:half]
        best_se[dst_upper, src_upper] = se_per_pair[half:]
    else:
        # Symmetric: same SE for (s,d) and (d,s)
        best_se[src_upper, dst_upper] = se_per_pair
        best_se[dst_upper, src_upper] = se_per_pair

    return jnp.array(best_se, dtype=jnp.int32)


def _find_traversed_cutsets(s, d, partition1, partition2):
    """Return boolean array (num_cutsets,) indicating which cut-sets are traversed
    by a request from node s to node d."""
    p1_s = partition1[:, s]
    p1_d = partition1[:, d]
    p2_s = partition2[:, s]
    p2_d = partition2[:, d]
    traversed = (p1_s & p2_d) | (p2_s & p1_d)
    return traversed.astype(jnp.bool_)


def _find_feasible_start_slots(traversed, cutset_slot_array, num_slots):
    """Find starting slot positions where a contiguous block of num_slots is feasible
    across all traversed cut-sets.

    Each cut-set is modelled as a virtual resource with integer capacity per slot
    (equal to the number of physical links in the cut-set).  A slot is available
    on a cut-set if its capacity is > 0.  A contiguous block [s, s+num_slots) is
    feasible if ALL slots in the block have capacity > 0 on ALL traversed cut-sets.

    Spectrum continuity: the same slot range is used across all cut-sets.

    Args:
        traversed: (C,) boolean — which cut-sets the request crosses.
        cutset_slot_array: (C, S) int — available capacity per slot per cut-set.
        num_slots: scalar int (may be traced) — number of contiguous slots needed.

    Returns:
        feasible_starts: (S,) boolean — True if starting at that slot is feasible.
    """
    num_total_slots = cutset_slot_array.shape[1]
    has_capacity = (cutset_slot_array > 0).astype(jnp.int32)  # (C, S)

    # For non-traversed cut-sets, treat all slots as available
    has_capacity = jnp.where(traversed[:, None], has_capacity, 1)  # (C, S)

    # All traversed cut-sets must have capacity at every slot in the block.
    # Compute per-slot: minimum across cut-sets (1 = all have capacity, 0 = at least one doesn't)
    all_available = jnp.min(has_capacity, axis=0)  # (S,)

    # Sliding-window: a start position s is feasible if all_available[s:s+num_slots] are all 1.
    # Use cumulative sum for efficient sliding window.
    cumsum = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(all_available)], axis=0
    )  # (S+1,)

    start_indices = jnp.arange(num_total_slots)  # (S,)
    end_indices = jnp.minimum(start_indices + num_slots, num_total_slots)  # (S,)
    block_sum = cumsum[end_indices] - cumsum[start_indices]  # (S,)

    valid_start = (start_indices + num_slots) <= num_total_slots  # (S,)
    feasible_starts = (block_sum >= num_slots) & valid_start  # (S,)
    return feasible_starts


def _expire_services(
    elapsed_time,
    cutset_slot_array,
    service_departure_times,
    service_traversed,
    service_direction,
    service_slot_start,
    service_slot_count,
    num_total_slots,
):
    """Remove expired services and restore capacity to cutset_slot_array.

    Departure times are stored as *relative* remaining time.  On each step
    the caller passes the inter-arrival time (``elapsed_time``).  Services
    whose remaining time <= elapsed_time have expired.  Surviving services
    have their remaining time decremented by elapsed_time.

    Args:
        elapsed_time: scalar — time elapsed since last step (arrival_time).
        cutset_slot_array: (C, 2, S) int — available capacity per direction per slot.
            Dimension 1: 0 = partition1→partition2, 1 = partition2→partition1.
        service_departure_times: (M,) float — remaining time per service (0 = empty).
        service_traversed: (M, C) int — which cut-sets each service traverses.
        service_direction: (M, C) int — direction used per cutset (0=1→2, 1=2→1).
        service_slot_start: (M,) int — start slot index per service.
        service_slot_count: (M,) int — number of slots per service.
        num_total_slots: int — number of slots (S).

    Returns:
        Updated cutset_slot_array, service_departure_times, service_traversed,
        service_direction, service_slot_start, service_slot_count.
    """
    elapsed_time = elapsed_time.squeeze()
    expired = (service_departure_times > 0) & (service_departure_times <= elapsed_time)  # (M,)

    # Build slot masks for expired services: (M, S)
    slot_indices = jnp.arange(num_total_slots)  # (S,)
    slot_mask = (slot_indices[None, :] >= service_slot_start[:, None]) & (
        slot_indices[None, :] < (service_slot_start[:, None] + service_slot_count[:, None])
    )  # (M, S)

    # Capacity to restore per direction:
    # expired[m] * traversed[m,c] * (direction[m,c]==d) * slot_mask[m,s] -> (C, 2, S)
    exp_trav = expired[:, None].astype(jnp.int32) * service_traversed  # (M, C)
    dir_is_0 = (service_direction == 0).astype(jnp.int32)  # (M, C)
    dir_is_1 = (service_direction == 1).astype(jnp.int32)  # (M, C)
    slot_mask_i = slot_mask.astype(jnp.int32)  # (M, S)

    delta_0 = jnp.einsum("mc,ms->cs", exp_trav * dir_is_0, slot_mask_i)  # (C, S)
    delta_1 = jnp.einsum("mc,ms->cs", exp_trav * dir_is_1, slot_mask_i)  # (C, S)
    delta = jnp.stack([delta_0, delta_1], axis=1)  # (C, 2, S)
    cutset_slot_array = cutset_slot_array + delta

    # Clear expired entries and shift surviving (active) departure times by -elapsed_time
    keep = ~expired
    active = service_departure_times > 0  # only shift entries that are actually in use
    keep_f = keep.astype(service_departure_times.dtype)
    active_f = active.astype(service_departure_times.dtype)
    service_departure_times = (service_departure_times - elapsed_time * active_f) * keep_f
    service_traversed = service_traversed * keep[:, None]
    service_direction = service_direction * keep[:, None]
    service_slot_start = service_slot_start * keep
    service_slot_count = service_slot_count * keep

    return (
        cutset_slot_array,
        service_departure_times,
        service_traversed,
        service_direction,
        service_slot_start,
        service_slot_count,
    )


def _simulation_step(
    carry,
    rng_key,
    partition1,
    partition2,
    best_se_matrix,
    params,
    num_total_slots,
    max_services,
):
    """One timestep of the capacity-bound simulation.

    Each cut-set is treated as a virtual resource with integer capacity per
    slot per direction.  cutset_slot_array has shape (C, 2, S) where
    dimension 1 is direction: 0 = partition1→partition2, 1 = partition2→partition1.

    carry = (state, cutset_slot_array,
             service_departure_times, service_traversed, service_direction,
             service_slot_start, service_slot_count,
             accepted_count, blocked_count, always_accepted_count,
             accepted_bitrate, blocked_bitrate, always_accepted_bitrate,
             total_bitrate)
    """
    (
        state,
        cutset_slot_array,
        service_departure_times,
        service_traversed,
        service_direction,
        service_slot_start,
        service_slot_count,
        accepted_count,
        blocked_count,
        always_accepted_count,
        accepted_bitrate,
        blocked_bitrate,
        always_accepted_bitrate,
        total_bitrate,
    ) = carry

    # --- 1. Generate request (advances time, generates source/dest/bw/holding_time) ---
    state = generate_request_rsa(rng_key, state, params)
    arrival_time = state.arrival_time  # inter-arrival time for this step
    holding_time = state.holding_time

    # --- 2. Expire services whose remaining time <= arrival_time, then shift ---
    (
        cutset_slot_array,
        service_departure_times,
        service_traversed,
        service_direction,
        service_slot_start,
        service_slot_count,
    ) = _expire_services(
        arrival_time,
        cutset_slot_array,
        service_departure_times,
        service_traversed,
        service_direction,
        service_slot_start,
        service_slot_count,
        num_total_slots,
    )

    # --- 3. Read request, compute SE and required slots ---
    nodes_sd, requested_datarate = read_rsa_request(state.request_array)
    source = nodes_sd[0].astype(jnp.int32)
    dest = nodes_sd[1].astype(jnp.int32)
    se = best_se_matrix[source, dest].astype(jnp.float32)
    num_slots = required_slots(
        requested_datarate,
        se,
        params.slot_size,
        guardband=params.guardband,
        temperature=params.temperature,
        differentiable=params.differentiable,
    )

    # --- 4. Find traversed cut-sets and direction ---
    traversed = _find_traversed_cutsets(source, dest, partition1, partition2)
    any_traversed = jnp.any(traversed)

    # Direction per cutset: 0 if source∈p1 and dest∈p2, else 1
    s_in_p1 = partition1[:, source]  # (C,)
    d_in_p2 = partition2[:, dest]  # (C,)
    goes_1to2 = s_in_p1 & d_in_p2  # (C,) — True means direction 0 (1→2)
    direction = 1 - goes_1to2.astype(jnp.int32)  # (C,) — 0=1→2, 1=2→1

    # --- 5. Build effective capacity view for this request's direction ---
    # cutset_slot_array is (C, 2, S). Select direction per cutset: effective_csa[c, s] = csa[c, dir[c], s]
    effective_csa = jnp.where(
        goes_1to2[:, None],
        cutset_slot_array[:, 0, :],  # direction 0: p1→p2
        cutset_slot_array[:, 1, :],  # direction 1: p2→p1
    )  # (C, S)

    # --- 6. Find feasible start slots ---
    feasible_starts = _find_feasible_start_slots(traversed, effective_csa, num_slots)
    start_slot = jnp.argmax(feasible_starts).astype(jnp.int32)
    has_feasible = feasible_starts[start_slot]
    accepted = any_traversed & has_feasible

    # --- 7. Allocate: decrement cutset_slot_array at the correct direction ---
    slot_indices = jnp.arange(num_total_slots)
    alloc_mask = ((slot_indices >= start_slot) & (slot_indices < (start_slot + num_slots))).astype(
        jnp.int32
    )  # (S,)
    trav_mask = accepted * traversed.astype(jnp.int32)  # (C,)
    # Build (C, 2, S) decrement: only at the correct direction
    dec_0 = (
        trav_mask[:, None] * goes_1to2.astype(jnp.int32)[:, None] * alloc_mask[None, :]
    )  # (C, S)
    dec_1 = (
        trav_mask[:, None] * (1 - goes_1to2.astype(jnp.int32))[:, None] * alloc_mask[None, :]
    )  # (C, S)
    decrement = jnp.stack([dec_0, dec_1], axis=1)  # (C, 2, S)
    cutset_slot_array = cutset_slot_array - decrement

    # --- 8. Record service in table (for later departure) ---
    empty_mask = service_departure_times == 0  # (M,)
    service_idx = jnp.argmax(empty_mask).astype(jnp.int32)
    departure_time = holding_time.astype(service_departure_times.dtype).squeeze()

    service_departure_times = jnp.where(
        accepted,
        service_departure_times.at[service_idx].set(departure_time),
        service_departure_times,
    )
    service_traversed = jnp.where(
        accepted,
        service_traversed.at[service_idx].set(traversed.astype(jnp.int32)),
        service_traversed,
    )
    service_direction = jnp.where(
        accepted,
        service_direction.at[service_idx].set(direction),
        service_direction,
    )
    service_slot_start = jnp.where(
        accepted,
        service_slot_start.at[service_idx].set(start_slot),
        service_slot_start,
    )
    service_slot_count = jnp.where(
        accepted,
        service_slot_count.at[service_idx].set(num_slots.astype(jnp.int32)),
        service_slot_count,
    )

    # --- 9. Update counters ---
    is_blocked = any_traversed & (~has_feasible)
    is_always_accepted = ~any_traversed

    accepted_count = (
        accepted_count + accepted.astype(jnp.int32) + is_always_accepted.astype(jnp.int32)
    )
    blocked_count = blocked_count + is_blocked.astype(jnp.int32)
    always_accepted_count = always_accepted_count + is_always_accepted.astype(jnp.int32)
    total_bitrate = total_bitrate + requested_datarate
    accepted_bitrate = accepted_bitrate + jnp.where(
        accepted | is_always_accepted, requested_datarate, 0.0
    )
    blocked_bitrate = blocked_bitrate + jnp.where(is_blocked, requested_datarate, 0.0)
    always_accepted_bitrate = always_accepted_bitrate + jnp.where(
        is_always_accepted, requested_datarate, 0.0
    )

    new_carry = (
        state,
        cutset_slot_array,
        service_departure_times,
        service_traversed,
        service_direction,
        service_slot_start,
        service_slot_count,
        accepted_count,
        blocked_count,
        always_accepted_count,
        accepted_bitrate,
        blocked_bitrate,
        always_accepted_bitrate,
        total_bitrate,
    )
    return new_carry, None


def run_single_trial(
    rng,
    initial_state,
    params,
    partition1,
    partition2,
    cutset_sizes_1to2,
    cutset_sizes_2to1,
    best_se_matrix,
    num_requests,
    max_services,
):
    """Run a single trial of the capacity-bound simulation.

    Each cut-set is treated as a virtual resource with integer capacity per
    slot per direction.  No physical link routing is performed.

    Args:
        rng: PRNGKey for this trial.
        initial_state: RSAEnvState (used only for traffic generation timing).
        params: RSAEnvParams.
        partition1, partition2: (C, N) int — node partition arrays.
        cutset_sizes_1to2: (C,) int — links per cut-set in p1→p2 direction.
        cutset_sizes_2to1: (C,) int — links per cut-set in p2→p1 direction.
        best_se_matrix: (N, N) int — best SE per (s,d) pair.
        num_requests: int — number of requests to simulate.
        max_services: int — size of the service tracking table.

    Returns:
        (accepted_count, blocked_count, always_accepted_count,
         accepted_bitrate, blocked_bitrate, always_accepted_bitrate, total_bitrate)
    """
    num_cutsets = partition1.shape[0]
    num_total_slots = params.link_resources

    # Initialize cutset_slot_array: (C, 2, S)
    # Dimension 1: 0 = p1→p2 capacity, 1 = p2→p1 capacity
    csa_0 = jnp.broadcast_to(cutset_sizes_1to2[:, None], (num_cutsets, num_total_slots))
    csa_1 = jnp.broadcast_to(cutset_sizes_2to1[:, None], (num_cutsets, num_total_slots))
    cutset_slot_array = jnp.stack([csa_0, csa_1], axis=1).copy()  # (C, 2, S)

    # Initialize service tracking table (all zeros = empty)
    service_departure_times = jnp.zeros(max_services, dtype=jnp.float32)
    service_traversed = jnp.zeros((max_services, num_cutsets), dtype=jnp.int32)
    service_direction = jnp.zeros((max_services, num_cutsets), dtype=jnp.int32)
    service_slot_start = jnp.zeros(max_services, dtype=jnp.int32)
    service_slot_count = jnp.zeros(max_services, dtype=jnp.int32)

    init_carry = (
        initial_state,
        cutset_slot_array,
        service_departure_times,
        service_traversed,
        service_direction,
        service_slot_start,
        service_slot_count,
        jnp.int32(0),  # accepted_count
        jnp.int32(0),  # blocked_count
        jnp.int32(0),  # always_accepted_count
        jnp.float32(0.0),  # accepted_bitrate
        jnp.float32(0.0),  # blocked_bitrate
        jnp.float32(0.0),  # always_accepted_bitrate
        jnp.float32(0.0),  # total_bitrate
        jnp.int32(0),  # step_idx
    )

    def step_fn(carry, _):
        step_idx = carry[-1]
        rng_key = jax.random.fold_in(rng, step_idx)
        sim_carry = carry[:-1]
        new_sim_carry, _ = _simulation_step(
            sim_carry,
            rng_key,
            partition1,
            partition2,
            best_se_matrix,
            params,
            num_total_slots,
            max_services,
        )
        return new_sim_carry + (step_idx + 1,), None

    final_carry, _ = jax.lax.scan(step_fn, init_carry, None, length=num_requests)

    (
        _final_state,
        _cutset_slot_array,
        _service_departure_times,
        _service_traversed,
        _service_direction,
        _service_slot_start,
        _service_slot_count,
        accepted_count,
        blocked_count,
        always_accepted_count,
        accepted_bitrate,
        blocked_bitrate,
        always_accepted_bitrate,
        total_bitrate,
        _step_idx,
    ) = final_carry

    return (
        accepted_count,
        blocked_count,
        always_accepted_count,
        accepted_bitrate,
        blocked_bitrate,
        always_accepted_bitrate,
        total_bitrate,
    )


def run_capacity_bound_simulation(
    heavy_cut_sets,
    env,
    params,
    best_se_matrix,
    loads,
    num_requests,
    num_trials,
    seed=0,
    max_services=2000,
    source_nodes=None,
    dest_nodes=None,
):
    """Run the full capacity-bound simulation across multiple loads and trials.

    Each cut-set is treated as a virtual resource with integer capacity per
    slot per direction.  No explicit link routing is performed.

    Args:
        heavy_cut_sets: dict with keys congestion, partition1, partition2, cutset_edges.
        env: RSAEnv instance (raw, unwrapped).
        params: RSAEnvParams (base params; arrival_rate overridden per load).
        best_se_matrix: (num_nodes, num_nodes) int — best SE per (s,d).
        loads: 1-D array of traffic loads (Erlangs) to sweep.
        num_requests: int, requests per trial.
        num_trials: int, trials per load.
        seed: int, base random seed.
        max_services: int — size of service tracking table for departures.
        source_nodes: HashableArrayWrapper — source node per link (for directional capacity).
        dest_nodes: HashableArrayWrapper — dest node per link (for directional capacity).

    Returns:
        results: dict mapping load -> {accepted, blocked, always_accepted, blocking_prob}
    """
    num_links = params.num_links
    mean_service_holding_time = params.mean_service_holding_time
    values_bw = jnp.array(params.values_bw.val, dtype=jnp.float32)

    # Precompute cut-set data (with directional sizes for directed graphs)
    cs_data = precompute_cutset_data(heavy_cut_sets, num_links, source_nodes, dest_nodes)
    partition1 = cs_data["partition1"]
    partition2 = cs_data["partition2"]
    cutset_sizes = cs_data["cutset_sizes"]
    num_cutsets = cs_data["num_cutsets"]

    # For directed graphs, use directional sizes; for undirected, each direction = total
    if "cutset_sizes_1to2" in cs_data:
        cutset_sizes_1to2 = cs_data["cutset_sizes_1to2"]
        cutset_sizes_2to1 = cs_data["cutset_sizes_2to1"]
    else:
        cutset_sizes_1to2 = cutset_sizes
        cutset_sizes_2to1 = cutset_sizes

    print("Capacity bound simulation setup (virtual cutset resources):")
    print(f"  Num cut-sets: {num_cutsets}")
    print(f"  Cut-set sizes (total): {np.asarray(cutset_sizes).tolist()}")
    print(f"  Cut-set sizes (1→2):   {np.asarray(cutset_sizes_1to2).tolist()}")
    print(f"  Cut-set sizes (2→1):   {np.asarray(cutset_sizes_2to1).tolist()}")
    print(f"  Num slots per cut-set: {params.link_resources}")
    print(f"  Slot size: {params.slot_size} GHz, Guardband: {params.guardband} slots")
    print(f"  Bandwidth values: {values_bw}")
    print(f"  Max slots per request: {params.max_slots}")
    print(f"  Consider modulation format: {params.consider_modulation_format}")
    print(f"  Max concurrent services: {max_services}")
    print(f"  Requests per trial: {num_requests}")
    print(f"  Num trials: {num_trials}")
    print(f"  Loads to sweep: {loads}")

    profiler = Profiler()

    # Get fresh initial state once (used as template for all loads)
    reset_key = jax.random.PRNGKey(seed)
    _, base_initial_state = env.reset(reset_key, params)
    trial_rngs = jax.random.split(jax.random.PRNGKey(seed + 1), num_trials)

    # Compile once with initial_state as a traced argument (not closed over)
    def single(rng, initial_state):
        return run_single_trial(
            rng,
            initial_state,
            params,
            partition1,
            partition2,
            cutset_sizes_1to2,
            cutset_sizes_2to1,
            best_se_matrix,
            num_requests,
            max_services,
        )

    print("Lowering and compiling simulation (this may take a while)...")
    with profiler.section("COMPILATION"):
        jitted_single = (
            jax.jit(jax.vmap(single, in_axes=(0, None)))
            .lower(trial_rngs, base_initial_state)
            .compile()
        )
    print("  Compilation done.")

    results = {}
    for load_idx, load_val in enumerate(loads):
        arrival_rate = float(load_val / mean_service_holding_time)

        # Update traced fields in state (no recompilation)
        initial_state = base_initial_state.replace(
            arrival_rate=jnp.array(arrival_rate, dtype=jnp.float32),
            mean_service_holding_time=jnp.array(mean_service_holding_time, dtype=jnp.float32),
        )

        print(f"\nLoad = {load_val:.1f} Erlang (arrival_rate = {arrival_rate:.4f})...")

        # Execution (reuses compiled function, no recompilation)
        with profiler.section(
            f"EXECUTION (load={load_val:.0f})",
            frames=num_requests * num_trials,
        ):
            (
                accepted,
                blocked,
                always_accepted,
                accepted_br,
                blocked_br,
                always_accepted_br,
                total_br,
            ) = jitted_single(trial_rngs, initial_state)
            # Block until results are ready for accurate timing
            jax.block_until_ready((accepted, blocked))

        accepted = np.asarray(accepted)
        blocked = np.asarray(blocked)
        always_accepted = np.asarray(always_accepted)
        accepted_br = np.asarray(accepted_br)
        blocked_br = np.asarray(blocked_br)
        always_accepted_br = np.asarray(always_accepted_br)
        total_br = np.asarray(total_br)

        total = accepted + blocked
        blocking_prob = blocked / np.maximum(total, 1)
        bitrate_blocking_prob = blocked_br / np.maximum(total_br, 1.0)

        mean_bp = float(np.mean(blocking_prob))
        mean_bbp = float(np.mean(bitrate_blocking_prob))
        print(
            f"  Blocking prob: {mean_bp:.6f}  Bitrate blocking prob: {mean_bbp:.6f} "
            f"(accepted={np.mean(accepted):.0f}, blocked={np.mean(blocked):.0f}, "
            f"always_accepted={np.mean(always_accepted):.0f})"
        )

        results[float(load_val)] = {
            "accepted": accepted,
            "blocked": blocked,
            "always_accepted": always_accepted,
            "blocking_prob": blocking_prob,
            "accepted_bitrate": accepted_br,
            "blocked_bitrate": blocked_br,
            "always_accepted_bitrate": always_accepted_br,
            "total_bitrate": total_br,
            "bitrate_blocking_prob": bitrate_blocking_prob,
        }

    profiler.summary()
    return results, profiler


def print_results_table(results):
    """Print a summary table of simulation results."""
    print("\n" + "=" * 100)
    print(
        f"{'Load':>10s}  {'Block Prob (mean)':>18s}  {'Block Prob (std)':>18s}  "
        f"{'BR Block Prob':>14s}  {'Accepted':>10s}  {'Blocked':>10s}  {'Always Acc':>10s}"
    )
    print("-" * 100)
    for load in sorted(results.keys()):
        r = results[load]
        bp = r["blocking_prob"]
        bbp = r["bitrate_blocking_prob"]
        print(
            f"{load:10.1f}  {float(np.mean(bp)):18.8f}  {float(np.std(bp)):18.8f}  "
            f"{float(np.mean(bbp)):14.8f}  {float(np.mean(r['accepted'])):10.0f}  "
            f"{float(np.mean(r['blocked'])):10.0f}  {float(np.mean(r['always_accepted'])):10.0f}"
        )
    print("=" * 100)


# =====================================================================
#  Main entry point
# =====================================================================


def main(argv):
    user_flags = get_user_flags(FLAGS)
    print(f"Available devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")

    if FLAGS.DISABLE_JIT:
        jax.config.update("jax_disable_jit", True)
        jax.numpy.set_printoptions(threshold=sys.maxsize, linewidth=220)

    # --- Build environment params ---
    graph = make_graph(FLAGS.topology_name, FLAGS.topology_directory)
    env, params = make(FLAGS)

    # Map node IDs to 0-based positional indices to match nx.adjacency_matrix
    node_list = sorted(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edges = sorted(graph.edges())
    source_nodes_np = jnp.array([node_to_idx[edge[0]] for edge in edges])
    destination_nodes_np = jnp.array([node_to_idx[edge[1]] for edge in edges])
    adj_matrix = nx.adjacency_matrix(graph, weight="").todense()

    print("Building weighted traffic matrix...")
    traffic_matrix_weighted = get_weighted_traffic_matrix(graph, params)
    print("  Done.")
    traffic_matrix_weighted = normalise_traffic_matrix(traffic_matrix_weighted)
    traffic_matrix_haw = HashableArrayWrapper(traffic_matrix_weighted)
    adj_matrix_haw = HashableArrayWrapper(jnp.array(adj_matrix))
    source_nodes_haw = HashableArrayWrapper(source_nodes_np)
    destination_nodes_haw = HashableArrayWrapper(destination_nodes_np)
    top_k = FLAGS.CUTSET_TOP_K

    # --- Find congested cut-sets ---
    use_exhaustive = FLAGS.CUTSET_EXHAUSTIVE
    if use_exhaustive and 2**params.num_nodes > jnp.iinfo(jnp.int32).max:
        print(
            f"WARNING: Topology has {params.num_nodes} nodes (2^{params.num_nodes} combinations), "
            f"which exceeds int32 max. Falling back to non-exhaustive (shortest-paths) cut-set method."
        )
        use_exhaustive = False
    if use_exhaustive:
        total_combinations = 2**params.num_nodes
        parallel_processes = FLAGS.CUTSET_PARALLEL_PROCESSES
        batch_size = min(FLAGS.CUTSET_BATCH_SIZE, total_combinations)
        batches_per_process = math.ceil(total_combinations / parallel_processes / batch_size)
        iterations_per_process = min(FLAGS.CUTSET_ITERATIONS, batches_per_process)
        batches_per_iteration = math.ceil(
            total_combinations / parallel_processes / iterations_per_process / batch_size
        )
        print(f"Top k: {top_k}")
        print(f"Num Nodes: {params.num_nodes}")
        print(f"Parallel processes: {parallel_processes}")
        print(f"Total cut-set combinations: {total_combinations}")
        print(f"Max batch size: {batch_size}")
        print(f"Total batches: {jnp.ceil(total_combinations / batch_size).astype(jnp.int32)}")
        print(f"Batches per process: {batches_per_process}")
        print(f"Batches per iteration: {batches_per_iteration}")
        print(f"Iterations per process: {iterations_per_process}")
        starts = jnp.arange(parallel_processes) * iterations_per_process * batch_size
        if FLAGS.VISIBLE_DEVICES:
            starts = jax.device_put(starts, jax.devices()[int(FLAGS.VISIBLE_DEVICES)])
        if FLAGS.DISABLE_JIT:
            heavy_cut_sets_raw = find_congested_cuts_exhaustive(
                starts,
                iterations_per_process,
                batches_per_iteration,
                adj_matrix_haw,
                traffic_matrix_haw,
                params.num_nodes,
                top_k,
                batch_size,
                source_nodes_haw,
                destination_nodes_haw,
                params.directed_graph,
            )
        else:
            with TimeIt("CUT-SET COMPILATION:"):
                func = (
                    jax.jit(
                        jax.vmap(
                            find_congested_cuts_exhaustive,
                            in_axes=(0, None, None, None, None, None, None, None, None, None, None),
                        ),
                        static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                    )
                    .lower(
                        starts,
                        iterations_per_process,
                        batches_per_iteration,
                        adj_matrix_haw,
                        traffic_matrix_haw,
                        params.num_nodes,
                        top_k,
                        batch_size,
                        source_nodes_haw,
                        destination_nodes_haw,
                        params.directed_graph,
                    )
                    .compile()
                )
            with TimeIt("CUT-SET EXECUTION:", frames=total_combinations):
                heavy_cut_sets_raw = func(starts)
                heavy_cut_sets_raw[0].block_until_ready()
    else:
        if FLAGS.DISABLE_JIT:
            heavy_cut_sets_raw = find_congested_cuts_simple(
                params.path_link_array,
                source_nodes_haw,
                destination_nodes_haw,
                adj_matrix_haw,
                traffic_matrix_haw,
                params.directed_graph,
            )
        else:
            with TimeIt("CUT-SET COMPILATION:"):
                func = (
                    jax.jit(find_congested_cuts_simple, static_argnums=(0, 1, 2, 3, 4, 5))
                    .lower(
                        params.path_link_array,
                        source_nodes_haw,
                        destination_nodes_haw,
                        adj_matrix_haw,
                        traffic_matrix_haw,
                        params.directed_graph,
                    )
                    .compile()
                )
            with TimeIt("CUT-SET EXECUTION:"):
                heavy_cut_sets_raw = func()
                heavy_cut_sets_raw[0].block_until_ready()

    # Post-process: reshape, deduplicate, select top-k
    print("Post-processing cut-sets...")
    print("  Reshaping...")
    congestions = jnp.reshape(heavy_cut_sets_raw[0], (-1,))
    partition1 = jnp.reshape(heavy_cut_sets_raw[1], (-1, params.num_nodes))
    partition2 = jnp.reshape(heavy_cut_sets_raw[2], (-1, params.num_nodes))
    print("  Finding cutset edges...")
    cutset_edges = jax.vmap(find_cutset_edges, in_axes=(0, 0, None, None))(
        partition1, partition2, source_nodes_haw, destination_nodes_haw
    )

    # Deduplicate and filter on CPU (avoids slow jnp.unique on GPU)
    print("  Deduplicating...")
    cutset_edges_np = np.asarray(cutset_edges)
    congestions_np = np.asarray(congestions)
    _, unique_indices = np.unique(cutset_edges_np, axis=0, return_index=True)
    nonzero_mask = congestions_np[unique_indices] > 0
    unique_indices = unique_indices[nonzero_mask]
    cutset_edges = jnp.array(cutset_edges_np[unique_indices])
    congestions = jnp.array(congestions_np[unique_indices])
    partition1 = partition1[unique_indices]
    partition2 = partition2[unique_indices]
    print(f"\nUnique cutsets with congestion > 0: {len(congestions)}")

    # Select top-k (or top-pct if specified)
    if FLAGS.CUTSET_TOP_PCT > 0:
        top_k = max(1, round(len(congestions) * FLAGS.CUTSET_TOP_PCT / 100))
        print(f"  Selecting top {FLAGS.CUTSET_TOP_PCT}% = {top_k} of {len(congestions)} cutsets...")
    else:
        print(f"  Selecting top-{top_k}...")
    top_k_indices = jnp.argsort(congestions)[-top_k:]
    cutset_edges = cutset_edges[top_k_indices]
    congestions = congestions[top_k_indices]
    partition1 = partition1[top_k_indices]
    partition2 = partition2[top_k_indices]
    print(f"Cutsets after top-{top_k} selection: {len(congestions)}")

    heavy_cut_sets = {
        "congestion": congestions,
        "partition1": partition1,
        "partition2": partition2,
        "cutset_edges": cutset_edges,
    }

    print(f"\nSelected {len(congestions)} cut-sets:")
    if not _has_gpu():
        for i, (cong, ce) in enumerate(
            zip(heavy_cut_sets["congestion"], heavy_cut_sets["cutset_edges"])
        ):
            links = [int(j) for j, v in enumerate(ce) if v > 0]
            print(f"  Cut-set {i}: congestion={float(cong):.4f}, links={links}")

    # --- Build best SE matrix for RMSA ---
    print("Building best spectral efficiency matrix...")
    best_se_matrix = build_best_se_matrix(params)
    print("  Done building SE matrix.")
    if params.consider_modulation_format:
        print(
            f"  SE range: {int(jnp.min(best_se_matrix[best_se_matrix > 0]))} - {int(jnp.max(best_se_matrix))}"
        )
    else:
        print("  Modulation format not considered (SE=1 everywhere)")

    # --- Get raw RSAEnv (unwrap LogWrapper) ---
    raw_env = env._env if hasattr(env, "_env") else env

    # --- Run capacity bound simulation ---
    if FLAGS.min_load is not None and FLAGS.max_load is not None and FLAGS.step_load is not None:
        loads = np.arange(FLAGS.min_load, FLAGS.max_load + FLAGS.step_load / 2, FLAGS.step_load)
    else:
        loads = np.array([FLAGS.load])

    results, sim_profiler = run_capacity_bound_simulation(
        heavy_cut_sets=heavy_cut_sets,
        env=raw_env,
        params=params,
        best_se_matrix=best_se_matrix,
        loads=loads,
        num_requests=int(FLAGS.max_requests),
        num_trials=FLAGS.num_trials,
        seed=FLAGS.SEED,
        source_nodes=source_nodes_haw if params.directed_graph else None,
        dest_nodes=destination_nodes_haw if params.directed_graph else None,
    )

    print_results_table(results)

    # Build and write standardized run summary for each load
    for load in sorted(results.keys()):
        r = results[load]
        bp = r["blocking_prob"]
        bbp = r["bitrate_blocking_prob"]

        metrics_dict = {
            "service_blocking_probability": {
                "mean": float(np.mean(bp)),
                "std": float(np.std(bp)),
                "iqr_lower": float(np.nanpercentile(bp, 25)),
                "iqr_upper": float(np.nanpercentile(bp, 75)),
            },
            "bitrate_blocking_probability": {
                "mean": float(np.mean(bbp)),
                "std": float(np.std(bbp)),
                "iqr_lower": float(np.nanpercentile(bbp, 25)),
                "iqr_upper": float(np.nanpercentile(bbp, 75)),
            },
            "accepted_count": {
                "mean": float(np.mean(r["accepted"])),
                "std": float(np.std(r["accepted"])),
                "iqr_lower": float(np.nanpercentile(r["accepted"], 25)),
                "iqr_upper": float(np.nanpercentile(r["accepted"], 75)),
            },
            "blocked_count": {
                "mean": float(np.mean(r["blocked"])),
                "std": float(np.std(r["blocked"])),
                "iqr_lower": float(np.nanpercentile(r["blocked"], 25)),
                "iqr_upper": float(np.nanpercentile(r["blocked"], 75)),
            },
            "always_accepted_count": {
                "mean": float(np.mean(r["always_accepted"])),
                "std": float(np.std(r["always_accepted"])),
                "iqr_lower": float(np.nanpercentile(r["always_accepted"], 25)),
                "iqr_upper": float(np.nanpercentile(r["always_accepted"], 75)),
            },
        }

        # Extract timing from profiler for this load
        timing = {}
        comp_key = f"COMPILATION (load={load:.0f})"
        exec_key = f"EXECUTION (load={load:.0f})"
        if comp_key in sim_profiler._records:
            timing["compilation_time_s"] = sum(e for e, _ in sim_profiler._records[comp_key])
        if exec_key in sim_profiler._records:
            exec_entries = sim_profiler._records[exec_key]
            timing["execution_time_s"] = sum(e for e, _ in exec_entries)
            total_frames = sum(f for _, f in exec_entries if f is not None)
            if total_frames and timing["execution_time_s"] > 0:
                timing["fps"] = total_frames / timing["execution_time_s"]

        # Override load in config for this specific run
        run_config = dict(user_flags)
        run_config["load"] = load

        summary = build_run_summary("cutset_bound", run_config, metrics_dict, timing=timing or None)
        write_run_summary(summary, FLAGS.DATA_OUTPUT_FILE, print_to_console=True)


if __name__ == "__main__":
    app.run(main)
