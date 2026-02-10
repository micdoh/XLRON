"""
Example command:
    uv run python -m experimental.capacity_bound_estimation.cutsets_bounds --topology_name=nsfnet_deeprmsa_directed --env_type=rmsa --truncate_holding_time --load=250 --link_resources=100 --k=5 --min_bw=25 --max_bw=100 --step_bw=1 --slot_size=12.5 --continuous_operation --num_sim_requests=100000 --num_trials=10 --sim_min_load=200 --sim_max_load=300 --sim_step_load=10 --CUTSET_EXHAUSTIVE --CUTSET_BATCH_SIZE=512 --CUTSET_ITERATIONS=32 --CUTSET_TOP_K=256 --link_selection_mode=least_congested
"""

import itertools
import math
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from absl import app, flags

import xlron.parameter_flags
from xlron.environments.dataclasses import ActionInfo, HashableArrayWrapper
from xlron.environments.env_funcs import (
    check_action_rsa,
    complete_step_rsa,
    generate_request_rsa,
    get_affected_slots_mask,
    get_paths,
    get_paths_se,
    implement_action_rsa,
    make_graph,
    normalise_traffic_matrix,
    read_rsa_request,
    required_slots,
)
from xlron.environments.make_env import make
from xlron.environments.wrappers import Profiler, TimeIt

FLAGS = flags.FLAGS

# ---------------------------------------------------------------------------
# Flags specific to capacity-bound simulation
# (registered at module level so they're available via FLAGS after flag parsing)
# ---------------------------------------------------------------------------
_CUTSET_SIM_FLAGS_REGISTERED = False
if not _CUTSET_SIM_FLAGS_REGISTERED:
    for name, defn in [
        ("num_sim_requests", (flags.DEFINE_integer, 100000, "Number of requests per trial")),
        (
            "num_trials",
            (flags.DEFINE_integer, 10, "Number of random-seed trials per traffic intensity"),
        ),
        ("sim_min_load", (flags.DEFINE_float, 10.0, "Minimum traffic load (Erlangs) for sweep")),
        ("sim_max_load", (flags.DEFINE_float, 200.0, "Maximum traffic load (Erlangs) for sweep")),
        ("sim_step_load", (flags.DEFINE_float, 10.0, "Step size for traffic load sweep")),
        (
            "max_concurrent_requests",
            (flags.DEFINE_integer, 5000, "Max concurrent connections for departure tracking"),
        ),
        (
            "USE_MEAN_CONGESTION_THRESHOLD",
            (
                flags.DEFINE_bool,
                False,
                "Filter cutsets to keep only those with congestion above the mean",
            ),
        ),
        (
            "link_selection_mode",
            (
                flags.DEFINE_string,
                "least_congested",
                "Secondary link selection heuristic: least_congested, most_congested, best_fit, random",
            ),
        ),
    ]:
        if not hasattr(FLAGS, name):
            defn[0](name, defn[1], defn[2])
    _CUTSET_SIM_FLAGS_REGISTERED = True

# =====================================================================
#  Cut-set discovery helpers (unchanged from original)
# =====================================================================


def get_weighted_traffic_matrix(graph, params):
    n_nodes = len(graph.nodes())
    traffic_matrix = jnp.zeros((n_nodes, n_nodes))
    for s in range(n_nodes):
        for d in range(n_nodes):
            if s != d:
                nodes = jnp.array([s, d])
                se = get_paths_se(params, nodes)
                traffic_matrix = jax.lax.dynamic_update_slice(
                    traffic_matrix, jnp.array(1 / se[0]).reshape((1, 1)), (s, d)
                )
            else:
                traffic_matrix = jax.lax.dynamic_update_slice(
                    traffic_matrix, jnp.array(0.0).reshape((1, 1)), (s, d)
                )
    return traffic_matrix


@partial(jax.jit, static_argnums=(1,))
def make_complete_subgraph(path_adj_matrix, adj_matrix):
    active = (path_adj_matrix.sum(axis=0) + path_adj_matrix.sum(axis=1)) > 0
    result = jnp.outer(active, active) * adj_matrix.val
    return result


@partial(jax.jit, static_argnums=(1, 2, 3))
def edges_to_adjacency(path_array, source_nodes, dest_nodes, num_nodes):
    adjacency = jnp.zeros((num_nodes, num_nodes))

    def update_adj(idx, adj):
        s, d = source_nodes.val[idx], dest_nodes.val[idx]
        update_val = jnp.array([[1.0]]) * path_array[idx]
        adj = jax.lax.dynamic_update_slice(adj, update_val, (s, d))
        adj = jax.lax.dynamic_update_slice(adj, update_val, (d, s))
        return adj

    adjacency = jax.lax.fori_loop(0, len(path_array), update_adj, adjacency)
    return adjacency


@partial(jax.jit, static_argnums=(1, 2, 3))
def get_cutset_from_path(path_array, adjacency, source_nodes, dest_nodes):
    num_nodes = adjacency.val.shape[0]
    path_adjacency = edges_to_adjacency(path_array, source_nodes, dest_nodes, num_nodes)
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
    def is_cut_edge(edge):
        u, v = edge
        return jnp.logical_or(jnp.logical_and(p1[u], p2[v]), jnp.logical_and(p1[v], p2[u]))

    offset = -jnp.min(jnp.concatenate([source_nodes.val, dest_nodes.val]))
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


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def calculate_congestion(
    partition_mask, adjacency_matrix, traffic_matrix, source_nodes, dest_nodes
):
    num_nodes = adjacency_matrix.shape[0]
    partition1 = partition_mask
    partition2 = 1 - partition_mask

    edges = find_cutset_edges(partition1, partition2, source_nodes, dest_nodes)
    cut_matrix = edges_to_adjacency(edges, source_nodes, dest_nodes, num_nodes)
    cut_size = jnp.sum(cut_matrix)

    traffic_across_cut = jnp.sum(traffic_matrix.val * (partition1[:, None] * partition2[None, :]))
    congestion = jnp.where(cut_size > 0, traffic_across_cut / cut_size, 0.0)

    partitioned_adj = adjacency_matrix.val - cut_matrix
    masked1 = jnp.where(jnp.outer(partition1, partition1) > 0, partitioned_adj, 0)
    masked2 = jnp.where(jnp.outer(partition2, partition2) > 0, partitioned_adj, 0)
    check1 = jnp.where(partition1 > 0, jnp.sum(masked1, axis=0) > 0, True)
    check2 = jnp.where(partition2 > 0, jnp.sum(masked2, axis=0) > 0, True)
    connected1 = jnp.all(check1)
    connected2 = jnp.all(check2)
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


@partial(jax.jit, static_argnums=(1, 2, 3, 4), donate_argnums=(0,))
def calculate_congestion_batch(
    partition_masks_batch, adjacency_matrix, traffic_matrix, source_nodes, dest_nodes
):
    return jax.vmap(calculate_congestion, in_axes=(0, None, None, None, None))(
        partition_masks_batch, adjacency_matrix, traffic_matrix, source_nodes, dest_nodes
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
):
    def find_congested_cuts_iter(_, i):
        batch_indices = jnp.arange(num_batches_per_iteration) + i

        def batch_eval_sort(_, j):
            j = j * max_batch_size
            new_masks = generate_gray_code_masks(num_nodes, max_batch_size, j)
            new_congestions = calculate_congestion_batch(
                new_masks, adj_matrix, traf_matrix, source_nodes, dest_nodes
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


def find_congested_cuts_simple(
    path_link_array, source_nodes, dest_nodes, adjacency_matrix, traffic_matrix
):
    def get_cutset_partitions_and_congestion(_, i):
        path = path_link_array.val[i]
        p1, p2 = get_cutset_from_path(path, adjacency_matrix, source_nodes, dest_nodes)
        congestion = calculate_congestion(
            p1, adjacency_matrix, traffic_matrix, source_nodes, dest_nodes
        )
        return None, (congestion, p1, p2)

    path_indices = jnp.arange(path_link_array.shape[0])
    _, (congestions, partition1, partition2) = jax.lax.scan(
        get_cutset_partitions_and_congestion, None, path_indices
    )
    return congestions, partition1, partition2


# =====================================================================
#  Capacity Bound Simulation
# =====================================================================


def precompute_cutset_data(heavy_cut_sets, num_links):
    """Precompute data structures for the capacity-bound simulation.

    From the heavy_cut_sets dict, build:
      - unique_link_indices: 1-D array of link indices that appear in any cut-set.
      - cutset_link_mask: (num_cutsets, num_unique_links) binary array mapping each
        cut-set to its member links *within the compressed index space*.
      - partition1, partition2: (num_cutsets, num_nodes) binary partition arrays.

    Returns a dict with keys:
        unique_link_indices, cutset_link_mask, partition1, partition2, num_cutsets,
        num_unique_links
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

    return {
        "unique_link_indices": unique_link_indices,  # (num_unique_links,)
        "cutset_link_mask": cutset_link_mask.astype(jnp.int32),  # (num_cutsets, num_unique_links)
        "partition1": partition1.astype(jnp.int32),  # (num_cutsets, num_nodes)
        "partition2": partition2.astype(jnp.int32),  # (num_cutsets, num_nodes)
        "num_cutsets": num_cutsets,
        "num_unique_links": num_unique_links,
        "link_to_compressed": link_to_compressed,  # (num_links,)
    }


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
    k = params.k_paths
    best_se = np.ones((num_nodes, num_nodes), dtype=np.int32)

    if not params.consider_modulation_format:
        return jnp.array(best_se)

    for s in range(num_nodes):
        for d in range(num_nodes):
            if s == d:
                continue
            nodes = jnp.array([s, d])
            se_vals = get_paths_se(params, nodes)  # (k,)
            best_se[s, d] = int(jnp.max(se_vals))

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


def _find_feasible_start_slots(traversed, cutset_link_mask, slot_array, num_slots):
    """Find starting slot positions where a contiguous block of num_slots is feasible
    across all traversed cut-sets.

    For each candidate starting position s, the block [s, s+num_slots) must be available.
    A block is available on cut-set i if ANY link in that cut-set has all slots in the
    block free. The starting position must work for ALL traversed cut-sets.

    Spectrum continuity: the same slot range is used across all cut-sets (and the
    intermediate links, which are unconstrained in the upper bound, would also use
    the same range).

    Args:
        traversed: (C,) boolean.
        cutset_link_mask: (C, L) binary.
        slot_array: (L, S) occupancy (1=used, 0=free).
        num_slots: scalar int (may be traced) — number of contiguous slots needed.

    Returns:
        feasible_starts: (S,) boolean — True if starting at that slot is feasible.
    """
    num_total_slots = slot_array.shape[1]
    free = 1 - slot_array  # (L, S), 1=free

    # Sliding-window sum using cumulative sum + dynamic indexing.
    # cumfree[link, i] = sum(free[link, 0:i])
    cumfree = jnp.concatenate(
        [jnp.zeros((free.shape[0], 1), dtype=free.dtype), jnp.cumsum(free, axis=1)], axis=1
    )  # (L, S+1)

    # For each start position s, block_free_count[link, s] =
    #   cumfree[link, s + num_slots] - cumfree[link, s]
    # We compute this using dynamic indexing that works with traced num_slots.
    start_indices = jnp.arange(num_total_slots)  # (S,)
    end_indices = start_indices + num_slots  # (S,) — may exceed S, handled below

    # cumfree has S+1 columns (indices 0..S). Gather columns at end_indices and start_indices.
    # Clamp end_indices to at most S (the last valid cumfree column index).
    end_indices_clamped = jnp.minimum(end_indices, num_total_slots)  # (S,)

    # Gather: cumfree[:, end_indices_clamped] and cumfree[:, start_indices]
    # Using advanced indexing: cumfree is (L, S+1)
    cumfree_at_end = cumfree[:, end_indices_clamped]  # (L, S)
    cumfree_at_start = cumfree[:, start_indices]  # (L, S)
    block_free_count = cumfree_at_end - cumfree_at_start  # (L, S)

    # Positions where the block would overflow the array are infeasible
    valid_start = (start_indices + num_slots) <= num_total_slots  # (S,)

    link_block_free = (block_free_count >= num_slots) & valid_start[None, :]  # (L, S)

    # Per cut-set: block at start s is feasible if ANY link in the cut-set has it free
    available = (
        jnp.einsum(
            "cl,ls->cs", cutset_link_mask.astype(jnp.int32), link_block_free.astype(jnp.int32)
        )
        > 0
    )  # (C, S)

    # Non-traversed cut-sets: treat as available
    available_or_skip = available | (~traversed[:, None])  # (C, S)

    # Feasible start = available on ALL traversed cut-sets
    feasible_starts = available_or_skip.all(axis=0)  # (S,)
    return feasible_starts


def _compute_link_tiebreaker(slot_array, start_slot, num_slots, link_selection_mode, rng_key):
    """Compute per-link tiebreaker score for greedy link assignment.

    Args:
        slot_array: (L, S) occupancy (1=used, 0=free).
        start_slot: scalar int — first slot index of the block.
        num_slots: scalar int — number of contiguous slots.
        link_selection_mode: static str — one of 'least_congested', 'most_congested',
            'best_fit', 'random'.
        rng_key: PRNG key (used only for 'random' mode).

    Returns:
        tiebreaker: (L,) float in [0, 1) — higher is preferred.
    """
    num_total_slots = slot_array.shape[1]
    num_links = slot_array.shape[0]
    free_fraction = jnp.sum(1 - slot_array, axis=1) / num_total_slots  # (L,)

    if link_selection_mode == "least_congested":
        # Prefer links with the most free slots overall
        return free_fraction
    elif link_selection_mode == "most_congested":
        # Prefer links with the fewest free slots overall
        return 1.0 - free_fraction
    elif link_selection_mode == "best_fit":
        # Prefer links where the contiguous free run containing the block is shortest
        # (i.e. least wasted space around the block).
        free = 1 - slot_array  # (L, S), 1=free

        # Compute contiguous free run length at each position using cumsum trick:
        # run_length[l, s] = number of consecutive free slots ending at position s.
        # We compute this cumulatively: reset on each occupied slot.
        def _run_lengths(free_row):
            # For a 1D free vector, compute contiguous run lengths ending at each position
            def scan_fn(run, f):
                new_run = (run + 1) * f  # reset to 0 on occupied, else increment
                return new_run, new_run

            _, runs = jax.lax.scan(scan_fn, jnp.int32(0), free_row.astype(jnp.int32))
            return runs

        run_at = jax.vmap(_run_lengths)(free)  # (L, S)
        # The run length of the contiguous free block containing [start, start+num_slots)
        # is run_at[l, end_of_run] where end_of_run is the last free slot in the run.
        # Approximate: use the run length at (start + num_slots - 1).
        end_pos = jnp.minimum(start_slot + num_slots - 1, num_total_slots - 1)
        run_at_end = run_at[:, end_pos]  # (L,)

        # Now find the full run by looking forward from end_pos for remaining free slots
        # Actually, run_at gives backward runs. We also need forward runs to get the full
        # contiguous block. Compute forward runs similarly.
        def _run_lengths_rev(free_row):
            def scan_fn(run, f):
                new_run = (run + 1) * f
                return new_run, new_run

            _, runs = jax.lax.scan(scan_fn, jnp.int32(0), free_row.astype(jnp.int32)[::-1])
            return runs[::-1]

        run_fwd = jax.vmap(_run_lengths_rev)(free)  # (L, S)
        run_fwd_at_start = run_fwd[:, jnp.minimum(start_slot, num_total_slots - 1)]  # (L,)
        # Total contiguous run = backward run at end + forward run at start - num_slots
        total_run = run_at_end + run_fwd_at_start - num_slots  # (L,)
        total_run = jnp.maximum(total_run, num_slots)  # floor at num_slots
        # Best fit: prefer smallest total run (tightest fit).
        # Invert so higher score = tighter fit. Normalize to [0, 1).
        return 1.0 - total_run.astype(jnp.float32) / (num_total_slots + 1)
    elif link_selection_mode == "random":
        return jax.random.uniform(rng_key, shape=(num_links,))
    else:
        # Default to least_congested
        return free_fraction


def _assign_links_for_block(
    start_slot,
    num_slots,
    traversed,
    cutset_link_mask,
    slot_array,
    max_links_to_assign,
    link_selection_mode,
    rng_key,
):
    """Greedy link assignment for a contiguous slot block [start_slot, start_slot+num_slots).

    For each traversed cut-set, we need at least one link in that cut-set with the
    entire block free. A single physical link can cover multiple cut-sets.

    Args:
        start_slot: scalar int — first slot index of the block.
        num_slots: scalar int — number of contiguous slots.
        traversed: (C,) boolean.
        cutset_link_mask: (C, L) binary.
        slot_array: (L, S) occupancy.
        max_links_to_assign: static int.
        link_selection_mode: static str — secondary link selection heuristic.
        rng_key: PRNG key (used for 'random' mode).

    Returns:
        assigned_links: (max_links_to_assign,) int, padded with -1.
        num_assigned: scalar int.
    """
    # Which links have the full block [start_slot, start_slot+num_slots) free?
    slot_indices = jnp.arange(slot_array.shape[1])
    in_block = (slot_indices >= start_slot) & (slot_indices < start_slot + num_slots)  # (S,)
    # For each link: are ALL slots in the block free?
    block_occupied = jnp.sum(
        slot_array * in_block[None, :], axis=1
    )  # (L,) — count of occupied in block
    link_block_free = block_occupied == 0  # (L,) bool

    # Compute tiebreaker based on selection mode
    tiebreaker = _compute_link_tiebreaker(
        slot_array, start_slot, num_slots, link_selection_mode, rng_key
    )  # (L,)

    init_covered = ~traversed  # non-traversed are "pre-covered"

    def greedy_body(i, state):
        covered, assigned_links, num_assigned = state
        uncovered = ~covered
        coverage_count = jnp.sum(cutset_link_mask * uncovered[:, None], axis=0)  # (L,)
        score = (coverage_count.astype(jnp.float32) + tiebreaker) * link_block_free
        score = jnp.where(coverage_count > 0, score, -1.0)

        best_link = jnp.argmax(score)
        best_score = score[best_link]

        newly_covered = cutset_link_mask[:, best_link].astype(jnp.bool_)
        new_covered = covered | newly_covered

        do_assign = best_score > 0
        covered = jnp.where(do_assign, new_covered, covered)
        assigned_links = assigned_links.at[i].set(jnp.where(do_assign, best_link, -1))
        num_assigned = num_assigned + do_assign.astype(jnp.int32)
        return (covered, assigned_links, num_assigned)

    assigned_links_init = jnp.full(max_links_to_assign, -1, dtype=jnp.int32)
    init_state = (init_covered, assigned_links_init, jnp.int32(0))
    _, assigned_links, num_assigned = jax.lax.fori_loop(
        0, max_links_to_assign, greedy_body, init_state
    )
    return assigned_links, num_assigned


def _simulation_step(
    carry,
    rng_key,
    partition1,
    partition2,
    cutset_link_mask,
    unique_link_indices,
    max_links_to_assign,
    best_se_matrix,
    params,
    link_selection_mode,
):
    """One timestep of the capacity-bound simulation.

    Uses standard RSA env functions for request generation, expiry,
    allocation, checking, and commit/rollback.

    carry = (state, always_accepted_count, blocked_count,
             always_accepted_bitrate, blocked_bitrate)
      state: RSAEnvState with link_slot_array (num_links, link_resources), etc.
    """
    state, always_accepted_count, blocked_count, always_accepted_bitrate, blocked_bitrate = carry

    rng_key, rng_link = jax.random.split(rng_key)

    # --- 1. Generate request and remove expired services ---
    state = generate_request_rsa(rng_key, state, params)

    # --- 2. Read request, compute SE and required slots ---
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

    # --- 3. Find traversed cut-sets ---
    traversed = _find_traversed_cutsets(source, dest, partition1, partition2)
    any_traversed = jnp.any(traversed)

    # --- 4. Find feasible start slot (using only cut-set links) ---
    slot_array_subset = state.link_slot_array[unique_link_indices]
    feasible_starts = _find_feasible_start_slots(
        traversed, cutset_link_mask, slot_array_subset, num_slots
    )
    start_slot = jnp.argmax(feasible_starts)
    has_feasible = feasible_starts[start_slot]

    # --- 5. Assign links (greedy set cover in compressed space) ---
    assigned_links_compressed, _ = _assign_links_for_block(
        start_slot,
        num_slots,
        traversed,
        cutset_link_mask,
        slot_array_subset,
        max_links_to_assign,
        link_selection_mode,
        rng_link,
    )

    # --- 6. Build path vector in full link space ---
    # Convert compressed link indices to full network indices.
    # For blocked requests (not feasible), path stays all-zeros which
    # causes check_real_path to fail in check_action_rsa.
    accept_with_cutset = any_traversed & has_feasible
    path = jnp.zeros(params.num_links, dtype=state.link_slot_array.dtype)

    def set_link_in_path(i, p):
        comp_idx = assigned_links_compressed[i]
        full_idx = jnp.where(comp_idx >= 0, unique_link_indices[jnp.maximum(comp_idx, 0)], -1)
        valid = (comp_idx >= 0) & accept_with_cutset
        return jnp.where(valid, p.at[full_idx].set(1.0), p)

    path = jax.lax.fori_loop(0, max_links_to_assign, set_link_in_path, path)

    # --- 7. Build ActionInfo ---
    affected_slots_mask = get_affected_slots_mask(
        state,
        start_slot.astype(state.link_slot_array.dtype),
        num_slots.astype(state.link_slot_array.dtype),
        path,
        params,
    )
    action_info = ActionInfo(
        action=jnp.array(0, dtype=state.link_slot_array.dtype),
        path_index=jnp.array(0, dtype=state.link_slot_array.dtype),
        initial_slot_index=start_slot.astype(state.link_slot_array.dtype),
        num_slots=num_slots.astype(state.link_slot_array.dtype),
        path=path,
        se=se,
        requested_datarate=requested_datarate,
        nodes_sd=nodes_sd,
        affected_slots_mask=affected_slots_mask,
    )

    # --- 8-10. Implement, check, complete (for traversed requests) ---
    state_after_impl = implement_action_rsa(state, action_info, params)
    check = check_action_rsa(state_after_impl, action_info, params)
    state_after_complete = complete_step_rsa(state_after_impl, action_info, check, params)

    # For non-traversed requests: always accepted, no allocation needed
    state_not_traversed = state.replace(
        accepted_services=state.accepted_services + 1,
        total_timesteps=state.total_timesteps + 1,
        total_bitrate=state.total_bitrate + requested_datarate,
        accepted_bitrate=state.accepted_bitrate + requested_datarate,
    )

    # Select between traversed and not-traversed branches
    final_state = jax.tree.map(
        lambda a, b: jnp.where(any_traversed, a, b),
        state_after_complete,
        state_not_traversed,
    )

    # Update custom counters
    is_blocked = any_traversed & (check > 0)
    always_accepted_count = always_accepted_count + (~any_traversed).astype(jnp.int32)
    blocked_count = blocked_count + is_blocked.astype(jnp.int32)
    always_accepted_bitrate = always_accepted_bitrate + jnp.where(
        ~any_traversed, requested_datarate, 0.0
    )
    blocked_bitrate = blocked_bitrate + jnp.where(is_blocked, requested_datarate, 0.0)

    return (
        final_state,
        always_accepted_count,
        blocked_count,
        always_accepted_bitrate,
        blocked_bitrate,
    ), None


def run_single_trial(
    rng,
    initial_state,
    params,
    partition1,
    partition2,
    cutset_link_mask,
    unique_link_indices,
    best_se_matrix,
    num_requests,
    max_links_to_assign,
    link_selection_mode,
):
    """Run a single trial of the capacity-bound simulation.

    Uses RSAEnvState from env.reset() as initial state. Departure tracking
    is handled by link_slot_departure_array inside the state.

    Returns:
        (accepted_count, blocked_count, always_accepted_count,
         accepted_bitrate, blocked_bitrate, always_accepted_bitrate, total_bitrate)
    """
    init_carry = (
        initial_state,
        jnp.int32(0),
        jnp.int32(0),
        jnp.float32(0.0),
        jnp.float32(0.0),
        jnp.int32(0),
    )

    def step_fn(carry, _):
        state, aa, bc, aabr, bbr, step_idx = carry
        rng_key = jax.random.fold_in(rng, step_idx)
        (state, aa, bc, aabr, bbr), _ = _simulation_step(
            (state, aa, bc, aabr, bbr),
            rng_key,
            partition1,
            partition2,
            cutset_link_mask,
            unique_link_indices,
            max_links_to_assign,
            best_se_matrix,
            params,
            link_selection_mode,
        )
        return (state, aa, bc, aabr, bbr, step_idx + 1), None

    final_carry, _ = jax.lax.scan(step_fn, init_carry, None, length=num_requests)

    (
        final_state,
        always_accepted_count,
        blocked_count,
        always_accepted_bitrate,
        blocked_bitrate,
        _,
    ) = final_carry
    accepted_count = final_state.accepted_services
    accepted_bitrate = final_state.accepted_bitrate
    total_bitrate = final_state.total_bitrate

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
    link_selection_mode="least_congested",
):
    """Run the full capacity-bound simulation across multiple loads and trials.

    Args:
        heavy_cut_sets: dict with keys congestion, partition1, partition2, cutset_edges.
        env: RSAEnv instance (raw, unwrapped).
        params: RSAEnvParams (base params; arrival_rate overridden per load).
        best_se_matrix: (num_nodes, num_nodes) int — best SE per (s,d).
        loads: 1-D array of traffic loads (Erlangs) to sweep.
        num_requests: int, requests per trial.
        num_trials: int, trials per load.
        seed: int, base random seed.
        link_selection_mode: str — secondary link selection heuristic.

    Returns:
        results: dict mapping load -> {accepted, blocked, always_accepted, blocking_prob}
    """
    num_links = params.num_links
    mean_service_holding_time = params.mean_service_holding_time
    values_bw = jnp.array(params.values_bw.val, dtype=jnp.float32)

    # Precompute cut-set data
    cs_data = precompute_cutset_data(heavy_cut_sets, num_links)
    partition1 = cs_data["partition1"]
    partition2 = cs_data["partition2"]
    cutset_link_mask = cs_data["cutset_link_mask"]
    unique_link_indices = cs_data["unique_link_indices"]
    num_unique_links = cs_data["num_unique_links"]
    num_cutsets = cs_data["num_cutsets"]

    max_links_to_assign = min(num_cutsets, num_unique_links, 20)

    print(f"Capacity bound simulation setup:")
    print(f"  Num cut-sets: {num_cutsets}")
    print(f"  Num unique links in cut-sets: {num_unique_links}")
    print(f"  Num slots per link: {params.link_resources}")
    print(f"  Slot size: {params.slot_size} GHz, Guardband: {params.guardband} slots")
    print(f"  Bandwidth values: {values_bw}")
    print(f"  Max slots per request: {params.max_slots}")
    print(f"  Consider modulation format: {params.consider_modulation_format}")
    print(f"  Max links to assign per request: {max_links_to_assign}")
    print(f"  Link selection mode: {link_selection_mode}")
    print(f"  Requests per trial: {num_requests}")
    print(f"  Num trials: {num_trials}")
    print(f"  Loads to sweep: {loads}")

    profiler = Profiler()

    results = {}
    for load_idx, load_val in enumerate(loads):
        arrival_rate = float(load_val / mean_service_holding_time)
        params_for_load = params.replace(arrival_rate=arrival_rate)

        # Get fresh initial state for each load (reset env)
        reset_key = jax.random.PRNGKey(seed)
        _, initial_state = env.reset(reset_key, params_for_load)

        trial_rngs = jax.random.split(jax.random.PRNGKey(seed + 1), num_trials)

        print(f"\nLoad = {load_val:.1f} Erlang (arrival_rate = {arrival_rate:.4f})...")

        def single(rng):
            return run_single_trial(
                rng,
                initial_state,
                params_for_load,
                partition1,
                partition2,
                cutset_link_mask,
                unique_link_indices,
                best_se_matrix,
                num_requests,
                max_links_to_assign,
                link_selection_mode,
            )

        # Compilation (each load recompiles because arrival_rate is static)
        with profiler.section(f"COMPILATION (load={load_val:.0f})"):
            jitted_single = jax.jit(jax.vmap(single)).lower(trial_rngs).compile()

        # Execution
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
            ) = jitted_single(trial_rngs)
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
    return results


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
    print(f"Available devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")

    if FLAGS.DISABLE_JIT:
        jax.config.update("jax_disable_jit", True)
        jax.numpy.set_printoptions(threshold=sys.maxsize, linewidth=220)

    # --- Build environment params ---
    graph = make_graph(FLAGS.topology_name, FLAGS.topology_directory)
    env, params = make(FLAGS)

    edges = sorted(graph.edges())
    source_nodes_np = jnp.array([edge[0] for edge in edges])
    destination_nodes_np = jnp.array([edge[1] for edge in edges])
    adj_matrix = nx.adjacency_matrix(graph, weight="").todense()

    traffic_matrix_weighted = get_weighted_traffic_matrix(graph, params)
    traffic_matrix_weighted = normalise_traffic_matrix(traffic_matrix_weighted)
    traffic_matrix_haw = HashableArrayWrapper(traffic_matrix_weighted)
    adj_matrix_haw = HashableArrayWrapper(jnp.array(adj_matrix))
    source_nodes_haw = HashableArrayWrapper(source_nodes_np)
    destination_nodes_haw = HashableArrayWrapper(destination_nodes_np)
    top_k = FLAGS.CUTSET_TOP_K

    # --- Find congested cut-sets ---
    if FLAGS.CUTSET_EXHAUSTIVE:
        total_combinations = 2**params.num_nodes
        parallel_processes = FLAGS.NUM_ENVS
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
            )
        else:
            with TimeIt("CUT-SET COMPILATION:"):
                func = (
                    jax.jit(
                        jax.vmap(
                            find_congested_cuts_exhaustive,
                            in_axes=(0, None, None, None, None, None, None, None, None, None),
                        ),
                        static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9),
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
            )
        else:
            with TimeIt("CUT-SET COMPILATION:"):
                func = (
                    jax.jit(find_congested_cuts_simple, static_argnums=(0, 1, 2, 3, 4))
                    .lower(
                        params.path_link_array,
                        source_nodes_haw,
                        destination_nodes_haw,
                        adj_matrix_haw,
                        traffic_matrix_haw,
                    )
                    .compile()
                )
            with TimeIt("CUT-SET EXECUTION:"):
                heavy_cut_sets_raw = func()
                heavy_cut_sets_raw[0].block_until_ready()

    # Post-process: reshape, deduplicate, select top-k
    congestions = jnp.reshape(heavy_cut_sets_raw[0], (-1,))
    partition1 = jnp.reshape(heavy_cut_sets_raw[1], (-1, params.num_nodes))
    partition2 = jnp.reshape(heavy_cut_sets_raw[2], (-1, params.num_nodes))
    cutset_edges = jax.vmap(find_cutset_edges, in_axes=(0, 0, None, None))(
        partition1, partition2, source_nodes_haw, destination_nodes_haw
    )

    # Deduplicate
    _, unique_indices = jnp.unique(cutset_edges, axis=0, return_index=True)
    cutset_edges = cutset_edges[unique_indices]
    congestions = congestions[unique_indices]
    partition1 = partition1[unique_indices]
    partition2 = partition2[unique_indices]

    # Filter out zero-congestion cutsets
    nonzero_mask = congestions > 0
    cutset_edges = cutset_edges[nonzero_mask]
    congestions = congestions[nonzero_mask]
    partition1 = partition1[nonzero_mask]
    partition2 = partition2[nonzero_mask]
    print(f"\nUnique cutsets with congestion > 0: {len(congestions)}")

    # Select top-k
    top_k_indices = jnp.argsort(congestions)[-top_k:]
    cutset_edges = cutset_edges[top_k_indices]
    congestions = congestions[top_k_indices]
    partition1 = partition1[top_k_indices]
    partition2 = partition2[top_k_indices]
    print(f"Cutsets after top-{top_k} selection: {len(congestions)}")

    # Optionally filter by mean congestion threshold
    if FLAGS.USE_MEAN_CONGESTION_THRESHOLD:
        threshold = float(jnp.mean(congestions))
        above_threshold = congestions >= threshold
        cutset_edges = cutset_edges[above_threshold]
        congestions = congestions[above_threshold]
        partition1 = partition1[above_threshold]
        partition2 = partition2[above_threshold]
        print(f"Mean congestion threshold: {threshold:.6f}")
        print(f"Cutsets after mean filtering: {len(congestions)}")

    heavy_cut_sets = {
        "congestion": congestions,
        "partition1": partition1,
        "partition2": partition2,
        "cutset_edges": cutset_edges,
    }

    print(f"\nSelected {len(congestions)} cut-sets:")
    for i, (cong, ce) in enumerate(
        zip(heavy_cut_sets["congestion"], heavy_cut_sets["cutset_edges"])
    ):
        links = [int(j) for j, v in enumerate(ce) if v > 0]
        print(f"  Cut-set {i}: congestion={float(cong):.4f}, links={links}")

    # --- Build best SE matrix for RMSA ---
    print("Building best spectral efficiency matrix...")
    best_se_matrix = build_best_se_matrix(params)
    if params.consider_modulation_format:
        print(
            f"  SE range: {int(jnp.min(best_se_matrix[best_se_matrix > 0]))} - {int(jnp.max(best_se_matrix))}"
        )
    else:
        print("  Modulation format not considered (SE=1 everywhere)")

    # --- Get raw RSAEnv (unwrap LogWrapper) ---
    raw_env = env._env if hasattr(env, "_env") else env

    # --- Run capacity bound simulation ---
    loads = np.arange(
        FLAGS.sim_min_load,
        FLAGS.sim_max_load + FLAGS.sim_step_load / 2,
        FLAGS.sim_step_load,
    )

    results = run_capacity_bound_simulation(
        heavy_cut_sets=heavy_cut_sets,
        env=raw_env,
        params=params,
        best_se_matrix=best_se_matrix,
        loads=loads,
        num_requests=FLAGS.num_sim_requests,
        num_trials=FLAGS.num_trials,
        seed=FLAGS.SEED,
        link_selection_mode=FLAGS.link_selection_mode,
    )

    print_results_table(results)

    # Print summary statistics for each load (parseable by shell scripts)
    for load in sorted(results.keys()):
        r = results[load]
        bp = r["blocking_prob"]
        bbp = r["bitrate_blocking_prob"]

        bp_mean = float(np.mean(bp))
        bp_std = float(np.std(bp))
        bp_iqr_lower = float(np.nanpercentile(bp, 25))
        bp_iqr_upper = float(np.nanpercentile(bp, 75))

        bbp_mean = float(np.mean(bbp))
        bbp_std = float(np.std(bbp))
        bbp_iqr_lower = float(np.nanpercentile(bbp, 25))
        bbp_iqr_upper = float(np.nanpercentile(bbp, 75))

        accepted_mean = float(np.mean(r["accepted"]))
        accepted_std = float(np.std(r["accepted"]))
        accepted_iqr_lower = float(np.nanpercentile(r["accepted"], 25))
        accepted_iqr_upper = float(np.nanpercentile(r["accepted"], 75))

        blocked_mean = float(np.mean(r["blocked"]))
        blocked_std = float(np.std(r["blocked"]))
        blocked_iqr_lower = float(np.nanpercentile(r["blocked"], 25))
        blocked_iqr_upper = float(np.nanpercentile(r["blocked"], 75))

        always_accepted_mean = float(np.mean(r["always_accepted"]))
        always_accepted_std = float(np.std(r["always_accepted"]))
        always_accepted_iqr_lower = float(np.nanpercentile(r["always_accepted"], 25))
        always_accepted_iqr_upper = float(np.nanpercentile(r["always_accepted"], 75))

        print(f"\n=== SUMMARY load={load:.1f} ===")
        print(f"Blocking Probability mean: {bp_mean:.8f}")
        print(f"Blocking Probability std: {bp_std:.8f}")
        print(f"Blocking Probability IQR lower: {bp_iqr_lower:.8f}")
        print(f"Blocking Probability IQR upper: {bp_iqr_upper:.8f}")
        print(f"Bitrate Blocking Probability mean: {bbp_mean:.8f}")
        print(f"Bitrate Blocking Probability std: {bbp_std:.8f}")
        print(f"Bitrate Blocking Probability IQR lower: {bbp_iqr_lower:.8f}")
        print(f"Bitrate Blocking Probability IQR upper: {bbp_iqr_upper:.8f}")
        print(f"Accepted Count mean: {accepted_mean:.5f}")
        print(f"Accepted Count std: {accepted_std:.5f}")
        print(f"Accepted Count IQR lower: {accepted_iqr_lower:.5f}")
        print(f"Accepted Count IQR upper: {accepted_iqr_upper:.5f}")
        print(f"Blocked Count mean: {blocked_mean:.5f}")
        print(f"Blocked Count std: {blocked_std:.5f}")
        print(f"Blocked Count IQR lower: {blocked_iqr_lower:.5f}")
        print(f"Blocked Count IQR upper: {blocked_iqr_upper:.5f}")
        print(f"Always Accepted Count mean: {always_accepted_mean:.5f}")
        print(f"Always Accepted Count std: {always_accepted_std:.5f}")
        print(f"Always Accepted Count IQR lower: {always_accepted_iqr_lower:.5f}")
        print(f"Always Accepted Count IQR upper: {always_accepted_iqr_upper:.5f}")


if __name__ == "__main__":
    app.run(main)
