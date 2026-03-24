"""Analyze shortest/longest path lengths (hops and distance) per node pair for usa100_directed."""

import numpy as np
import networkx as nx
from xlron.environments.env_funcs import init_path_link_array, init_link_length_array, make_graph

TOPOLOGY = "usa100_directed"
TOPOLOGY = "tataind_directed"
K_VALUES = [50, 100]


def analyze(topology_name, k):
    graph = make_graph(topology_name)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    path_link_array = init_path_link_array(
        graph, k, directed=True, topology_name=topology_name,
        cache_dir="xlron/data/topologies/ksp",
        path_sort_criteria="spectral_resources",
    )
    pla = np.asarray(path_link_array)  # shape: (num_pairs * k, num_edges)

    # Link lengths
    link_lengths = np.array([graph.edges[e]["distance"] for e in sorted(graph.edges)])

    # Hops per path
    hops = pla.sum(axis=1)
    # Distance per path
    distances = pla.astype(np.float64) @ link_lengths.astype(np.float64)

    num_pairs = num_nodes * (num_nodes - 1)
    hops = hops.reshape(num_pairs, k)
    distances = distances.reshape(num_pairs, k)

    # For each node pair: min/max hops and distance (ignoring zero-padded paths)
    # Zero rows = no path found for that k-index
    valid = hops > 0

    print(f"\n{'='*70}")
    print(f"Topology: {topology_name}, K={k}")
    print(f"Nodes: {num_nodes}, Edges: {num_edges}, Node pairs: {num_pairs}")
    print(f"Path-link array shape: {pla.shape}")
    print(f"{'='*70}")

    # Count how many valid paths per pair
    valid_counts = valid.sum(axis=1)
    print(f"\nPaths found per node pair:")
    print(f"  Min: {valid_counts.min()}, Max: {valid_counts.max()}, "
          f"Mean: {valid_counts.mean():.1f}, Median: {np.median(valid_counts):.0f}")
    pairs_with_fewer = (valid_counts < k).sum()
    print(f"  Pairs with fewer than {k} paths: {pairs_with_fewer}/{num_pairs}")

    # Per-pair min/max hops
    hops_masked = np.where(valid, hops, np.nan)
    min_hops = np.nanmin(hops_masked, axis=1)
    max_hops = np.nanmax(hops_masked, axis=1)
    hop_ratios = max_hops / min_hops

    print(f"\nHops (per node pair):")
    print(f"  Shortest path hops — Min: {np.nanmin(min_hops):.0f}, Max: {np.nanmax(min_hops):.0f}, "
          f"Mean: {np.nanmean(min_hops):.1f}")
    print(f"  Longest path hops  — Min: {np.nanmin(max_hops):.0f}, Max: {np.nanmax(max_hops):.0f}, "
          f"Mean: {np.nanmean(max_hops):.1f}")
    print(f"  Ratio (longest/shortest) — Min: {np.nanmin(hop_ratios):.2f}, "
          f"Max: {np.nanmax(hop_ratios):.2f}, Mean: {np.nanmean(hop_ratios):.2f}")

    # Per-pair min/max distance
    dist_masked = np.where(valid, distances, np.nan)
    min_dist = np.nanmin(dist_masked, axis=1)
    max_dist = np.nanmax(dist_masked, axis=1)
    dist_ratios = max_dist / min_dist

    print(f"\nDistance (per node pair):")
    print(f"  Shortest path dist — Min: {np.nanmin(min_dist):.0f}, Max: {np.nanmax(min_dist):.0f}, "
          f"Mean: {np.nanmean(min_dist):.0f}")
    print(f"  Longest path dist  — Min: {np.nanmin(max_dist):.0f}, Max: {np.nanmax(max_dist):.0f}, "
          f"Mean: {np.nanmean(max_dist):.0f}")
    print(f"  Ratio (longest/shortest) — Min: {np.nanmin(dist_ratios):.2f}, "
          f"Max: {np.nanmax(dist_ratios):.2f}, Mean: {np.nanmean(dist_ratios):.2f}")

    # Distribution of hop ratios
    print(f"\nHop ratio distribution:")
    for threshold in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]:
        count = (hop_ratios > threshold).sum()
        print(f"  Pairs with ratio > {threshold:4.1f}: {count:5d} ({100*count/num_pairs:.1f}%)")

    print(f"\nDistance ratio distribution:")
    for threshold in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]:
        count = (dist_ratios > threshold).sum()
        print(f"  Pairs with ratio > {threshold:4.1f}: {count:5d} ({100*count/num_pairs:.1f}%)")

    # Worst 10 node pairs by hop ratio
    worst_idx = np.argsort(hop_ratios)[::-1][:10]
    nodes = sorted(graph.nodes)
    print(f"\nTop 10 worst node pairs by hop ratio:")
    print(f"  {'Src':>4} {'Dst':>4} {'MinHops':>8} {'MaxHops':>8} {'Ratio':>7} {'MinDist':>9} {'MaxDist':>9}")
    for idx in worst_idx:
        src = idx // (num_nodes - 1)
        dst = idx % (num_nodes - 1)
        if dst >= src:
            dst += 1
        print(f"  {nodes[src]:>4} {nodes[dst]:>4} {min_hops[idx]:>8.0f} {max_hops[idx]:>8.0f} "
              f"{hop_ratios[idx]:>7.2f} {min_dist[idx]:>9.0f} {max_dist[idx]:>9.0f}")

    # --- Distance cap analysis ---
    print(f"\n{'='*70}")
    print(f"DISTANCE CAP ANALYSIS (K={k})")
    print(f"{'='*70}")
    for cap in [5000, 7500, 10000, 12500, 15000]:
        # For each pair: keep paths under cap, or just the shortest if shortest > cap
        # Count surviving paths per pair
        surviving = np.zeros(num_pairs, dtype=int)
        for i in range(num_pairs):
            pair_dists = distances[i]
            pair_valid = valid[i]
            pair_dists_valid = pair_dists[pair_valid]
            if len(pair_dists_valid) == 0:
                surviving[i] = 0
                continue
            under_cap = (pair_dists_valid <= cap).sum()
            if under_cap == 0:
                # Shortest path exceeds cap, keep only shortest
                surviving[i] = 1
            else:
                surviving[i] = under_cap

        paths_removed = k * num_pairs - surviving.sum()
        total_paths = k * num_pairs
        pct_removed = 100 * paths_removed / total_paths

        # Surviving path stats
        pairs_only_shortest = (surviving == 1).sum()
        pairs_capped = (surviving < k).sum()

        # New max distance after filtering
        new_max_hops = np.zeros(num_pairs)
        new_max_dist = np.zeros(num_pairs)
        for i in range(num_pairs):
            n_surv = surviving[i]
            if n_surv == 0:
                continue
            pair_dists_valid = distances[i][valid[i]]
            pair_hops_valid = hops[i][valid[i]]
            if n_surv == 1 and pair_dists_valid.min() > cap:
                # Only shortest kept
                new_max_dist[i] = pair_dists_valid.min()
                new_max_hops[i] = pair_hops_valid[np.argmin(pair_dists_valid)]
            else:
                under = pair_dists_valid <= cap
                new_max_dist[i] = pair_dists_valid[under].max()
                new_max_hops[i] = pair_hops_valid[under].max()

        new_hop_ratios = new_max_hops / min_hops
        new_dist_ratios = new_max_dist / min_dist

        print(f"\n  Cap = {cap} km:")
        print(f"    Paths removed: {paths_removed}/{total_paths} ({pct_removed:.1f}%)")
        print(f"    Surviving paths per pair — Min: {surviving.min()}, Max: {surviving.max()}, "
              f"Mean: {surviving.mean():.1f}")
        print(f"    Pairs reduced to shortest only: {pairs_only_shortest}")
        print(f"    Pairs with any paths removed: {pairs_capped}")
        print(f"    New max hop ratio  — Mean: {np.nanmean(new_hop_ratios):.2f}, "
              f"Max: {np.nanmax(new_hop_ratios):.2f}")
        print(f"    New max dist ratio — Mean: {np.nanmean(new_dist_ratios):.2f}, "
              f"Max: {np.nanmax(new_dist_ratios):.2f}")


    # --- Breakdown by shortest path hop count ---
    print(f"\n{'='*70}")
    print(f"PATH DIVERSITY BY SHORTEST HOP COUNT (K={k})")
    print(f"{'='*70}")
    print(f"  {'SP Hops':>7} {'Pairs':>6} {'MeanValid':>10} {'MeanMaxHops':>12} "
          f"{'MeanHopRatio':>13} {'MeanDistRatio':>14} "
          f"{'Paths<=1.5x':>12} {'Paths<=2x':>10} {'Paths<=3x':>10}")

    for sp_hops in range(1, int(np.nanmax(min_hops)) + 1):
        mask = min_hops == sp_hops
        if mask.sum() == 0:
            continue
        n_pairs = mask.sum()
        mean_valid = valid_counts[mask].mean()
        mean_max_h = max_hops[mask].mean()
        mean_hr = hop_ratios[mask].mean()
        mean_dr = dist_ratios[mask].mean()

        # How many paths per pair are within Nx the shortest distance
        pair_indices = np.where(mask)[0]
        within_1_5x = 0
        within_2x = 0
        within_3x = 0
        for i in pair_indices:
            pair_dists = distances[i][valid[i]]
            sp_dist = pair_dists.min()
            within_1_5x += (pair_dists <= 1.5 * sp_dist).sum()
            within_2x += (pair_dists <= 2.0 * sp_dist).sum()
            within_3x += (pair_dists <= 3.0 * sp_dist).sum()
        mean_1_5x = within_1_5x / n_pairs
        mean_2x = within_2x / n_pairs
        mean_3x = within_3x / n_pairs

        print(f"  {sp_hops:>7} {n_pairs:>6} {mean_valid:>10.1f} {mean_max_h:>12.1f} "
              f"{mean_hr:>13.2f} {mean_dr:>14.2f} "
              f"{mean_1_5x:>12.1f} {mean_2x:>10.1f} {mean_3x:>10.1f}")


if __name__ == "__main__":
    for k in K_VALUES:
        analyze(TOPOLOGY, k)
