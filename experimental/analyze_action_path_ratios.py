"""Run FF-KSP heuristic on usa100_directed with K=70 and analyze
the hop/distance ratios of chosen paths vs shortest paths.

Step 1: Run the heuristic eval with --log_actions --TRAJ_DATA_OUTPUT_FILE
Step 2: Analyze the output CSV
"""

import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from xlron.environments.env_funcs import (
    init_path_link_array, init_link_length_array, make_graph,
)

TOPOLOGY = "usa100_directed"
K = 70
TRAJ_FILE = "experimental/usa100_k70_ff_ksp_traj.csv"


def run_heuristic():
    """Run the heuristic eval and save trajectory data."""
    cmd = [
        sys.executable, "-m", "xlron.train.train",
        "--env_type=rmsa",
        f"--topology_name={TOPOLOGY}",
        "--link_resources=320",
        "--slot_size=12.5",
        "--guardband=1",
        "--values_bw=100",
        "--mean_service_holding_time=25",
        "--continuous_operation",
        f"--k={K}",
        "--path_heuristic=ff_ksp",
        "--EVAL_HEURISTIC",
        "--load=250",
        "--NUM_ENVS=1",
        "--TOTAL_TIMESTEPS=100000",
        "--STEPS_PER_INCREMENT=100000",
        "--ROLLOUT_LENGTH=128",
        "--log_actions",
        f"--TRAJ_DATA_OUTPUT_FILE={TRAJ_FILE}",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("Heuristic eval failed!")
        sys.exit(1)


def analyze():
    """Analyze the trajectory data."""
    df = pd.read_csv(TRAJ_FILE)
    print(f"\nLoaded trajectory data: {len(df)} actions")
    print(f"Columns: {list(df.columns)}")

    # Get path arrays for shortest path comparison
    graph = make_graph(TOPOLOGY)
    link_length_array = np.asarray(init_link_length_array(graph), dtype=np.float64)
    pla_full = init_path_link_array(
        graph, K, directed=True, topology_name=TOPOLOGY,
        cache_dir="xlron/data/topologies/ksp",
        path_sort_criteria="spectral_resources",
    )
    pla = np.asarray(pla_full)

    # Precompute all path distances and hops
    all_distances = pla.astype(np.float64) @ link_length_array
    all_hops = pla.sum(axis=1)

    num_nodes = graph.number_of_nodes()
    num_pairs = num_nodes * (num_nodes - 1)
    all_distances_by_pair = all_distances.reshape(num_pairs, K)
    all_hops_by_pair = all_hops.reshape(num_pairs, K)
    shortest_dist_by_pair = all_distances_by_pair[:, 0]
    shortest_hops_by_pair = all_hops_by_pair[:, 0]

    # Extract data from trajectory
    path_indices = df["path_indices"].values
    returns = df["returns"].values
    path_lengths = df["path_length"].values
    num_hops = df["num_hops"].values

    # Compute pair index and relative path index
    pair_indices = path_indices // K
    relative_path_indices = path_indices % K

    # Get shortest path stats for each action
    sp_dist = shortest_dist_by_pair[pair_indices]
    sp_hops = shortest_hops_by_pair[pair_indices]

    # Compute ratios
    dist_ratio = path_lengths / np.maximum(sp_dist, 1e-6)
    hop_ratio = num_hops / np.maximum(sp_hops, 1)

    # Filter valid actions (non-zero hops)
    valid = num_hops > 0
    accepted = returns >= 0  # continuous_operation: 0=accept, -1=block

    rpi = relative_path_indices[valid]
    dr = dist_ratio[valid]
    hr = hop_ratio[valid]
    sp_h = sp_hops[valid]
    acc = accepted[valid]

    print(f"\nTotal actions: {len(path_indices)}")
    print(f"Valid actions: {valid.sum()}")
    print(f"Accepted actions: {accepted.sum()} ({100*accepted.sum()/len(path_indices):.1f}%)")

    print(f"\n{'='*70}")
    print(f"ACTION PATH RATIO ANALYSIS (FF-KSP, K={K})")
    print(f"{'='*70}")

    for label, mask in [("All valid", np.ones(len(rpi), dtype=bool)), ("Accepted only", acc)]:
        r = rpi[mask]
        d = dr[mask]
        h = hr[mask]

        print(f"\n--- {label} ({mask.sum()} actions) ---")
        if mask.sum() == 0:
            print("  (no actions)")
            continue

        print(f"\nRelative path index chosen:")
        print(f"  Mean: {r.mean():.2f}, Median: {np.median(r):.0f}, "
              f"Max: {r.max()}, P95: {np.percentile(r, 95):.0f}, P99: {np.percentile(r, 99):.0f}")

        print(f"\nHop ratio (chosen / shortest):")
        print(f"  Mean: {h.mean():.3f}, Median: {np.median(h):.3f}, "
              f"Max: {h.max():.2f}, P95: {np.percentile(h, 95):.3f}, P99: {np.percentile(h, 99):.3f}")

        print(f"\nDistance ratio (chosen / shortest):")
        print(f"  Mean: {d.mean():.3f}, Median: {np.median(d):.3f}, "
              f"Max: {d.max():.2f}, P95: {np.percentile(d, 95):.3f}, P99: {np.percentile(d, 99):.3f}")

        # Distribution of relative path index
        print(f"\nRelative path index distribution:")
        for threshold in [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 69]:
            if threshold >= K:
                break
            count = (r == threshold).sum()
            cum_count = (r <= threshold).sum()
            print(f"  Path {threshold:>3}: {count:>7} ({100*count/len(r):.2f}%)  "
                  f"Cumulative <= {threshold}: {cum_count:>7} ({100*cum_count/len(r):.1f}%)")

        # Hop ratio distribution
        print(f"\nHop ratio distribution:")
        for threshold in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0]:
            count = (h > threshold).sum()
            print(f"  Actions with ratio > {threshold:4.2f}: {count:>7} ({100*count/len(h):.2f}%)")

    # Breakdown by shortest path hop count (accepted only)
    sp_h_acc = sp_h[acc]
    rpi_acc = rpi[acc]
    hr_acc = hr[acc]
    dr_acc = dr[acc]

    print(f"\n{'='*70}")
    print(f"ACCEPTED ACTION RATIOS BY SHORTEST HOP COUNT")
    print(f"{'='*70}")
    print(f"  {'SP Hops':>7} {'Actions':>8} {'MeanPathIdx':>12} {'P95PathIdx':>11} "
          f"{'MaxPathIdx':>11} {'MeanHopRatio':>13} {'MeanDistRatio':>14}")

    for sp in range(1, int(sp_h_acc.max()) + 1):
        mask = sp_h_acc == sp
        if mask.sum() == 0:
            continue
        n = mask.sum()
        print(f"  {sp:>7} {n:>8} {rpi_acc[mask].mean():>12.1f} "
              f"{np.percentile(rpi_acc[mask], 95):>11.0f} {rpi_acc[mask].max():>11} "
              f"{hr_acc[mask].mean():>13.3f} {dr_acc[mask].mean():>14.3f}")

    # What fraction of accepted actions use paths beyond various K thresholds?
    print(f"\n{'='*70}")
    print(f"ACCEPTED ACTIONS USING PATH INDEX >= K_THRESHOLD")
    print(f"{'='*70}")
    for k_thresh in [5, 10, 15, 20, 30, 40, 50, 60, 70]:
        if k_thresh > K:
            break
        count = (rpi_acc >= k_thresh).sum()
        print(f"  Actions using path >= {k_thresh:>3}: {count:>7} ({100*count/len(rpi_acc):.2f}%)")


if __name__ == "__main__":
    traj_path = Path(TRAJ_FILE)
    if not traj_path.exists():
        print("Trajectory file not found, running heuristic eval first...")
        run_heuristic()
    else:
        print(f"Using existing trajectory file: {TRAJ_FILE}")
    analyze()
