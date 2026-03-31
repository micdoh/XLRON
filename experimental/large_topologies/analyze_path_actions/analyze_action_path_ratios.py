"""Run heuristic and analyze the hop/distance ratios of chosen paths vs shortest paths.

Step 1: Run the heuristic eval with --log_actions --TRAJ_DATA_OUTPUT_FILE
Step 2: Analyze the output CSV
"""

import argparse
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from xlron.environments.env_funcs import (
    init_path_link_array, init_link_length_array, make_graph,
)

BASE_DIR = "experimental/large_topologies/analyze_path_actions"

TOPOLOGIES = {
    "usa100": {"name": "usa100_directed", "k": 70, "load": 620},
    "tataind": {"name": "tataind_directed", "k": 90, "load": 450},
}

HEURISTICS = ["ff_ksp", "ksp_ff"]


def traj_filename(topo_key, heuristic):
    """Return trajectory file path for a topology/heuristic combo."""
    topo = TOPOLOGIES[topo_key]
    return f"{BASE_DIR}/{topo['name'].replace('_directed', '')}_k{topo['k']}_{heuristic}_traj.csv"


def run_heuristic(topology_name, k, heuristic, load, traj_file):
    """Run the heuristic eval and save trajectory data."""
    cmd = [
        sys.executable, "-m", "xlron.train.train",
        "--env_type=rmsa",
        f"--topology_name={topology_name}",
        "--link_resources=320",
        "--slot_size=12.5",
        "--guardband=1",
        "--values_bw=100",
        "--mean_service_holding_time=25",
        "--continuous_operation",
        f"--k={k}",
        f"--path_heuristic={heuristic}",
        "--EVAL_HEURISTIC",
        f"--load={load}",
        "--NUM_ENVS=1",
        "--TOTAL_TIMESTEPS=100000",
        "--STEPS_PER_INCREMENT=100000",
        "--ROLLOUT_LENGTH=128",
        "--log_actions",
        f"--TRAJ_DATA_OUTPUT_FILE={traj_file}",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("Heuristic eval failed!")
        sys.exit(1)


def analyze(topology_name, k, traj_file, heuristic="ff_ksp"):
    """Analyze the trajectory data."""
    df = pd.read_csv(traj_file)
    print(f"\nLoaded trajectory data: {len(df)} actions")
    print(f"Columns: {list(df.columns)}")

    # Get path arrays for shortest path comparison
    graph = make_graph(topology_name)
    link_length_array = np.asarray(init_link_length_array(graph), dtype=np.float64)
    pla_full = init_path_link_array(
        graph, k, directed=True, topology_name=topology_name,
        cache_dir="xlron/data/topologies/ksp",
        path_sort_criteria="spectral_resources",
    )
    pla = np.asarray(pla_full)

    # Precompute all path distances and hops
    all_distances = pla.astype(np.float64) @ link_length_array
    all_hops = pla.sum(axis=1)

    num_nodes = graph.number_of_nodes()
    num_pairs = num_nodes * (num_nodes - 1)
    all_distances_by_pair = all_distances.reshape(num_pairs, k)
    all_hops_by_pair = all_hops.reshape(num_pairs, k)
    shortest_dist_by_pair = all_distances_by_pair[:, 0]
    shortest_hops_by_pair = all_hops_by_pair[:, 0]

    # Extract data from trajectory
    path_indices = df["path_indices"].values
    returns = df["returns"].values
    path_lengths = df["path_length"].values
    num_hops = df["num_hops"].values

    # Compute pair index and relative path index
    pair_indices = path_indices // k
    relative_path_indices = path_indices % k

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
    print(f"ACTION PATH RATIO ANALYSIS ({heuristic.upper().replace('_', '-')}, {topology_name}, K={k})")
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
        rpi_thresholds = sorted(set([0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]) & set(range(k)))
        if k - 1 not in rpi_thresholds:
            rpi_thresholds.append(k - 1)
            rpi_thresholds.sort()
        for threshold in rpi_thresholds:
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
    thresholds = sorted(set([5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]) & set(range(1, k + 1)))
    if k not in thresholds:
        thresholds.append(k)
        thresholds.sort()
    for k_thresh in thresholds:
        count = (rpi_acc >= k_thresh).sum()
        print(f"  Actions using path >= {k_thresh:>3}: {count:>7} ({100*count/len(rpi_acc):.2f}%)")

    # Distance distribution of accepted actions (250 km buckets)
    acc_dist = path_lengths[valid][acc]
    acc_hops = num_hops[valid][acc]
    n_acc = len(acc_dist)

    print(f"\n{'='*70}")
    print(f"ACCEPTED PATH DISTANCE DISTRIBUTION (250 km buckets)")
    print(f"{'='*70}")
    dist_bucket_size = 250
    max_dist = int(np.ceil(acc_dist.max() / dist_bucket_size) * dist_bucket_size)
    cum = 0
    print(f"  {'Distance (km)':>20} {'Count':>8} {'%':>7} {'Cumul %':>8}")
    for lo in range(0, max_dist, dist_bucket_size):
        hi = lo + dist_bucket_size
        count = ((acc_dist >= lo) & (acc_dist < hi)).sum()
        cum += count
        if count > 0:
            print(f"  {lo:>8} - {hi:>6}    {count:>8} {100*count/n_acc:>6.2f}% {100*cum/n_acc:>7.1f}%")
    print(f"\n  Summary: Min={acc_dist.min():.0f} km, Max={acc_dist.max():.0f} km, "
          f"Mean={acc_dist.mean():.0f} km, Median={np.median(acc_dist):.0f} km, "
          f"P95={np.percentile(acc_dist, 95):.0f} km, P99={np.percentile(acc_dist, 99):.0f} km")

    # Hop count distribution of accepted actions
    print(f"\n{'='*70}")
    print(f"ACCEPTED PATH HOP COUNT DISTRIBUTION")
    print(f"{'='*70}")
    cum = 0
    print(f"  {'Hops':>6} {'Count':>8} {'%':>7} {'Cumul %':>8}")
    for h in range(1, int(acc_hops.max()) + 1):
        count = (acc_hops == h).sum()
        cum += count
        if count > 0:
            print(f"  {h:>6}   {count:>8} {100*count/n_acc:>6.2f}% {100*cum/n_acc:>7.1f}%")
    print(f"\n  Summary: Min={acc_hops.min():.0f}, Max={acc_hops.max():.0f}, "
          f"Mean={acc_hops.mean():.1f}, Median={np.median(acc_hops):.0f}, "
          f"P95={np.percentile(acc_hops, 95):.0f}, P99={np.percentile(acc_hops, 99):.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze heuristic action path ratios")
    parser.add_argument("--topology", "-t", nargs="+", choices=list(TOPOLOGIES.keys()),
                        default=list(TOPOLOGIES.keys()),
                        help="Topologies to analyze (default: all)")
    parser.add_argument("--heuristic", "-H", nargs="+", choices=HEURISTICS,
                        default=HEURISTICS,
                        help="Heuristics to analyze (default: all)")
    parser.add_argument("--rerun", action="store_true",
                        help="Force re-run heuristic even if trajectory file exists")
    args = parser.parse_args()

    for topo_key in args.topology:
        topo = TOPOLOGIES[topo_key]
        topo_name, k, load = topo["name"], topo["k"], topo["load"]
        for heuristic in args.heuristic:
            traj_file = traj_filename(topo_key, heuristic)
            print(f"\n{'#'*70}")
            print(f"# {topo_name} (K={k}) — {heuristic.upper().replace('_', '-')}, load={load}")
            print(f"{'#'*70}")
            traj_path = Path(traj_file)
            if args.rerun or not traj_path.exists():
                print("Trajectory file not found (or --rerun), running heuristic eval...")
                run_heuristic(topo_name, k, heuristic, load, traj_file)
            else:
                print(f"Using existing trajectory file: {traj_file}")
            analyze(topo_name, k, traj_file, heuristic)
