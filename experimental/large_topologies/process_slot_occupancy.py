"""Compute per-link, per-slot occupancy heatmaps from trajectory data.

For each request the occupied (link, slot) pairs are held for
(departure_time - arrival_time) time units.  The output is a 2-D array
of shape (num_links, num_slots) giving the total occupied time for each
link-slot combination, saved as a compressed .npz alongside a JSON file
with the link labels (nodeA→nodeB notation).
"""

import json
import pathlib

import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent
RESULTS = BASE / "results"

TOPOLOGIES = {
    "tataind": "tataind_directed",
    "usa100": "usa100_directed",
}

METHODS = ["ff_ksp", "transformer"]

NUM_SLOTS = 320  # link_resources used in evaluation


def load_topology_edges(topo_file: str) -> list[tuple[int, int]]:
    path = pathlib.Path(__file__).resolve().parents[1] / ".." / "xlron" / "data" / "topologies" / f"{topo_file}.json"
    with open(path) as f:
        data = json.load(f)
    return [(e["source"], e["target"]) for e in data.get("links", data.get("edges", []))]


def compute_occupancy(traj_path: pathlib.Path, num_links: int) -> np.ndarray:
    """Return array of shape (num_links, NUM_SLOTS) with total occupied time."""
    df = pd.read_csv(traj_path)
    occupancy = np.zeros((num_links, NUM_SLOTS), dtype=np.float64)

    holding_times = (df["departure_time"] - df["arrival_time"]).values
    slot_starts = df["slot_indices"].values.astype(int)
    required_slots = df["required_slots"].values.astype(int)
    path_links_strs = df["path_links"].values

    for i in range(len(df)):
        ht = holding_times[i]
        if ht <= 0:
            continue
        s0 = slot_starts[i]
        ns = required_slots[i]
        pl = path_links_strs[i]
        # Find which links are used
        for link_idx in range(num_links):
            if pl[link_idx] == '1':
                occupancy[link_idx, s0:s0 + ns] += ht

    return occupancy


def compute_link_usage(traj_path: pathlib.Path, num_links: int) -> np.ndarray:
    """Return array of shape (num_links,) with count of requests using each link."""
    df = pd.read_csv(traj_path, usecols=["path_links"])
    counts = np.zeros(num_links, dtype=np.int64)
    for pl in df["path_links"].values:
        for link_idx in range(num_links):
            if pl[link_idx] == '1':
                counts[link_idx] += 1
    return counts


def main():
    for topo_key, topo_file in TOPOLOGIES.items():
        edges = load_topology_edges(topo_file)
        num_links = len(edges)
        labels = [f"{s}->{t}" for s, t in edges]

        out_dir = RESULTS / topo_key
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save link labels once per topology
        with open(out_dir / f"{topo_key}_link_labels.json", "w") as f:
            json.dump(labels, f)

        for method in METHODS:
            traj_path = out_dir / f"{topo_key}_{method}_traj.csv"
            if not traj_path.exists():
                print(f"  SKIP {topo_key}/{method} (no traj)")
                continue

            print(f"  Processing {topo_key} / {method} ...")
            occupancy = compute_occupancy(traj_path, num_links)
            np.savez_compressed(
                out_dir / f"{topo_key}_{method}_slot_occupancy.npz",
                occupancy=occupancy,
            )
            print(f"    -> saved occupancy ({occupancy.shape}, nonzero frac: {(occupancy > 0).mean():.3f})")

            link_usage = compute_link_usage(traj_path, num_links)
            np.save(out_dir / f"{topo_key}_{method}_link_usage.npy", link_usage)
            print(f"    -> saved link_usage ({link_usage.shape})")

    print("\nDone.")


if __name__ == "__main__":
    main()
