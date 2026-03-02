"""Shared configuration for the all-topology capacity bounds survey."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root (XLRON repo)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOPOLOGY_STATS_CSV = PROJECT_ROOT / "experimental" / "benchmarks" / "results" / "topology_stats.csv"

# Traffic model
SHARED_FLAGS = {
    "env_type": "rmsa",
    "link_resources": 320,
    "slot_size": 12.5,
    "guardband": 1,
    "values_bw": "100",
    "mean_service_holding_time": 25,
    "continuous_operation": True,
    "ENV_WARMUP_STEPS": 0,
    "modulations_csv_filepath": str(PROJECT_ROOT / "xlron" / "data" / "modulations" / "modulations_deeprmsa.csv"),
    "path_heuristic": "ksp_ff",
    "k": 50,
}

# Quick scan parameters (for load discovery)
QUICK_SCAN_PARAMS = {
    "NUM_ENVS": 100,
    "TOTAL_TIMESTEPS": 1_300_000,
    "ROLLOUT_LENGTH": 128,
}

# Full heuristic evaluation parameters
FULL_HEURISTIC_PARAMS = {
    "NUM_ENVS": 100,
    "TOTAL_TIMESTEPS": 1_300_000,
    "ROLLOUT_LENGTH": 128,
}

# Cut-set bounds parameters
FULL_CUTSET_PARAMS = {
    "max_requests": 13000,
    "num_trials": 10,
}

# Reconfigurable routing bounds parameters
FULL_RR_PARAMS = {
    "TOTAL_TIMESTEPS": 13000,
    "NUM_ENVS": 1,
    "COMPILE_RR_BOUNDS": True,
}

# K-sensitivity parameters
K_SENSITIVITY_VALUES = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
K_SENSITIVITY_MIN_NODES = 30

# Topologies to exclude
EXCLUDED_TOPOLOGIES = {
    # Disconnected (edge_connectivity=0)
    "cernet_directed",
    "germany50_directed",
    "internet2_directed",
    "italy_directed",
    "loni_directed",
    "renater_directed",
    "sanet_directed",
    # Trivial
    "three_node_chain_directed",
}

# Deduplicate: map duplicates to the canonical name (skip the duplicate)
DUPLICATE_MAP = {
    "york_directed": "tataind_directed",
    "conus6079_directed": "coronet_directed",
    "japan48_directed": "jpn48_directed",
}


def load_topology_stats() -> pd.DataFrame:
    """Load topology stats CSV and return DataFrame."""
    return pd.read_csv(TOPOLOGY_STATS_CSV)


def get_topology_list() -> list[dict]:
    """Return filtered list of directed topologies with their stats.

    Each entry is a dict with keys from the CSV columns.
    Excludes disconnected, trivial, undirected, and duplicate topologies.
    """
    df = load_topology_stats()

    # Keep only directed topologies
    df = df[df["directed"] == True]

    # Exclude bad topologies
    df = df[~df["topology_name"].isin(EXCLUDED_TOPOLOGIES)]

    # Exclude duplicates
    df = df[~df["topology_name"].isin(DUPLICATE_MAP.keys())]

    # Exclude topologies with infinite diameter/path length (disconnected but
    # edge_connectivity might not be exactly 0 due to float issues)
    df = df[np.isfinite(df["avg_path_length"])]
    df = df[np.isfinite(df["diameter"])]

    return df.to_dict("records")


def estimate_initial_load(row: dict) -> float:
    """Estimate load at ~0.1% blocking for the 320-slot RMSA setup.

    Uses topology stats to estimate capacity, accounting for
    distance-dependent modulation and guardband.

    The formula: load ~ C * num_undir_edges * link_resources / (avg_path_length * avg_slots)
    where avg_slots depends on average path distance and modulation format.
    """
    avg_path_distance_km = row["avg_path_length"] * row["avg_link_distance_km"]

    # Estimate average slots per request (including guardband=1)
    # based on modulations_deeprmsa.csv reaches: 16QAM≤625km, 8QAM≤1250km, QPSK≤2500km, BPSK>2500km
    # Slot counts with guardband: 3, 4, 5, 9
    if avg_path_distance_km <= 625:
        avg_slots = 3.0
    elif avg_path_distance_km <= 1250:
        avg_slots = 3.5
    elif avg_path_distance_km <= 2500:
        avg_slots = 4.5
    else:
        avg_slots = 7.0

    num_undir_edges = row["num_edges"] / 2  # directed edges → undirected count
    link_resources = SHARED_FLAGS["link_resources"]

    load = 0.5 * num_undir_edges * link_resources / (row["avg_path_length"] * avg_slots)
    return max(5.0, round(load))


def build_command(
    script: str,
    topology: str,
    extra_flags: dict | None = None,
    output_file: str | None = None,
) -> list[str]:
    """Build a command list for subprocess.run().

    Args:
        script: Either a module path like "xlron.train.train" (run with -m)
                or a file path like "xlron/bounds/reconfigurable_routing_bounds.py".
        topology: Topology name.
        extra_flags: Additional flags to merge with SHARED_FLAGS.
        output_file: Path for --DATA_OUTPUT_FILE.
    """
    cmd = ["uv", "run", "python"]
    if not script.endswith(".py"):
        cmd.extend(["-m", script])
    else:
        cmd.append(script)

    # Merge flags
    flags = dict(SHARED_FLAGS)
    if extra_flags:
        flags.update(extra_flags)

    flags["topology_name"] = topology

    if output_file:
        flags["DATA_OUTPUT_FILE"] = output_file

    for key, val in flags.items():
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}={val}")

    return cmd


def run_command(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a command and return the result. Prints stdout/stderr on failure."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"  FAILED: {' '.join(cmd[:8])}...", file=sys.stderr)
        if result.stderr:
            # Print last 20 lines of stderr
            lines = result.stderr.strip().split("\n")
            for line in lines[-20:]:
                print(f"    {line}", file=sys.stderr)
    return result


def parse_jsonl_blocking(jsonl_path: str | Path) -> list[dict]:
    """Parse a JSONL file and extract load + blocking probability.

    Returns list of dicts with keys: load, blocking_mean, blocking_std.
    """
    results = []
    path = Path(jsonl_path)
    if not path.exists():
        return results

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            config = obj.get("config", {})
            metrics = obj.get("metrics", {})
            bp = metrics.get("service_blocking_probability", {})
            results.append({
                "load": config.get("load", 0),
                "blocking_mean": bp.get("mean", 0),
                "blocking_std": bp.get("std", 0),
            })
    return results


TARGET_BP = 0.001  # 0.1% blocking probability as a fraction


def compute_heuristic_gradient(entry: dict) -> float | None:
    """Compute dBP/dLoad from the heuristic bracket points in load_ranges.json.

    The gradient approximates how quickly blocking changes with load near the
    0.1% transition. Bounds methods are assumed to have a similar gradient,
    just shifted horizontally.
    """
    load_low = entry.get("load_low", 0)
    load_high = entry.get("load_high", 0)
    bp_low = entry.get("bp_low", 0)
    bp_high = entry.get("bp_high", 0)
    if load_high > load_low and bp_high > bp_low:
        return (bp_high - bp_low) / (load_high - load_low)
    return None


def estimate_next_load(
    probe_load: float,
    bp_at_probe: float,
    gradient: float | None,
    target_bp: float = TARGET_BP,
) -> int:
    """Estimate a load on the other side of target_bp from the current probe.

    Uses the heuristic gradient to approximate how much load change is needed
    to cross from the current blocking probability to the target. Applies a
    2x safety factor to overshoot rather than undershoot.
    """
    if bp_at_probe <= 0:
        # Zero blocking - use gentle multiplier to avoid overshooting
        # the narrow transition region
        return round(probe_load * 1.5)

    if gradient is None or gradient <= 0:
        # No usable gradient - use multiplier
        if bp_at_probe < target_bp:
            return round(probe_load * 2)
        else:
            return max(round(probe_load * 0.5), 1)

    if bp_at_probe < target_bp:
        # Below target - need higher load
        delta = (target_bp - bp_at_probe) / gradient
        return max(round(probe_load + delta * 2), round(probe_load * 1.1))
    else:
        # Above target - need lower load
        delta = (bp_at_probe - target_bp) / gradient
        return max(round(probe_load - delta * 2), 1)


def load_load_ranges() -> dict:
    """Load the discovered load ranges from JSON."""
    path = RESULTS_DIR / "load_ranges.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_load_ranges(ranges: dict):
    """Save load ranges to JSON."""
    path = RESULTS_DIR / "load_ranges.json"
    with open(path, "w") as f:
        json.dump(ranges, f, indent=2)


def load_heuristic_selection() -> dict:
    """Load per-topology heuristic selection from JSON.

    Returns dict mapping topology_name -> heuristic name (e.g. "ksp_ff" or "ff_ksp").
    """
    path = RESULTS_DIR / "heuristic_selection.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}
