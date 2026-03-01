"""Phase 6: Analyze results, interpolate to 0.1% blocking, generate summary table and plots.

Reads all JSONL outputs from Phases 2-5, performs interpolation to find
the load at 0.1% blocking for each method, and generates:
- summary_table.csv: One row per topology with loads at 0.1% blocking
- figures/bounds_comparison_scatter.png: Gap analysis
- figures/k_sensitivity_*.png: K-sensitivity plots
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from config import (
    K_SENSITIVITY_MIN_NODES,
    K_SENSITIVITY_VALUES,
    RESULTS_DIR,
    get_topology_list,
    load_topology_stats,
)


TARGET_BP = 0.001  # 0.1% blocking probability (as fraction)


def load_all_jsonl(directory: Path) -> dict[str, list[dict]]:
    """Load all JSONL files from a directory.

    Returns dict mapping topology_name -> list of {load, blocking_mean, blocking_std}.
    """
    results = {}
    if not directory.exists():
        return results

    for jsonl_file in sorted(directory.glob("*.jsonl")):
        topo_name = jsonl_file.stem
        entries = []
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                config = obj.get("config", {})
                metrics = obj.get("metrics", {})
                bp = metrics.get("service_blocking_probability", {})
                entries.append({
                    "load": config.get("load", 0),
                    "blocking_mean": bp.get("mean", 0),
                    "blocking_std": bp.get("std", 0),
                })
        if entries:
            # Sort by load and deduplicate
            entries.sort(key=lambda x: x["load"])
            results[topo_name] = entries
    return results


def find_load_at_blocking(entries: list[dict], target_bp: float = TARGET_BP) -> float | None:
    """Interpolate to find load where blocking probability crosses target_bp.

    Uses scipy.interpolate.interp1d (same approach as summarise_bounds_table.py:find_x_at_y).
    Returns None if interpolation fails.
    """
    if len(entries) < 2:
        return None

    loads = np.array([e["load"] for e in entries])
    bps = np.array([e["blocking_mean"] for e in entries])

    # Filter out zero blocking (can't interpolate with these on log scale)
    valid = bps > 0
    if valid.sum() < 2:
        return None

    loads_valid = loads[valid]
    bps_valid = bps[valid]

    # Check that target is within range
    if target_bp < bps_valid.min() or target_bp > bps_valid.max():
        return None

    try:
        # Interpolate: blocking -> load (inverse function)
        f = interpolate.interp1d(bps_valid, loads_valid, bounds_error=False)
        result = f(target_bp)
        if np.isnan(result):
            return None
        return float(result)
    except Exception:
        return None


def generate_summary_table(
    heuristic_data: dict,
    cutset_data: dict,
    rr_data: dict,
    topologies: list[dict],
) -> pd.DataFrame:
    """Generate the main summary table with interpolated loads at 0.1% blocking."""
    rows = []
    for topo in topologies:
        name = topo["topology_name"]
        row = {
            "topology": name,
            "nodes": topo["num_nodes"],
            "edges": topo["num_edges"],
            "avg_degree": round(topo["avg_degree"], 2),
            "diameter": topo["diameter"],
            "avg_path_length": round(topo["avg_path_length"], 3),
            "avg_link_distance_km": round(topo["avg_link_distance_km"], 1),
        }

        heur_load = None
        if name in heuristic_data:
            heur_load = find_load_at_blocking(heuristic_data[name])
        row["heuristic_load_01pct"] = round(heur_load, 1) if heur_load else None

        cutset_load = None
        if name in cutset_data:
            cutset_load = find_load_at_blocking(cutset_data[name])
        row["cutset_load_01pct"] = round(cutset_load, 1) if cutset_load else None

        rr_load = None
        if name in rr_data:
            rr_load = find_load_at_blocking(rr_data[name])
        row["rr_load_01pct"] = round(rr_load, 1) if rr_load else None

        # Compute gaps
        if heur_load and cutset_load:
            row["gap_cutset_pct"] = round(100 * (cutset_load - heur_load) / heur_load, 1)
        else:
            row["gap_cutset_pct"] = None

        if heur_load and rr_load:
            row["gap_rr_pct"] = round(100 * (rr_load - heur_load) / heur_load, 1)
        else:
            row["gap_rr_pct"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("topology").reset_index(drop=True)
    return df


def plot_gap_scatter(df: pd.DataFrame, figures_dir: Path):
    """Plot gap analysis: scatter of gap vs topology properties."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, gap_col, label in [
        (axes[0], "gap_cutset_pct", "Cut-set bound gap (%)"),
        (axes[1], "gap_rr_pct", "RR bound gap (%)"),
        (axes[2], "gap_cutset_pct", "Cut-set bound gap (%)"),
    ]:
        valid = df.dropna(subset=[gap_col])
        if valid.empty:
            ax.set_title("No data")
            continue

        if ax == axes[0]:
            ax.scatter(valid["nodes"], valid[gap_col], alpha=0.6, s=30)
            ax.set_xlabel("Number of nodes")
            ax.set_ylabel(label)
            ax.set_title("Gap vs Network Size")
        elif ax == axes[1]:
            ax.scatter(valid["avg_degree"], valid[gap_col], alpha=0.6, s=30)
            ax.set_xlabel("Average degree")
            ax.set_ylabel(label)
            ax.set_title("Gap vs Connectivity")
        else:
            ax.scatter(valid["avg_path_length"], valid[gap_col], alpha=0.6, s=30)
            ax.set_xlabel("Average path length (hops)")
            ax.set_ylabel(label)
            ax.set_title("Gap vs Path Length")

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "bounds_gap_scatter.png", dpi=150)
    plt.close()
    print(f"  Saved {figures_dir / 'bounds_gap_scatter.png'}")


def plot_bounds_overview(df: pd.DataFrame, figures_dir: Path):
    """Plot overview: heuristic vs bounds loads for all topologies."""
    valid = df.dropna(subset=["heuristic_load_01pct"])
    if valid.empty:
        print("  No data for bounds overview plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(valid))
    width = 0.25

    ax.bar(x - width, valid["heuristic_load_01pct"], width, label="Heuristic (KSP-FF)", alpha=0.8)
    if valid["cutset_load_01pct"].notna().any():
        ax.bar(x, valid["cutset_load_01pct"], width, label="Cut-set bound", alpha=0.8)
    if valid["rr_load_01pct"].notna().any():
        ax.bar(x + width, valid["rr_load_01pct"], width, label="RR bound", alpha=0.8)

    ax.set_xlabel("Topology")
    ax.set_ylabel("Load at 0.1% blocking (Erlang)")
    ax.set_title("Capacity Comparison Across Topologies")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace("_directed", "") for n in valid["topology"]],
        rotation=90,
        fontsize=6,
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(figures_dir / "bounds_overview.png", dpi=150)
    plt.close()
    print(f"  Saved {figures_dir / 'bounds_overview.png'}")


def load_k_sensitivity_data(k_sens_dir: Path) -> dict[str, list[dict]]:
    """Load K-sensitivity results.

    Returns dict mapping topology -> list of {k, blocking_mean, blocking_std}.
    """
    results = {}
    if not k_sens_dir.exists():
        return results

    for jsonl_file in sorted(k_sens_dir.glob("*.jsonl")):
        topo_name = jsonl_file.stem
        entries = []
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                config = obj.get("config", {})
                metrics = obj.get("metrics", {})
                bp = metrics.get("service_blocking_probability", {})
                entries.append({
                    "k": config.get("k", 0),
                    "blocking_mean": bp.get("mean", 0),
                    "blocking_std": bp.get("std", 0),
                })
        if entries:
            entries.sort(key=lambda x: x["k"])
            results[topo_name] = entries
    return results


def plot_k_sensitivity(k_data: dict, figures_dir: Path):
    """Generate K-sensitivity plots."""
    if not k_data:
        print("  No K-sensitivity data to plot")
        return

    # Plot 1: Individual topology curves
    fig, ax = plt.subplots(figsize=(12, 8))
    for topo_name, entries in sorted(k_data.items()):
        ks = [e["k"] for e in entries]
        bps = [e["blocking_mean"] * 100 for e in entries]  # convert to %
        label = topo_name.replace("_directed", "")
        ax.plot(ks, bps, marker="o", markersize=3, label=label, alpha=0.7)

    ax.set_xlabel("K (number of shortest paths)")
    ax.set_ylabel("Service Blocking Probability (%)")
    ax.set_title("K-Sensitivity: Blocking vs Number of Paths")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Only show legend if reasonable number of topologies
    if len(k_data) <= 20:
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(figures_dir / "k_sensitivity_absolute.png", dpi=150)
    plt.close()
    print(f"  Saved {figures_dir / 'k_sensitivity_absolute.png'}")

    # Plot 2: Normalized to K=50 value
    fig, ax = plt.subplots(figsize=(12, 8))
    for topo_name, entries in sorted(k_data.items()):
        ks = [e["k"] for e in entries]
        bps = [e["blocking_mean"] for e in entries]

        # Find K=50 value for normalization
        k50_bp = None
        for e in entries:
            if e["k"] == 50:
                k50_bp = e["blocking_mean"]
                break

        if k50_bp is None or k50_bp == 0:
            continue

        normalized = [bp / k50_bp for bp in bps]
        label = topo_name.replace("_directed", "")
        ax.plot(ks, normalized, marker="o", markersize=3, label=label, alpha=0.7)

    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="K=50 baseline")
    ax.set_xlabel("K (number of shortest paths)")
    ax.set_ylabel("Blocking Probability (normalized to K=50)")
    ax.set_title("K-Sensitivity: Normalized Blocking vs Number of Paths")
    ax.grid(True, alpha=0.3)

    if len(k_data) <= 20:
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(figures_dir / "k_sensitivity_normalized.png", dpi=150)
    plt.close()
    print(f"  Saved {figures_dir / 'k_sensitivity_normalized.png'}")

    # Plot 3: Median curve with IQR band
    fig, ax = plt.subplots(figsize=(10, 6))
    all_normalized = {}
    for entries in k_data.values():
        k50_bp = None
        for e in entries:
            if e["k"] == 50:
                k50_bp = e["blocking_mean"]
                break
        if k50_bp is None or k50_bp == 0:
            continue
        for e in entries:
            k = e["k"]
            norm = e["blocking_mean"] / k50_bp
            all_normalized.setdefault(k, []).append(norm)

    if all_normalized:
        ks_sorted = sorted(all_normalized.keys())
        medians = [np.median(all_normalized[k]) for k in ks_sorted]
        q25 = [np.percentile(all_normalized[k], 25) for k in ks_sorted]
        q75 = [np.percentile(all_normalized[k], 75) for k in ks_sorted]

        ax.plot(ks_sorted, medians, "o-", color="steelblue", linewidth=2, label="Median")
        ax.fill_between(ks_sorted, q25, q75, alpha=0.2, color="steelblue", label="IQR")
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("K (number of shortest paths)")
        ax.set_ylabel("Blocking Probability (normalized to K=50)")
        ax.set_title(f"K-Sensitivity Summary ({len(k_data)} topologies)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "k_sensitivity_summary.png", dpi=150)
    plt.close()
    print(f"  Saved {figures_dir / 'k_sensitivity_summary.png'}")


def main():
    print("=" * 60)
    print("Phase 6: Analyzing results")
    print("=" * 60)

    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    topologies = get_topology_list()

    # Load all results
    print("Loading results...")
    heuristic_data = load_all_jsonl(RESULTS_DIR / "heuristic_eval")
    cutset_data = load_all_jsonl(RESULTS_DIR / "cutset_bounds")
    rr_data = load_all_jsonl(RESULTS_DIR / "rr_bounds")

    print(f"  Heuristic: {len(heuristic_data)} topologies")
    print(f"  Cut-set:   {len(cutset_data)} topologies")
    print(f"  RR bounds: {len(rr_data)} topologies")

    # Generate summary table
    print("\nGenerating summary table...")
    df = generate_summary_table(heuristic_data, cutset_data, rr_data, topologies)
    table_path = RESULTS_DIR / "summary_table.csv"
    df.to_csv(table_path, index=False)
    print(f"  Saved {table_path}")

    # Print summary statistics
    valid_heur = df["heuristic_load_01pct"].notna().sum()
    valid_cutset = df["cutset_load_01pct"].notna().sum()
    valid_rr = df["rr_load_01pct"].notna().sum()
    print(f"\n  Topologies with valid 0.1% blocking interpolation:")
    print(f"    Heuristic: {valid_heur}/{len(df)}")
    print(f"    Cut-set:   {valid_cutset}/{len(df)}")
    print(f"    RR bounds: {valid_rr}/{len(df)}")

    if df["gap_cutset_pct"].notna().any():
        print(f"\n  Cut-set gap statistics:")
        print(f"    Mean:   {df['gap_cutset_pct'].mean():.1f}%")
        print(f"    Median: {df['gap_cutset_pct'].median():.1f}%")
        print(f"    Min:    {df['gap_cutset_pct'].min():.1f}%")
        print(f"    Max:    {df['gap_cutset_pct'].max():.1f}%")

    if df["gap_rr_pct"].notna().any():
        print(f"\n  RR bound gap statistics:")
        print(f"    Mean:   {df['gap_rr_pct'].mean():.1f}%")
        print(f"    Median: {df['gap_rr_pct'].median():.1f}%")
        print(f"    Min:    {df['gap_rr_pct'].min():.1f}%")
        print(f"    Max:    {df['gap_rr_pct'].max():.1f}%")

    # Generate plots
    print("\nGenerating plots...")
    plot_bounds_overview(df, figures_dir)
    plot_gap_scatter(df, figures_dir)

    # K-sensitivity analysis
    print("\nK-sensitivity analysis...")
    k_data = load_k_sensitivity_data(RESULTS_DIR / "k_sensitivity")
    print(f"  K-sensitivity data: {len(k_data)} topologies")
    plot_k_sensitivity(k_data, figures_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
