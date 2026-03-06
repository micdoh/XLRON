"""Phase 6: Analyze results, interpolate to 0.1% blocking, generate summary table and plots.

Reads all JSONL outputs from Phases 1-4, performs interpolation to find
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

# plot_style lives one directory up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_style import PALETTE, REFERENCE_LINE_COLOR, configure_style

from config import (
    K_SENSITIVITY_MIN_NODES,
    K_SENSITIVITY_VALUES,
    RESULTS_DIR,
    get_topology_list,
    load_heuristic_selection,
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
            # Sort by load and deduplicate (keep last entry per load)
            entries.sort(key=lambda x: x["load"])
            seen = {}
            for e in entries:
                seen[e["load"]] = e
            entries = sorted(seen.values(), key=lambda x: x["load"])
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
    heur_selection = load_heuristic_selection()
    rows = []
    for topo in topologies:
        name = topo["topology_name"]
        row = {
            "topology": name,
            "heuristic_used": heur_selection.get(name, "ksp_ff"),
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

    scatter_configs = [
        (axes[0], "gap_cutset_pct", "Cut-set bound gap (%)", "nodes", "Number of nodes", "Gap vs Network Size"),
        (axes[1], "gap_rr_pct", "RR bound gap (%)", "avg_degree", "Average degree", "Gap vs Connectivity"),
        (axes[2], "gap_cutset_pct", "Cut-set bound gap (%)", "avg_path_length", "Average path length (hops)", "Gap vs Path Length"),
    ]

    for i, (ax, gap_col, ylabel, xcol, xlabel, title) in enumerate(scatter_configs):
        valid = df.dropna(subset=[gap_col])
        if valid.empty:
            ax.set_title("No data")
            continue

        ax.scatter(valid[xcol], valid[gap_col], alpha=0.7, s=60,
                   color=PALETTE[i], edgecolor="white", linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(figures_dir / "bounds_gap_scatter.png")
    plt.close()
    print(f"  Saved {figures_dir / 'bounds_gap_scatter.png'}")


def plot_bounds_overview(df: pd.DataFrame, figures_dir: Path):
    """Plot overview: heuristic vs bounds loads for all topologies."""
    valid = df.dropna(subset=["heuristic_load_01pct"])
    if valid.empty:
        print("  No data for bounds overview plot")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(valid))
    width = 0.25

    ax.bar(x - width, valid["heuristic_load_01pct"], width,
           label="Heuristic (best)", color=PALETTE[0], edgecolor="white", linewidth=0.5)
    if valid["cutset_load_01pct"].notna().any():
        ax.bar(x, valid["cutset_load_01pct"], width,
               label="Cut-set bound", color=PALETTE[1], edgecolor="white", linewidth=0.5)
    if valid["rr_load_01pct"].notna().any():
        ax.bar(x + width, valid["rr_load_01pct"], width,
               label="RR bound", color=PALETTE[2], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Topology")
    ax.set_ylabel("Load at 0.1% blocking (Erlang)")
    ax.set_title("Capacity Comparison Across Topologies")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace("_directed", "") for n in valid["topology"]],
        rotation=90,
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "bounds_overview.png")
    plt.close()
    print(f"  Saved {figures_dir / 'bounds_overview.png'}")


def plot_bounds_overview_normalized(df: pd.DataFrame, figures_dir: Path, sort_by: str = "topology"):
    """Plot normalized bounds overview: bars show % difference from heuristic load.

    Args:
        sort_by: "topology" to sort alphabetically, "edges" to sort by num edges.
    """
    # Need heuristic + at least one bound
    valid = df.dropna(subset=["heuristic_load_01pct"]).copy()
    has_cutset = valid["cutset_load_01pct"].notna()
    has_rr = valid["rr_load_01pct"].notna()
    valid = valid[has_cutset | has_rr]
    if valid.empty:
        print(f"  No data for normalized bounds overview ({sort_by})")
        return

    if sort_by == "edges":
        valid = valid.sort_values("edges").reset_index(drop=True)
        suffix = "by_edges"
    else:
        valid = valid.sort_values("topology").reset_index(drop=True)
        suffix = "by_name"

    # Compute normalized gaps (%)
    cutset_gap = (valid["cutset_load_01pct"] - valid["heuristic_load_01pct"]) / valid["heuristic_load_01pct"] * 100
    rr_gap = (valid["rr_load_01pct"] - valid["heuristic_load_01pct"]) / valid["heuristic_load_01pct"] * 100

    fig, ax = plt.subplots(figsize=(max(14, len(valid) * 0.4), 8))

    x = np.arange(len(valid))
    width = 0.35

    if cutset_gap.notna().any():
        ax.bar(x - width / 2, cutset_gap, width,
               label="Cut-set bound", color=PALETTE[1], edgecolor="white", linewidth=0.5)
    if rr_gap.notna().any():
        ax.bar(x + width / 2, rr_gap, width,
               label="Resource-prioritized defragmentation", color=PALETTE[2], edgecolor="white", linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Topology")
    ax.set_ylabel("Load difference from heuristic (%)")
    order_label = "(by number of edges)" if sort_by == "edges" else "(alphabetical)"
    ax.set_title(f"Bound Gap Relative to Best Heuristic {order_label}")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace("_directed", "") for n in valid["topology"]],
        rotation=75,
        ha="right",
        fontsize=8,
    )
    ax.legend()

    plt.tight_layout()
    fname = f"bounds_overview_normalized_{suffix}.png"
    plt.savefig(figures_dir / fname)
    plt.close()
    print(f"  Saved {figures_dir / fname}")


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
    sorted_topos = sorted(k_data.items())
    for i, (topo_name, entries) in enumerate(sorted_topos):
        ks = [e["k"] for e in entries]
        bps = [e["blocking_mean"] * 100 for e in entries]  # convert to %
        label = topo_name.replace("_directed", "")
        ax.plot(ks, bps, marker="o", label=label, alpha=0.7,
                color=PALETTE[i % len(PALETTE)])

    ax.set_xlabel("K (number of shortest paths)")
    ax.set_ylabel("Service Blocking Probability (%)")
    ax.set_title("K-Sensitivity: Blocking vs Number of Paths")
    ax.set_yscale("log")

    if len(k_data) <= 20:
        ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(figures_dir / "k_sensitivity_absolute.png")
    plt.close()
    print(f"  Saved {figures_dir / 'k_sensitivity_absolute.png'}")

    # Plot 2: Normalized to K=50 value
    fig, ax = plt.subplots(figsize=(12, 8))
    color_idx = 0
    for topo_name, entries in sorted_topos:
        ks = [e["k"] for e in entries]
        bps = [e["blocking_mean"] for e in entries]

        k50_bp = None
        for e in entries:
            if e["k"] == 50:
                k50_bp = e["blocking_mean"]
                break

        if k50_bp is None or k50_bp == 0:
            continue

        normalized = [bp / k50_bp for bp in bps]
        label = topo_name.replace("_directed", "")
        ax.plot(ks, normalized, marker="o", label=label, alpha=0.7,
                color=PALETTE[color_idx % len(PALETTE)])
        color_idx += 1

    ax.axhline(y=1.0, color=REFERENCE_LINE_COLOR, linestyle="--", alpha=0.7,
               label="K=50 baseline")
    ax.set_xlabel("K (number of shortest paths)")
    ax.set_ylabel("Blocking Probability (normalized to K=50)")
    ax.set_title("K-Sensitivity: Normalized Blocking vs Number of Paths")

    if len(k_data) <= 20:
        ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(figures_dir / "k_sensitivity_normalized.png")
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

        ax.plot(ks_sorted, medians, "o-", color=PALETTE[0], label="Median")
        ax.fill_between(ks_sorted, q25, q75, alpha=0.2, color=PALETTE[0], label="IQR")
        ax.axhline(y=1.0, color=REFERENCE_LINE_COLOR, linestyle="--", alpha=0.7)
        ax.set_xlabel("K (number of shortest paths)")
        ax.set_ylabel("Blocking Probability (normalized to K=50)")
        ax.set_title(f"K-Sensitivity Summary ({len(k_data)} topologies)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "k_sensitivity_summary.png")
    plt.close()
    print(f"  Saved {figures_dir / 'k_sensitivity_summary.png'}")


def plot_k_best_bar(k_data: dict, figures_dir: Path):
    """Bar chart showing the value of K that gives lowest mean blocking per topology."""
    if not k_data:
        print("  No K-sensitivity data for best-K bar chart")
        return

    topos = []
    best_ks = []
    for topo_name in sorted(k_data.keys()):
        entries = k_data[topo_name]
        if not entries:
            continue
        best = min(entries, key=lambda e: e["blocking_mean"])
        topos.append(topo_name.replace("_directed", ""))
        best_ks.append(best["k"])

    if not topos:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(topos) * 0.5), 6))
    x = np.arange(len(topos))
    ax.bar(x, best_ks, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Topology")
    ax.set_ylabel("Best K (lowest blocking)")
    ax.set_title("Optimal K per Topology")
    ax.set_xticks(x)
    ax.set_xticklabels(topos, rotation=75, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / "k_sensitivity_best_k.png")
    plt.close()
    print(f"  Saved {figures_dir / 'k_sensitivity_best_k.png'}")


def plot_k_best_improvement(k_data: dict, figures_dir: Path):
    """Bar chart showing % blocking reduction at optimal K vs K=50 per topology."""
    if not k_data:
        print("  No K-sensitivity data for best-K improvement chart")
        return

    topos = []
    improvements = []
    for topo_name in sorted(k_data.keys()):
        entries = k_data[topo_name]
        if not entries:
            continue

        k50_bp = None
        for e in entries:
            if e["k"] == 50:
                k50_bp = e["blocking_mean"]
                break
        if k50_bp is None or k50_bp == 0:
            continue

        best = min(entries, key=lambda e: e["blocking_mean"])
        pct_diff = (best["blocking_mean"] - k50_bp) / k50_bp * 100
        topos.append(topo_name.replace("_directed", ""))
        improvements.append(pct_diff)

    if not topos:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(topos) * 0.5), 6))
    x = np.arange(len(topos))
    colors = [PALETTE[0] if v <= 0 else PALETTE[1] for v in improvements]
    ax.bar(x, improvements, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Topology")
    ax.set_ylabel("Blocking change vs K=50 (%)")
    ax.set_title("Blocking Improvement at Optimal K Relative to K=50")
    ax.set_xticks(x)
    ax.set_xticklabels(topos, rotation=75, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / "k_sensitivity_best_k_improvement.png")
    plt.close()
    print(f"  Saved {figures_dir / 'k_sensitivity_best_k_improvement.png'}")


def print_heuristic_selection_table(df: pd.DataFrame):
    """Print a detailed table of heuristic selections."""
    sel = df[["topology", "heuristic_used", "nodes", "edges", "avg_degree",
              "heuristic_load_01pct"]].copy()
    sel = sel.sort_values("topology")

    ff_ksp = sel[sel["heuristic_used"] == "ff_ksp"]
    ksp_ff = sel[sel["heuristic_used"] == "ksp_ff"]

    print(f"\n  Heuristic selection breakdown:")
    print(f"    KSP-FF: {len(ksp_ff)} topologies")
    print(f"    FF-KSP: {len(ff_ksp)} topologies")

    if not ff_ksp.empty:
        print(f"\n  Topologies where FF-KSP is superior:")
        print(f"    {'Topology':<40s} {'Nodes':>5s} {'Edges':>5s} {'Deg':>5s} {'Load@0.1%':>10s}")
        print(f"    {'-'*40} {'-'*5} {'-'*5} {'-'*5} {'-'*10}")
        for _, row in ff_ksp.iterrows():
            name = row["topology"].replace("_directed", "")
            load_str = f"{row['heuristic_load_01pct']:.1f}" if pd.notna(row["heuristic_load_01pct"]) else "N/A"
            print(f"    {name:<40s} {row['nodes']:>5d} {row['edges']:>5d} "
                  f"{row['avg_degree']:>5.1f} {load_str:>10s}")

    if not ff_ksp.empty:
        print(f"\n  FF-KSP topology stats (vs KSP-FF topologies):")
        for col, label in [("nodes", "Nodes"), ("avg_degree", "Avg degree")]:
            ff_med = ff_ksp[col].median()
            ksp_med = ksp_ff[col].median() if not ksp_ff.empty else float("nan")
            print(f"    {label}: FF-KSP median={ff_med:.1f}, KSP-FF median={ksp_med:.1f}")


def plot_heuristic_selection(df: pd.DataFrame, figures_dir: Path):
    """Plot heuristic selection summary: bar chart + topology property comparison."""
    ff_ksp = df[df["heuristic_used"] == "ff_ksp"]
    ksp_ff = df[df["heuristic_used"] == "ksp_ff"]
    n_ff = len(ff_ksp)
    n_ksp = len(ksp_ff)

    color_ksp = PALETTE[0]  # teal
    color_ff = PALETTE[1]   # coral

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [1, 1.5, 1.5]})

    # Panel 1: Count bar chart
    ax = axes[0]
    bars = ax.bar(["KSP-FF", "FF-KSP"], [n_ksp, n_ff],
                  color=[color_ksp, color_ff], edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, [n_ksp, n_ff]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Best Heuristic Selection")
    ax.set_ylim(0, max(n_ksp, n_ff, 1) * 1.15)

    # Panel 2: Node count distribution by heuristic
    ax = axes[1]
    bins = np.arange(0, df["nodes"].max() + 10, 10)
    if not ksp_ff.empty:
        ax.hist(ksp_ff["nodes"], bins=bins, alpha=0.6, color=color_ksp,
                label="KSP-FF", edgecolor="white", linewidth=0.5)
    if not ff_ksp.empty:
        ax.hist(ff_ksp["nodes"], bins=bins, alpha=0.6, color=color_ff,
                label="FF-KSP", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of nodes")
    ax.set_title("Node Count Distribution by Heuristic")
    ax.legend()

    # Panel 3: Avg degree distribution by heuristic
    ax = axes[2]
    bins_deg = np.arange(0, df["avg_degree"].max() + 1, 0.5)
    if not ksp_ff.empty:
        ax.hist(ksp_ff["avg_degree"], bins=bins_deg, alpha=0.6, color=color_ksp,
                label="KSP-FF", edgecolor="white", linewidth=0.5)
    if not ff_ksp.empty:
        ax.hist(ff_ksp["avg_degree"], bins=bins_deg, alpha=0.6, color=color_ff,
                label="FF-KSP", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Average node degree")
    ax.set_title("Avg Degree Distribution by Heuristic")
    ax.legend()

    # Shared y-axis label
    fig.supylabel("Number of topologies")

    plt.tight_layout()
    plt.savefig(figures_dir / "heuristic_selection.png")
    plt.close()
    print(f"  Saved {figures_dir / 'heuristic_selection.png'}")


def main():
    print("=" * 60)
    print("Phase 6: Analyzing results")
    print("=" * 60)

    # Apply publication style (smaller sizes for multi-panel dashboard figures)
    configure_style(
        font_size=16,
        axes_label_size=18,
        tick_size=14,
        legend_size=13,
    )

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

    # Heuristic selection analysis
    print_heuristic_selection_table(df)

    # Generate plots
    print("\nGenerating plots...")
    plot_bounds_overview(df, figures_dir)
    plot_bounds_overview_normalized(df, figures_dir, sort_by="topology")
    plot_bounds_overview_normalized(df, figures_dir, sort_by="edges")
    plot_gap_scatter(df, figures_dir)
    plot_heuristic_selection(df, figures_dir)

    # K-sensitivity analysis
    print("\nK-sensitivity analysis...")
    k_data = load_k_sensitivity_data(RESULTS_DIR / "k_sensitivity")
    print(f"  K-sensitivity data: {len(k_data)} topologies")
    plot_k_sensitivity(k_data, figures_dir)
    plot_k_best_bar(k_data, figures_dir)
    plot_k_best_improvement(k_data, figures_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
