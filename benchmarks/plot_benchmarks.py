#!/usr/bin/env python3
"""Generate benchmark plots from aggregated results.

Usage:
    python benchmarks/plot_benchmarks.py --input=benchmarks/results/benchmark_results.csv
    python benchmarks/plot_benchmarks.py --input=benchmarks/results/benchmark_results.csv --plots=fps_vs_num_envs,cpu_vs_gpu
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from plot_style import (
    BAND_COLORS,
    BAND_DISPLAY,
    DEVICE_COLORS,
    ENV_TYPE_COLORS,
    ENV_TYPE_DISPLAY,
    TOPOLOGY_COLORS,
    TOPOLOGY_DISPLAY,
    configure_style,
    format_fps,
    get_topology_order,
)


def _filter_group(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Filter dataframe to rows whose 'file' column starts with prefix."""
    return df[df["file"].str.startswith(prefix)].copy()


def _save(fig: plt.Figure, output_dir: Path, name: str):
    """Save figure as both PDF and PNG."""
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# -- Plot 1: FPS vs NUM_ENVS (log-log) ---------------------------------------


def plot_fps_vs_num_envs(df: pd.DataFrame, output_dir: Path):
    """Log-log plot of FPS vs NUM_ENVS, faceted by env_type."""
    data = _filter_group(df, "num_envs_")
    if data.empty:
        print("  No data for num_envs group")
        return

    env_types = sorted(data["config_env_type"].unique())
    fig, axes = plt.subplots(1, len(env_types), figsize=(14 * len(env_types), 10))
    if len(env_types) == 1:
        axes = [axes]

    for ax, env_type in zip(axes, env_types):
        subset = data[data["config_env_type"] == env_type].sort_values("config_NUM_ENVS")
        color = ENV_TYPE_COLORS.get(env_type, "black")
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)

        ax.plot(
            subset["config_NUM_ENVS"], subset["timing_fps"], marker="o", color=color, label=display
        )

        # Ideal linear scaling reference
        if len(subset) > 0:
            x0 = subset["config_NUM_ENVS"].iloc[0]
            y0 = subset["timing_fps"].iloc[0]
            x_range = subset["config_NUM_ENVS"].values
            ax.plot(
                x_range, y0 * (x_range / x0), "--", color="gray", alpha=0.5, label="Linear scaling"
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("NUM_ENVS")
        ax.set_ylabel("Frames per Second (FPS)")
        ax.set_title(display)
        ax.legend()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    _save(fig, output_dir, "fps_vs_num_envs")


# -- Plot 2: FPS vs Link Resources -------------------------------------------


def plot_fps_vs_link_resources(df: pd.DataFrame, output_dir: Path):
    """FPS vs link_resources, faceted by env_type."""
    data = _filter_group(df, "link_resources_")
    if data.empty:
        print("  No data for link_resources group")
        return

    env_types = sorted(data["config_env_type"].unique())
    fig, axes = plt.subplots(1, len(env_types), figsize=(14 * len(env_types), 10))
    if len(env_types) == 1:
        axes = [axes]

    for ax, env_type in zip(axes, env_types):
        subset = data[data["config_env_type"] == env_type].sort_values("config_link_resources")
        color = ENV_TYPE_COLORS.get(env_type, "black")
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)

        ax.plot(subset["config_link_resources"], subset["timing_fps"], marker="o", color=color)
        ax.set_xlabel("Link Resources (slots)")
        ax.set_ylabel("FPS")
        ax.set_title(display)

    fig.tight_layout()
    _save(fig, output_dir, "fps_vs_link_resources")


# -- Plot 3: FPS vs K-paths --------------------------------------------------


def plot_fps_vs_k(df: pd.DataFrame, output_dir: Path):
    """FPS vs K, faceted by env_type."""
    data = _filter_group(df, "k_paths_")
    if data.empty:
        print("  No data for k_paths group")
        return

    env_types = sorted(data["config_env_type"].unique())
    fig, axes = plt.subplots(1, len(env_types), figsize=(14 * len(env_types), 10))
    if len(env_types) == 1:
        axes = [axes]

    for ax, env_type in zip(axes, env_types):
        subset = data[data["config_env_type"] == env_type].sort_values("config_k")
        color = ENV_TYPE_COLORS.get(env_type, "black")
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)

        ax.plot(subset["config_k"], subset["timing_fps"], marker="o", color=color)
        ax.set_xlabel("K (shortest paths)")
        ax.set_ylabel("FPS")
        ax.set_title(display)

    fig.tight_layout()
    _save(fig, output_dir, "fps_vs_k")


# -- Plot 4: FPS vs Topology Size (scatter) -----------------------------------


def plot_fps_vs_topology(
    df: pd.DataFrame, output_dir: Path, topo_stats: pd.DataFrame | None = None
):
    """Scatter of FPS vs num_nodes, colored by env_type."""
    data = _filter_group(df, "topology_")
    if data.empty:
        print("  No data for topology group")
        return

    if topo_stats is None:
        topo_stats_path = Path("benchmarks/topology_stats.csv")
        if not topo_stats_path.exists():
            print("  No topology_stats.csv found, skipping topology plot")
            return
        topo_stats = pd.read_csv(topo_stats_path)

    merged = data.merge(
        topo_stats, left_on="config_topology_name", right_on="topology_name", how="left"
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    for env_type in sorted(merged["config_env_type"].unique()):
        subset = merged[merged["config_env_type"] == env_type].sort_values("num_nodes")
        color = ENV_TYPE_COLORS.get(env_type, "black")
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)

        ax.scatter(
            subset["num_nodes"], subset["timing_fps"], c=color, label=display, s=120, zorder=5
        )

        for _, row in subset.iterrows():
            topo_label = TOPOLOGY_DISPLAY.get(row["config_topology_name"], "")
            ax.annotate(
                topo_label,
                (row["num_nodes"], row["timing_fps"]),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=14,
            )

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("FPS")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "fps_vs_topology_size")


# -- Plot 5: Compilation Time vs NUM_ENVS ------------------------------------


def plot_compilation_time(df: pd.DataFrame, output_dir: Path):
    """Compilation time vs NUM_ENVS."""
    data = _filter_group(df, "num_envs_")
    if data.empty or "timing_compilation_time_s" not in data.columns:
        print("  No compilation time data")
        return

    env_types = sorted(data["config_env_type"].unique())
    fig, ax = plt.subplots(figsize=(14, 10))

    for env_type in env_types:
        subset = data[data["config_env_type"] == env_type].sort_values("config_NUM_ENVS")
        color = ENV_TYPE_COLORS.get(env_type, "black")
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)

        ax.plot(
            subset["config_NUM_ENVS"],
            subset["timing_compilation_time_s"],
            marker="o",
            color=color,
            label=display,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("NUM_ENVS")
    ax.set_ylabel("Compilation Time (s)")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig.tight_layout()
    _save(fig, output_dir, "compilation_time_vs_num_envs")


# -- Plot 6: CPU vs GPU Speedup ----------------------------------------------


def plot_cpu_vs_gpu(df: pd.DataFrame, output_dir: Path):
    """GPU speedup over CPU at different NUM_ENVS."""
    data = _filter_group(df, "device_")
    if data.empty:
        print("  No data for device group")
        return

    # Separate CPU and GPU runs
    cpu_data = data[data["file"].str.startswith("device_cpu_")]
    gpu_data = data[data["file"].str.startswith("device_gpu_")]

    env_types = sorted(data["config_env_type"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(28, 10))

    # Left: raw FPS comparison
    ax = axes[0]
    for env_type in env_types:
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)
        for device, dev_data, color, ls in [
            ("CPU", cpu_data, DEVICE_COLORS["cpu"], "--"),
            ("GPU", gpu_data, DEVICE_COLORS["gpu"], "-"),
        ]:
            subset = dev_data[dev_data["config_env_type"] == env_type].sort_values(
                "config_NUM_ENVS"
            )
            if subset.empty:
                continue
            ax.plot(
                subset["config_NUM_ENVS"],
                subset["timing_fps"],
                marker="o",
                color=color,
                linestyle=ls,
                label=f"{display} ({device})",
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("NUM_ENVS")
    ax.set_ylabel("FPS")
    ax.set_title("Raw Throughput")
    ax.legend(fontsize=16)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Right: speedup
    ax = axes[1]
    for env_type in env_types:
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)
        color = ENV_TYPE_COLORS.get(env_type, "black")

        cpu_subset = cpu_data[cpu_data["config_env_type"] == env_type].sort_values(
            "config_NUM_ENVS"
        )
        gpu_subset = gpu_data[gpu_data["config_env_type"] == env_type].sort_values(
            "config_NUM_ENVS"
        )

        if cpu_subset.empty or gpu_subset.empty:
            continue

        merged = cpu_subset.merge(gpu_subset, on="config_NUM_ENVS", suffixes=("_cpu", "_gpu"))
        if merged.empty:
            continue

        speedup = merged["timing_fps_gpu"] / merged["timing_fps_cpu"]
        ax.plot(merged["config_NUM_ENVS"], speedup, marker="s", color=color, label=display)

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Parity")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NUM_ENVS")
    ax.set_ylabel("GPU Speedup (x)")
    ax.set_title("GPU / CPU Speedup")
    ax.legend(fontsize=16)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    _save(fig, output_dir, "cpu_vs_gpu")


# -- Plot 7: GN Model Band Scaling -------------------------------------------


def plot_gn_band_scaling(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart of FPS for C / C+L / C+L+S bands."""
    data = _filter_group(df, "gn_bands_")
    if data.empty:
        print("  No data for gn_bands group")
        return

    env_types = sorted(data["config_env_type"].unique())
    bands = ["C", "C,L", "C,L,S"]

    fig, ax = plt.subplots(figsize=(14, 10))
    x = np.arange(len(bands))
    width = 0.35
    offsets = np.linspace(-width / 2, width / 2, len(env_types))

    for i, env_type in enumerate(env_types):
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)
        color = ENV_TYPE_COLORS.get(env_type, "black")
        fps_values = []
        for band in bands:
            subset = data[
                (data["config_env_type"] == env_type) & (data["config_band_preference"] == band)
            ]
            fps_values.append(subset["timing_fps"].iloc[0] if not subset.empty else 0)

        ax.bar(x + offsets[i], fps_values, width, label=display, color=color)

    ax.set_xlabel("Band Configuration")
    ax.set_ylabel("FPS")
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_DISPLAY.get(b, b) for b in bands])
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "gn_band_scaling")


# -- Plot 8: Heatmap (topology x NUM_ENVS) -----------------------------------


def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    """Heatmap of FPS with topology on y-axis and NUM_ENVS on x-axis.

    Uses data from the num_envs group (single topology) and topology group
    (single NUM_ENVS). For a full heatmap, a dedicated cross-sweep is needed.
    This plot uses the topology group data (all topos at NUM_ENVS=64).
    """
    data = _filter_group(df, "topology_")
    if data.empty:
        print("  No data for heatmap")
        return

    # Use only one env type for the heatmap
    for env_type in ["rwa", "rmsa"]:
        subset = data[data["config_env_type"] == env_type]
        if subset.empty:
            continue

        display = ENV_TYPE_DISPLAY.get(env_type, env_type)
        topo_order = [t for t in get_topology_order() if t in subset["config_topology_name"].values]
        topo_labels = [TOPOLOGY_DISPLAY.get(t, t) for t in topo_order]
        fps_values = []
        for topo in topo_order:
            row = subset[subset["config_topology_name"] == topo]
            fps_values.append(row["timing_fps"].iloc[0] if not row.empty else 0)

        fig, ax = plt.subplots(figsize=(10, 8))
        fps_arr = np.array(fps_values).reshape(-1, 1)
        im = ax.imshow(fps_arr, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(topo_labels)))
        ax.set_yticklabels(topo_labels)
        ax.set_xticks([0])
        ax.set_xticklabels(["NUM_ENVS=64"])

        # Annotate cells with FPS values
        for i, fps in enumerate(fps_values):
            ax.text(
                0,
                i,
                format_fps(fps),
                ha="center",
                va="center",
                fontsize=18,
                color="white" if fps > max(fps_values) * 0.6 else "black",
            )

        cbar = fig.colorbar(im, ax=ax, label="FPS")
        ax.set_title(f"{display} Throughput by Topology")
        fig.tight_layout()
        _save(fig, output_dir, f"heatmap_{env_type}")


# -- Plot 9: Scaling Efficiency -----------------------------------------------


def plot_scaling_efficiency(df: pd.DataFrame, output_dir: Path):
    """FPS(N) / (N * FPS(1)) showing parallelization efficiency."""
    data = _filter_group(df, "num_envs_")
    if data.empty:
        print("  No data for scaling efficiency")
        return

    env_types = sorted(data["config_env_type"].unique())
    fig, ax = plt.subplots(figsize=(14, 10))

    for env_type in env_types:
        subset = data[data["config_env_type"] == env_type].sort_values("config_NUM_ENVS")
        color = ENV_TYPE_COLORS.get(env_type, "black")
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)

        # Get FPS at NUM_ENVS=1
        base = subset[subset["config_NUM_ENVS"] == 1]
        if base.empty:
            continue
        fps_1 = base["timing_fps"].iloc[0]

        efficiency = subset["timing_fps"] / (subset["config_NUM_ENVS"] * fps_1)
        ax.plot(subset["config_NUM_ENVS"], efficiency, marker="o", color=color, label=display)

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Perfect scaling")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NUM_ENVS")
    ax.set_ylabel("Scaling Efficiency")
    ax.set_ylim(0, 1.5)
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig.tight_layout()
    _save(fig, output_dir, "scaling_efficiency")


# -- Plot 10: Cross-env-type Comparison (bar chart) ---------------------------


def plot_cross_env(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing FPS across env types on same config."""
    data = _filter_group(df, "cross_env_")
    if data.empty:
        print("  No data for cross_env group")
        return

    # Single-env comparison (NUM_ENVS=1)
    single = data[data["config_NUM_ENVS"] == 1].sort_values("timing_fps", ascending=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = range(len(single))
    colors = [ENV_TYPE_COLORS.get(et, "black") for et in single["config_env_type"]]
    labels = [ENV_TYPE_DISPLAY.get(et, et) for et in single["config_env_type"]]

    bars = ax.barh(y_pos, single["timing_fps"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("FPS")
    ax.set_title("Throughput by Environment Type (NUM_ENVS=1)")
    ax.set_xscale("log")

    # Annotate bars
    for bar, fps in zip(bars, single["timing_fps"]):
        ax.text(
            bar.get_width() * 1.05,
            bar.get_y() + bar.get_height() / 2,
            format_fps(fps),
            va="center",
            fontsize=18,
        )

    fig.tight_layout()
    _save(fig, output_dir, "cross_env_comparison")


# -- Main ---------------------------------------------------------------------

PLOT_FUNCTIONS = {
    "fps_vs_num_envs": plot_fps_vs_num_envs,
    "fps_vs_link_resources": plot_fps_vs_link_resources,
    "fps_vs_k": plot_fps_vs_k,
    "fps_vs_topology": plot_fps_vs_topology,
    "compilation_time": plot_compilation_time,
    "cpu_vs_gpu": plot_cpu_vs_gpu,
    "gn_band_scaling": plot_gn_band_scaling,
    "heatmap": plot_heatmap,
    "scaling_efficiency": plot_scaling_efficiency,
    "cross_env": plot_cross_env,
}


def main():
    parser = argparse.ArgumentParser(description="Generate XLRON benchmark plots")
    parser.add_argument("--input", required=True, help="Path to aggregated CSV")
    parser.add_argument(
        "--topo_stats", default="benchmarks/topology_stats.csv", help="Path to topology stats CSV"
    )
    parser.add_argument(
        "--output_dir", default="benchmarks/figures", help="Output directory for figures"
    )
    parser.add_argument(
        "--plots",
        default=None,
        help=f"Comma-separated plot names (default: all). Available: {', '.join(PLOT_FUNCTIONS)}",
    )
    args = parser.parse_args()

    configure_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    topo_stats = None
    if Path(args.topo_stats).exists():
        topo_stats = pd.read_csv(args.topo_stats)

    plots_to_make = (
        [p.strip() for p in args.plots.split(",")] if args.plots else list(PLOT_FUNCTIONS)
    )

    unknown = [p for p in plots_to_make if p not in PLOT_FUNCTIONS]
    if unknown:
        print(f"Unknown plots: {unknown}. Available: {list(PLOT_FUNCTIONS)}")
        return

    for plot_name in plots_to_make:
        print(f"Generating: {plot_name}")
        fn = PLOT_FUNCTIONS[plot_name]
        if plot_name == "fps_vs_topology":
            fn(df, output_dir, topo_stats)
        else:
            fn(df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
