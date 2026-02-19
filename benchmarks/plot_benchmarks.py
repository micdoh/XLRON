#!/usr/bin/env python3
"""Generate benchmark plots from aggregated results.

Usage:
    python benchmarks/plot_benchmarks.py --input=benchmarks/results/benchmark_results.csv
    python benchmarks/plot_benchmarks.py --input=benchmarks/results/benchmark_results.csv --device=gpu
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


# -- Helpers ------------------------------------------------------------------


def _filter_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Filter dataframe to rows belonging to a sweep group.

    Uses the 'group' column added by aggregate_results.py. Falls back to
    regex matching on the 'file' column if the 'group' column is missing.
    """
    if "group" in df.columns:
        return df[df["group"] == group].copy()
    import re
    pattern = rf"^(?:cpu_|gpu_)?{re.escape(group)}_"
    return df[df["file"].str.contains(pattern, regex=True)].copy()


def _filter_device(df: pd.DataFrame, device: str | None) -> pd.DataFrame:
    """Filter to a single device.  If device is None, return as-is."""
    if device is None or "device" not in df.columns:
        return df
    return df[df["device"] == device].copy()


def _save(fig: plt.Figure, output_dir: Path, name: str):
    """Save figure as PNG."""
    fig.savefig(output_dir / f"{name}.png")
    plt.close(fig)
    print(f"  Saved {name}.png")


# -- Plot 1: FPS vs NUM_ENVS (log-log) ---------------------------------------


def plot_fps_vs_num_envs(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Log-log plot of FPS vs NUM_ENVS. Only shows RMSA.

    Generates two plots:
    - fps_vs_num_envs: filtered to --device (default GPU if available)
    - fps_vs_num_envs_cpu_gpu: both CPU and GPU series
    """
    data = _filter_group(df, "num_envs")
    if data.empty:
        print("  No data for num_envs group")
        return

    # Only show RMSA
    data = data[data["config_env_type"] == "rmsa"]
    if data.empty:
        print("  No RMSA data for num_envs group")
        return

    # Plot 1: single-device (respect --device filter)
    single_dev = _filter_device(data, device)
    if not single_dev.empty:
        fig, ax = plt.subplots(figsize=(14, 10))
        subset = single_dev.sort_values("config_NUM_ENVS")
        color = ENV_TYPE_COLORS.get("rmsa", "black")

        ax.plot(
            subset["config_NUM_ENVS"], subset["timing_fps"],
            marker="o", color=color, label="RMSA",
        )

        # Ideal linear scaling reference
        if len(subset) > 0:
            x0 = subset["config_NUM_ENVS"].iloc[0]
            y0 = subset["timing_fps"].iloc[0]
            x_range = subset["config_NUM_ENVS"].values
            ax.plot(
                x_range, y0 * (x_range / x0), "--",
                color="gray", alpha=0.5, label="Linear scaling",
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("# Parallel Environments")
        ax.set_ylabel("FPS")
        ax.legend()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        fig.tight_layout()
        _save(fig, output_dir, "fps_vs_num_envs")

    # Plot 2: CPU + GPU overlay
    if "device" in data.columns:
        cpu_data = data[data["device"] == "cpu"]
        gpu_data = data[data["device"] == "gpu"]
        if not cpu_data.empty and not gpu_data.empty:
            fig, ax = plt.subplots(figsize=(14, 10))
            for label, dev_data, color, ls in [
                ("RMSA (CPU)", cpu_data, DEVICE_COLORS["cpu"], "--"),
                ("RMSA (GPU)", gpu_data, DEVICE_COLORS["gpu"], "-"),
            ]:
                subset = dev_data.sort_values("config_NUM_ENVS")
                ax.plot(
                    subset["config_NUM_ENVS"], subset["timing_fps"],
                    marker="o", color=color, linestyle=ls, label=label,
                )

            # Linear scaling reference from GPU
            gpu_sorted = gpu_data.sort_values("config_NUM_ENVS")
            if len(gpu_sorted) > 0:
                x0 = gpu_sorted["config_NUM_ENVS"].iloc[0]
                y0 = gpu_sorted["timing_fps"].iloc[0]
                x_range = gpu_sorted["config_NUM_ENVS"].values
                ax.plot(
                    x_range, y0 * (x_range / x0), "--",
                    color="gray", alpha=0.5, label="Linear scaling",
                )

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("# Parallel Environments")
            ax.set_ylabel("FPS")
            ax.legend()
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            fig.tight_layout()
            _save(fig, output_dir, "fps_vs_num_envs_cpu_gpu")


# -- Plot 2: FPS vs Link Resources -------------------------------------------


def plot_fps_vs_link_resources(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """FPS vs link_resources. Only shows RMSA."""
    data = _filter_device(_filter_group(df, "link_resources"), device)
    if data.empty:
        print("  No data for link_resources group")
        return

    # Only show RMSA
    data = data[data["config_env_type"] == "rmsa"]
    if data.empty:
        print("  No RMSA data for link_resources group")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    subset = data.sort_values("config_link_resources")
    color = ENV_TYPE_COLORS.get("rmsa", "black")

    ax.plot(subset["config_link_resources"], subset["timing_fps"],
            marker="o", color=color)
    ax.set_xlabel("FSU per Link")
    ax.set_ylabel("FPS")

    fig.tight_layout()
    _save(fig, output_dir, "fps_vs_link_resources")


# -- Plot 3: FPS vs K-paths --------------------------------------------------


def plot_fps_vs_k(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """FPS vs K. Only shows RMSA."""
    data = _filter_device(_filter_group(df, "k_paths"), device)
    if data.empty:
        print("  No data for k_paths group")
        return

    # Only show RMSA
    data = data[data["config_env_type"] == "rmsa"]
    if data.empty:
        print("  No RMSA data for k_paths group")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    subset = data.sort_values("config_k")
    color = ENV_TYPE_COLORS.get("rmsa", "black")

    ax.plot(subset["config_k"], subset["timing_fps"], marker="o", color=color)
    ax.set_xlabel("K (shortest paths)")
    ax.set_ylabel("FPS")

    fig.tight_layout()
    _save(fig, output_dir, "fps_vs_k")


# -- Plot 4: FPS vs Topology (multiple subplots) -----------------------------


def _deduplicate_topologies(data: pd.DataFrame) -> pd.DataFrame:
    """Keep one topology variant per base network.

    Preference order: deeprmsa > nevin > ptrnet_real > ptrnet_published > other.
    """
    priority = {"deeprmsa": 0, "nevin": 1, "ptrnet_real": 2, "ptrnet_published": 3}

    def _variant_priority(name: str) -> int:
        for key, p in priority.items():
            if key in name:
                return p
        return 99

    def _base_name(name: str) -> str:
        # e.g. "cost239_deeprmsa_directed" -> "cost239"
        # "usnet_gcnrnn_directed" -> "usnet"
        parts = name.replace("_directed", "").replace("_undirected", "").split("_")
        return parts[0]

    data = data.copy()
    data["_base"] = data["config_topology_name"].apply(_base_name)
    data["_prio"] = data["config_topology_name"].apply(_variant_priority)
    data = data.sort_values("_prio").drop_duplicates(subset=["_base"], keep="first")
    data = data.drop(columns=["_base", "_prio"])
    return data


def plot_fps_vs_topology(
    df: pd.DataFrame, output_dir: Path, topo_stats: pd.DataFrame | None = None,
    device: str | None = None,
):
    """Multiple topology plots: FPS vs nodes, FPS vs edges, FPS vs avg_degree.

    Only shows RMSA. Deduplicates topology variants (prefers deeprmsa).
    Also generates a 3D surface plot of FPS vs num_nodes vs num_edges.
    """
    data = _filter_device(_filter_group(df, "topology"), device)
    if data.empty:
        print("  No data for topology group")
        return

    # Only show RMSA
    data = data[data["config_env_type"] == "rmsa"]
    if data.empty:
        print("  No RMSA data for topology group")
        return

    # Deduplicate topology variants
    data = _deduplicate_topologies(data)

    if topo_stats is None:
        topo_stats_path = Path("benchmarks/results/topology_stats.csv")
        if not topo_stats_path.exists():
            print("  No topology_stats.csv found, skipping topology plot")
            return
        topo_stats = pd.read_csv(topo_stats_path)

    merged = data.merge(
        topo_stats, left_on="config_topology_name", right_on="topology_name", how="left",
    )

    # Generate FPS vs each graph metric
    metrics = [
        ("num_nodes", "Number of Nodes"),
        ("num_edges", "Number of Directed Edges"),
        ("avg_degree", "Average Node Degree"),
        ("avg_path_length", "Average Shortest Path Length"),
    ]

    for metric_col, metric_label in metrics:
        fig, ax = plt.subplots(figsize=(16, 10))
        subset = merged.sort_values(metric_col)
        color = ENV_TYPE_COLORS.get("rmsa", "black")

        ax.scatter(
            subset[metric_col], subset["timing_fps"],
            c=color, s=120, zorder=5,
        )

        for _, row in subset.iterrows():
            topo_label = TOPOLOGY_DISPLAY.get(row["config_topology_name"], "")
            ax.annotate(
                topo_label, (row[metric_col], row["timing_fps"]),
                textcoords="offset points", xytext=(5, 8), fontsize=14,
            )

        ax.set_xlabel(metric_label)
        ax.set_ylabel("FPS")
        fig.tight_layout()
        safe_name = metric_col.replace("_", "-")
        _save(fig, output_dir, f"fps_vs_{safe_name}")

    # 3D surface plot: FPS vs num_nodes and num_edges
    if "num_nodes" in merged.columns and "num_edges" in merged.columns:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection="3d")
        color = ENV_TYPE_COLORS.get("rmsa", "black")

        ax.scatter(
            merged["num_nodes"], merged["num_edges"], merged["timing_fps"],
            c=color, s=120, depthshade=True,
        )

        for _, row in merged.iterrows():
            topo_label = TOPOLOGY_DISPLAY.get(row["config_topology_name"], "")
            ax.text(
                row["num_nodes"], row["num_edges"], row["timing_fps"],
                f"  {topo_label}", fontsize=12,
            )

        ax.set_xlabel("Number of Nodes")
        ax.set_ylabel("Number of Directed Edges")
        ax.set_zlabel("FPS")
        fig.tight_layout()
        _save(fig, output_dir, "fps_vs_nodes_edges_3d")


# -- Plot 5: Compilation Time vs NUM_ENVS ------------------------------------


def plot_compilation_time(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Compilation time vs NUM_ENVS."""
    data = _filter_device(_filter_group(df, "num_envs"), device)
    if data.empty or "timing_compilation_time_s" not in data.columns:
        print("  No compilation time data")
        return

    data = data.dropna(subset=["timing_compilation_time_s"])
    if data.empty:
        print("  No compilation time data (all NaN)")
        return

    env_types = sorted(data["config_env_type"].unique())
    fig, ax = plt.subplots(figsize=(14, 10))

    for env_type in env_types:
        subset = data[data["config_env_type"] == env_type].sort_values("config_NUM_ENVS")
        color = ENV_TYPE_COLORS.get(env_type, "black")
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)

        ax.plot(
            subset["config_NUM_ENVS"], subset["timing_compilation_time_s"],
            marker="o", color=color, label=display,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("# Parallel Environments")
    ax.set_ylabel("Compilation Time (s)")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig.tight_layout()
    _save(fig, output_dir, "compilation_time_vs_num_envs")


# -- Plot 6: CPU vs GPU Speedup ----------------------------------------------


def plot_cpu_vs_gpu(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """GPU speedup over CPU at different NUM_ENVS.

    Uses the 'device' group first; falls back to num_envs group if both
    CPU and GPU data are available there.  Ignores the --device filter since
    this plot inherently needs both.  Only shows RMSA data.
    """
    # Try dedicated device group
    data = _filter_group(df, "device")
    if data.empty:
        # Fall back to num_envs group which may have both cpu and gpu
        data = _filter_group(df, "num_envs")

    if data.empty or "device" not in data.columns:
        print("  No data for cpu_vs_gpu plot")
        return

    # Only show RMSA
    data = data[data["config_env_type"] == "rmsa"]
    if data.empty:
        print("  No RMSA data for cpu_vs_gpu plot")
        return

    # Average duplicate rows (multiple runs per config)
    group_cols = ["config_NUM_ENVS", "config_env_type", "device"]
    data = data.groupby(group_cols, as_index=False)["timing_fps"].mean()

    cpu_data = data[data["device"] == "cpu"]
    gpu_data = data[data["device"] == "gpu"]

    if cpu_data.empty or gpu_data.empty:
        print("  Need both CPU and GPU data for cpu_vs_gpu plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(28, 10))

    # Left: raw FPS comparison
    ax = axes[0]
    for device_label, dev_data, color, ls in [
        ("CPU", cpu_data, DEVICE_COLORS["cpu"], "--"),
        ("GPU", gpu_data, DEVICE_COLORS["gpu"], "-"),
    ]:
        subset = dev_data.sort_values("config_NUM_ENVS")
        ax.plot(
            subset["config_NUM_ENVS"], subset["timing_fps"],
            marker="o", color=color, linestyle=ls,
            label=f"RMSA ({device_label})",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("# Parallel Environments")
    ax.set_ylabel("FPS")
    ax.legend(fontsize=16)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Right: speedup
    ax = axes[1]
    cpu_subset = cpu_data.sort_values("config_NUM_ENVS")
    gpu_subset = gpu_data.sort_values("config_NUM_ENVS")

    merged = cpu_subset.merge(
        gpu_subset, on="config_NUM_ENVS", suffixes=("_cpu", "_gpu"),
    )
    if not merged.empty:
        speedup = merged["timing_fps_gpu"] / merged["timing_fps_cpu"]
        ax.plot(merged["config_NUM_ENVS"], speedup,
                marker="s", color=ENV_TYPE_COLORS.get("rmsa", "black"), label="RMSA")

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Parity")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("# Parallel Environments")
    ax.set_ylabel("GPU Speedup (x)")
    ax.legend(fontsize=16)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    _save(fig, output_dir, "cpu_vs_gpu")


# -- Plot 7: GN Model Band Scaling -------------------------------------------


def plot_gn_band_scaling(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Grouped bar chart of FPS for C / C+L / C+L+S bands, one subplot per device."""
    data = _filter_group(df, "gn_bands")
    if data.empty:
        print("  No data for gn_bands group")
        return

    devices = sorted(data["device"].unique()) if "device" in data.columns else [None]
    # If user specified --device, just show that one
    if device is not None and "device" in data.columns:
        devices = [device]
        data = data[data["device"] == device]

    bands = ["C", "C,L", "C,L,S"]
    # Only include bands that actually have data
    bands = [b for b in bands if b in data["config_band_preference"].values]

    for dev in devices:
        dev_data = data[data["device"] == dev] if dev is not None else data
        if dev_data.empty:
            continue

        env_types = sorted(dev_data["config_env_type"].unique())
        fig, ax = plt.subplots(figsize=(14, 10))
        x = np.arange(len(bands))
        n = len(env_types)
        width = 0.8 / max(n, 1)

        for i, env_type in enumerate(env_types):
            display = ENV_TYPE_DISPLAY.get(env_type, env_type)
            color = ENV_TYPE_COLORS.get(env_type, "black")
            fps_values = []
            for band in bands:
                subset = dev_data[
                    (dev_data["config_env_type"] == env_type)
                    & (dev_data["config_band_preference"] == band)
                ]
                fps_values.append(subset["timing_fps"].iloc[0] if not subset.empty else 0)

            offset = (i - (n - 1) / 2) * width
            ax.bar(x + offset, fps_values, width, label=display, color=color)

        ax.set_xlabel("Band Configuration")
        ax.set_ylabel("FPS")
        ax.set_xticks(x)
        ax.set_xticklabels([BAND_DISPLAY.get(b, b) for b in bands])
        ax.legend()
        dev_label = f" ({dev.upper()})" if dev else ""
        ax.set_title(f"GN Model Throughput by Band{dev_label}")
        fig.tight_layout()
        suffix = f"_{dev}" if dev else ""
        _save(fig, output_dir, f"gn_band_scaling{suffix}")


# -- Plot 8: Heatmap (topology x NUM_ENVS) -----------------------------------


def plot_heatmap(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Heatmap of FPS with topology on y-axis, colored by FPS."""
    data = _filter_device(_filter_group(df, "topology"), device)
    if data.empty:
        print("  No data for heatmap")
        return

    for env_type in ["rwa", "rmsa"]:
        subset = data[data["config_env_type"] == env_type]
        if subset.empty:
            continue

        display = ENV_TYPE_DISPLAY.get(env_type, env_type)
        topo_order = [
            t for t in get_topology_order()
            if t in subset["config_topology_name"].values
        ]
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
        num_envs_val = int(subset["config_NUM_ENVS"].iloc[0])
        ax.set_xticklabels([f"NUM_ENVS={num_envs_val}"])

        for i, fps in enumerate(fps_values):
            ax.text(
                0, i, format_fps(fps), ha="center", va="center", fontsize=18,
                color="white" if fps > max(fps_values) * 0.6 else "black",
            )

        fig.colorbar(im, ax=ax, label="FPS")
        ax.set_title(f"{display} Throughput by Topology")
        fig.tight_layout()
        _save(fig, output_dir, f"heatmap_{env_type}")


# -- Plot 9: Scaling Efficiency -----------------------------------------------


def plot_scaling_efficiency(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """FPS(N) / (N * FPS(1)) showing parallelization efficiency."""
    data = _filter_device(_filter_group(df, "num_envs"), device)
    if data.empty:
        print("  No data for scaling efficiency")
        return

    env_types = sorted(data["config_env_type"].unique())
    fig, ax = plt.subplots(figsize=(14, 10))

    for env_type in env_types:
        subset = data[data["config_env_type"] == env_type].sort_values("config_NUM_ENVS")
        color = ENV_TYPE_COLORS.get(env_type, "black")
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)

        base = subset[subset["config_NUM_ENVS"] == 1]
        if base.empty:
            continue
        fps_1 = base["timing_fps"].iloc[0]

        efficiency = subset["timing_fps"] / (subset["config_NUM_ENVS"] * fps_1)
        ax.plot(subset["config_NUM_ENVS"], efficiency,
                marker="o", color=color, label=display)

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


def plot_cross_env(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Bar chart comparing FPS across env types on same config."""
    data = _filter_device(_filter_group(df, "cross_env"), device)
    if data.empty:
        print("  No data for cross_env group")
        return

    # Single-env comparison (NUM_ENVS=1)
    single = data[data["config_NUM_ENVS"] == 1].sort_values("timing_fps", ascending=True)
    if single.empty:
        print("  No NUM_ENVS=1 data for cross_env group")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    y_pos = range(len(single))
    colors = [ENV_TYPE_COLORS.get(et, "black") for et in single["config_env_type"]]
    labels = [ENV_TYPE_DISPLAY.get(et, et) for et in single["config_env_type"]]

    bars = ax.barh(y_pos, single["timing_fps"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    # Use log scale but show plain decimal tick labels with exponent in axis label
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x / 1e3:.1f}" if x >= 1e3 else f"{x:.0f}"
    ))
    # Determine the exponent from the data range
    max_fps = single["timing_fps"].max()
    if max_fps >= 1e6:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x / 1e6:.1f}"
        ))
        ax.set_xlabel(r"FPS ($\times 10^6$)")
    elif max_fps >= 1e3:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x / 1e3:.1f}"
        ))
        ax.set_xlabel(r"FPS ($\times 10^3$)")
    else:
        ax.set_xlabel("FPS")

    for bar, fps in zip(bars, single["timing_fps"]):
        ax.text(
            bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
            format_fps(fps), va="center", fontsize=18,
        )

    fig.tight_layout()
    _save(fig, output_dir, "cross_env_comparison")


# -- Plot 11: Config Grid (L-S-E bar chart) ----------------------------------


def plot_config_grid(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Horizontal bar chart of FPS for different (link_resources, NUM_ENVS) combos.

    Each row is labelled Lxx-Sxx-Exx (Links, Slots/FSU, Envs).
    Shows paired CPU and GPU bars.  Uses config_grid group if available,
    otherwise assembles data from existing groups.
    """
    # Try dedicated config_grid group first
    data = _filter_group(df, "config_grid")
    if data.empty:
        # Fall back: gather RMSA data from all groups that have both cpu and gpu
        data = df[df["config_env_type"] == "rmsa"].copy()

    if data.empty or "device" not in data.columns:
        print("  No data for config_grid plot")
        return

    # Only RMSA
    data = data[data["config_env_type"] == "rmsa"]

    # Average duplicate rows per (link_resources, NUM_ENVS, device)
    group_cols = ["config_link_resources", "config_NUM_ENVS", "device"]
    data = data.groupby(group_cols, as_index=False)["timing_fps"].mean()

    # Pivot so we have cpu and gpu columns
    piv = data.pivot_table(
        index=["config_link_resources", "config_NUM_ENVS"],
        columns="device", values="timing_fps",
    )
    # Keep only rows with both CPU and GPU
    piv = piv.dropna(subset=["cpu", "gpu"])
    if piv.empty:
        print("  No matching CPU+GPU data for config_grid plot")
        return

    piv = piv.reset_index()
    # Sort by link_resources then NUM_ENVS for consistent ordering
    piv = piv.sort_values(
        ["config_link_resources", "config_NUM_ENVS"], ascending=[True, True],
    )

    # Build labels: Lxx-Sxx-Exx (L=num directed edges, S=FSU/slots, E=parallel envs)
    # Topology is fixed (NSFNET=44 directed edges) in fallback data;
    # config_grid also uses NSFNET.  Look up num_edges if available.
    num_edges = 44  # NSFNET default
    labels = [
        f"L{num_edges}"
        f"-S{int(row['config_link_resources'])}"
        f"-E{int(row['config_NUM_ENVS'])}"
        for _, row in piv.iterrows()
    ]

    y = np.arange(len(labels))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(16, max(8, len(labels) * 0.7)))

    bars_cpu = ax.barh(
        y + bar_height / 2, piv["cpu"], bar_height,
        label="CPU", color=DEVICE_COLORS["cpu"],
    )
    bars_gpu = ax.barh(
        y - bar_height / 2, piv["gpu"], bar_height,
        label="GPU", color=DEVICE_COLORS["gpu"],
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("FPS")
    ax.legend(fontsize=16)
    ax.invert_yaxis()  # top-to-bottom reading order

    # Annotate bars with FPS values
    for bar, fps in zip(bars_cpu, piv["cpu"]):
        ax.text(
            bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
            format_fps(fps), va="center", fontsize=14,
        )
    for bar, fps in zip(bars_gpu, piv["gpu"]):
        ax.text(
            bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
            format_fps(fps), va="center", fontsize=14,
        )

    fig.tight_layout()
    _save(fig, output_dir, "config_grid")


# -- Plot 12: Configuration Summary ------------------------------------------


def plot_config_summary(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Generate a text-based PNG summarizing the configuration of each benchmark group."""
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.axis("off")

    lines = []
    lines.append("Benchmark Configuration Summary")
    lines.append("=" * 60)

    group_descriptions = {
        "num_envs": ("FPS vs # Parallel Environments", "fps_vs_num_envs"),
        "link_resources": ("FPS vs FSU per Link", "fps_vs_link_resources"),
        "k_paths": ("FPS vs K Shortest Paths", "fps_vs_k"),
        "topology": ("FPS vs Topology", "fps_vs_topology"),
        "device": ("CPU vs GPU Comparison", "cpu_vs_gpu"),
        "cross_env": ("Cross-Environment Comparison", "cross_env_comparison"),
        "gn_bands": ("GN Model Band Scaling", "gn_band_scaling"),
    }

    for group, (title, filename) in group_descriptions.items():
        sub = df[df["group"] == group] if "group" in df.columns else pd.DataFrame()
        if sub.empty:
            continue

        lines.append("")
        lines.append(f"{title}  ({filename}.png)")
        lines.append("-" * 50)

        env_types = sorted(sub["config_env_type"].unique())
        devices = sorted(sub["device"].unique()) if "device" in sub.columns else ["N/A"]
        num_envs = sorted(sub["config_NUM_ENVS"].unique())
        topologies = sorted(sub["config_topology_name"].unique())
        k_vals = sorted(sub["config_k"].dropna().unique())
        lr_vals = sorted(sub["config_link_resources"].dropna().unique())

        lines.append(f"  Env types:   {', '.join(env_types)}")
        lines.append(f"  Devices:     {', '.join(str(d) for d in devices)}")
        lines.append(f"  NUM_ENVS:    {', '.join(str(int(n)) for n in num_envs)}")
        lines.append(f"  Topology:    {', '.join(topologies)}")
        if k_vals:
            lines.append(f"  K:           {', '.join(str(int(k)) for k in k_vals)}")
        if lr_vals:
            lines.append(f"  Link res:    {', '.join(str(int(lr)) for lr in lr_vals)}")

        load_vals = sorted(sub["config_load"].dropna().unique())
        if load_vals:
            lines.append(f"  Load:        {', '.join(str(int(l)) for l in load_vals)} Erlangs")
        heuristic = sub["config_path_heuristic"].dropna().unique()
        if len(heuristic) > 0:
            lines.append(f"  Heuristic:   {', '.join(str(h) for h in heuristic)}")

    text = "\n".join(lines)
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes, verticalalignment="top",
        fontfamily="monospace", fontsize=14, wrap=False,
    )

    fig.tight_layout()
    _save(fig, output_dir, "config_summary")


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
    "config_grid": plot_config_grid,
    "config_summary": plot_config_summary,
}


def main():
    parser = argparse.ArgumentParser(description="Generate XLRON benchmark plots")
    parser.add_argument("--input", required=True, help="Path to aggregated CSV")
    parser.add_argument(
        "--topo_stats", default="benchmarks/topology_stats.csv",
        help="Path to topology stats CSV",
    )
    parser.add_argument(
        "--output_dir", default="benchmarks/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--plots", default=None,
        help=f"Comma-separated plot names (default: all). "
             f"Available: {', '.join(PLOT_FUNCTIONS)}",
    )
    parser.add_argument(
        "--device", default=None, choices=["cpu", "gpu"],
        help="Filter to a single device (default: use all data). "
             "The cpu_vs_gpu plot always uses both regardless.",
    )
    args = parser.parse_args()

    configure_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    topo_stats = None
    if Path(args.topo_stats).exists():
        topo_stats = pd.read_csv(args.topo_stats)

    # Show data summary
    if "group" in df.columns:
        print("Data summary:")
        print(df.groupby(["group", "device"]).size().to_string())
        print()

    if args.device:
        print(f"Filtering to device: {args.device}")
        print()

    plots_to_make = (
        [p.strip() for p in args.plots.split(",")]
        if args.plots
        else list(PLOT_FUNCTIONS)
    )

    unknown = [p for p in plots_to_make if p not in PLOT_FUNCTIONS]
    if unknown:
        print(f"Unknown plots: {unknown}. Available: {list(PLOT_FUNCTIONS)}")
        return

    for plot_name in plots_to_make:
        print(f"Generating: {plot_name}")
        fn = PLOT_FUNCTIONS[plot_name]
        if plot_name == "fps_vs_topology":
            fn(df, output_dir, topo_stats, args.device)
        else:
            fn(df, output_dir, args.device)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
