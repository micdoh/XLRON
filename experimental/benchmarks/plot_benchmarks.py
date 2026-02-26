#!/usr/bin/env python3
"""Generate benchmark plots from aggregated results.

Automatically aggregates raw JSONL results before plotting.
All paths default to the benchmarks/ directory structure.

Usage:
    python benchmarks/plot_benchmarks.py
    python benchmarks/plot_benchmarks.py --device=gpu
    python benchmarks/plot_benchmarks.py --plots=fps_vs_num_envs,cpu_vs_gpu
    python benchmarks/plot_benchmarks.py --input=custom/path.csv --skip_aggregate
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Resolve paths relative to this script so they work from any cwd
_BENCHMARKS_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _BENCHMARKS_DIR / "results"
_FIGURES_DIR = _BENCHMARKS_DIR / "figures"
_DEFAULT_CSV = _RESULTS_DIR / "benchmark_results.csv"

from experimental.plot_style import (
    BAND_DISPLAY,
    DEVICE_COLORS,
    ENV_TYPE_COLORS,
    ENV_TYPE_DISPLAY,
    HEATMAP_CPU_COLORS,
    HEATMAP_GPU_COLORS,
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


def _agg(df: pd.DataFrame, group_cols: list[str], value_col: str = "timing_fps") -> pd.DataFrame:
    """Aggregate repeat runs: compute mean and std per configuration.

    Returns a DataFrame with columns: *group_cols, {value_col}_mean, {value_col}_std.
    """
    agg = df.groupby(group_cols, as_index=False)[value_col].agg(["mean", "std"])
    agg = agg.reset_index()
    agg = agg.rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std"})
    agg[f"{value_col}_std"] = agg[f"{value_col}_std"].fillna(0)
    return agg


# -- Plot 1: FPS vs NUM_ENVS (log-log) ---------------------------------------


def plot_fps_vs_num_envs(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Log-log plot of FPS vs NUM_ENVS with CPU and GPU series. Only shows RMSA."""
    data = _filter_group(df, "num_envs")
    if data.empty:
        print("  No data for num_envs group")
        return

    # Only show RMSA
    data = data[data["config_env_type"] == "rmsa"]
    if data.empty:
        print("  No RMSA data for num_envs group")
        return

    devices = sorted(data["device"].unique()) if "device" in data.columns else [None]
    fig, ax = plt.subplots(figsize=(14, 10))

    device_styles = {"cpu": "--", "gpu": "-"}
    ref_x, ref_y0 = None, None

    for dev in devices:
        dev_data = data[data["device"] == dev] if dev else data
        if dev_data.empty:
            continue
        agg = _agg(dev_data, ["config_NUM_ENVS"]).sort_values("config_NUM_ENVS")
        x = agg["config_NUM_ENVS"].values
        y = agg["timing_fps_mean"].values
        yerr = agg["timing_fps_std"].values
        color = DEVICE_COLORS.get(dev, ENV_TYPE_COLORS.get("rmsa", "black")) if dev else ENV_TYPE_COLORS.get("rmsa", "black")
        ls = device_styles.get(dev, "-")
        label = f"RMSA ({dev.upper()})" if dev and len(devices) > 1 else "RMSA"
        ax.plot(x, y, marker="o", color=color, linestyle=ls, label=label)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
        # Use the fastest device for the linear scaling reference
        if ref_y0 is None or y[0] > ref_y0:
            ref_x, ref_y0 = x, y[0]

    if ref_x is not None:
        ax.plot(ref_x, ref_y0 * (ref_x / ref_x[0]), "--",
                color="gray", alpha=0.5, label="Linear scaling")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("# Parallel Environments")
    ax.set_ylabel("FPS")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig.tight_layout()
    _save(fig, output_dir, "fps_vs_num_envs")


# -- Plot 2: FPS vs Link Resources -------------------------------------------


def plot_fps_vs_link_resources_and_k(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Combined figure: FPS vs FSU per Link (left) and FPS vs K (right)."""
    data_lr = _filter_device(_filter_group(df, "link_resources"), device)
    data_k = _filter_device(_filter_group(df, "k_paths"), device)
    if data_lr.empty and data_k.empty:
        print("  No data for link_resources or k_paths groups")
        return

    # Only RMSA
    data_lr = data_lr[data_lr["config_env_type"] == "rmsa"] if not data_lr.empty else data_lr
    data_k = data_k[data_k["config_env_type"] == "rmsa"] if not data_k.empty else data_k

    fig, (ax_lr, ax_k) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Helper to plot one panel
    def _plot_panel(ax, data, x_col, x_label):
        if data.empty:
            return
        group_cols = [x_col, "device"] if "device" in data.columns else [x_col]
        agg = _agg(data, group_cols)
        devices = sorted(agg["device"].unique(), reverse=True) if "device" in agg.columns else [None]
        for dev in devices:
            sub = agg[agg["device"] == dev] if dev else agg
            sub = sub.sort_values(x_col)
            color = DEVICE_COLORS.get(dev, ENV_TYPE_COLORS.get("rmsa", "black")) if dev else ENV_TYPE_COLORS.get("rmsa", "black")
            label = dev.upper() if dev and len(devices) > 1 else "RMSA"
            x = sub[x_col].values
            y = sub["timing_fps_mean"].values
            yerr = sub["timing_fps_std"].values
            ax.plot(x, y, marker="o", color=color, label=label)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
        ax.set_xlabel(x_label, fontsize=16)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_yscale("log")

    _plot_panel(ax_lr, data_lr, "config_link_resources", "FSU per Link")
    _plot_panel(ax_k, data_k, "config_k", "K (shortest paths)")

    ax_lr.set_ylabel("FPS", fontsize=16)
    # More y-axis ticks
    from matplotlib.ticker import LogLocator
    ax_lr.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax_lr.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=20))
    ax_lr.yaxis.set_minor_formatter(plt.NullFormatter())

    # Legend on right panel, center right
    ax_k.legend(loc="center right")

    fig.tight_layout()
    _save(fig, output_dir, "fps_vs_link_resources_and_k")


# -- Plot 4: Compilation Time vs NUM_ENVS ------------------------------------


def plot_compilation_time(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Compilation time vs NUM_ENVS.

    Uses env-type colours and distinguishes CPU/GPU via marker and linestyle.
    The legend has two sections: one showing device styles, one showing env-type colours.
    """
    import matplotlib.lines as mlines

    data = _filter_device(_filter_group(df, "num_envs"), device)
    if data.empty or "timing_compilation_time_s" not in data.columns:
        print("  No compilation time data")
        return

    data = data.dropna(subset=["timing_compilation_time_s"])
    if data.empty:
        print("  No compilation time data (all NaN)")
        return

    env_types = sorted(data["config_env_type"].unique())
    devices = sorted(data["device"].unique()) if "device" in data.columns else [None]
    fig, ax = plt.subplots(figsize=(14, 10))

    device_linestyles = {"cpu": "--", "gpu": "-"}
    device_markers = {"cpu": "s", "gpu": "o"}

    for env_type in env_types:
        et_data = data[data["config_env_type"] == env_type]
        group_cols = ["config_NUM_ENVS", "device"] if "device" in et_data.columns else ["config_NUM_ENVS"]
        agg = _agg(et_data, group_cols, "timing_compilation_time_s")

        for dev in devices:
            if dev is not None and "device" in agg.columns:
                sub = agg[agg["device"] == dev].sort_values("config_NUM_ENVS")
            else:
                sub = agg.sort_values("config_NUM_ENVS")
            if sub.empty:
                continue

            color = ENV_TYPE_COLORS.get(env_type, "black")
            linestyle = device_linestyles.get(dev, "-")
            marker = device_markers.get(dev, "o")

            x = sub["config_NUM_ENVS"].values
            y = sub["timing_compilation_time_s_mean"].values
            yerr = sub["timing_compilation_time_s_std"].values

            ax.plot(x, y, marker=marker, color=color, linestyle=linestyle)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.15)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("# Parallel Environments")
    ax.set_ylabel("Compilation Time (s)")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Build two-section legend: device styles (black) + env-type colours
    legend_handles = []
    if len(devices) > 1:
        for dev in devices:
            h = mlines.Line2D(
                [], [], color="black",
                linestyle=device_linestyles.get(dev, "-"),
                marker=device_markers.get(dev, "o"),
                label=dev.upper(),
            )
            legend_handles.append(h)
    for env_type in env_types:
        h = mlines.Line2D(
            [], [], color=ENV_TYPE_COLORS.get(env_type, "black"),
            linestyle="-", marker="o",
            label=ENV_TYPE_DISPLAY.get(env_type, env_type),
        )
        legend_handles.append(h)
    ax.legend(handles=legend_handles)

    fig.tight_layout()
    _save(fig, output_dir, "compilation_time_vs_num_envs")


# -- Plot 6: GPU Speedup ------------------------------------------------------


def plot_gpu_speedup(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """GPU speedup over CPU at different NUM_ENVS.

    Prefers the 'num_envs' group (which scales TOTAL_TIMESTEPS with
    NUM_ENVS to avoid compilation overhead distorting FPS at high env
    counts).  Falls back to the 'device' group if num_envs doesn't have
    both CPU and GPU data.  Ignores the --device filter since this plot
    inherently needs both.  Only shows RMSA data.
    """
    # Prefer num_envs group (better TOTAL_TIMESTEPS scaling)
    data = _filter_group(df, "num_envs")
    if data.empty or "device" not in data.columns:
        data = _filter_group(df, "device")
    else:
        # Check it actually has both cpu and gpu; fall back otherwise
        rmsa_ne = data[data["config_env_type"] == "rmsa"]
        devs = rmsa_ne["device"].unique() if not rmsa_ne.empty else []
        if "cpu" not in devs or "gpu" not in devs:
            data = _filter_group(df, "device")

    if data.empty or "device" not in data.columns:
        print("  No data for gpu_speedup plot")
        return

    # Only show RMSA
    data = data[data["config_env_type"] == "rmsa"]
    if data.empty:
        print("  No RMSA data for gpu_speedup plot")
        return

    # Aggregate repeat runs
    agg = _agg(data, ["config_NUM_ENVS", "config_env_type", "device"])

    cpu_agg = agg[agg["device"] == "cpu"].sort_values("config_NUM_ENVS")
    gpu_agg = agg[agg["device"] == "gpu"].sort_values("config_NUM_ENVS")

    if cpu_agg.empty or gpu_agg.empty:
        print("  Need both CPU and GPU data for gpu_speedup plot")
        return

    merged = cpu_agg.merge(
        gpu_agg, on="config_NUM_ENVS", suffixes=("_cpu", "_gpu"),
    )
    if merged.empty:
        print("  No overlapping NUM_ENVS values for gpu_speedup plot")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    speedup = merged["timing_fps_mean_gpu"] / merged["timing_fps_mean_cpu"]
    ax.plot(merged["config_NUM_ENVS"], speedup,
            marker="s", color=ENV_TYPE_COLORS.get("rmsa", "black"), label="RMSA")

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Parity")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("# Parallel Environments")
    ax.set_ylabel("GPU Speedup (x)")
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    _save(fig, output_dir, "gpu_speedup")


# -- Plot 7: GN Model Band Scaling -------------------------------------------


def plot_gn_band_scaling(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Grouped bar chart of FPS for C / C+L / C+L+S bands.

    Each band configuration has one bar per (env_type, device) combination.
    """
    data = _filter_group(df, "gn_bands")
    if data.empty:
        print("  No data for gn_bands group")
        return

    bands = ["C", "C,L", "C,L,S"]
    bands = [b for b in bands if b in data["config_band_preference"].values]

    group_cols = ["config_env_type", "config_band_preference"]
    if "device" in data.columns:
        group_cols.append("device")
    agg = _agg(data, group_cols)

    env_types = sorted(agg["config_env_type"].unique())
    devices = sorted(agg["device"].unique()) if "device" in agg.columns else [None]
    # Put GPU first for consistent ordering
    dev_order = []
    if "gpu" in devices:
        dev_order.append("gpu")
    if "cpu" in devices:
        dev_order.append("cpu")
    for d in devices:
        if d not in dev_order and d is not None:
            dev_order.append(d)
    if not dev_order:
        dev_order = [None]

    # Build series: one bar group per (device, env_type) — device as color, env as hatch
    series = []
    for dev in dev_order:
        for env_type in env_types:
            series.append((env_type, dev))

    env_hatches = {et: h for et, h in zip(sorted(env_types), ["", "//", "\\\\", "xx"])}

    fig, ax = plt.subplots(figsize=(14, 10))
    x = np.arange(len(bands))
    n = len(series)
    width = 0.8 / max(n, 1)

    for i, (env_type, dev) in enumerate(series):
        display = ENV_TYPE_DISPLAY.get(env_type, env_type)
        color = DEVICE_COLORS.get(dev, "black") if dev else "black"
        dev_label = dev.upper() if dev else ""
        label = f"{dev_label} {display}" if dev and len(dev_order) > 1 else display
        hatch = env_hatches.get(env_type, "")

        fps_means = []
        fps_stds = []
        for band in bands:
            mask = (
                (agg["config_env_type"] == env_type)
                & (agg["config_band_preference"] == band)
            )
            if dev is not None and "device" in agg.columns:
                mask = mask & (agg["device"] == dev)
            row = agg[mask]
            fps_means.append(row["timing_fps_mean"].iloc[0] if not row.empty else 0)
            fps_stds.append(row["timing_fps_std"].iloc[0] if not row.empty else 0)

        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, fps_means, width, yerr=fps_stds,
               label=label, color=color, hatch=hatch, capsize=4,
               edgecolor="white" if hatch else None)

    ax.set_xlabel("Band Configuration")
    ax.set_yscale("log")
    ax.set_ylabel("FPS")
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_DISPLAY.get(b, b) for b in bands])
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "gn_band_scaling")


# -- Plot 8: Heatmap (topology x env_type) ------------------------------------


def plot_heatmap(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Combined heatmap: GPU on top, CPU below, separate colorbars that blend.

    GPU uses a warm colormap (dark orange → yellow → white).
    CPU uses a cool colormap (dark blue → cyan → pale yellow-green).
    The CPU top color blends into a paler version of the GPU bottom color,
    giving visual continuity: CPU-fast ≈ GPU-slow in hue.
    """
    data = _filter_group(df, "topology")
    if data.empty:
        print("  No data for heatmap")
        return

    # For env types with multiple NUM_ENVS runs (e.g. ne1 + ne64), keep only
    # the highest NUM_ENVS per (device, topology, env_type) for the heatmap.
    if "config_NUM_ENVS" in data.columns:
        group_cols = ["config_topology_name", "config_env_type"]
        if "device" in data.columns:
            group_cols = ["device"] + group_cols
        max_ne = data.groupby(group_cols, dropna=False)["config_NUM_ENVS"].transform("max")
        data = data[data["config_NUM_ENVS"] == max_ne]

    from matplotlib.colors import LogNorm, LinearSegmentedColormap

    cmap_gpu = LinearSegmentedColormap.from_list("gpu_heat", HEATMAP_GPU_COLORS, N=256)
    cmap_cpu = LinearSegmentedColormap.from_list("cpu_heat", HEATMAP_CPU_COLORS, N=256)

    devices = sorted(data["device"].unique()) if "device" in data.columns else [None]
    # Ensure GPU is first (top) if both present
    dev_order = []
    if "gpu" in devices:
        dev_order.append("gpu")
    if "cpu" in devices:
        dev_order.append("cpu")
    for d in devices:
        if d not in dev_order and d is not None:
            dev_order.append(d)
    if not dev_order:
        dev_order = [None]

    dev_cmaps = {"gpu": cmap_gpu, "cpu": cmap_cpu}

    # Collect all env types and topologies across devices for consistent axes
    all_env_types = sorted(data["config_env_type"].unique())
    all_topos = [t for t in get_topology_order()
                 if t in data["config_topology_name"].values]
    topo_labels = [TOPOLOGY_DISPLAY.get(t, t) for t in all_topos]

    # Build one matrix and norm per device
    dev_data_map = {}
    for dev in dev_order:
        dd = data[data["device"] == dev] if dev is not None else data
        if dd.empty:
            continue
        agg = _agg(dd, ["config_topology_name", "config_env_type"])
        matrix = np.full((len(all_topos), len(all_env_types)), np.nan)
        for j, env_type in enumerate(all_env_types):
            for i, topo in enumerate(all_topos):
                row = agg[
                    (agg["config_topology_name"] == topo)
                    & (agg["config_env_type"] == env_type)
                ]
                if not row.empty:
                    matrix[i, j] = row["timing_fps_mean"].iloc[0]
        vals = matrix[~np.isnan(matrix) & (matrix > 0)]
        if len(vals) == 0:
            continue
        dev_norm = LogNorm(vmin=vals.min(), vmax=vals.max())
        dev_data_map[dev] = (matrix, dev_norm)

    if not dev_data_map:
        print("  No device data for heatmap")
        return

    n_devs = len(dev_data_map)
    n_cols = len(all_env_types)
    # gridspec: heatmaps + one colorbar column per device (stacked)
    fig = plt.figure(figsize=(5 + 2.2 * n_cols, 4 + 5 * n_devs))
    gs = fig.add_gridspec(n_devs, 2, width_ratios=[1, 0.04], wspace=0.05)

    axes = []
    for idx, dev in enumerate(dev_data_map):
        matrix, dev_norm = dev_data_map[dev]
        dev_cmap = dev_cmaps.get(dev, plt.get_cmap("turbo"))

        ax = fig.add_subplot(gs[idx, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        im = ax.imshow(matrix, aspect="auto", cmap=dev_cmap, norm=dev_norm)

        ax.set_yticks(range(len(topo_labels)))
        ax.set_yticklabels(topo_labels, fontsize=22, fontweight="bold")
        ax.set_xticks(range(len(all_env_types)))

        # Only show x tick labels on the bottom subplot
        if idx == n_devs - 1:
            ax.set_xticklabels([ENV_TYPE_DISPLAY.get(et, et) for et in all_env_types],
                               rotation=45, ha="right", fontsize=22, fontweight="bold")
        else:
            ax.tick_params(labelbottom=False)

        dev_label = dev.upper() if dev else ""
        ax.set_ylabel(dev_label, fontsize=28, fontweight="bold")

        # Annotate cells
        for i in range(len(all_topos)):
            for j in range(len(all_env_types)):
                val = matrix[i, j]
                if np.isnan(val) or val == 0:
                    txt = "—"
                    txt_color = "gray"
                else:
                    txt = format_fps(val)
                    rgba = dev_cmap(dev_norm(val))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    txt_color = "white" if lum < 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=18,
                        fontweight="bold", color=txt_color)

        # Per-device colorbar in the same row
        cbar_ax = fig.add_subplot(gs[idx, 1])
        fig.colorbar(im, cax=cbar_ax)

    # Single "FPS" label centered on the colorbar column, close to colorbars
    cbar_right = gs.get_grid_positions(fig)[3][-1]  # right edge of colorbar column
    fig.text(cbar_right + 0.03, 0.5, "FPS", va="center", ha="left", rotation=90,
             fontsize=28, fontweight="bold", transform=fig.transFigure)

    fig.tight_layout()
    _save(fig, output_dir, "heatmap")


# -- Plot 10: Cross-env-type Comparison (bar chart) ---------------------------


def plot_cross_env(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Grouped bar chart comparing FPS across env types, with CPU and GPU bars side-by-side."""
    data = _filter_group(df, "cross_env")
    if data.empty:
        print("  No data for cross_env group")
        return

    # Single-env comparison (NUM_ENVS=1)
    single = data[data["config_NUM_ENVS"] == 1]
    if single.empty:
        print("  No NUM_ENVS=1 data for cross_env group")
        return

    group_cols = ["config_env_type"]
    if "device" in single.columns:
        group_cols.append("device")
    agg = _agg(single, group_cols)

    # Sort env types by max FPS across devices
    env_order = (
        agg.groupby("config_env_type")["timing_fps_mean"]
        .max()
        .sort_values()
        .index.tolist()
    )
    devices = sorted(agg["device"].unique()) if "device" in agg.columns else [None]
    # Put GPU first for consistent ordering
    dev_order = []
    if "gpu" in devices:
        dev_order.append("gpu")
    if "cpu" in devices:
        dev_order.append("cpu")
    for d in devices:
        if d not in dev_order and d is not None:
            dev_order.append(d)
    if not dev_order:
        dev_order = [None]

    fig, ax = plt.subplots(figsize=(14, 10))
    y = np.arange(len(env_order))
    n_dev = len(dev_order)
    height = 0.8 / max(n_dev, 1)

    all_bars = []
    all_fps = []
    for i, dev in enumerate(dev_order):
        fps_means = []
        fps_stds = []
        for et in env_order:
            mask = agg["config_env_type"] == et
            if dev is not None and "device" in agg.columns:
                mask = mask & (agg["device"] == dev)
            row = agg[mask]
            fps_means.append(row["timing_fps_mean"].iloc[0] if not row.empty else 0)
            fps_stds.append(row["timing_fps_std"].iloc[0] if not row.empty else 0)

        offset = (i - (n_dev - 1) / 2) * height
        color = DEVICE_COLORS.get(dev, "black") if dev else "black"
        label = dev.upper() if dev and n_dev > 1 else None
        for j, et in enumerate(env_order):
            bar = ax.barh(
                y[j] + offset, fps_means[j], height, xerr=fps_stds[j],
                color=color, capsize=4,
                label=label if j == 0 else None,
            )
            all_bars.extend(bar)
            all_fps.append(fps_means[j])

    ax.set_yticks(y)
    ax.set_yticklabels([ENV_TYPE_DISPLAY.get(et, et) for et in env_order])
    ax.set_xscale("log")
    ax.set_xlabel("FPS")

    for bar, fps in zip(all_bars, all_fps):
        if fps > 0:
            ax.text(
                bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                format_fps(fps), va="center", fontsize=14,
            )

    if n_dev > 1:
        ax.legend()
    fig.tight_layout()
    _save(fig, output_dir, "cross_env_comparison")


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

        if group == "gn_bands" and "config_band_preference" in sub.columns:
            band_vals = sorted(sub["config_band_preference"].dropna().unique())
            band_labels = [BAND_DISPLAY.get(b, b) for b in band_vals]
            lines.append(f"  Bands:       {', '.join(band_labels)}")

    text = "\n".join(lines)
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes, verticalalignment="top",
        fontfamily="monospace", fontsize=14, wrap=False,
    )

    fig.tight_layout()
    _save(fig, output_dir, "config_summary")


# -- Plot 13: FPS vs FSU / K by NUM_ENVS (side-by-side) -----------------------


def plot_fps_vs_fsu_and_k_by_num_envs(df: pd.DataFrame, output_dir: Path, device: str | None = None):
    """Side-by-side: FPS vs FSU per Link (left) and FPS vs K (right), one line per NUM_ENVS.

    Lines go from transparent (low NUM_ENVS) to opaque (high NUM_ENVS).
    Color is determined by device (GPU/CPU).  GPU is listed first in the legend.
    A shared legend sits outside the right panel.
    """
    import matplotlib.lines as mlines

    # -- Load left panel data (FSU per Link) --
    data_lr = _filter_group(df, "config_grid")
    if data_lr.empty:
        data_lr = _filter_group(df, "link_resources")
    data_lr = data_lr[data_lr["config_env_type"] == "rmsa"] if not data_lr.empty else data_lr
    if device and not data_lr.empty:
        data_lr = _filter_device(data_lr, device)
    if not data_lr.empty and data_lr["config_link_resources"].nunique() < 2:
        data_lr = pd.DataFrame()

    # -- Load right panel data (K) --
    data_k = _filter_group(df, "k_grid")
    if data_k.empty:
        data_k = _filter_group(df, "k_paths")
    data_k = data_k[data_k["config_env_type"] == "rmsa"] if not data_k.empty else data_k
    if device and not data_k.empty:
        data_k = _filter_device(data_k, device)
    if not data_k.empty and data_k["config_k"].nunique() < 2:
        data_k = pd.DataFrame()

    if data_lr.empty and data_k.empty:
        print("  No data for fps_vs_fsu_and_k_by_num_envs")
        return

    fig, (ax_lr, ax_k) = plt.subplots(1, 2, figsize=(24, 10), sharey=True)

    # Collect all NUM_ENVS and devices across both panels for a unified legend
    all_ne_set = set()
    dev_set = set()

    def _plot_panel(ax, data, x_col, x_label):
        if data.empty:
            return
        group_cols = [x_col, "config_NUM_ENVS"]
        if "device" in data.columns:
            group_cols = group_cols + ["device"]
        agg = _agg(data, group_cols)

        # Determine devices — put GPU first
        if "device" in agg.columns:
            devices = sorted(agg["device"].unique())
            dev_order = []
            if "gpu" in devices:
                dev_order.append("gpu")
            if "cpu" in devices:
                dev_order.append("cpu")
            for d in devices:
                if d not in dev_order:
                    dev_order.append(d)
        else:
            dev_order = [None]

        for dev in dev_order:
            sub = agg[agg["device"] == dev] if dev else agg
            if sub.empty:
                continue
            if dev is not None:
                dev_set.add(dev)
            color = DEVICE_COLORS.get(dev, "black") if dev else "black"
            num_envs_vals = sorted(sub["config_NUM_ENVS"].unique())
            n = len(num_envs_vals)

            for idx, ne in enumerate(num_envs_vals):
                all_ne_set.add(ne)
                alpha = 0.25 + 0.75 * (idx / max(n - 1, 1))
                row = sub[sub["config_NUM_ENVS"] == ne].sort_values(x_col)
                x = row[x_col].values
                y = row["timing_fps_mean"].values
                yerr = row["timing_fps_std"].values
                ax.plot(x, y, marker="o", color=color, alpha=alpha)
                ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha * 0.2)

        ax.set_xlabel(x_label)
        ax.set_yscale("log")

    _plot_panel(ax_lr, data_lr, "config_link_resources", "FSU per Link")
    _plot_panel(ax_k, data_k, "config_k", "K (shortest paths)")

    ax_lr.set_ylabel("FPS")

    # Shared legend on the right side of the right panel
    handles = []
    # Device entries (colored dots) — GPU first
    dev_legend_order = []
    if "gpu" in dev_set:
        dev_legend_order.append("gpu")
    if "cpu" in dev_set:
        dev_legend_order.append("cpu")
    for d in sorted(dev_set):
        if d not in dev_legend_order:
            dev_legend_order.append(d)
    for dev in dev_legend_order:
        color = DEVICE_COLORS.get(dev, "black")
        handles.append(mlines.Line2D(
            [], [], marker="o", color="none", markerfacecolor=color,
            markeredgecolor=color, markersize=10, label=dev.upper(),
        ))
    # NUM_ENVS entries (black/grey alpha-scaled lines)
    all_ne = sorted(all_ne_set)
    n_all = len(all_ne)
    for idx, ne in enumerate(all_ne):
        alpha = 0.25 + 0.75 * (idx / max(n_all - 1, 1))
        handles.append(mlines.Line2D(
            [], [], marker="o", color="black", alpha=alpha,
            markersize=8, label=f"{int(ne)} envs",
        ))
    ax_k.legend(
        handles=handles, fontsize=16,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=True, borderaxespad=0,
    )

    fig.tight_layout()
    _save(fig, output_dir, "fps_vs_fsu_and_k_by_num_envs")


# -- Main ---------------------------------------------------------------------

PLOT_FUNCTIONS = {
    "fps_vs_num_envs": plot_fps_vs_num_envs,
    "fps_vs_link_resources_and_k": plot_fps_vs_link_resources_and_k,
    "compilation_time": plot_compilation_time,
    "gpu_speedup": plot_gpu_speedup,
    "gn_band_scaling": plot_gn_band_scaling,
    "heatmap": plot_heatmap,
    "cross_env": plot_cross_env,
    "config_summary": plot_config_summary,
    "fps_vs_fsu_and_k_by_num_envs": plot_fps_vs_fsu_and_k_by_num_envs,
}


def main():
    parser = argparse.ArgumentParser(description="Generate XLRON benchmark plots")
    parser.add_argument(
        "--input", default=str(_DEFAULT_CSV),
        help="Path to aggregated CSV (default: benchmarks/results/benchmark_results.csv)",
    )
    parser.add_argument(
        "--results_dir", default=str(_RESULTS_DIR),
        help="Directory containing raw JSONL result files (default: benchmarks/results/)",
    )
    parser.add_argument(
        "--output_dir", default=str(_FIGURES_DIR),
        help="Output directory for figures (default: benchmarks/figures/)",
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
    parser.add_argument(
        "--skip_aggregate", action="store_true",
        help="Skip re-aggregating JSONL files (use existing CSV as-is)",
    )
    args = parser.parse_args()

    configure_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-aggregate raw JSONL results into CSV before plotting
    if not args.skip_aggregate:
        from aggregate_results import aggregate
        print("Aggregating JSONL results...")
        aggregate(results_dir=args.results_dir, output_csv=args.input)
        print()

    df = pd.read_csv(args.input)

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
        fn(df, output_dir, args.device)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
