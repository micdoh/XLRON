"""Plotting script for large topology (TataInd, USA100) evaluation results.

Compares FF-KSP heuristic vs Transformer model across blocking probability,
path characteristics, utilisation, and spectral efficiency.
"""

import json
import pathlib

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from plot_style import (
    configure_style, PRIMARY_COLORS, ACCENT_COLORS, increase_legend_line_thickness,
)

configure_style()

# Font sizes for single-column paper figures
FS_TITLE = 42
FS_LABEL = 38
FS_TICK = 32
FS_LEGEND = 32

# Custom colormaps aligned with project palette
_OCCUPANCY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "occupancy", ["#FFFFFF", PRIMARY_COLORS[1], ACCENT_COLORS[2], PRIMARY_COLORS[0], PRIMARY_COLORS[3]],
)
# Diverging: purple (Transformer higher) ← white → teal (FF-KSP higher)
_DIFF_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "occupancy_diff", ["#4A2870", ACCENT_COLORS[0], "#FFFFFF", PRIMARY_COLORS[0], PRIMARY_COLORS[3]],
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = pathlib.Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
FIGURES.mkdir(exist_ok=True)

MEAN_HT = 25  # mean_service_holding_time (default)
TRAJ_REQUESTS = 100_000

TOPOLOGIES = {
    "tataind": {"display": "TataInd", "eval_loads": (350, 600, 10), "traj_load": 450},
    "usa100": {"display": "USA100", "eval_loads": (550, 750, 50), "traj_load": 620},
}

METHODS = {
    "ff_ksp": {"display": "FF-KSP", "color": PRIMARY_COLORS[3], "marker": "o"},       # dark teal
    "transformer": {"display": "Transformer", "color": ACCENT_COLORS[0], "marker": "s"},  # purple
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_eval_results(topo: str, method: str) -> list[dict]:
    path = RESULTS / topo / f"{topo}_{method}_eval_results.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_traj(topo: str, method: str) -> pd.DataFrame:
    path = RESULTS / topo / f"{topo}_{method}_traj.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. Service blocking probability vs traffic load
# ---------------------------------------------------------------------------

def plot_blocking_vs_load():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    for ax, (topo, tinfo) in zip(axes, TOPOLOGIES.items()):
        for method, minfo in METHODS.items():
            results = load_eval_results(topo, method)
            loads = [r["config"]["load"] for r in results]
            sbp_mean = [r["metrics"]["service_blocking_probability"]["mean"] * 100 for r in results]
            sbp_iqr_lo = [r["metrics"]["service_blocking_probability"]["iqr_lower"] * 100 for r in results]
            sbp_iqr_hi = [r["metrics"]["service_blocking_probability"]["iqr_upper"] * 100 for r in results]
            # Sort by load
            order = np.argsort(loads)
            loads = [loads[i] for i in order]
            sbp_mean = [sbp_mean[i] for i in order]
            sbp_iqr_lo = [sbp_iqr_lo[i] for i in order]
            sbp_iqr_hi = [sbp_iqr_hi[i] for i in order]

            mean_arr = np.array(sbp_mean)
            lo = np.array(sbp_iqr_lo)
            hi = np.array(sbp_iqr_hi)
            ax.plot(
                loads, mean_arr,
                marker=minfo["marker"], color=minfo["color"],
                label=minfo["display"], markersize=10,
            )
            ax.fill_between(
                loads, lo, hi,
                alpha=0.2, color=minfo["color"],
            )

        ax.set_xlabel("Traffic Load (Erlang)", fontsize=FS_LABEL)
        ax.set_title(tinfo["display"], fontsize=FS_TITLE)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-2)
        if topo == "tataind":
            ax.set_xlim(left=400)
        ax.tick_params(labelsize=FS_TICK)
        if topo == "usa100":
            ax.legend(fontsize=FS_LEGEND)

    axes[0].set_ylabel("Bitrate Blocking Probability (%)", fontsize=FS_LABEL)
    plt.tight_layout()
    fig.savefig(FIGURES / "blocking_vs_load.png")
    plt.close(fig)
    print("  -> blocking_vs_load")


def load_cutset_results(topo: str, topk: int | None = None) -> list[dict] | None:
    if topk is not None:
        path = RESULTS / topo / f"{topo}_cutset_bound_topk{topk}_results.jsonl"
    else:
        path = RESULTS / topo / f"{topo}_cutset_bound_results.jsonl"
    if not path.exists():
        return None
    with open(path) as f:
        return [json.loads(line) for line in f]


CUTSET_VARIANTS = [
    {"topk": 4, "label": "Cut-Sets Top 4", "color": PRIMARY_COLORS[1], "marker": "^", "ls": "--"},
    {"topk": 10, "label": "Cut-Sets Top 10", "color": ACCENT_COLORS[2], "marker": "v", "ls": "-."},
    {"topk": None, "label": "Cut-Sets Top 64", "color": ACCENT_COLORS[1], "marker": "D", "ls": ":"},
]


def plot_blocking_vs_load_with_cutset():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    for ax, (topo, tinfo) in zip(axes, TOPOLOGIES.items()):
        # Plot heuristic and transformer methods
        for method, minfo in METHODS.items():
            results = load_eval_results(topo, method)
            loads = [r["config"]["load"] for r in results]
            sbp_mean = [r["metrics"]["service_blocking_probability"]["mean"] * 100 for r in results]
            sbp_iqr_lo = [r["metrics"]["service_blocking_probability"]["iqr_lower"] * 100 for r in results]
            sbp_iqr_hi = [r["metrics"]["service_blocking_probability"]["iqr_upper"] * 100 for r in results]
            order = np.argsort(loads)
            loads = [loads[i] for i in order]
            sbp_mean = [sbp_mean[i] for i in order]
            sbp_iqr_lo = [sbp_iqr_lo[i] for i in order]
            sbp_iqr_hi = [sbp_iqr_hi[i] for i in order]

            ax.plot(
                loads, sbp_mean,
                marker=minfo["marker"], color=minfo["color"],
                label=minfo["display"], markersize=10,
            )
            ax.fill_between(loads, sbp_iqr_lo, sbp_iqr_hi, alpha=0.2, color=minfo["color"])

        # Plot cutset bound variants
        for cv in CUTSET_VARIANTS:
            cutset = load_cutset_results(topo, cv["topk"])
            if not cutset:
                continue
            loads_c = [r["config"]["load"] for r in cutset]
            sbp_c = [r["metrics"]["service_blocking_probability"]["mean"] * 100 for r in cutset]
            sbp_c_lo = [r["metrics"]["service_blocking_probability"]["iqr_lower"] * 100 for r in cutset]
            sbp_c_hi = [r["metrics"]["service_blocking_probability"]["iqr_upper"] * 100 for r in cutset]
            order = np.argsort(loads_c)
            loads_c = [loads_c[i] for i in order]
            sbp_c = [sbp_c[i] for i in order]
            sbp_c_lo = [sbp_c_lo[i] for i in order]
            sbp_c_hi = [sbp_c_hi[i] for i in order]

            ax.plot(
                loads_c, sbp_c,
                marker=cv["marker"], color=cv["color"], linestyle=cv["ls"],
                label=cv["label"], markersize=10,
            )
            ax.fill_between(loads_c, sbp_c_lo, sbp_c_hi, alpha=0.15, color=cv["color"])

        ax.set_xlabel("Traffic Load (Erlang)", fontsize=FS_LABEL)
        ax.set_title(tinfo["display"], fontsize=FS_TITLE)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-2)
        if topo == "tataind":
            ax.set_xlim(left=400)
        ax.tick_params(labelsize=FS_TICK)
        if topo == "usa100":
            ax.legend(fontsize=FS_LEGEND - 4)

    axes[0].set_ylabel("Service Blocking Probability (%)", fontsize=FS_LABEL)
    plt.tight_layout()
    fig.savefig(FIGURES / "blocking_vs_load_with_cutset.png")
    plt.close(fig)
    print("  -> blocking_vs_load_with_cutset")


# ---------------------------------------------------------------------------
# 2. Box plots: path length (km) and hops for each method/topology
# ---------------------------------------------------------------------------

def plot_path_boxplots():
    # Collect available traj data
    data_entries = []
    for topo, tinfo in TOPOLOGIES.items():
        for method, minfo in METHODS.items():
            df = load_traj(topo, method)
            if df is not None:
                data_entries.append((topo, tinfo, method, minfo, df))

    if not data_entries:
        print("  -> SKIPPED path_boxplots (no traj data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    labels = []
    path_lengths = []
    num_hops = []
    colors = []

    for topo, tinfo, method, minfo, df in data_entries:
        method_label = minfo["display"].replace("Transformer", "Trans.")
        labels.append(f"{tinfo['display']}\n{method_label}")
        path_lengths.append(df["path_length"].values)
        num_hops.append(df["num_hops"].values)
        colors.append(minfo["color"])

    boxplot_kw = dict(
        tick_labels=labels, patch_artist=True, showfliers=False, widths=0.6,
        showmeans=True,
        meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="black",
                       markersize=12, markeredgewidth=2),
        medianprops=dict(color="black", linewidth=3),
    )

    # Path length box plot
    bp1 = axes[0].boxplot(path_lengths, **boxplot_kw)
    for patch, c in zip(bp1["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Path Length (km)", fontsize=FS_LABEL)
    axes[0].tick_params(labelsize=FS_TICK)

    # Hops box plot
    bp2 = axes[1].boxplot(num_hops, **boxplot_kw)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Path Length (hops)", fontsize=FS_LABEL)
    axes[1].tick_params(labelsize=FS_TICK)

    plt.tight_layout()
    fig.savefig(FIGURES / "path_boxplots.png")
    plt.close(fig)
    print("  -> path_boxplots")


# ---------------------------------------------------------------------------
# 3. Utilisation over steps (from traj data)
# ---------------------------------------------------------------------------

def plot_utilisation_over_steps():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    has_data = [False, False]
    for idx, (topo, tinfo) in enumerate(TOPOLOGIES.items()):
        ax = axes[idx]
        for method, minfo in METHODS.items():
            df = load_traj(topo, method)
            if df is None:
                continue
            has_data[idx] = True
            # Downsample for plotting
            window = max(1, len(df) // 500)
            util_smooth = df["utilization"].rolling(window=window, min_periods=1).mean()
            steps = np.arange(len(util_smooth))
            ax.plot(
                steps[::window], util_smooth.values[::window],
                color=minfo["color"], label=minfo["display"], linewidth=2,
            )
        ax.set_xlabel("Training Episode", fontsize=FS_LABEL)
        ax.set_title(tinfo["display"], fontsize=FS_TITLE)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}"))
        ax.tick_params(labelsize=FS_TICK)
        if topo == "usa100":
            ax.legend(fontsize=FS_LEGEND)

    axes[0].set_ylabel("Utilisation", fontsize=FS_LABEL)
    # Only save if we have some data
    if any(has_data):
        plt.tight_layout()
        fig.savefig(FIGURES / "utilisation_over_steps.png")
        print("  -> utilisation_over_steps")
    else:
        print("  -> SKIPPED utilisation_over_steps (no traj data)")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Bitrate blocking probability over steps (from traj data)
# ---------------------------------------------------------------------------

def plot_bitrate_blocking_over_steps():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    has_data = [False, False]
    for idx, (topo, tinfo) in enumerate(TOPOLOGIES.items()):
        ax = axes[idx]
        for method, minfo in METHODS.items():
            df = load_traj(topo, method)
            if df is None:
                continue
            has_data[idx] = True
            window = max(1, len(df) // 500)
            bbp_smooth = df["bitrate_blocking_probability"].rolling(window=window, min_periods=1).mean()
            steps = np.arange(len(bbp_smooth))
            ax.plot(
                steps[::window], bbp_smooth.values[::window] * 100,
                color=minfo["color"], label=minfo["display"], linewidth=2,
            )
        ax.set_xlabel(r"Request Index ($\times 10^3$)", fontsize=FS_LABEL)
        ax.set_title(tinfo["display"], fontsize=FS_TITLE)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}"))
        ax.tick_params(labelsize=FS_TICK)
        if topo == "usa100":
            ax.legend(fontsize=FS_LEGEND)

    axes[0].set_ylabel("Bitrate Blocking Probability (%)", fontsize=FS_LABEL)
    if any(has_data):
        plt.tight_layout()
        fig.savefig(FIGURES / "bitrate_blocking_over_steps.png")
        print("  -> bitrate_blocking_over_steps")
    else:
        print("  -> SKIPPED bitrate_blocking_over_steps (no traj data)")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Box plots: spectral efficiency and required slots
# ---------------------------------------------------------------------------

def plot_se_slots_boxplots():
    data_entries = []
    for topo, tinfo in TOPOLOGIES.items():
        for method, minfo in METHODS.items():
            df = load_traj(topo, method)
            if df is not None:
                data_entries.append((topo, tinfo, method, minfo, df))

    if not data_entries:
        print("  -> SKIPPED se_slots_boxplots (no traj data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    labels = []
    se_data = []
    slots_data = []
    colors = []

    for topo, tinfo, method, minfo, df in data_entries:
        labels.append(f"{tinfo['display']}\n{minfo['display']}")
        se_data.append(df["path_spectral_efficiency"].values)
        slots_data.append(df["required_slots"].values)
        colors.append(minfo["color"])

    boxplot_kw = dict(
        tick_labels=labels, patch_artist=True, showfliers=False, widths=0.6,
        showmeans=True,
        meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="black",
                       markersize=12, markeredgewidth=2),
        medianprops=dict(color="black", linewidth=3),
    )

    # Spectral efficiency box plot
    bp1 = axes[0].boxplot(se_data, **boxplot_kw)
    for patch, c in zip(bp1["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Spectral Efficiency (b/s/Hz)", fontsize=FS_LABEL)
    axes[0].set_title("Spectral Efficiency Distribution", fontsize=FS_TITLE)
    axes[0].tick_params(labelsize=FS_TICK)

    # Required slots box plot
    bp2 = axes[1].boxplot(slots_data, **boxplot_kw)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Required Slots", fontsize=FS_LABEL)
    axes[1].set_title("Required Slots Distribution", fontsize=FS_TITLE)
    axes[1].tick_params(labelsize=FS_TICK)

    plt.tight_layout()
    fig.savefig(FIGURES / "se_slots_boxplots.png")
    plt.close(fig)
    print("  -> se_slots_boxplots")


# ---------------------------------------------------------------------------
# 6. Per-request path comparison (hops and km, FF-KSP vs Transformer)
# ---------------------------------------------------------------------------

def plot_path_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(24, 16), sharex="col")

    window = 1000  # smoothing window for readability

    for col, (topo, tinfo) in enumerate(TOPOLOGIES.items()):
        for method, minfo in METHODS.items():
            df = load_traj(topo, method)
            if df is None:
                continue
            steps = np.arange(len(df))
            pl_smooth = df["path_length"].rolling(window=window, min_periods=1).mean()
            hops_smooth = df["num_hops"].rolling(window=window, min_periods=1).mean()

            # Top row: path length in km
            axes[0, col].plot(
                steps[::window], pl_smooth.values[::window],
                color=minfo["color"], label=minfo["display"], linewidth=2,
            )

            # Bottom row: path length in hops
            axes[1, col].plot(
                steps[::window], hops_smooth.values[::window],
                color=minfo["color"], label=minfo["display"], linewidth=2,
            )

        axes[0, col].set_title(tinfo["display"], fontsize=FS_TITLE)
        axes[1, col].set_xlabel(r"Request Index ($\times 10^3$)", fontsize=FS_LABEL)
        for row in range(2):
            axes[row, col].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}"))
            axes[row, col].tick_params(labelsize=FS_TICK)
        # Legend only on upper USA100
        if col == 1:
            axes[0, col].legend(fontsize=FS_LEGEND)

        # Tighten y-axis to data range for better differentiation
        for row in range(2):
            ymin, ymax = axes[row, col].get_ylim()
            margin = (ymax - ymin) * 0.05
            axes[row, col].set_ylim(ymin - margin, ymax + margin)
            axes[row, col].margins(y=0.02)

    axes[0, 0].set_ylabel("Path Length (km)", fontsize=FS_LABEL)
    axes[1, 0].set_ylabel("Path Length (hops)", fontsize=FS_LABEL)

    plt.tight_layout()
    fig.savefig(FIGURES / "path_comparison.png")
    plt.close(fig)
    print("  -> path_comparison")


# ---------------------------------------------------------------------------
# 7. Delta between FF-KSP and Transformer path choices
# ---------------------------------------------------------------------------

def plot_path_delta():
    fig, axes = plt.subplots(2, 2, figsize=(24, 16), sharex="col")

    window = 500  # smooth the delta for readability

    for col, (topo, tinfo) in enumerate(TOPOLOGIES.items()):
        df_ff = load_traj(topo, "ff_ksp")
        df_tr = load_traj(topo, "transformer")
        if df_ff is None or df_tr is None:
            continue

        n = min(len(df_ff), len(df_tr))
        steps = np.arange(n)

        # Delta: Transformer minus FF-KSP (negative = Transformer chose shorter)
        delta_km = df_tr["path_length"].values[:n] - df_ff["path_length"].values[:n]
        delta_hops = df_tr["num_hops"].values[:n] - df_ff["num_hops"].values[:n]

        delta_km_smooth = pd.Series(delta_km).rolling(window=window, min_periods=1).mean()
        delta_hops_smooth = pd.Series(delta_hops).rolling(window=window, min_periods=1).mean()

        # Top row: delta km
        axes[0, col].plot(steps, delta_km_smooth, color="black", linewidth=0.5)
        axes[0, col].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        axes[0, col].fill_between(
            steps, 0, delta_km_smooth, where=delta_km_smooth > 0,
            alpha=0.3, color=PRIMARY_COLORS[3], label="FF-KSP shorter",
        )
        axes[0, col].fill_between(
            steps, 0, delta_km_smooth, where=delta_km_smooth < 0,
            alpha=0.3, color=ACCENT_COLORS[0], label="Transformer shorter",
        )

        # Bottom row: delta hops
        axes[1, col].plot(steps, delta_hops_smooth, color="black", linewidth=0.5)
        axes[1, col].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        axes[1, col].fill_between(
            steps, 0, delta_hops_smooth, where=delta_hops_smooth > 0,
            alpha=0.3, color=PRIMARY_COLORS[3], label="FF-KSP shorter",
        )
        axes[1, col].fill_between(
            steps, 0, delta_hops_smooth, where=delta_hops_smooth < 0,
            alpha=0.3, color=ACCENT_COLORS[0], label="Transformer shorter",
        )

        axes[0, col].set_title(tinfo["display"], fontsize=FS_TITLE)
        axes[1, col].set_xlabel(r"Request Index ($\times 10^3$)", fontsize=FS_LABEL)
        for row in range(2):
            axes[row, col].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}"))
            axes[row, col].tick_params(labelsize=FS_TICK)
        if col == 1:
            axes[0, col].legend(fontsize=FS_LEGEND)

    axes[0, 0].set_ylabel(r"$\Delta$ Path Length (km)", fontsize=FS_LABEL)
    axes[1, 0].set_ylabel(r"$\Delta$ Path Length (hops)", fontsize=FS_LABEL)

    plt.tight_layout()
    fig.savefig(FIGURES / "path_delta.png")
    plt.close(fig)
    print("  -> path_delta")


# ---------------------------------------------------------------------------
# 8. Slot occupancy heatmaps
# ---------------------------------------------------------------------------

def _load_occupancy_data(topo):
    """Load link labels and occupancy arrays normalised by total holding time sum.

    Each cell becomes: (accumulated time on this link-slot) / (sum of all holding times),
    giving the fraction of total network holding time consumed by each link-slot.
    """
    labels_path = RESULTS / topo / f"{topo}_link_labels.json"
    if not labels_path.exists():
        return None, {}
    with open(labels_path) as f:
        link_labels = json.load(f)
    occ = {}
    for method in METHODS:
        npz_path = RESULTS / topo / f"{topo}_{method}_slot_occupancy.npz"
        traj_path = RESULTS / topo / f"{topo}_{method}_traj.csv"
        if npz_path.exists() and traj_path.exists():
            raw = np.load(npz_path)["occupancy"]
            # Normalise by total holding time across all requests
            df = pd.read_csv(traj_path, usecols=["departure_time"])
            total_ht = df["departure_time"].sum()  # arrival=0, so ht = departure
            occ[method] = raw / total_ht
    return link_labels, occ


def _set_heatmap_axes(ax, num_links, num_slots):
    """Configure y-axis (Link ID every 20, large labels) and x-axis (grid every 10, labels every 40)."""
    # Y-axis: numeric link IDs every 20, no horizontal grid
    ytick_step = 20
    yticks = np.arange(0, num_links, ytick_step)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(i) for i in yticks], fontsize=14)

    # X-axis: 1-indexed so range is [1, num_slots]. Grid every 10, labels every 40.
    major_xticks = np.arange(40, num_slots + 1, 40)  # 40, 80, ..., 320
    minor_xticks = np.arange(10, num_slots + 1, 10)  # 10, 20, ..., 320
    ax.set_xticks(major_xticks - 1, minor=False)      # pixel positions (0-indexed)
    ax.set_xticklabels([str(x) for x in major_xticks])
    ax.set_xticks(minor_xticks - 1, minor=True)
    ax.grid(which="minor", axis="x", color="white", linewidth=0.3, alpha=0.5)
    ax.grid(which="major", axis="y", visible=False)
    ax.tick_params(which="minor", bottom=False)


def plot_slot_occupancy():
    fig, axes = plt.subplots(2, 2, figsize=(28, 24))
    method_order = list(METHODS.keys())

    all_occ = {}
    all_labels = {}
    for topo, tinfo in TOPOLOGIES.items():
        link_labels, occ = _load_occupancy_data(topo)
        if link_labels is None:
            print(f"  -> SKIPPED slot_occupancy (no labels for {topo})")
            return
        all_labels[topo] = link_labels
        all_occ[topo] = occ

    for col, (topo, tinfo) in enumerate(TOPOLOGIES.items()):
        # Per-topology normalisation range (shared across methods within topology)
        topo_vals = [all_occ[topo][m] for m in method_order if m in all_occ[topo]]
        vmin = min(v.min() for v in topo_vals)
        vmax = max(v.max() for v in topo_vals)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        for row, method in enumerate(method_order):
            if method not in all_occ[topo]:
                continue
            occupancy = all_occ[topo][method]
            minfo = METHODS[method]
            ax = axes[row, col]
            im = ax.imshow(
                occupancy, aspect="auto", origin="lower",
                interpolation="nearest", cmap=_OCCUPANCY_CMAP, norm=norm,
            )
            ax.set_title(f"{tinfo['display']} \u2014 {minfo['display']}", fontsize=FS_TITLE)
            _set_heatmap_axes(ax, occupancy.shape[0], occupancy.shape[1])
            ax.tick_params(labelsize=FS_TICK)
            if row == 1:
                ax.set_xlabel("FSU Index", fontsize=FS_LABEL)
            if col == 0:
                ax.set_ylabel("Link ID", fontsize=FS_LABEL)

        # One colorbar per topology column
        cbar = fig.colorbar(im, ax=axes[:, col], label="Relative FSU Occupancy", shrink=0.6)
        cbar.set_label("Relative FSU Occupancy", fontsize=FS_LABEL)
        cbar.ax.tick_params(labelsize=FS_TICK)
    fig.savefig(FIGURES / "slot_occupancy.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> slot_occupancy")


# ---------------------------------------------------------------------------
# 9. Slot occupancy difference heatmaps (FF-KSP minus Transformer)
# ---------------------------------------------------------------------------

def plot_slot_occupancy_diff():
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), gridspec_kw={"wspace": 0.15})

    diffs = {}
    all_labels = {}
    for topo, tinfo in TOPOLOGIES.items():
        link_labels, occ = _load_occupancy_data(topo)
        if link_labels is None or "ff_ksp" not in occ or "transformer" not in occ:
            print(f"  -> SKIPPED slot_occupancy_diff ({topo})")
            continue
        diffs[topo] = occ["ff_ksp"] - occ["transformer"]
        all_labels[topo] = link_labels

    if not diffs:
        return

    topo_list = [t for t in TOPOLOGIES if t in diffs]
    for col, topo in enumerate(topo_list):
        tinfo = TOPOLOGIES[topo]
        ax = axes[col]
        vmax = np.abs(diffs[topo]).max()
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(
            diffs[topo], aspect="auto", origin="lower",
            interpolation="nearest", cmap=_DIFF_CMAP, norm=norm,
        )
        ax.set_title(tinfo["display"], fontsize=FS_TITLE)
        _set_heatmap_axes(ax, diffs[topo].shape[0], diffs[topo].shape[1])
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel("FSU Index", fontsize=FS_LABEL)
        if col == 0:
            ax.set_ylabel("Link ID", fontsize=FS_LABEL)

        # Only add colorbar on the right subplot
        if col == len(topo_list) - 1:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_ticks([])
            cbar.set_label(r"$\Delta$ Relative FSU Occupancy", labelpad=15, fontsize=FS_LABEL)
            # Place text labels at top and bottom of colorbar
            cbar.ax.text(
                1.8, 1.02, "FF-KSP", transform=cbar.ax.transAxes,
                ha="center", va="bottom", fontsize=30, fontweight="bold",
            )
            cbar.ax.text(
                1.2, -0.02, "RL", transform=cbar.ax.transAxes,
                ha="center", va="top", fontsize=30, fontweight="bold",
            )

    fig.savefig(FIGURES / "slot_occupancy_diff.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> slot_occupancy_diff")


# ---------------------------------------------------------------------------
# 10. Relative link usage difference (Transformer vs FF-KSP)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 10b. USA100 ablation: blocking probability vs training steps
# ---------------------------------------------------------------------------

ABLATIONS = {
    "original": {"display": "All Features", "color": ACCENT_COLORS[1], "linestyle": "dashdot"},
    "onpolicy": {"display": "No Off-Policy IAM", "color": ACCENT_COLORS[0]},
    "nodamping": {"display": "No Damping", "color": PRIMARY_COLORS[0]},
    "nogating": {"display": "No Gating", "color": ACCENT_COLORS[3]},
    "nogating_nodamping": {"display": "No Gating +\nNo Damping", "color": PRIMARY_COLORS[2]},
    "novml": {"display": "No VML", "color": "#B8860B"},
}

FFKSP_LABELS = {
    "usa100": "FF-KSP",
    "tataind": "FF-KSP",
}


def _load_ablation(ablation_dir, name: str, smooth: int = 3):
    """Load mean, iqr_lower, iqr_upper CSVs for an ablation variant."""
    d = ablation_dir / name
    mean_df = pd.read_csv(d / f"{name}_mean.csv")
    lo_df = pd.read_csv(d / f"{name}_iqr_lower.csv")
    hi_df = pd.read_csv(d / f"{name}_iqr_upper.csv")
    # Use episode_count (col 0) as x-axis
    episodes = mean_df.iloc[:, 0].values.astype(float)
    mean_vals = mean_df.iloc[:, 1].values.astype(float) * 100
    lo_vals = lo_df.iloc[:, 1].values.astype(float) * 100
    hi_vals = hi_df.iloc[:, 1].values.astype(float) * 100
    # Running average to smooth (use pandas for proper edge handling)
    if smooth > 1:
        mean_vals = pd.Series(mean_vals).rolling(smooth, min_periods=1).mean().values
        lo_vals = pd.Series(lo_vals).rolling(smooth, min_periods=1).mean().values
        hi_vals = pd.Series(hi_vals).rolling(smooth, min_periods=1).mean().values
    return episodes, mean_vals, lo_vals, hi_vals


def _plot_ablation_panel(ax, topo: str):
    """Plot ablation series for a single topology on the given axes."""
    ablation_dir = RESULTS / topo / "ablations"
    if not ablation_dir.exists():
        return

    # Plot FF-KSP as horizontal band
    ffksp_dir = ablation_dir / "ffksp"
    if ffksp_dir.exists():
        steps, mean_vals, lo_vals, hi_vals = _load_ablation(ablation_dir, "ffksp")
        ax.plot(
            steps, mean_vals,
            color=PRIMARY_COLORS[3], label=FFKSP_LABELS[topo],
            linestyle="--",
        )
        ax.fill_between(steps, lo_vals, hi_vals, alpha=0.2, color=PRIMARY_COLORS[3])

    # Plot each ablation variant
    for name, info in ABLATIONS.items():
        d = ablation_dir / name
        if not d.exists():
            continue
        steps, mean_vals, lo_vals, hi_vals = _load_ablation(ablation_dir, name)
        ax.plot(
            steps, mean_vals,
            color=info["color"], label=info["display"],
            linestyle=info.get("linestyle", "-"),
        )
        ax.fill_between(steps, lo_vals, hi_vals, alpha=0.2, color=info["color"])

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=20))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6], numticks=20))
    _fmt = mticker.FuncFormatter(lambda x, _: f"{x:g}")
    ax.yaxis.set_minor_formatter(_fmt)
    ax.yaxis.set_major_formatter(_fmt)
    ax.grid(axis="y", which="both", linewidth=0.5, alpha=0.4)
    ax.set_title(TOPOLOGIES[topo]["display"])


def plot_ablation_blocking():
    fig, axes = plt.subplots(1, 2, figsize=(28, 10.5), sharey=True)

    for col, topo in enumerate(["tataind", "usa100"]):
        _plot_ablation_panel(axes[col], topo)
        axes[col].set_xlabel("Training Episode")

    axes[0].set_ylabel("Bitrate Blocking Probability (%)")
    # Collect legend handles/labels from the second panel (has all entries)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.01), fontsize=FS_LEGEND,
               columnspacing=1.0, handletextpad=0.4)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.23)
    fig.savefig(FIGURES / "ablation_blocking.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> ablation_blocking")


def plot_link_usage_delta():
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)

    for col, (topo, tinfo) in enumerate(TOPOLOGIES.items()):
        ff_path = RESULTS / topo / f"{topo}_ff_ksp_link_usage.npy"
        tr_path = RESULTS / topo / f"{topo}_transformer_link_usage.npy"
        if not ff_path.exists() or not tr_path.exists():
            continue

        ff_usage = np.load(ff_path).astype(float)
        tr_usage = np.load(tr_path).astype(float)

        # Relative difference: (Transformer - FF-KSP) / FF-KSP * 100
        delta_pct = (tr_usage - ff_usage) / ff_usage * 100

        ax = axes[col]
        link_ids = np.arange(len(delta_pct))
        colors = [ACCENT_COLORS[0] if d < 0 else PRIMARY_COLORS[3] for d in delta_pct]
        ax.bar(link_ids, delta_pct, color=colors, width=1.0, edgecolor="none")
        ax.axhline(0, color="black", linewidth=1, linestyle="-")
        ax.set_xlabel("Link ID", fontsize=FS_LABEL)
        ax.set_title(tinfo["display"], fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PRIMARY_COLORS[3], label="Transformer uses more"),
            Patch(facecolor=ACCENT_COLORS[0], label="Transformer uses less"),
        ]
        if col == 1:
            ax.legend(handles=legend_elements, fontsize=FS_LEGEND)

    axes[0].set_ylabel(r"$\Delta$ Link Usage (%)", fontsize=FS_LABEL)
    plt.tight_layout()
    fig.savefig(FIGURES / "link_usage_delta.png")
    plt.close(fig)
    print("  -> link_usage_delta")


LOSS_COMPONENTS = {
    "total_loss": {"display": "Total Loss", "color": "black", "linestyle": "--"},
    "actor_loss": {"display": "Actor Loss", "color": "red"},
    "validmass_loss": {"display": "Valid Mass Loss", "color": ACCENT_COLORS[2]},
    "value_loss": {"display": "Value Loss", "color": ACCENT_COLORS[0]},
    "entropy_loss": {"display": "Entropy Loss", "color": PRIMARY_COLORS[0]},
}


def _load_loss(loss_dir, name: str, smooth: int = 50):
    """Load a loss CSV and return (steps, values)."""
    path = loss_dir / f"{name}.csv"
    df = pd.read_csv(path)
    steps = df.iloc[:, 4].values.astype(float)
    vals = df.iloc[:, 1].values.astype(float)
    if name == "entropy_loss":
        vals = -vals
    if name == "total_loss":
        steps, vals = steps[5:], vals[5:]
    if smooth > 1:
        vals = pd.Series(vals).rolling(smooth, min_periods=1).mean().values
    return steps, vals


def _plot_loss_panel(ax, topo: str):
    """Plot all loss components for a single topology on the given axes."""
    loss_dir = RESULTS / topo / "loss"
    if not loss_dir.exists():
        return
    for name, info in LOSS_COMPONENTS.items():
        path = loss_dir / f"{name}.csv"
        if not path.exists():
            continue
        steps, vals = _load_loss(loss_dir, name)
        ax.plot(
            steps, vals,
            color=info["color"], label=info["display"],
            linestyle=info.get("linestyle", "-"),
        )
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.grid(axis="y", which="both", linewidth=0.5, alpha=0.4)
    ax.set_title(TOPOLOGIES[topo]["display"])


def plot_loss_components():
    fig, axes = plt.subplots(1, 2, figsize=(28, 10), sharey=True)
    for col, topo in enumerate(["tataind", "usa100"]):
        _plot_loss_panel(axes[col], topo)
    axes[0].set_xlabel("Update Step")
    axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Update Step")
    # Collect legend handles/labels from either panel
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.01), fontsize=FS_LEGEND)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.savefig(FIGURES / "loss_components.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> loss_components")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating large topology comparison plots...")
    plot_blocking_vs_load()
    plot_blocking_vs_load_with_cutset()
    plot_path_boxplots()
    plot_utilisation_over_steps()
    plot_bitrate_blocking_over_steps()
    plot_se_slots_boxplots()
    plot_path_comparison()
    plot_path_delta()
    plot_slot_occupancy()
    plot_slot_occupancy_diff()
    plot_link_usage_delta()
    plot_ablation_blocking()
    plot_loss_components()
    print("\nAll plots saved to:", FIGURES)
