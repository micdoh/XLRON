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
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

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

        ax.set_xlabel("Traffic Load (Erlang)")
        ax.set_title(tinfo["display"])
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-2)
        if topo == "tataind":
            ax.set_xlim(left=400)
        ax.legend()

    axes[0].set_ylabel("Service Blocking Probability (%)")
    plt.tight_layout()
    fig.savefig(FIGURES / "blocking_vs_load.png")
    plt.close(fig)
    print("  -> blocking_vs_load")


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

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    labels = []
    path_lengths = []
    num_hops = []
    colors = []

    for topo, tinfo, method, minfo, df in data_entries:
        labels.append(f"{tinfo['display']}\n{minfo['display']}")
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
    axes[0].set_ylabel("Path Length (km)")
    axes[0].set_title("Path Length Distribution")

    # Hops box plot
    bp2 = axes[1].boxplot(num_hops, **boxplot_kw)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Number of Hops")
    axes[1].set_title("Path Hops Distribution")

    plt.tight_layout()
    fig.savefig(FIGURES / "path_boxplots.png")
    plt.close(fig)
    print("  -> path_boxplots")


# ---------------------------------------------------------------------------
# 3. Utilisation over steps (from traj data)
# ---------------------------------------------------------------------------

def plot_utilisation_over_steps():
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

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
        ax.set_xlabel(r"Traffic Request (x10$^3$)")
        ax.set_title(tinfo["display"])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}"))
        ax.legend()

    axes[0].set_ylabel("Utilisation")
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
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

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
        ax.set_xlabel(r"Traffic Request (x10$^3$)")
        ax.set_title(tinfo["display"])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}"))
        ax.legend()

    axes[0].set_ylabel("Bitrate Blocking Probability (%)")
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

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

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
    axes[0].set_ylabel("Spectral Efficiency (b/s/Hz)")
    axes[0].set_title("Spectral Efficiency Distribution")

    # Required slots box plot
    bp2 = axes[1].boxplot(slots_data, **boxplot_kw)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Required Slots")
    axes[1].set_title("Required Slots Distribution")

    plt.tight_layout()
    fig.savefig(FIGURES / "se_slots_boxplots.png")
    plt.close(fig)
    print("  -> se_slots_boxplots")


# ---------------------------------------------------------------------------
# 6. Per-request path comparison (hops and km, FF-KSP vs Transformer)
# ---------------------------------------------------------------------------

def plot_path_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(24, 12), sharex="col")

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

        axes[0, col].set_title(tinfo["display"])
        axes[1, col].set_xlabel(r"Traffic Request (x10$^3$)")
        for row in range(2):
            axes[row, col].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}"))
        # Legend only on right-hand figures
        if col == 1:
            axes[0, col].legend()
            axes[1, col].legend()

        # Tighten y-axis to data range for better differentiation
        for row in range(2):
            ymin, ymax = axes[row, col].get_ylim()
            margin = (ymax - ymin) * 0.05
            axes[row, col].set_ylim(ymin - margin, ymax + margin)
            axes[row, col].margins(y=0.02)

    axes[0, 0].set_ylabel("Path Length (km)")
    axes[1, 0].set_ylabel("Path Length (hops)")

    plt.tight_layout()
    fig.savefig(FIGURES / "path_comparison.png")
    plt.close(fig)
    print("  -> path_comparison")


# ---------------------------------------------------------------------------
# 7. Delta between FF-KSP and Transformer path choices
# ---------------------------------------------------------------------------

def plot_path_delta():
    fig, axes = plt.subplots(2, 2, figsize=(24, 12), sharex="col")

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

        axes[0, col].set_title(tinfo["display"])
        axes[1, col].set_xlabel(r"Traffic Request (x10$^3$)")
        for row in range(2):
            axes[row, col].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}"))
        if col == 1:
            axes[0, col].legend()

    axes[0, 0].set_ylabel(r"$\Delta$ Path Length (km)")
    axes[1, 0].set_ylabel(r"$\Delta$ Path Length (hops)")

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
            ax.set_title(f"{tinfo['display']} \u2014 {minfo['display']}")
            _set_heatmap_axes(ax, occupancy.shape[0], occupancy.shape[1])
            if row == 1:
                ax.set_xlabel("FSU Index")
            if col == 0:
                ax.set_ylabel("Link ID")

        # One colorbar per topology column
        fig.colorbar(im, ax=axes[:, col], label="Relative FSU Occupancy", shrink=0.6)
    fig.savefig(FIGURES / "slot_occupancy.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> slot_occupancy")


# ---------------------------------------------------------------------------
# 9. Slot occupancy difference heatmaps (FF-KSP minus Transformer)
# ---------------------------------------------------------------------------

def plot_slot_occupancy_diff():
    fig, axes = plt.subplots(1, 2, figsize=(28, 14), gridspec_kw={"wspace": 0.08})

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
        ax.set_title(tinfo["display"])
        _set_heatmap_axes(ax, diffs[topo].shape[0], diffs[topo].shape[1])
        ax.set_xlabel("FSU Index")
        if col == 0:
            ax.set_ylabel("Link ID")

        # Only add colorbar on the right subplot
        if col == len(topo_list) - 1:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_ticks([])
            cbar.set_label(r"$\Delta$ Relative FSU Occupancy", labelpad=15)
            # Place text labels at top and bottom of colorbar
            cbar.ax.text(
                1.5, 1.02, "FF-KSP", transform=cbar.ax.transAxes,
                ha="center", va="bottom", fontsize=22, fontweight="bold",
            )
            cbar.ax.text(
                1.5, -0.02, "Transformer", transform=cbar.ax.transAxes,
                ha="center", va="top", fontsize=22, fontweight="bold",
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
    "original": {"display": "Original", "color": PRIMARY_COLORS[3]},
    "gating1": {"display": "No Gating", "color": ACCENT_COLORS[1]},
    "nodamping": {"display": "No Damping", "color": PRIMARY_COLORS[0]},
    "onpolicy": {"display": "On-Policy IAM", "color": ACCENT_COLORS[0]},
}

ABLATION_DIR = RESULTS / "usa100" / "ablations"


def _load_ablation(name: str):
    """Load mean, iqr_lower, iqr_upper CSVs for an ablation variant."""
    d = ABLATION_DIR / name
    mean_df = pd.read_csv(d / f"{name}_mean.csv")
    lo_df = pd.read_csv(d / f"{name}_iqr_lower.csv")
    hi_df = pd.read_csv(d / f"{name}_iqr_upper.csv")
    # Step column has the run name prefix — grab by position (col index 4)
    steps = mean_df.iloc[:, 4].values.astype(float)
    mean_vals = mean_df.iloc[:, 1].values.astype(float) * 100
    lo_vals = lo_df.iloc[:, 1].values.astype(float) * 100
    hi_vals = hi_df.iloc[:, 1].values.astype(float) * 100
    return steps, mean_vals, lo_vals, hi_vals


def plot_ablation_blocking():
    if not ABLATION_DIR.exists():
        print("  -> SKIPPED ablation_blocking (no ablation data)")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    for name, info in ABLATIONS.items():
        d = ABLATION_DIR / name
        if not d.exists():
            continue
        steps, mean_vals, lo_vals, hi_vals = _load_ablation(name)
        ax.plot(
            steps, mean_vals,
            color=info["color"], label=info["display"],
        )
        ax.fill_between(steps, lo_vals, hi_vals, alpha=0.2, color=info["color"])

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=20))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6], numticks=20))
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_minor_formatter().set_scientific(False)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Service Blocking Probability (%)")
    ax.set_title("USA100")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e3:.0f}k"))
    ax.grid(axis="y", which="both", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    fig.savefig(FIGURES / "usa100_ablation_blocking.png")
    plt.close(fig)
    print("  -> usa100_ablation_blocking")


def plot_link_usage_delta():
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=True)

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
        ax.set_xlabel("Link ID")
        ax.set_title(tinfo["display"])

        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PRIMARY_COLORS[3], label="Transformer uses more"),
            Patch(facecolor=ACCENT_COLORS[0], label="Transformer uses less"),
        ]
        if col == 1:
            ax.legend(handles=legend_elements)

    axes[0].set_ylabel(r"$\Delta$ Link Usage (%)")
    plt.tight_layout()
    fig.savefig(FIGURES / "link_usage_delta.png")
    plt.close(fig)
    print("  -> link_usage_delta")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating large topology comparison plots...")
    plot_blocking_vs_load()
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
    print("\nAll plots saved to:", FIGURES)
