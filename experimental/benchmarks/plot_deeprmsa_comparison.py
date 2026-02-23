#!/usr/bin/env python3
"""Compare DeepRMSA training curves: Original vs XLRON.

Reads deeprmsa_benchmark.csv (XLRON) and deeprmsa_original_training_results.csv
(Original) from the results directory and produces three plots:
  1. Service Blocking Probability (%) vs Environment Steps
  2. Service Blocking Probability (%) vs Time (s) [log scale, execution only]
  3. Service Blocking Probability (%) vs Time (s) [log scale, with compilation]

Usage:
    python experimental/benchmarks/plot_deeprmsa_comparison.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experimental.plot_style import COMPARISON_COLORS, REFERENCE_LINE_COLOR, configure_style

# -- Configurable constants ---------------------------------------------------

# XLRON training config.
XLRON_TRAINING_FPS = 1.77e6  # total env steps/s (execution only, excluding compilation)
XLRON_NUM_ENVS = 512  # parallel envs; CSV "step" column is per-env steps
XLRON_COMPILATION_TIME = 75.39  # seconds; JIT compilation before training starts

# Original DeepRMSA config.
ORIGINAL_NUM_AGENTS = 10  # parallel training agents

# -- Paths --------------------------------------------------------------------

_BENCHMARKS_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _BENCHMARKS_DIR / "results"
_FIGURES_DIR = _BENCHMARKS_DIR / "figures"

_XLRON_CSV = _RESULTS_DIR / "deeprmsa_benchmark.csv"
_ORIGINAL_CSV = _RESULTS_DIR / "deeprmsa_original_training_results.csv"
_OPTICALGRLGYM_CSV = _RESULTS_DIR / "training_iqr_data.csv"


def _load_xlron() -> pd.DataFrame:
    df = pd.read_csv(_XLRON_CSV)
    df["bp_pct"] = df["service_blocking_probability_mean"] * 100
    df["bp_lower_pct"] = df["service_blocking_probability_iqr_lower"] * 100
    df["bp_upper_pct"] = df["service_blocking_probability_iqr_upper"] * 100
    # CSV "step" is per-env; total env steps = step * NUM_ENVS
    df["total_steps"] = df["step"] * XLRON_NUM_ENVS
    df["execution_time"] = df["total_steps"] / XLRON_TRAINING_FPS
    df["wall_time"] = XLRON_COMPILATION_TIME + df["execution_time"]
    return df


def _load_training_iqr() -> pd.DataFrame:
    df = pd.read_csv(_OPTICALGRLGYM_CSV)
    df["bp_pct"] = df["sbp_mean"] * 100
    df["bp_lower_pct"] = df["sbp_q1"] * 100
    df["bp_upper_pct"] = df["sbp_q3"] * 100
    df["total_steps"] = df["global_step"]  # already total steps
    # Estimate compilation offset by linear extrapolation to step 0.
    n = min(10, len(df))
    steps = df["global_step"].iloc[:n].values
    times = df["time_elapsed"].iloc[:n].values
    coeffs = np.polyfit(steps, times, 1)
    compilation_offset = max(coeffs[1], 0.0)
    df["execution_time"] = df["time_elapsed"] - compilation_offset
    # Rebase so execution time starts from 0
    df["execution_time"] = df["execution_time"] - df["execution_time"].iloc[0]
    df["wall_time"] = df["time_elapsed"] - df["time_elapsed"].iloc[0]
    # Drop first 3000 rows (early noisy data) and rebase times to 0
    df = df.iloc[3000:].reset_index(drop=True)
    df["execution_time"] = df["execution_time"] - df["execution_time"].iloc[0]
    df["wall_time"] = df["wall_time"] - df["wall_time"].iloc[0]
    return df


def _load_original() -> pd.DataFrame:
    df = pd.read_csv(_ORIGINAL_CSV)
    df["bp_pct"] = df["bp_median"] * 100
    df["bp_lower_pct"] = df["bp_q1"] * 100
    df["bp_upper_pct"] = df["bp_q3"] * 100
    # Total env steps across all parallel agents
    df["total_steps"] = df["steps_median"] * ORIGINAL_NUM_AGENTS
    # Estimate compilation overhead by linear extrapolation to step 0.
    n = min(10, len(df))
    steps = df["steps_median"].iloc[:n].values
    times = df["time_median"].iloc[:n].values
    coeffs = np.polyfit(steps, times, 1)  # slope, intercept
    compilation_offset = max(coeffs[1], 0.0)  # intercept = compilation time
    df["execution_time"] = df["time_median"] - compilation_offset
    # Rebase so execution time starts from 0
    df["execution_time"] = df["execution_time"] - df["execution_time"].iloc[0]
    return df


def plot_bp_vs_steps(xlron: pd.DataFrame, original: pd.DataFrame,
                     training_iqr: pd.DataFrame, output_dir: Path):
    """Plot 1: Service Blocking Probability (%) vs Environment Steps."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Original (total steps across all agents)
    ax.plot(original["total_steps"], original["bp_pct"],
            color=COMPARISON_COLORS["deeprmsa_original"], label="DeepRMSA", linewidth=1.5)
    ax.fill_between(original["total_steps"],
                    original["bp_lower_pct"], original["bp_upper_pct"],
                    color=COMPARISON_COLORS["deeprmsa_original"], alpha=0.2)

    # XLRON (total steps across all envs)
    ax.plot(xlron["total_steps"], xlron["bp_pct"],
            color=COMPARISON_COLORS["xlron"], label="XLRON", linewidth=1.5)
    ax.fill_between(xlron["total_steps"],
                    xlron["bp_lower_pct"], xlron["bp_upper_pct"],
                    color=COMPARISON_COLORS["xlron"], alpha=0.2)

    # Optical-RL-Gym
    ax.plot(training_iqr["total_steps"], training_iqr["bp_pct"],
            color=COMPARISON_COLORS["optical_rl_gym"], label="Optical-RL-Gym", linewidth=1.5)
    ax.fill_between(training_iqr["total_steps"],
                    training_iqr["bp_lower_pct"], training_iqr["bp_upper_pct"],
                    color=COMPARISON_COLORS["optical_rl_gym"], alpha=0.2)

    # KSP-FF reference line
    ax.axhline(y=5.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.5, label=r"5-SP-FF$_{km}$")

    ax.set_yscale("log")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Service Blocking Probability (%)")
    ax.legend()
    ax.set_xlim(left=0)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "deeprmsa_bp_vs_steps.png", dpi=200)
    plt.close(fig)
    print(f"  Saved deeprmsa_bp_vs_steps.png")


def plot_bp_vs_time(xlron: pd.DataFrame, original: pd.DataFrame,
                    training_iqr: pd.DataFrame, output_dir: Path):
    """Plot 2: Service Blocking Probability (%) vs Time (s) [log scale, execution only]."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Original
    ax.plot(original["execution_time"], original["bp_pct"],
            color=COMPARISON_COLORS["deeprmsa_original"], label="DeepRMSA", linewidth=1.5)
    ax.fill_between(original["execution_time"],
                    original["bp_lower_pct"], original["bp_upper_pct"],
                    color=COMPARISON_COLORS["deeprmsa_original"], alpha=0.2)

    # XLRON
    ax.plot(xlron["execution_time"], xlron["bp_pct"],
            color=COMPARISON_COLORS["xlron"], label="XLRON", linewidth=1.5)
    ax.fill_between(xlron["execution_time"],
                    xlron["bp_lower_pct"], xlron["bp_upper_pct"],
                    color=COMPARISON_COLORS["xlron"], alpha=0.2)

    # Optical-RL-Gym
    ax.plot(training_iqr["execution_time"], training_iqr["bp_pct"],
            color=COMPARISON_COLORS["optical_rl_gym"], label="Optical-RL-Gym", linewidth=1.5)
    ax.fill_between(training_iqr["execution_time"],
                    training_iqr["bp_lower_pct"], training_iqr["bp_upper_pct"],
                    color=COMPARISON_COLORS["optical_rl_gym"], alpha=0.2)

    # KSP-FF reference line
    ax.axhline(y=5.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.5, label=r"5-SP-FF$_{km}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Service Blocking Probability (%)")
    ax.legend()
    ax.set_xlim(left=0.1)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "deeprmsa_bp_vs_time.png", dpi=200)
    plt.close(fig)
    print(f"  Saved deeprmsa_bp_vs_time.png")


def plot_bp_vs_wall_time(xlron: pd.DataFrame, original: pd.DataFrame,
                         training_iqr: pd.DataFrame, output_dir: Path):
    """Plot 3: Service Blocking Probability (%) vs Wall Time (s) [with compilation]."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Compilation shaded region
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 55
    ax.axvspan(0, XLRON_COMPILATION_TIME, alpha=0.15, color="gray",
               label="XLRON Compilation")

    # Original
    ax.plot(original["execution_time"], original["bp_pct"],
            color=COMPARISON_COLORS["deeprmsa_original"], label="DeepRMSA", linewidth=1.5)
    ax.fill_between(original["execution_time"],
                    original["bp_lower_pct"], original["bp_upper_pct"],
                    color=COMPARISON_COLORS["deeprmsa_original"], alpha=0.2)

    # XLRON (shifted by compilation time)
    ax.plot(xlron["wall_time"], xlron["bp_pct"],
            color=COMPARISON_COLORS["xlron"], label="XLRON", linewidth=1.5)
    ax.fill_between(xlron["wall_time"],
                    xlron["bp_lower_pct"], xlron["bp_upper_pct"],
                    color=COMPARISON_COLORS["xlron"], alpha=0.2)

    # Optical-RL-Gym
    ax.plot(training_iqr["wall_time"], training_iqr["bp_pct"],
            color=COMPARISON_COLORS["optical_rl_gym"], label="Optical-RL-Gym", linewidth=1.5)
    ax.fill_between(training_iqr["wall_time"],
                    training_iqr["bp_lower_pct"], training_iqr["bp_upper_pct"],
                    color=COMPARISON_COLORS["optical_rl_gym"], alpha=0.2)

    # KSP-FF reference line
    ax.axhline(y=5.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.5,
               label=r"5-SP-FF$_{km}$")

    # Speedup annotations
    xlron_end = xlron["wall_time"].max()
    irl_end = training_iqr["wall_time"].max()
    original_end = original["execution_time"].max()
    xlron_exec = xlron["execution_time"].max()

    # Faint vertical lines at end of each data series
    for x_end in (xlron_end, original_end, irl_end):
        ax.axvline(x=x_end, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

    # Speedup of XLRON relative to each baseline
    exec_speedup_vs_deeprmsa = original_end / xlron_exec
    exec_speedup_vs_irl = training_iqr["execution_time"].max() / xlron_exec

    # Arrow 1: XLRON ↔ DeepRMSA
    arrow_y1 = 10.0
    ax.annotate("", xy=(original_end, arrow_y1), xytext=(xlron_end, arrow_y1),
                arrowprops=dict(arrowstyle="<->", color="black", lw=2,
                                shrinkA=0, shrinkB=0))

    # Arrow 2: same x-start (XLRON end) ↔ Optical-RL-Gym end (slightly below)
    arrow_y2 = arrow_y1 * 0.85
    ax.annotate("", xy=(irl_end, arrow_y2), xytext=(xlron_end, arrow_y2),
                arrowprops=dict(arrowstyle="<->", color="black", lw=2,
                                shrinkA=0, shrinkB=0))

    label = f"Execution speedup: {exec_speedup_vs_deeprmsa:.0f}x to {exec_speedup_vs_irl:.0f}x"
    ax.text(1e3, 6.2, label, ha="center", va="bottom",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.5, alpha=0.9))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Service Blocking Probability (%)")
    ax.legend(loc="upper right")
    ax.set_xlim(left=10)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "deeprmsa_bp_vs_wall_time.png", dpi=200)
    plt.close(fig)
    print(f"  Saved deeprmsa_bp_vs_wall_time.png")


def plot_bp_steps_and_wall_time(xlron: pd.DataFrame, original: pd.DataFrame,
                                training_iqr: pd.DataFrame, output_dir: Path):
    """Combined: BP vs Steps (left) and BP vs Wall Time (right), side by side."""
    fig, (ax_steps, ax_wt) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # -- Helper to plot all three series on an axis --
    def _plot_series(ax, x_col_xlron, x_col_original, x_col_irl):
        ax.plot(original[x_col_original], original["bp_pct"],
                color=COMPARISON_COLORS["deeprmsa_original"], label="DeepRMSA", linewidth=1.5)
        ax.fill_between(original[x_col_original],
                        original["bp_lower_pct"], original["bp_upper_pct"],
                        color=COMPARISON_COLORS["deeprmsa_original"], alpha=0.2)

        ax.plot(xlron[x_col_xlron], xlron["bp_pct"],
                color=COMPARISON_COLORS["xlron"], label="XLRON", linewidth=1.5)
        ax.fill_between(xlron[x_col_xlron],
                        xlron["bp_lower_pct"], xlron["bp_upper_pct"],
                        color=COMPARISON_COLORS["xlron"], alpha=0.2)

        ax.plot(training_iqr[x_col_irl], training_iqr["bp_pct"],
                color=COMPARISON_COLORS["optical_rl_gym"], label="Optical-RL-Gym", linewidth=1.5)
        ax.fill_between(training_iqr[x_col_irl],
                        training_iqr["bp_lower_pct"], training_iqr["bp_upper_pct"],
                        color=COMPARISON_COLORS["optical_rl_gym"], alpha=0.2)

        ax.axhline(y=5.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.5,
                   label=r"5-SP-FF$_{km}$")
        ax.set_yscale("log")

    # Left panel: BP vs Steps
    _plot_series(ax_steps, "total_steps", "total_steps", "total_steps")
    ax_steps.set_xlabel("Environment Steps")
    ax_steps.set_ylabel("Service Blocking Probability (%)")
    ax_steps.set_xlim(left=0)

    # Right panel: BP vs Wall Time (with compilation region and annotations)
    ax_wt.axvspan(0, XLRON_COMPILATION_TIME, alpha=0.15, color="gray",
                  label="XLRON Compilation")
    _plot_series(ax_wt, "wall_time", "execution_time", "wall_time")
    ax_wt.set_xscale("log")
    ax_wt.set_xlabel("Time (s)")
    ax_wt.set_xlim(left=10)

    # Speedup annotations on right panel
    xlron_end = xlron["wall_time"].max()
    irl_end = training_iqr["wall_time"].max()
    original_end = original["execution_time"].max()
    xlron_exec = xlron["execution_time"].max()

    for x_end in (xlron_end, original_end, irl_end):
        ax_wt.axvline(x=x_end, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

    exec_speedup_vs_deeprmsa = original_end / xlron_exec
    exec_speedup_vs_irl = training_iqr["execution_time"].max() / xlron_exec

    # Arrow 1: XLRON ↔ DeepRMSA
    arrow_y1 = 10.0
    ax_wt.annotate("", xy=(original_end, arrow_y1), xytext=(xlron_end, arrow_y1),
                   arrowprops=dict(arrowstyle="<->", color="black", lw=2,
                                   shrinkA=0, shrinkB=0))

    # Arrow 2: same x-start (XLRON end) ↔ Optical-RL-Gym end (slightly below)
    arrow_y2 = arrow_y1 * 0.85
    ax_wt.annotate("", xy=(irl_end, arrow_y2), xytext=(xlron_end, arrow_y2),
                   arrowprops=dict(arrowstyle="<->", color="black", lw=2,
                                   shrinkA=0, shrinkB=0))

    label = f"Speedup: {exec_speedup_vs_deeprmsa:.0f}x to {exec_speedup_vs_irl:.0f}x"
    ax_wt.text(1e3, 6.2, label, ha="center", va="bottom",
               fontsize=10, family="monospace",
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.5, alpha=0.9))

    # Single legend from left panel (avoid duplicates from right)
    handles, labels = ax_steps.get_legend_handles_labels()
    # Add compilation region handle from right panel
    wt_handles, wt_labels = ax_wt.get_legend_handles_labels()
    for h, l in zip(wt_handles, wt_labels):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    ax_wt.legend(handles, labels, loc="upper right", fontsize=9)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "deeprmsa_bp_combined.png", dpi=200)
    plt.close(fig)
    print(f"  Saved deeprmsa_bp_combined.png")


def main():
    configure_style(font_size=14, axes_label_size=16, tick_size=12, legend_size=12)
    xlron = _load_xlron()
    original = _load_original()
    training_iqr = _load_training_iqr()

    print(f"XLRON: {len(xlron)} rows, total steps {xlron['total_steps'].min():,.0f}-{xlron['total_steps'].max():,.0f} "
          f"(per-env steps * {XLRON_NUM_ENVS} envs)")
    print(f"Original: {len(original)} rows, total steps {original['total_steps'].min():,.0f}-{original['total_steps'].max():,.0f} "
          f"(per-agent steps * {ORIGINAL_NUM_AGENTS} agents)")
    print(f"Training IQR: {len(training_iqr)} rows, total steps {training_iqr['total_steps'].min():,.0f}-{training_iqr['total_steps'].max():,.0f}")
    print(f"XLRON execution time: {xlron['execution_time'].min():.1f}-{xlron['execution_time'].max():.1f}s "
          f"(FPS={XLRON_TRAINING_FPS:.0f}), compilation: {XLRON_COMPILATION_TIME:.0f}s")
    print(f"Original execution time: {original['execution_time'].min():.1f}-{original['execution_time'].max():.1f}s")
    print(f"Training IQR wall time: {training_iqr['wall_time'].min():.1f}-{training_iqr['wall_time'].max():.1f}s")

    plot_bp_vs_steps(xlron, original, training_iqr, _FIGURES_DIR)
    plot_bp_vs_time(xlron, original, training_iqr, _FIGURES_DIR)
    plot_bp_vs_wall_time(xlron, original, training_iqr, _FIGURES_DIR)
    plot_bp_steps_and_wall_time(xlron, original, training_iqr, _FIGURES_DIR)


if __name__ == "__main__":
    main()
