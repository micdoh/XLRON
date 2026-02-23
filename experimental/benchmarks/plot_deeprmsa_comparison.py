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

from experimental.plot_style import configure_style

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


def plot_bp_vs_steps(xlron: pd.DataFrame, original: pd.DataFrame, output_dir: Path):
    """Plot 1: Service Blocking Probability (%) vs Environment Steps."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Original (total steps across all agents)
    ax.plot(original["total_steps"], original["bp_pct"],
            color="#1f77b4", label="DeepRMSA Original", linewidth=1.5)
    ax.fill_between(original["total_steps"],
                    original["bp_lower_pct"], original["bp_upper_pct"],
                    color="#1f77b4", alpha=0.2)

    # XLRON (total steps across all envs)
    ax.plot(xlron["total_steps"], xlron["bp_pct"],
            color="#ff7f0e", label="DeepRMSA XLRON", linewidth=1.5)
    ax.fill_between(xlron["total_steps"],
                    xlron["bp_lower_pct"], xlron["bp_upper_pct"],
                    color="#ff7f0e", alpha=0.2)

    # KSP-FF reference line
    ax.axhline(y=4.0, color="green", linestyle="--", linewidth=1.5, label="KSP-FF")

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


def plot_bp_vs_time(xlron: pd.DataFrame, original: pd.DataFrame, output_dir: Path):
    """Plot 2: Service Blocking Probability (%) vs Time (s) [log scale, execution only]."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Original
    ax.plot(original["execution_time"], original["bp_pct"],
            color="#1f77b4", label="DeepRMSA Original", linewidth=1.5)
    ax.fill_between(original["execution_time"],
                    original["bp_lower_pct"], original["bp_upper_pct"],
                    color="#1f77b4", alpha=0.2)

    # XLRON
    ax.plot(xlron["execution_time"], xlron["bp_pct"],
            color="#ff7f0e", label="DeepRMSA XLRON", linewidth=1.5)
    ax.fill_between(xlron["execution_time"],
                    xlron["bp_lower_pct"], xlron["bp_upper_pct"],
                    color="#ff7f0e", alpha=0.2)

    # KSP-FF reference line
    ax.axhline(y=4.0, color="green", linestyle="--", linewidth=1.5, label="KSP-FF")

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


def plot_bp_vs_wall_time(xlron: pd.DataFrame, original: pd.DataFrame, output_dir: Path):
    """Plot 3: Service Blocking Probability (%) vs Wall Time (s) [with compilation]."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Compilation shaded region
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 55
    ax.axvspan(0, XLRON_COMPILATION_TIME, alpha=0.15, color="gray",
               label="XLRON Compilation")

    # Original
    ax.plot(original["execution_time"], original["bp_pct"],
            color="#1f77b4", label="DeepRMSA Original", linewidth=1.5)
    ax.fill_between(original["execution_time"],
                    original["bp_lower_pct"], original["bp_upper_pct"],
                    color="#1f77b4", alpha=0.2)

    # XLRON (shifted by compilation time)
    ax.plot(xlron["wall_time"], xlron["bp_pct"],
            color="#ff7f0e", label="DeepRMSA XLRON", linewidth=1.5)
    ax.fill_between(xlron["wall_time"],
                    xlron["bp_lower_pct"], xlron["bp_upper_pct"],
                    color="#ff7f0e", alpha=0.2)

    # KSP-FF reference line
    ax.axhline(y=4.0, color="green", linestyle="--", linewidth=1.5, label="KSP-FF")

    # Speedup annotation arrow at y=10%
    xlron_end = xlron["wall_time"].max()
    original_end = original["execution_time"].max()
    xlron_exec = xlron["execution_time"].max()
    exec_speedup = original_end / xlron_exec
    wall_speedup = original_end / xlron_end
    xlron_final_bp = xlron["bp_pct"].iloc[-1]
    original_final_bp = original["bp_pct"].iloc[-1]
    blocking_reduction = original_final_bp / xlron_final_bp
    arrow_y = 10.0
    ax.annotate("", xy=(original_end, arrow_y), xytext=(xlron_end, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="red", lw=2))
    label = (f"Execution speedup  : {exec_speedup:>5.0f}x\n"
             f"Wall-clock speedup : {wall_speedup:>5.0f}x\n"
             f"Blocking reduction : {blocking_reduction:>5.1f}x")
    # Place text box just below the arrow, centred in log-space
    text_x = np.sqrt(xlron_end * original_end) * 1.5  # geometric mean shifted right
    ax.text(text_x, arrow_y * 0.75, label, ha="center", va="top",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.5, alpha=0.9))

    # Vertical arrow showing blocking reduction
    ax.annotate("", xy=(150, 3.4), xytext=(150, original_final_bp),
                arrowprops=dict(arrowstyle="<->", color="red", lw=2))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Service Blocking Probability (%)")
    ax.legend()
    ax.set_xlim(left=10)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "deeprmsa_bp_vs_wall_time.png", dpi=200)
    plt.close(fig)
    print(f"  Saved deeprmsa_bp_vs_wall_time.png")


def main():
    configure_style(font_size=14, axes_label_size=16, tick_size=12, legend_size=12)
    xlron = _load_xlron()
    original = _load_original()

    print(f"XLRON: {len(xlron)} rows, total steps {xlron['total_steps'].min():,.0f}-{xlron['total_steps'].max():,.0f} "
          f"(per-env steps * {XLRON_NUM_ENVS} envs)")
    print(f"Original: {len(original)} rows, total steps {original['total_steps'].min():,.0f}-{original['total_steps'].max():,.0f} "
          f"(per-agent steps * {ORIGINAL_NUM_AGENTS} agents)")
    print(f"XLRON execution time: {xlron['execution_time'].min():.1f}-{xlron['execution_time'].max():.1f}s "
          f"(FPS={XLRON_TRAINING_FPS:.0f}), compilation: {XLRON_COMPILATION_TIME:.0f}s")
    print(f"Original execution time: {original['execution_time'].min():.1f}-{original['execution_time'].max():.1f}s")

    plot_bp_vs_steps(xlron, original, _FIGURES_DIR)
    plot_bp_vs_time(xlron, original, _FIGURES_DIR)
    plot_bp_vs_wall_time(xlron, original, _FIGURES_DIR)


if __name__ == "__main__":
    main()
