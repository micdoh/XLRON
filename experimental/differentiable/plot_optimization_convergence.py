"""Plot optimization convergence: accepted services vs optimization step.

Loads results JSON files and plots:
  - Case 0: initialized from scratch
  - Case 1: initialized from heuristic
  - Horizontal line for heuristic baseline

Usage:
  uv run python experimental/differentiable/plot_optimization_convergence.py
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plot_style import configure_style, PALETTE, ACCENT_COLORS, PRIMARY_COLORS

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


def main():
    configure_style(
        font_size=24,
        axes_label_size=28,
        tick_size=22,
        legend_size=20,
        figure_dpi=150,
    )

    # Load data
    with open(os.path.join(FIGURES_DIR, "optimization_results_nsfnet_nevin_undirected_2000req.json")) as f:
        case0 = json.load(f)
    with open(os.path.join(FIGURES_DIR, "optimization_results_nsfnet_nevin_undirected_2000req_heur.json")) as f:
        case1 = json.load(f)

    max_requests = 2000.0

    # Convert rewards to accepted services
    heuristic_accepted = max_requests + case0["heuristic_reward"]
    case0_rewards = np.array(case0["rewards_per_iteration"])
    case0_accepted = max_requests + case0_rewards
    case1_rewards = np.array(case1["rewards_per_iteration"])
    case1_accepted = max_requests + case1_rewards
    case1_best = max_requests + case1["best_reward"]

    steps_case0 = np.arange(len(case0_accepted))
    steps_case1 = np.arange(len(case1_accepted))

    fig, ax = plt.subplots(figsize=(11, 7))

    # Heuristic baseline
    ax.axhline(y=heuristic_accepted, color=ACCENT_COLORS[0], linestyle='--',
               linewidth=3.0, label=f'KSP-FF (best: {heuristic_accepted:.0f})', zorder=3)

    # Case 0: from scratch
    ax.plot(steps_case0, case0_accepted, color=PRIMARY_COLORS[0], linewidth=2.2,
            alpha=0.85, label=f'Case 0: From scratch (best: {max_requests + case0["best_reward"]:.0f})', zorder=2)

    # Case 1: from heuristic (show trajectory)
    ax.plot(steps_case1, case1_accepted, color=ACCENT_COLORS[1], linewidth=2.2,
            alpha=0.85, label=f'Case 1: From heuristic (best: {case1_best:.0f})', zorder=2)

    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Accepted Services')
    ax.legend(loc='lower right', framealpha=0.9)

    ax.set_xlim(0, max(len(case0_accepted), len(case1_accepted)))

    plt.tight_layout()
    fname = "optimization_convergence.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname))
    print(f"Saved {fname}")


if __name__ == "__main__":
    main()
