"""Histogram showing the effect of truncating holding time to < 2*mean.

Extracted from truncation.ipynb.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add experimental/ to path so plot_style is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from plot_style import configure_style

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def deeprmsa_sample(mean):
    """Sample from exponential distribution truncated at 2*mean."""
    holding_time = np.random.exponential(mean)
    while holding_time > 2 * mean:
        holding_time = np.random.exponential(mean)
    return holding_time


def plot_truncation(mean=25, n_samples=1_000_000):
    deeprmsa = np.array([deeprmsa_sample(mean) for _ in range(n_samples)])
    rmsa = np.random.exponential(mean, size=n_samples)

    max_ht = mean
    deeprmsa_norm = deeprmsa / max_ht
    rmsa_norm = rmsa / max_ht
    mean_ht_truncated = np.mean(deeprmsa_norm)
    mean_ht = np.mean(rmsa_norm)

    bins = np.arange(0, 10.01, 0.01)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(deeprmsa_norm, bins=bins, label="Truncated", alpha=0.5, density=True)
    ax.hist(rmsa_norm, bins=bins, label="Not truncated", alpha=0.5, density=True)
    ax.axvline(mean_ht_truncated, color='b', linestyle='dashed', linewidth=1,
               label="Mean truncated")
    ax.axvline(mean_ht, color='r', linestyle='dashed', linewidth=1,
               label="Mean not truncated")
    ax.legend(fontsize=11)
    ax.set_xlabel("Normalised holding time", size=14)
    ax.set_ylabel("Probability density", size=14)
    ax.set_xlim([0, 6])
    ax.tick_params(labelsize=12)

    # Arrow and annotation showing the mean decrease
    pct_decrease = int(((mean_ht - mean_ht_truncated) / mean_ht) * 100)
    ax.arrow(mean_ht, 0.8, -mean_ht + mean_ht_truncated, 0,
             head_width=0.05, head_length=0.1, fc='k', ec='k',
             length_includes_head=True)
    ax.text(mean_ht + 0.15, 0.75,
            f"{pct_decrease}% decrease in\nmean holding time",
            bbox=dict(facecolor='white', alpha=0.5), size=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'truncation.png')


def main():
    configure_style(font_size=20, axes_label_size=20, tick_size=16, legend_size=14)
    plot_truncation()


if __name__ == '__main__':
    main()
