"""Heuristic comparison plots: blocking probability vs K and vs traffic load.

Extracted from heuristic_comparison.ipynb.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add experimental/ to path so plot_style is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from plot_style import configure_style

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

TOPOLOGIES = np.array([
    'nsfnet_deeprmsa_directed',
    'cost239_deeprmsa_directed',
    'usnet_gcnrnn_directed',
    'jpn48_directed',
])


def load_data():
    data_dir = Path(__file__).resolve().parents[4] / 'experiment_data' / 'JOCN2024'
    heur_data = pd.read_csv(data_dir / 'heuristic_comparison_high_traffic_new.csv')
    k_data = pd.read_csv(data_dir / 'k_traffic_comparison_new_new.csv')
    traffic_data = pd.read_csv(data_dir / 'experiment_results_traffic.csv')

    # Keep only relevant columns
    cols = [
        'TOPOLOGY', 'HEUR', 'K', 'LOAD',
        'service_blocking_probability_mean',
        'service_blocking_probability_std',
        'service_blocking_probability_iqr_lower',
        'service_blocking_probability_iqr_upper',
    ]
    heur_data = heur_data[cols]

    # Filter out kmc_ff and kmf_ff
    heur_data = heur_data[~heur_data['HEUR'].str.contains('kmc_ff|kmf_ff')]

    # Filter to matching traffic levels
    heur_data = heur_data[
        ~((heur_data['TOPOLOGY'].str.contains('cost239')) & (heur_data['LOAD'] != 317))
    ]
    heur_data = heur_data[
        ~((heur_data['TOPOLOGY'].str.contains('jpn48')) & (heur_data['LOAD'] != 115))
    ]

    return heur_data, k_data, traffic_data


def plot_heuristic_comparison(heur_data):
    """Blocking probability vs K for each topology, one heuristic per line."""
    fig, axs = plt.subplots(1, len(TOPOLOGIES), figsize=(20, 5))
    axes = []
    for n, topology in enumerate(TOPOLOGIES):
        ax = axs[n]
        ax.set_title(topology.split('_')[0].upper())
        ax.set_yscale('log')
        if n == 0:
            ax.set_ylim(0.3, 6)
            ax.set_xlabel('K')
            ax.set_ylabel('Service Blocking Prob. (%)')
            ax.set_yticks([0.3, 0.5, 1, 2, 3, 4, 5, 6], minor=True)
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.1f}'))
            ax.yaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter('{x:.1f}'))
            ax.set_xticks(heur_data['K'].unique())
        else:
            ax.sharey(axes[0])
            ax.sharex(axes[0])
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.2, axis='y')
        ax.grid(which='major', color='black', linestyle='-', linewidth=0.2, axis='y')

        for heuristic in heur_data['HEUR'].unique():
            df = heur_data[
                (heur_data['TOPOLOGY'] == topology) & (heur_data['HEUR'] == heuristic)
            ]
            ax.plot(
                df['K'],
                df['service_blocking_probability_mean'] * 100,
                label=heuristic.upper().replace('_', '-'),
            )
            upper = df['service_blocking_probability_mean'] + df['service_blocking_probability_std']
            lower = df['service_blocking_probability_mean'] - df['service_blocking_probability_std']
            ax.fill_between(df['K'], lower * 100, upper * 100, alpha=0.2)
        axes.append(ax)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'heuristic_comparison.png')
    plt.show()


def plot_k_traffic_comparison(k_data):
    """Blocking probability vs K for each topology, one traffic level per line."""
    # Fix IQR values
    k_data = k_data.copy()
    k_data['service_blocking_probability_iqr_lower'] = (
        k_data['service_blocking_probability_iqr_lower'].ffill()
    )
    k_data['service_blocking_probability_iqr_upper'] = np.where(
        k_data['service_blocking_probability_iqr_upper'] == 0,
        k_data['service_blocking_probability_mean'],
        k_data['service_blocking_probability_iqr_upper'],
    )

    topologies = k_data['TOPOLOGY'].unique()
    fig, axs = plt.subplots(1, len(topologies), figsize=(20, 5))
    axes = []
    for n, topology in enumerate(topologies):
        ax = axs[n]
        ax.set_title(topology.split('_')[0].upper())
        ax.set_yscale('log')
        if n == 0:
            ax.set_ylim(0.0005, 10)
            ax.set_ylabel('Service Blocking Prob. (%)')
            ax.set_xlabel('K')
            ax.set_xticks([2, 5, 8, 12, 16, 20, 25, 30, 40])
        else:
            ax.sharey(axes[0])
            ax.sharex(axes[0])
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        ax.grid(which='major', color='black', linestyle='-', linewidth=0.2, axis='y')

        labels = ["High", " ", "", " ", " ", "Low"]
        loads = k_data[k_data['TOPOLOGY'] == topology]['LOAD'].unique()
        if topology == 'jpn48_directed':
            loads = np.array([52, 61, 72, 91, 100, 120])

        colors = plt.cm.viridis(np.linspace(0, 1, len(loads)))
        for i, traffic in enumerate(loads[::-1]):
            df = k_data[(k_data['TOPOLOGY'] == topology) & (k_data['LOAD'] == traffic)]
            ax.plot(
                df['K'],
                df['service_blocking_probability_mean'] * 100,
                color=colors[i],
                label=labels[i],
            )
            upper = df['service_blocking_probability_mean'] + df['service_blocking_probability_std']
            lower = df['service_blocking_probability_mean'] - df['service_blocking_probability_std']
            ax.fill_between(df['K'], lower * 100, upper * 100, alpha=0.2, color=colors[i])
        axes.append(ax)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.text(46, 2.8, s='Traffic')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'k_traffic_comparison.png')
    plt.show()


def plot_heuristic_traffic_comparison(traffic_data):
    """Blocking probability vs traffic load for each topology, one heuristic per line."""
    # Fix IQR values
    traffic_data = traffic_data.copy()
    traffic_data['service_blocking_probability_iqr_lower'] = (
        traffic_data['service_blocking_probability_iqr_lower'].ffill()
    )
    traffic_data['service_blocking_probability_iqr_upper'] = np.where(
        traffic_data['service_blocking_probability_iqr_upper'] == 0,
        traffic_data['service_blocking_probability_mean'],
        traffic_data['service_blocking_probability_iqr_upper'],
    )

    fig, axs = plt.subplots(1, len(TOPOLOGIES), figsize=(20, 5))
    axes = []
    for n, topology in enumerate(TOPOLOGIES):
        ax = axs[n]
        ax.set_title(topology.split('_')[0].upper())
        ax.set_yscale('log')
        if n == 0:
            ax.set_ylabel('Service Blocking Prob. (%)')
            ax.set_xlabel('Traffic Load (Erlang)')
        else:
            ax.sharey(axes[0])
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        ax.grid(which='major', color='black', linestyle='-', linewidth=0.2, axis='y')

        for heuristic in traffic_data['HEUR'].unique():
            df = traffic_data[
                (traffic_data['TOPOLOGY'] == topology) & (traffic_data['HEUR'] == heuristic)
            ]
            ax.plot(
                df['LOAD'],
                df['service_blocking_probability_mean'] * 100,
                label=heuristic.upper().replace('_', '-'),
            )
            upper = df['service_blocking_probability_iqr_upper']
            lower = df['service_blocking_probability_iqr_lower']
            ax.fill_between(df['LOAD'], lower * 100, upper * 100, alpha=0.2)
        axes.append(ax)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'heuristic_50_traffic_comparison.png')
    plt.show()


def main():
    configure_style(font_size=20, axes_label_size=20, tick_size=16, legend_size=14)
    heur_data, k_data, traffic_data = load_data()
    plot_heuristic_comparison(heur_data)
    plot_k_traffic_comparison(k_data)
    plot_heuristic_traffic_comparison(traffic_data)


if __name__ == '__main__':
    main()
