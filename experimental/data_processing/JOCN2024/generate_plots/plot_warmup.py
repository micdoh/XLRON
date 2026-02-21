"""Box plots of warm-up period estimation using mean-crossing rule.

Extracted from warmup.ipynb (first method -- mean crossing rule).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add experimental/ to path so plot_style is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from plot_style import configure_style

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def simulate_warmup(n_repeats=1000, num_requests=100_000):
    """Run mean-crossing warmup simulation across loads and arrival rates."""
    results = []
    for _rep in range(n_repeats):
        for load in np.arange(50, 1050, 50):
            for arrival_rate in [5, 10, 15, 20, 25]:
                mean_service_holding_time = load / arrival_rate
                services = []
                active_services = []
                current_time = 0
                for i in range(num_requests):
                    current_time += np.random.exponential(1 / arrival_rate)
                    holding_time = np.random.exponential(mean_service_holding_time)
                    services.append(current_time + holding_time)
                    active_services = [x for x in services if x > current_time]
                    if len(active_services) >= load:
                        results.append({
                            "arrival_rate": arrival_rate,
                            "mean_service_holding_time": mean_service_holding_time,
                            "load": load,
                            "current_time": current_time,
                            "active_services": len(active_services),
                            "num_requests": i,
                        })
                        break
    return pd.DataFrame(results)


def plot_warmup_boxplots(df):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Get series of num_requests for each load
    loads = sorted(df["load"].unique())
    requests = [df[df["load"] == load]["num_requests"] for load in loads]

    # Create box plot
    bp = ax.boxplot(
        requests, positions=loads, widths=20, patch_artist=True,
        showfliers=False, showmeans=True,
    )

    # Customize box appearance
    for box in bp['boxes']:
        box.set(facecolor='white', edgecolor='black')
    for whisker in bp['whiskers']:
        whisker.set(color='black')
    for cap in bp['caps']:
        cap.set(color='black')
    for median in bp['medians']:
        median.set(color='red')

    # Extract upper whisker values and fit a line
    upper_whiskers = [item.get_ydata()[1] for item in bp['whiskers'][1::2]]
    coeffs = np.polyfit(loads, upper_whiskers, 1)
    poly = np.poly1d(coeffs)

    # Calculate R-squared
    y_pred = poly(loads)
    y_mean = np.mean(upper_whiskers)
    ss_tot = np.sum((upper_whiskers - y_mean) ** 2)
    ss_res = np.sum((upper_whiskers - y_pred) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    label = f'Fitted line: y = {coeffs[0]:.2f}x\nR\u00b2 = {r_squared:.4f}'
    ax.plot(loads, poly(loads), color='r', linestyle='--', label=label)

    ax.set_xlabel("Traffic load (Erlang)", size=18)
    ax.set_ylabel("Requests until steady state", size=18)
    ax.legend(fontsize=16)
    ax.tick_params(labelsize=16)
    ax.set_xticks(loads[1::2])
    ax.set_xticklabels([f'{load:.0f}' for load in loads[1::2]])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'steady_state_boxplots.png')
    plt.show()


def main():
    configure_style(font_size=18, axes_label_size=18, tick_size=16, legend_size=16)
    print("Running warmup simulation (this may take a while)...")
    df = simulate_warmup(n_repeats=1000)
    plot_warmup_boxplots(df)


if __name__ == '__main__':
    main()
