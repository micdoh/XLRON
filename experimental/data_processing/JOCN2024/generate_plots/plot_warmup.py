"""Box plots of warm-up period estimation using mean-crossing rule.

Extracted from warmup.ipynb (first method -- mean crossing rule).
JAX-accelerated: uses jax.lax.fori_loop + jax.vmap for ~100x speedup.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add experimental/ to path so plot_style is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from plot_style import configure_style

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Circular buffer size must exceed max load (1000).
# Safe because expired entries depart before the buffer wraps:
#   mean_holding_time = load/arrival_rate < BUF_SIZE/arrival_rate  ⟹  load < BUF_SIZE
_BUF_SIZE = 2048
_MAX_REQUESTS = 5_000  # Convergence typically < 3×load; 5K is safe up to load=1000


def simulate_warmup(n_repeats=1000, max_requests=_MAX_REQUESTS, seed=42):
    """JAX-accelerated warmup simulation across loads and arrival rates.

    For each (load, arrival_rate) combo and each repeat, finds the number of
    requests until the number of active services first reaches the offered load.

    Uses the same exponential sampling pattern as
    ``xlron.environments.env_funcs.generate_arrival_holding_times``:
      - inter-arrival times:  exponential / arrival_rate
      - holding times:        exponential * mean_service_holding_time
    """

    @jax.jit
    def _single_sim(key, load_f, arrival_rate_f):
        """Simulate one trial, return request index at first crossing (or -1)."""
        mean_ht = load_f / arrival_rate_f
        key_a, key_h = jax.random.split(key)

        # Pre-generate all random variates (vectorised)
        inter_arrivals = (
            jax.random.exponential(key_a, shape=(max_requests,), dtype=jnp.float32)
            / arrival_rate_f
        )
        holding_times = (
            jax.random.exponential(key_h, shape=(max_requests,), dtype=jnp.float32)
            * mean_ht
        )

        def body(i, carry):
            current_time, buf, result_i = carry
            current_time = current_time + inter_arrivals[i]
            departure = current_time + holding_times[i]
            # Write departure to circular buffer position
            buf = buf.at[i % _BUF_SIZE].set(departure)
            # Count active services (departure > current_time)
            active = jnp.sum(buf > current_time)
            # Record first crossing only
            crossed = active >= load_f
            result_i = jnp.where((result_i < 0) & crossed, i, result_i)
            return (current_time, buf, result_i)

        init = (
            jnp.float32(0.0),
            jnp.zeros(_BUF_SIZE, dtype=jnp.float32),
            jnp.int32(-1),
        )
        _, _, result_i = jax.lax.fori_loop(0, max_requests, body, init)
        return result_i

    # Vectorise across repeats
    _batch_sim = jax.jit(jax.vmap(_single_sim, in_axes=(0, None, None)))

    master_key = jax.random.PRNGKey(seed)
    results = []
    loads = np.arange(50, 1050, 50)
    arrival_rates = [5, 10, 15, 20, 25]
    total_configs = len(loads) * len(arrival_rates)

    t0 = time.perf_counter()
    for cfg_idx, load in enumerate(loads):
        for ar_idx, arrival_rate in enumerate(arrival_rates):
            done = cfg_idx * len(arrival_rates) + ar_idx + 1
            elapsed = time.perf_counter() - t0
            print(
                f"\r  config {done}/{total_configs}  "
                f"(load={load}, ar={arrival_rate})  "
                f"[{elapsed:.1f}s elapsed]",
                end="", flush=True,
            )

            keys = jax.random.split(master_key, n_repeats + 1)
            master_key, sim_keys = keys[0], keys[1:]

            nr_arr = np.asarray(
                _batch_sim(sim_keys, jnp.float32(load), jnp.float32(arrival_rate))
            )
            valid = nr_arr >= 0
            for nr in nr_arr[valid]:
                results.append({
                    "arrival_rate": arrival_rate,
                    "mean_service_holding_time": load / arrival_rate,
                    "load": load,
                    "active_services": int(load),
                    "num_requests": int(nr),
                })

    print(f"\n  Simulation complete in {time.perf_counter() - t0:.1f}s")
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


def main():
    configure_style(font_size=18, axes_label_size=18, tick_size=16, legend_size=16)
    print("Running warmup simulation (JAX-accelerated)...")
    df = simulate_warmup(n_repeats=1000)
    plot_warmup_boxplots(df)


if __name__ == '__main__':
    main()
