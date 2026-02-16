#!/usr/bin/env python3
"""
Gerard et al. 2025 Validation Script
=====================================

Reproduces key results from:
  "C+L-band long-haul transmission supporting 800 Gb/s per channel
   across 9.6 THz over 1504 km" (Gerard et al., 2025)

Uses XLRON's GN model with DRA to simulate 90-channel C+L-band loading
on a 3-node chain (node 0 -> node 1 -> node 2) matching the paper's
link configuration (700 km + 800 km = 1500 km).

Produces:
  Plot 1: Per-channel GOSNR vs frequency (C+L combined)
  Plot 2: OSNR_ASE and OSNR_NL vs frequency
  Plot 3: Band-averaged metrics vs Gerard Table I
  Plot 4: Launch power sweep vs average GOSNR
  Plot 5: Capacity vs distance
  Plot 6: Spectral efficiency vs distance
  Plot 7: Per-channel launch power profile
  Summary: Combined 3-panel validation figure

Usage:
  cd /path/to/XLRON
  python experimental/gerard2025_validation.py
"""

import os
import sys
import importlib.util

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import c as speed_of_light

# Import plot_style from benchmarks/ (not a package, so use importlib)
_ps_spec = importlib.util.spec_from_file_location(
    "plot_style", os.path.join(_project_root, "benchmarks", "plot_style.py")
)
_plot_style = importlib.util.module_from_spec(_ps_spec)
_ps_spec.loader.exec_module(_plot_style)
configure_style = _plot_style.configure_style

from xlron.environments.make_env import make, process_config
from xlron.environments.gn_model import isrs_gn_model, isrs_gn_model_dra
from xlron.environments.env_funcs import (
    calculate_throughput_from_active_lightpaths,
    get_launch_power,
)
from xlron.heuristics.heuristics import ksp_ff
from xlron import dtype_config
from xlron.gui.presets import PRESETS

# Apply publication-quality plot style (smaller fonts for multi-panel figures)
configure_style(font_size=16, axes_label_size=18, tick_size=14, legend_size=12)

# Band colors consistent with plot_style palette
C_COLOR = "#1f77b4"   # Blue
L_COLOR = "#d62728"   # Red
C_COLOR_LIGHT = "#aec7e8"
L_COLOR_LIGHT = "#ff9896"
GERARD_MARKER = "D"

# ---------------------------------------------------------------------------
# Gerard 2025 reference values (Table I, approximate)
# ---------------------------------------------------------------------------
GERARD_REF = {
    "C_band": {
        "GOSNR_dB": 20.0,
        "OSNR_ASE_dB": 21.5,
        "OSNR_NL_dB": 27.0,
    },
    "L_band": {
        "GOSNR_dB": 19.7,
        "OSNR_ASE_dB": 21.2,
        "OSNR_NL_dB": 26.5,
    },
    "total_capacity_tbps": 72.0,
    "spectral_efficiency_bps_hz": 7.29,
    "distance_km": 1504,
    "num_channels": 90,
    "channel_rate_gbps": 800,
    "total_bandwidth_thz": 9.6,
}


def run_simulation(config_overrides=None, quiet=False):
    """Run the simulation and return the final loaded state (before auto-reset).

    Uses the raw env's step_env to avoid auto-reset after max_requests.

    Returns:
        (final_state, env_params, raw_env, config)
    """
    config = {
        **PRESETS["Gerard 2025 recreation"],
        # Defaults required by process_config
        "ROLLOUT_LENGTH": 90,
        "NUM_MINIBATCHES": 1,
        "seed": 0,
        "load": 250,
        "mean_service_holding_time": 25,
    }
    if config_overrides:
        config.update(config_overrides)

    processed = process_config(config)
    env_wrapped, env_params = make(processed)
    raw_env = env_wrapped._env  # Unwrap LogWrapper to avoid auto-reset

    rng = jax.random.PRNGKey(config.get("seed", 0))
    rng, reset_key = jax.random.split(rng)
    obs, state = raw_env.reset(reset_key, env_params)

    num_channels = config.get("max_requests", 90)

    for step in range(num_channels):
        rng, action_key, step_key = jax.random.split(rng, 3)

        action = ksp_ff(state, env_params)
        launch_power = get_launch_power(state, action, action, env_params)
        full_action = jnp.concatenate(
            [action.reshape((1,)), launch_power.reshape((1,))], axis=0
        )

        obs, state, reward, terminal, truncated, info = raw_env.step_env(
            step_key, state, full_action, env_params
        )

        if not quiet and ((step + 1) % 30 == 0 or step == num_channels - 1):
            num_occupied = int(jnp.sum(state.channel_centre_bw_array[0] > 0))
            print(
                f"  Step {step+1}/{num_channels}: "
                f"{num_occupied} slots occupied, reward={float(reward):.1f}"
            )

    num_occ = int(jnp.sum(state.channel_centre_bw_array[0] > 0))
    if not quiet:
        print(
            f"\nFinal: {num_occ} slots occupied on link 0, "
            f"{int(jnp.sum(state.channel_centre_bw_array[1] > 0))} on link 1"
        )

    return state, env_params, raw_env, processed


def compute_snr_diagnostics(state, params):
    """Compute per-link SNR diagnostics returning noise power breakdown."""

    def get_link_diagnostics(link_index):
        link_lengths = params.link_length_array[link_index, :]
        num_spans = jnp.ceil(jnp.sum(link_lengths) / params.max_span_length).astype(
            jnp.int32
        )
        bw_link = state.channel_centre_bw_array[link_index, :]
        ch_power_link = state.channel_power_array[link_index, :]
        ch_centres_link = state.channel_centre_freq_array[link_index, :]

        P = dict(
            num_channels=params.link_resources,
            num_spans=num_spans,
            max_spans=params.max_spans,
            ref_lambda=params.ref_lambda,
            length=link_lengths,
            attenuation_i=jnp.array(params.attenuation),
            attenuation_bar_i=jnp.array(params.attenuation_bar),
            nonlinear_coeff=jnp.array(params.nonlinear_coeff),
            raman_gain_slope_i=jnp.array(params.raman_gain_slope),
            dispersion_coeff=jnp.array(params.dispersion_coeff),
            dispersion_slope=jnp.array(params.dispersion_slope),
            coherent=params.coherent,
            amplifier_noise_figure=params.amplifier_noise_figure.val,
            transceiver_snr=params.transceiver_snr.val,
            ch_power_w_i=ch_power_link,
            ch_centre_i=ch_centres_link * 1e9,
            ch_bandwidth_i=bw_link * 1e9,
            excess_kurtosis_i=jnp.zeros(params.link_resources, dtype=jnp.float32),
            uniform_spans=params.uniform_spans,
            num_subchannels=params.num_subchannels,
        )

        if params.use_raman_amp:
            snr, diagnostics = isrs_gn_model_dra.get_snr_dra(
                **P,
                fit_params_ij=params.raman_fit_params.val,
                raman_pump_power_fw=params.raman_pump_power_fw.val,
                raman_pump_power_bw=params.raman_pump_power_bw.val,
                raman_pump_freq_fw=params.raman_pump_freq_fw.val,
                raman_pump_freq_bw=params.raman_pump_freq_bw.val,
            )
        else:
            snr, diagnostics = isrs_gn_model.get_snr(
                **P,
                num_roadms=params.num_roadms,
                roadm_loss=params.roadm_loss,
                mod_format_correction=False,
            )

        eta_nli, _, _, p_ase_inline, p_ase_roadm, p_nli, transceiver_noise = diagnostics
        return snr, p_ase_inline, p_nli, transceiver_noise, ch_power_link

    results = jax.vmap(lambda i: get_link_diagnostics(i))(
        jnp.arange(params.num_links)
    )
    snr_all, p_ase_all, p_nli_all, trx_all, ch_power_all = results

    return {
        "snr_linear": snr_all,
        "p_ase": p_ase_all,
        "p_nli": p_nli_all,
        "transceiver_noise": trx_all,
        "ch_power": ch_power_all,
    }


def compute_path_diagnostics(diag, state, params):
    """Compute end-to-end path SNR by summing noise across both links."""
    ch_power = diag["ch_power"][0]
    occupied = state.channel_centre_bw_array[0] > 0

    p_ase_path = jnp.sum(diag["p_ase"], axis=0)
    p_nli_path = jnp.sum(diag["p_nli"], axis=0)
    trx_path = diag["transceiver_noise"][0]

    # ROADM ASE for 1 express node (node 1)
    if hasattr(params, "roadm_express_loss"):
        ch_centres_hz = state.channel_centre_freq_array[0] * 1e9
        ch_bw_hz = state.channel_centre_bw_array[0] * 1e9
        roadm_ase = isrs_gn_model.calculate_roadm_ase(
            roadm_express_loss=params.roadm_express_loss.val,
            roadm_add_drop_loss=params.roadm_add_drop_loss.val,
            roadm_noise_figure=params.roadm_noise_figure.val,
            num_roadm_express=1,
            ref_lambda=params.ref_lambda,
            ch_centre_i=ch_centres_hz,
            ch_bandwidth_i=ch_bw_hz,
        )
    else:
        roadm_ase = jnp.zeros_like(p_ase_path)

    total_ase = p_ase_path + roadm_ase
    total_noise = total_ase + p_nli_path + trx_path

    def safe_db(x):
        return 10 * jnp.log10(jnp.maximum(x, 1e-30))

    gosnr = jnp.where(occupied, ch_power / total_noise, jnp.nan)
    osnr_ase = jnp.where(occupied, ch_power / jnp.maximum(total_ase, 1e-30), jnp.nan)
    osnr_nl = jnp.where(occupied, ch_power / jnp.maximum(p_nli_path, 1e-30), jnp.nan)

    return {
        "gosnr_db": jnp.where(occupied, safe_db(gosnr), jnp.nan),
        "osnr_ase_db": jnp.where(occupied, safe_db(osnr_ase), jnp.nan),
        "osnr_nl_db": jnp.where(occupied, safe_db(osnr_nl), jnp.nan),
        "occupied": occupied,
        "ch_power": ch_power,
    }


def get_frequencies_thz(state, params):
    """Get absolute channel frequencies in THz.

    channel_centre_freq_array stores *relative* frequencies (GHz) from the
    reference wavelength.  Convert to absolute THz for plotting.
    """
    ref_freq_ghz = speed_of_light / float(params.ref_lambda) / 1e9
    rel_ghz = np.array(state.channel_centre_freq_array[0])
    abs_ghz = ref_freq_ghz + rel_ghz  # relative can be negative (L-band)
    return abs_ghz / 1e3  # THz


def get_occupied_mask(state):
    """Return boolean mask of occupied slots (BW > 0, works for neg freqs)."""
    return np.array(state.channel_centre_bw_array[0]) > 0


def identify_bands(freqs_thz, occ):
    """Identify C-band vs L-band by finding the inter-band frequency gap."""
    f_occ = freqs_thz[occ]
    if len(f_occ) < 2:
        return occ, np.zeros_like(occ)
    f_sorted = np.sort(f_occ)
    gap_idx = int(np.argmax(np.diff(f_sorted)))
    boundary = (f_sorted[gap_idx] + f_sorted[gap_idx + 1]) / 2
    c_mask = occ & np.array(freqs_thz > boundary)
    l_mask = occ & np.array(freqs_thz <= boundary)
    return c_mask, l_mask


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _band_scatter(ax, freqs, vals, c_mask, l_mask, occ, marker="o", s=50):
    """Helper: scatter C and L band points with consistent styling."""
    ax.scatter(
        freqs[occ & c_mask], vals[occ & c_mask],
        c=C_COLOR, s=s, marker=marker, label="C-band",
        zorder=3, edgecolors="k", linewidths=0.3,
    )
    ax.scatter(
        freqs[occ & l_mask], vals[occ & l_mask],
        c=L_COLOR, s=s, marker=marker, label="L-band",
        zorder=3, edgecolors="k", linewidths=0.3,
    )


def plot1_gosnr_vs_frequency(freqs, path_d, c_mask, l_mask, occ, out_dir):
    """Per-channel GOSNR vs frequency with Gerard reference lines."""
    gosnr = np.array(path_d["gosnr_db"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    _band_scatter(ax1, freqs, gosnr, c_mask, l_mask, occ)
    ax1.axhline(
        GERARD_REF["C_band"]["GOSNR_dB"], color=C_COLOR, ls="--", alpha=0.6,
        label=f'Gerard C avg ({GERARD_REF["C_band"]["GOSNR_dB"]:.1f} dB)',
    )
    ax1.axhline(
        GERARD_REF["L_band"]["GOSNR_dB"], color=L_COLOR, ls="--", alpha=0.6,
        label=f'Gerard L avg ({GERARD_REF["L_band"]["GOSNR_dB"]:.1f} dB)',
    )
    ax1.set_ylabel("GOSNR (dB)")
    ax1.set_title("Per-Channel GOSNR vs Frequency")
    ax1.legend(loc="best")

    fec_threshold = 15.0
    margin = gosnr - fec_threshold
    _band_scatter(ax2, freqs, margin, c_mask, l_mask, occ)
    ax2.axhline(0, color="black", ls="-", alpha=0.5, label="FEC threshold")
    ax2.set_xlabel("Frequency (THz)")
    ax2.set_ylabel("Margin above FEC (dB)")
    ax2.set_title("SNR Margin above 16QAM FEC Threshold")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot1_gosnr_vs_frequency.png"))
    plt.close()
    print("  Saved plot1_gosnr_vs_frequency.png")


def plot2_osnr_components(freqs, path_d, c_mask, l_mask, occ, out_dir):
    """OSNR_ASE and OSNR_NL vs frequency."""
    osnr_ase = np.array(path_d["osnr_ase_db"])
    osnr_nl = np.array(path_d["osnr_nl_db"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    _band_scatter(ax1, freqs, osnr_ase, c_mask, l_mask, occ)
    ax1.axhline(
        GERARD_REF["C_band"]["OSNR_ASE_dB"], color=C_COLOR, ls="--", alpha=0.6,
        label=f'Gerard C avg ({GERARD_REF["C_band"]["OSNR_ASE_dB"]:.1f} dB)',
    )
    ax1.axhline(
        GERARD_REF["L_band"]["OSNR_ASE_dB"], color=L_COLOR, ls="--", alpha=0.6,
        label=f'Gerard L avg ({GERARD_REF["L_band"]["OSNR_ASE_dB"]:.1f} dB)',
    )
    ax1.set_ylabel("OSNR$_{ASE}$ (dB)")
    ax1.set_title("Per-Channel OSNR$_{ASE}$ vs Frequency")
    ax1.legend(loc="best")

    _band_scatter(ax2, freqs, osnr_nl, c_mask, l_mask, occ)
    ax2.axhline(
        GERARD_REF["C_band"]["OSNR_NL_dB"], color=C_COLOR, ls="--", alpha=0.6,
        label=f'Gerard C avg ({GERARD_REF["C_band"]["OSNR_NL_dB"]:.1f} dB)',
    )
    ax2.axhline(
        GERARD_REF["L_band"]["OSNR_NL_dB"], color=L_COLOR, ls="--", alpha=0.6,
        label=f'Gerard L avg ({GERARD_REF["L_band"]["OSNR_NL_dB"]:.1f} dB)',
    )
    ax2.set_xlabel("Frequency (THz)")
    ax2.set_ylabel("OSNR$_{NL}$ (dB)")
    ax2.set_title("Per-Channel OSNR$_{NL}$ vs Frequency")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot2_osnr_components.png"))
    plt.close()
    print("  Saved plot2_osnr_components.png")


def plot3_band_comparison(path_d, c_mask, l_mask, occ, out_dir):
    """Band-averaged metrics bar chart: XLRON vs Gerard."""
    gosnr = np.array(path_d["gosnr_db"])
    osnr_ase = np.array(path_d["osnr_ase_db"])
    osnr_nl = np.array(path_d["osnr_nl_db"])

    c_occ, l_occ = occ & c_mask, occ & l_mask

    def _safe_mean(arr, mask):
        return float(np.nanmean(arr[mask])) if np.any(mask) else np.nan

    xlron_c = {
        "GOSNR": _safe_mean(gosnr, c_occ),
        "OSNR_ASE": _safe_mean(osnr_ase, c_occ),
        "OSNR_NL": _safe_mean(osnr_nl, c_occ),
    }
    xlron_l = {
        "GOSNR": _safe_mean(gosnr, l_occ),
        "OSNR_ASE": _safe_mean(osnr_ase, l_occ),
        "OSNR_NL": _safe_mean(osnr_nl, l_occ),
    }
    gerard_c = GERARD_REF["C_band"]
    gerard_l = GERARD_REF["L_band"]

    # Print table
    print("\n" + "=" * 70)
    print("BAND-AVERAGED METRICS COMPARISON")
    print("=" * 70)
    header = f"{'Metric':<15} {'XLRON C':>10} {'Gerard C':>10} {'XLRON L':>10} {'Gerard L':>10}"
    print(header)
    print("-" * 70)
    for m in ["GOSNR", "OSNR_ASE", "OSNR_NL"]:
        xc = f"{xlron_c[m]:.2f}" if not np.isnan(xlron_c[m]) else "N/A"
        xl = f"{xlron_l[m]:.2f}" if not np.isnan(xlron_l[m]) else "N/A"
        print(
            f"{m + ' (dB)':<15} {xc:>10} {gerard_c[m+'_dB']:>10.1f} "
            f"{xl:>10} {gerard_l[m+'_dB']:>10.1f}"
        )
    print("=" * 70)

    # Bar chart
    metrics_list = ["GOSNR", "OSNR_ASE", "OSNR_NL"]
    x = np.arange(len(metrics_list))
    w = 0.18

    fig, ax = plt.subplots(figsize=(12, 7))
    xlron_c_vals = [xlron_c[m] if not np.isnan(xlron_c[m]) else 0 for m in metrics_list]
    xlron_l_vals = [xlron_l[m] if not np.isnan(xlron_l[m]) else 0 for m in metrics_list]
    ger_c_vals = [gerard_c[f"{m}_dB"] for m in metrics_list]
    ger_l_vals = [gerard_l[f"{m}_dB"] for m in metrics_list]

    bars = [
        ax.bar(x - 1.5 * w, xlron_c_vals, w, label="XLRON C-band", color=C_COLOR),
        ax.bar(x - 0.5 * w, ger_c_vals, w, label="Gerard C-band", color=C_COLOR_LIGHT,
               edgecolor=C_COLOR, hatch="//"),
        ax.bar(x + 0.5 * w, xlron_l_vals, w, label="XLRON L-band", color=L_COLOR),
        ax.bar(x + 1.5 * w, ger_l_vals, w, label="Gerard L-band", color=L_COLOR_LIGHT,
               edgecolor=L_COLOR, hatch="//"),
    ]
    for bar_group in bars:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["GOSNR", "OSNR$_{ASE}$", "OSNR$_{NL}$"],
    )
    ax.set_ylabel("Value (dB)")
    ax.set_title("Band-Averaged Metrics: XLRON vs Gerard 2025")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot3_band_comparison.png"))
    plt.close()
    print("  Saved plot3_band_comparison.png")

    return xlron_c, xlron_l


def plot4_launch_power_sweep(out_dir):
    """Sweep total fibre launch power and plot average GOSNR."""
    print("\n--- Launch Power Sweep ---")
    power_dbm_values = np.arange(14.0, 26.0, 1.0)
    avg_gosnr_values = []

    for pdbm in power_dbm_values:
        print(f"  Power = {pdbm:.1f} dBm...", end="", flush=True)
        try:
            state, params, _, _ = run_simulation(
                {"max_power_per_fibre": float(pdbm)}, quiet=True
            )
            diag = compute_snr_diagnostics(state, params)
            pd_ = compute_path_diagnostics(diag, state, params)
            occ = np.array(pd_["occupied"])
            g = np.array(pd_["gosnr_db"])
            avg = float(np.nanmean(g[occ])) if np.any(occ) else np.nan
            avg_gosnr_values.append(avg)
            print(f" avg GOSNR = {avg:.2f} dB, {int(np.sum(occ))} ch")
        except Exception as e:
            print(f" ERROR: {e}")
            avg_gosnr_values.append(np.nan)

    avg_gosnr_values = np.array(avg_gosnr_values)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(power_dbm_values, avg_gosnr_values, "ko-")

    valid = ~np.isnan(avg_gosnr_values)
    if np.any(valid):
        opt_idx = int(np.nanargmax(avg_gosnr_values))
        ax.scatter(
            [power_dbm_values[opt_idx]], [avg_gosnr_values[opt_idx]],
            c="red", s=200, zorder=5, marker="*",
            label=f"Optimum: {power_dbm_values[opt_idx]:.1f} dBm, "
                  f"{avg_gosnr_values[opt_idx]:.1f} dB",
        )

    ax.axvspan(18.0, 19.5, alpha=0.1, color="green", label="Gerard optimal range")
    ax.set_xlabel("Total Fibre Launch Power (dBm)")
    ax.set_ylabel("Average GOSNR (dB)")
    ax.set_title("Average GOSNR vs Total Launch Power")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot4_launch_power_sweep.png"))
    plt.close()
    print("  Saved plot4_launch_power_sweep.png")


def plot5_capacity_and_se_vs_distance(out_dir):
    """Capacity and spectral efficiency vs distance by scaling span_length."""
    print("\n--- Capacity vs Distance Sweep ---")

    total_distances_km = [500, 750, 1000, 1500, 2000, 2500, 3000]
    capacities = []
    spectral_effs = []
    base_total_km = 700 + 800

    for total_km in total_distances_km:
        scale = total_km / base_total_km
        new_span_length = 100 * scale

        print(
            f"  Distance = {total_km} km (span = {new_span_length:.0f} km)...",
            end="", flush=True,
        )
        try:
            state, params, _, _ = run_simulation(
                {"span_length": new_span_length}, quiet=True
            )

            # Use Shannon-Hartley throughput from the env
            throughput_gbps = float(
                calculate_throughput_from_active_lightpaths(state, params)
            )
            cap_tbps = throughput_gbps / 1000.0
            capacities.append(cap_tbps)

            occ = np.array(state.channel_centre_bw_array[0] > 0)
            bw_thz = float(np.sum(occ)) * 100e-3
            se = (throughput_gbps) / (bw_thz * 1e3) if bw_thz > 0 else 0
            spectral_effs.append(se)
            print(f" throughput = {cap_tbps:.1f} Tb/s, SE = {se:.2f} b/s/Hz")
        except Exception as e:
            print(f" ERROR: {e}")
            capacities.append(0)
            spectral_effs.append(0)

    # Capacity plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(total_distances_km, capacities, "o-", color=C_COLOR, label="XLRON")
    ax.axhline(
        GERARD_REF["total_capacity_tbps"], color="#2ca02c", ls="--",
        label=f'Gerard: {GERARD_REF["total_capacity_tbps"]} Tb/s',
    )
    ax.scatter(
        [GERARD_REF["distance_km"]], [GERARD_REF["total_capacity_tbps"]],
        c="#2ca02c", s=200, marker="*", zorder=5,
    )
    ax.set_xlabel("Total Distance (km)")
    ax.set_ylabel("Total Capacity (Tb/s)")
    ax.set_title("Capacity vs Distance")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot5_capacity_vs_distance.png"))
    plt.close()
    print("  Saved plot5_capacity_vs_distance.png")

    # SE plot
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(total_distances_km, spectral_effs, "o-", color=L_COLOR, label="XLRON")
    ax.axhline(
        GERARD_REF["spectral_efficiency_bps_hz"], color="#2ca02c", ls="--",
        label=f'Gerard: {GERARD_REF["spectral_efficiency_bps_hz"]} b/s/Hz',
    )
    ax.set_xlabel("Total Distance (km)")
    ax.set_ylabel("Spectral Efficiency (b/s/Hz)")
    ax.set_title("Spectral Efficiency vs Distance")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot6_spectral_efficiency_vs_distance.png"))
    plt.close()
    print("  Saved plot6_spectral_efficiency_vs_distance.png")


def plot7_power_profile(state, freqs, c_mask, l_mask, occ, out_dir):
    """Per-channel launch power vs frequency."""
    ch_power_w = np.array(state.channel_power_array[0])
    with np.errstate(divide="ignore"):
        launch_dbm = np.where(
            ch_power_w > 0, 10 * np.log10(ch_power_w / 1e-3), np.nan
        )

    fig, ax = plt.subplots(figsize=(14, 6))
    _band_scatter(ax, freqs, launch_dbm, c_mask, l_mask, occ)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Launch Power per Channel (dBm)")
    ax.set_title("Per-Channel Launch Power vs Frequency")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot7_power_profile.png"))
    plt.close()
    print("  Saved plot7_power_profile.png")


def plot_summary(freqs, path_d, c_mask, l_mask, occ, out_dir):
    """Thesis-ready 3-panel validation figure."""
    gosnr = np.array(path_d["gosnr_db"])
    osnr_ase = np.array(path_d["osnr_ase_db"])
    osnr_nl = np.array(path_d["osnr_nl_db"])
    c_occ, l_occ = occ & c_mask, occ & l_mask

    fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

    # Panel 1: GOSNR
    ax = axes[0]
    _band_scatter(ax, freqs, gosnr, c_mask, l_mask, occ)
    ax.axhline(
        GERARD_REF["C_band"]["GOSNR_dB"], color=C_COLOR, ls="--", alpha=0.6,
        label=f'Gerard C avg ({GERARD_REF["C_band"]["GOSNR_dB"]:.1f} dB)',
    )
    ax.axhline(
        GERARD_REF["L_band"]["GOSNR_dB"], color=L_COLOR, ls="--", alpha=0.6,
        label=f'Gerard L avg ({GERARD_REF["L_band"]["GOSNR_dB"]:.1f} dB)',
    )
    ax.set_ylabel("GOSNR (dB)")
    ax.set_title("Gerard 2025 Validation: C+L Band 90-Channel Performance")
    ax.legend(loc="best")

    # Panel 2: OSNR components
    ax = axes[1]
    ax.scatter(
        freqs[c_occ], osnr_ase[c_occ], c=C_COLOR, s=40, marker="o",
        label="OSNR$_{ASE}$ C", zorder=3, edgecolors="k", linewidths=0.3,
    )
    ax.scatter(
        freqs[l_occ], osnr_ase[l_occ], c=L_COLOR, s=40, marker="o",
        label="OSNR$_{ASE}$ L", zorder=3, edgecolors="k", linewidths=0.3,
    )
    ax.scatter(
        freqs[c_occ], osnr_nl[c_occ], c=C_COLOR, s=40, marker="^",
        label="OSNR$_{NL}$ C", zorder=3, edgecolors="k", linewidths=0.3,
    )
    ax.scatter(
        freqs[l_occ], osnr_nl[l_occ], c=L_COLOR, s=40, marker="^",
        label="OSNR$_{NL}$ L", zorder=3, edgecolors="k", linewidths=0.3,
    )
    ax.set_ylabel("OSNR (dB)")
    ax.set_title("OSNR Components")
    ax.legend(loc="best", ncol=2)

    # Panel 3: Band bar chart
    ax = axes[2]
    metrics_list = ["GOSNR", "OSNR_ASE", "OSNR_NL"]

    def _safe(arr, mask):
        return float(np.nanmean(arr[mask])) if np.any(mask) else 0.0

    xlron_c_v = [_safe(gosnr, c_occ), _safe(osnr_ase, c_occ), _safe(osnr_nl, c_occ)]
    xlron_l_v = [_safe(gosnr, l_occ), _safe(osnr_ase, l_occ), _safe(osnr_nl, l_occ)]
    ger_c = [GERARD_REF["C_band"][f"{m}_dB"] for m in metrics_list]
    ger_l = [GERARD_REF["L_band"][f"{m}_dB"] for m in metrics_list]

    x = np.arange(len(metrics_list))
    w = 0.18
    ax.bar(x - 1.5 * w, xlron_c_v, w, label="XLRON C", color=C_COLOR)
    ax.bar(x - 0.5 * w, ger_c, w, label="Gerard C", color=C_COLOR_LIGHT,
           edgecolor=C_COLOR, hatch="//")
    ax.bar(x + 0.5 * w, xlron_l_v, w, label="XLRON L", color=L_COLOR)
    ax.bar(x + 1.5 * w, ger_l, w, label="Gerard L", color=L_COLOR_LIGHT,
           edgecolor=L_COLOR, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(["GOSNR", "OSNR$_{ASE}$", "OSNR$_{NL}$"])
    ax.set_ylabel("Value (dB)")
    ax.set_title("Band-Averaged: XLRON vs Gerard 2025")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "validation_summary.png"))
    plt.close()
    print("  Saved validation_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gerard2025_results"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # --- Baseline simulation ---
    print("\n" + "=" * 70)
    print("RUNNING BASELINE SIMULATION (Gerard 2025 preset)")
    print("=" * 70)

    state, params, env, config = run_simulation()

    # --- Throughput via Shannon-Hartley ---
    throughput_gbps = float(
        calculate_throughput_from_active_lightpaths(state, params)
    )
    throughput_tbps = throughput_gbps / 1000.0

    # --- SNR diagnostics ---
    print("\nComputing SNR diagnostics...")
    diag = compute_snr_diagnostics(state, params)
    path_d = compute_path_diagnostics(diag, state, params)

    freqs = get_frequencies_thz(state, params)
    occ = get_occupied_mask(state)
    c_mask, l_mask = identify_bands(freqs, occ)

    gosnr_db = np.array(path_d["gosnr_db"])
    num_ch = int(np.sum(occ))
    print(f"\nChannels loaded: {num_ch}")
    print(f"C-band channels: {int(np.sum(occ & c_mask))}")
    print(f"L-band channels: {int(np.sum(occ & l_mask))}")
    print(f"Shannon-Hartley throughput: {throughput_tbps:.1f} Tb/s "
          f"(Gerard: {GERARD_REF['total_capacity_tbps']} Tb/s)")
    if num_ch > 0:
        print(
            f"GOSNR range: {np.nanmin(gosnr_db[occ]):.2f} - "
            f"{np.nanmax(gosnr_db[occ]):.2f} dB"
        )
        print(
            f"GOSNR variation: "
            f"{np.nanmax(gosnr_db[occ]) - np.nanmin(gosnr_db[occ]):.2f} dB"
        )

    # --- Generate core plots ---
    print("\nGenerating plots...")
    plot1_gosnr_vs_frequency(freqs, path_d, c_mask, l_mask, occ, out_dir)
    plot2_osnr_components(freqs, path_d, c_mask, l_mask, occ, out_dir)
    plot3_band_comparison(path_d, c_mask, l_mask, occ, out_dir)
    plot7_power_profile(state, freqs, c_mask, l_mask, occ, out_dir)
    plot_summary(freqs, path_d, c_mask, l_mask, occ, out_dir)

    # --- Sweep plots (slower) ---
    plot4_launch_power_sweep(out_dir)
    plot5_capacity_and_se_vs_distance(out_dir)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"All plots saved to: {out_dir}")

    print(f"\n  Shannon-Hartley capacity: {throughput_tbps:.1f} Tb/s "
          f"(Gerard: {GERARD_REF['total_capacity_tbps']} Tb/s)")
    bw_thz = num_ch * 100e-3
    if bw_thz > 0:
        se = throughput_gbps / (bw_thz * 1e3)
        print(
            f"  Spectral efficiency: {se:.2f} b/s/Hz "
            f"(Gerard: {GERARD_REF['spectral_efficiency_bps_hz']} b/s/Hz)"
        )
    if num_ch > 0:
        variation = float(np.nanmax(gosnr_db[occ]) - np.nanmin(gosnr_db[occ]))
        print(
            f"  GOSNR variation across C+L: {variation:.2f} dB "
            f"(Gerard target: < 2 dB)"
        )
        fec_threshold = 15.0
        above = int(np.sum(occ & (gosnr_db > fec_threshold)))
        print(
            f"  Channels above FEC threshold ({fec_threshold} dB): "
            f"{above}/{num_ch}"
        )


if __name__ == "__main__":
    main()
