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
  Plot 1: Per-channel GOSNR vs frequency with QAM FEC thresholds
  Plot 2: OSNR_ASE and OSNR_NL vs frequency
  Plot 3: Band-averaged metrics vs Gerard Table I
  Plot 4: Launch power sweep vs average GOSNR
  Plot 5: Ablation study (power sweeps with different model configurations)
  Summary: Combined 3-panel validation figure
  Printed: Comprehensive validation summary table

Usage:
  cd /path/to/XLRON
  python experimental/gerard2025_validation.py
"""

import importlib.util
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as speed_of_light

# Import plot_style from benchmarks/ (not a package, so use importlib)
_ps_spec = importlib.util.spec_from_file_location(
    "plot_style", os.path.join(_project_root, "benchmarks", "plot_style.py")
)
_plot_style = importlib.util.module_from_spec(_ps_spec)
_ps_spec.loader.exec_module(_plot_style)
configure_style = _plot_style.configure_style

from xlron import dtype_config
from xlron.environments.env_funcs import (
    calculate_throughput_from_active_lightpaths,
    get_launch_power,
)
from xlron.environments.gn_model import isrs_gn_model, isrs_gn_model_dra
from xlron.environments.make_env import _gsnr_threshold_db, make, process_config
from xlron.gui.presets import PRESETS
from xlron.heuristics.heuristics import ksp_ff

# Apply publication-quality plot style (smaller fonts for multi-panel figures)
configure_style(font_size=16, axes_label_size=18, tick_size=14, legend_size=12)

# ---------------------------------------------------------------------------
# Data caching
# ---------------------------------------------------------------------------

BASELINE_DATA_FILE = "baseline_data.npz"
SWEEP_DATA_FILE = "sweep_data.npz"


def save_baseline_data(data_dir, freqs, path_d, occ, c_mask, l_mask, throughput_gbps):
    """Save baseline simulation results to disk."""
    os.makedirs(data_dir, exist_ok=True)
    np.savez(
        os.path.join(data_dir, BASELINE_DATA_FILE),
        freqs=freqs,
        gosnr_db=np.array(path_d["gosnr_db"]),
        osnr_ase_db=np.array(path_d["osnr_ase_db"]),
        osnr_nl_db=np.array(path_d["osnr_nl_db"]),
        occupied=np.array(path_d["occupied"]),
        ch_power=np.array(path_d["ch_power"]),
        occ=occ,
        c_mask=c_mask,
        l_mask=l_mask,
        throughput_gbps=np.array(throughput_gbps),
    )
    print(f"  Saved baseline data to {data_dir}/{BASELINE_DATA_FILE}")


def load_baseline_data(data_dir):
    """Load cached baseline data. Returns (freqs, path_d, occ, c_mask, l_mask, throughput_gbps) or None."""
    path = os.path.join(data_dir, BASELINE_DATA_FILE)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    path_d = {
        "gosnr_db": d["gosnr_db"],
        "osnr_ase_db": d["osnr_ase_db"],
        "osnr_nl_db": d["osnr_nl_db"],
        "occupied": d["occupied"],
        "ch_power": d["ch_power"],
    }
    return d["freqs"], path_d, d["occ"], d["c_mask"], d["l_mask"], float(d["throughput_gbps"])


def save_sweep_data(data_dir, power_dbm_values, avg_gosnr_values):
    """Save launch power sweep results to disk."""
    os.makedirs(data_dir, exist_ok=True)
    np.savez(
        os.path.join(data_dir, SWEEP_DATA_FILE),
        power_dbm_values=power_dbm_values,
        avg_gosnr_values=avg_gosnr_values,
    )
    print(f"  Saved sweep data to {data_dir}/{SWEEP_DATA_FILE}")


def load_sweep_data(data_dir):
    """Load cached sweep data. Returns (power_dbm_values, avg_gosnr_values) or None."""
    path = os.path.join(data_dir, SWEEP_DATA_FILE)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["power_dbm_values"], d["avg_gosnr_values"]


# Band colors consistent with plot_style palette
C_COLOR = "#1f77b4"  # Blue
L_COLOR = "#d62728"  # Red
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
        full_action = jnp.concatenate([action.reshape((1,)), launch_power.reshape((1,))], axis=0)

        obs, state, reward, terminal, truncated, info = raw_env.step_env(
            step_key, state, full_action, env_params
        )

        if not quiet and ((step + 1) % 30 == 0 or step == num_channels - 1):
            num_occupied = int(jnp.sum(state.channel_centre_bw_array[0] > 0))
            print(
                f"  Step {step + 1}/{num_channels}: "
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
        num_spans = jnp.ceil(jnp.sum(link_lengths) / params.max_span_length).astype(jnp.int32)
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

    results = jax.vmap(lambda i: get_link_diagnostics(i))(jnp.arange(params.num_links))
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
        freqs[occ & c_mask],
        vals[occ & c_mask],
        c=C_COLOR,
        s=s,
        marker=marker,
        label="C-band",
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.scatter(
        freqs[occ & l_mask],
        vals[occ & l_mask],
        c=L_COLOR,
        s=s,
        marker=marker,
        label="L-band",
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )


def _qam_fec_thresholds(beta_fec=1.5e-2):
    """Compute GSNR thresholds for QAM formats using the same formula as the env."""
    formats = [
        ("32QAM", 5),
        ("64QAM", 6),
        ("128QAM", 7),
        ("256QAM", 8),
    ]
    return {name: _gsnr_threshold_db(beta_fec, m) for name, m in formats}


QAM_COLORS = {
    "32QAM": "#2ca02c",
    "64QAM": "#ff7f0e",
    "128QAM": "#9467bd",
    "256QAM": "#8c564b",
}


def plot1_gosnr_vs_frequency(freqs, path_d, c_mask, l_mask, occ, out_dir):
    """Per-channel GOSNR vs frequency with Gerard reference and QAM FEC lines."""
    gosnr = np.array(path_d["gosnr_db"])

    qam_thresholds = _qam_fec_thresholds()

    fig, ax = plt.subplots(figsize=(14, 7))

    _band_scatter(ax, freqs, gosnr, c_mask, l_mask, occ)
    ax.axhline(
        GERARD_REF["C_band"]["GOSNR_dB"],
        color=C_COLOR,
        ls="-",
        alpha=0.6,
        label=f"Gerard C avg ({GERARD_REF['C_band']['GOSNR_dB']:.1f} dB)",
    )
    ax.axhline(
        GERARD_REF["L_band"]["GOSNR_dB"],
        color=L_COLOR,
        ls="-",
        alpha=0.6,
        label=f"Gerard L avg ({GERARD_REF['L_band']['GOSNR_dB']:.1f} dB)",
    )

    fec_dash_styles = ["--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]
    for (qam, thresh), ds in zip(qam_thresholds.items(), fec_dash_styles):
        ax.axhline(
            thresh,
            color="0.5",
            ls=ds,
            alpha=0.7,
            label=f"{qam} FEC ({thresh:.1f} dB)",
        )

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("GOSNR (dB)")

    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot1_gosnr_vs_frequency.png"))
    plt.close()
    print("  Saved plot1_gosnr_vs_frequency.png")


def plot2_osnr_components(freqs, path_d, c_mask, l_mask, occ, out_dir):
    """OSNR_ASE and OSNR_NL vs frequency on a single axis."""
    osnr_ase = np.array(path_d["osnr_ase_db"])
    osnr_nl = np.array(path_d["osnr_nl_db"])
    c_occ, l_occ = occ & c_mask, occ & l_mask

    fig, ax = plt.subplots(figsize=(14, 7))

    # ASE: circles
    ax.scatter(
        freqs[c_occ],
        osnr_ase[c_occ],
        c=C_COLOR,
        s=50,
        marker="o",
        label="OSNR$_{ASE}$ C-band",
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.scatter(
        freqs[l_occ],
        osnr_ase[l_occ],
        c=L_COLOR,
        s=50,
        marker="o",
        label="OSNR$_{ASE}$ L-band",
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )
    # NL: triangles
    ax.scatter(
        freqs[c_occ],
        osnr_nl[c_occ],
        c=C_COLOR,
        s=50,
        marker="^",
        label="OSNR$_{NL}$ C-band",
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.scatter(
        freqs[l_occ],
        osnr_nl[l_occ],
        c=L_COLOR,
        s=50,
        marker="^",
        label="OSNR$_{NL}$ L-band",
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )

    # Gerard reference lines
    ax.axhline(
        GERARD_REF["C_band"]["OSNR_ASE_dB"],
        color=C_COLOR,
        ls="--",
        alpha=0.5,
        label=f"Gerard OSNR$_{{ASE}}$ C ({GERARD_REF['C_band']['OSNR_ASE_dB']:.1f} dB)",
    )
    ax.axhline(
        GERARD_REF["L_band"]["OSNR_ASE_dB"],
        color=L_COLOR,
        ls="--",
        alpha=0.5,
        label=f"Gerard OSNR$_{{ASE}}$ L ({GERARD_REF['L_band']['OSNR_ASE_dB']:.1f} dB)",
    )
    ax.axhline(
        GERARD_REF["C_band"]["OSNR_NL_dB"],
        color=C_COLOR,
        ls=":",
        alpha=0.5,
        label=f"Gerard OSNR$_{{NL}}$ C ({GERARD_REF['C_band']['OSNR_NL_dB']:.1f} dB)",
    )
    ax.axhline(
        GERARD_REF["L_band"]["OSNR_NL_dB"],
        color=L_COLOR,
        ls=":",
        alpha=0.5,
        label=f"Gerard OSNR$_{{NL}}$ L ({GERARD_REF['L_band']['OSNR_NL_dB']:.1f} dB)",
    )

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("OSNR (dB)")

    ax.legend(loc="lower right", ncol=2)

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
            f"{m + ' (dB)':<15} {xc:>10} {gerard_c[m + '_dB']:>10.1f} "
            f"{xl:>10} {gerard_l[m + '_dB']:>10.1f}"
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
        ax.bar(
            x - 0.5 * w,
            ger_c_vals,
            w,
            label="Gerard C-band",
            color=C_COLOR_LIGHT,
            edgecolor=C_COLOR,
            hatch="//",
        ),
        ax.bar(x + 0.5 * w, xlron_l_vals, w, label="XLRON L-band", color=L_COLOR),
        ax.bar(
            x + 1.5 * w,
            ger_l_vals,
            w,
            label="Gerard L-band",
            color=L_COLOR_LIGHT,
            edgecolor=L_COLOR,
            hatch="//",
        ),
    ]
    for bar_group in bars:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["GOSNR", "OSNR$_{ASE}$", "OSNR$_{NL}$"],
    )
    ax.set_ylabel("Value (dB)")

    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot3_band_comparison.png"))
    plt.close()
    print("  Saved plot3_band_comparison.png")

    return xlron_c, xlron_l


def run_launch_power_sweep():
    """Run sweep of total fibre launch power. Returns (power_dbm_values, avg_gosnr_values)."""
    print("\n--- Launch Power Sweep ---")
    power_dbm_values = np.arange(14.0, 26.0, 0.1)
    avg_gosnr_values = []

    for pdbm in power_dbm_values:
        print(f"  Power = {pdbm:.1f} dBm...", end="", flush=True)
        try:
            state, params, _, _ = run_simulation({"max_power_per_fibre": float(pdbm)}, quiet=True)
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

    return power_dbm_values, np.array(avg_gosnr_values)


def plot4_launch_power_sweep(power_dbm_values, avg_gosnr_values, out_dir):
    """Plot average GOSNR vs total fibre launch power."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(power_dbm_values, avg_gosnr_values, "ko-", markersize=3)

    valid = ~np.isnan(avg_gosnr_values)
    if np.any(valid):
        opt_idx = int(np.nanargmax(avg_gosnr_values))
        ax.scatter(
            [power_dbm_values[opt_idx]],
            [avg_gosnr_values[opt_idx]],
            c="red",
            s=200,
            zorder=5,
            marker="*",
            label=f"Optimum: {power_dbm_values[opt_idx]:.1f} dBm, "
            f"{avg_gosnr_values[opt_idx]:.1f} dB",
        )

    ax.axvspan(18.0, 19.5, alpha=0.1, color="green", label="Gerard optimal range")
    ax.set_xlabel("Total Fibre Launch Power (dBm)")
    ax.set_ylabel("Average GOSNR (dB)")

    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot4_launch_power_sweep.png"))
    plt.close()
    print("  Saved plot4_launch_power_sweep.png")


# ---------------------------------------------------------------------------
# Ablation study: power sweep with different configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "Full model (baseline)": {},
    "No Raman": {"use_raman_amp": False},
    "1 subchannel": {"num_subchannels": 1},
    "Incoherent": {"coherent": False},
    "No Raman + 1 subchannel": {"use_raman_amp": False, "num_subchannels": 1},
}

ABLATION_DATA_FILE = "ablation_data.npz"


def run_ablation_sweeps():
    """Run power sweeps for each ablation configuration.

    Returns dict: {config_name: (power_dbm_values, avg_gosnr_values)}.
    """
    power_dbm_values = np.arange(14.0, 26.0, 0.25)
    results = {}

    for name, overrides in ABLATION_CONFIGS.items():
        print(f"\n--- Ablation sweep: {name} ---")
        avg_gosnr_values = []

        for pdbm in power_dbm_values:
            print(f"  Power = {pdbm:.2f} dBm...", end="", flush=True)
            try:
                cfg = {"max_power_per_fibre": float(pdbm)}
                cfg.update(overrides)
                state, params, _, _ = run_simulation(cfg, quiet=True)
                diag = compute_snr_diagnostics(state, params)
                pd_ = compute_path_diagnostics(diag, state, params)
                occ = np.array(pd_["occupied"])
                g = np.array(pd_["gosnr_db"])
                avg = float(np.nanmean(g[occ])) if np.any(occ) else np.nan
                avg_gosnr_values.append(avg)
                print(f" avg GOSNR = {avg:.2f} dB")
            except Exception as e:
                print(f" ERROR: {e}")
                avg_gosnr_values.append(np.nan)

        results[name] = (power_dbm_values, np.array(avg_gosnr_values))

    return results


def save_ablation_data(data_dir, results):
    """Save ablation sweep results to disk."""
    os.makedirs(data_dir, exist_ok=True)
    save_dict = {}
    for i, (name, (powers, gosnrs)) in enumerate(results.items()):
        save_dict[f"name_{i}"] = name
        save_dict[f"powers_{i}"] = powers
        save_dict[f"gosnrs_{i}"] = gosnrs
    save_dict["n_configs"] = np.array(len(results))
    np.savez(os.path.join(data_dir, ABLATION_DATA_FILE), **save_dict)
    print(f"  Saved ablation data to {data_dir}/{ABLATION_DATA_FILE}")


def load_ablation_data(data_dir):
    """Load cached ablation data. Returns dict or None."""
    path = os.path.join(data_dir, ABLATION_DATA_FILE)
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    n = int(d["n_configs"])
    results = {}
    for i in range(n):
        name = str(d[f"name_{i}"])
        results[name] = (d[f"powers_{i}"], d[f"gosnrs_{i}"])
    return results


def plot5_ablation_sweep(ablation_results, out_dir):
    """Plot ablation study: power sweep curves for different configurations."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    for i, (name, (powers, gosnrs)) in enumerate(ablation_results.items()):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        ax.plot(powers, gosnrs, color=c, marker=m, markersize=3, label=name, linewidth=1.5)

        valid = ~np.isnan(gosnrs)
        if np.any(valid):
            opt_idx = int(np.nanargmax(gosnrs))
            ax.scatter(
                [powers[opt_idx]],
                [gosnrs[opt_idx]],
                c=c,
                s=150,
                zorder=5,
                marker="*",
                edgecolors="k",
                linewidths=0.5,
            )

    ax.axvspan(18.0, 19.5, alpha=0.1, color="green", label="Gerard optimal range")
    ax.set_xlabel("Total Fibre Launch Power (dBm)")
    ax.set_ylabel("Average GOSNR (dB)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot5_ablation_sweep.png"))
    plt.close()
    print("  Saved plot5_ablation_sweep.png")


def plot_summary(freqs, path_d, c_mask, l_mask, occ, out_dir):
    """Thesis-ready 3-panel validation figure."""
    gosnr = np.array(path_d["gosnr_db"])
    osnr_ase = np.array(path_d["osnr_ase_db"])
    osnr_nl = np.array(path_d["osnr_nl_db"])
    c_occ, l_occ = occ & c_mask, occ & l_mask

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 1, hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])  # Independent x-axis for bar chart

    # Panel 1: GOSNR
    _band_scatter(ax1, freqs, gosnr, c_mask, l_mask, occ)
    ax1.axhline(
        GERARD_REF["C_band"]["GOSNR_dB"],
        color=C_COLOR,
        ls="--",
        alpha=0.6,
        label=f"Gerard C avg ({GERARD_REF['C_band']['GOSNR_dB']:.1f} dB)",
    )
    ax1.axhline(
        GERARD_REF["L_band"]["GOSNR_dB"],
        color=L_COLOR,
        ls="--",
        alpha=0.6,
        label=f"Gerard L avg ({GERARD_REF['L_band']['GOSNR_dB']:.1f} dB)",
    )
    ax1.set_ylabel("GOSNR (dB)")

    ax1.legend(loc="best")

    # Panel 2: OSNR components
    ax2.scatter(
        freqs[c_occ],
        osnr_ase[c_occ],
        c=C_COLOR,
        s=40,
        marker="o",
        label="OSNR$_{ASE}$ C",
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )
    ax2.scatter(
        freqs[l_occ],
        osnr_ase[l_occ],
        c=L_COLOR,
        s=40,
        marker="o",
        label="OSNR$_{ASE}$ L",
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )
    ax2.scatter(
        freqs[c_occ],
        osnr_nl[c_occ],
        c=C_COLOR,
        s=40,
        marker="^",
        label="OSNR$_{NL}$ C",
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )
    ax2.scatter(
        freqs[l_occ],
        osnr_nl[l_occ],
        c=L_COLOR,
        s=40,
        marker="^",
        label="OSNR$_{NL}$ L",
        zorder=3,
        edgecolors="k",
        linewidths=0.3,
    )
    ax2.set_xlabel("Frequency (THz)")
    ax2.set_ylabel("OSNR (dB)")

    ax2.legend(loc="best", ncol=2)

    # Panel 3: Band bar chart (independent x-axis)
    metrics_list = ["GOSNR", "OSNR_ASE", "OSNR_NL"]

    def _safe(arr, mask):
        return float(np.nanmean(arr[mask])) if np.any(mask) else 0.0

    xlron_c_v = [_safe(gosnr, c_occ), _safe(osnr_ase, c_occ), _safe(osnr_nl, c_occ)]
    xlron_l_v = [_safe(gosnr, l_occ), _safe(osnr_ase, l_occ), _safe(osnr_nl, l_occ)]
    ger_c = [GERARD_REF["C_band"][f"{m}_dB"] for m in metrics_list]
    ger_l = [GERARD_REF["L_band"][f"{m}_dB"] for m in metrics_list]

    x = np.arange(len(metrics_list))
    w = 0.18
    ax3.bar(x - 1.5 * w, xlron_c_v, w, label="XLRON C", color=C_COLOR)
    ax3.bar(
        x - 0.5 * w, ger_c, w, label="Gerard C", color=C_COLOR_LIGHT, edgecolor=C_COLOR, hatch="//"
    )
    ax3.bar(x + 0.5 * w, xlron_l_v, w, label="XLRON L", color=L_COLOR)
    ax3.bar(
        x + 1.5 * w, ger_l, w, label="Gerard L", color=L_COLOR_LIGHT, edgecolor=L_COLOR, hatch="//"
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(["GOSNR", "OSNR$_{ASE}$", "OSNR$_{NL}$"])
    ax3.set_ylabel("Value (dB)")

    ax3.legend(loc="upper left")

    plt.savefig(os.path.join(out_dir, "validation_summary.png"))
    plt.close()
    print("  Saved validation_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gerard2025_results")
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Data directory:   {data_dir}")

    # --- Baseline: load cached or run simulation ---
    cached = load_baseline_data(data_dir)
    if cached is not None:
        print("\nLoaded cached baseline data.")
        freqs, path_d, occ, c_mask, l_mask, throughput_gbps = cached
    else:
        print("\n" + "=" * 70)
        print("RUNNING BASELINE SIMULATION (Gerard 2025 preset)")
        print("=" * 70)

        state, params, env, config = run_simulation()

        throughput_gbps = float(calculate_throughput_from_active_lightpaths(state, params))

        print("\nComputing SNR diagnostics...")
        diag = compute_snr_diagnostics(state, params)
        path_d = compute_path_diagnostics(diag, state, params)

        freqs = get_frequencies_thz(state, params)
        occ = get_occupied_mask(state)
        c_mask, l_mask = identify_bands(freqs, occ)

        save_baseline_data(data_dir, freqs, path_d, occ, c_mask, l_mask, throughput_gbps)

    throughput_tbps = throughput_gbps / 1000.0
    gosnr_db = np.array(path_d["gosnr_db"])
    num_ch = int(np.sum(occ))
    print(f"\nChannels loaded: {num_ch}")
    print(f"C-band channels: {int(np.sum(occ & c_mask))}")
    print(f"L-band channels: {int(np.sum(occ & l_mask))}")
    print(
        f"Shannon-Hartley throughput: {throughput_tbps:.1f} Tb/s "
        f"(Gerard: {GERARD_REF['total_capacity_tbps']} Tb/s)"
    )
    if num_ch > 0:
        print(f"GOSNR range: {np.nanmin(gosnr_db[occ]):.2f} - {np.nanmax(gosnr_db[occ]):.2f} dB")
        print(f"GOSNR variation: {np.nanmax(gosnr_db[occ]) - np.nanmin(gosnr_db[occ]):.2f} dB")

    # --- Generate core plots ---
    print("\nGenerating plots...")
    plot1_gosnr_vs_frequency(freqs, path_d, c_mask, l_mask, occ, out_dir)
    plot2_osnr_components(freqs, path_d, c_mask, l_mask, occ, out_dir)
    plot3_band_comparison(path_d, c_mask, l_mask, occ, out_dir)
    plot_summary(freqs, path_d, c_mask, l_mask, occ, out_dir)

    # --- Sweep: load cached or run ---
    sweep_cached = load_sweep_data(data_dir)
    if sweep_cached is not None:
        print("\nLoaded cached sweep data.")
        power_dbm_values, avg_gosnr_values = sweep_cached
    else:
        power_dbm_values, avg_gosnr_values = run_launch_power_sweep()
        save_sweep_data(data_dir, power_dbm_values, avg_gosnr_values)

    plot4_launch_power_sweep(power_dbm_values, avg_gosnr_values, out_dir)

    # --- Ablation study: load cached or run ---
    ablation_cached = load_ablation_data(data_dir)
    if ablation_cached is not None:
        print("\nLoaded cached ablation data.")
        ablation_results = ablation_cached
    else:
        ablation_results = run_ablation_sweeps()
        save_ablation_data(data_dir, ablation_results)

    plot5_ablation_sweep(ablation_results, out_dir)

    # --- Comprehensive summary table ---
    osnr_ase_db = np.array(path_d["osnr_ase_db"])
    osnr_nl_db = np.array(path_d["osnr_nl_db"])
    c_occ, l_occ = occ & c_mask, occ & l_mask

    def _sm(arr, mask):
        return float(np.nanmean(arr[mask])) if np.any(mask) else np.nan

    xlron_c_gosnr = _sm(gosnr_db, c_occ)
    xlron_l_gosnr = _sm(gosnr_db, l_occ)
    xlron_c_osnr_ase = _sm(osnr_ase_db, c_occ)
    xlron_l_osnr_ase = _sm(osnr_ase_db, l_occ)
    xlron_c_osnr_nl = _sm(osnr_nl_db, c_occ)
    xlron_l_osnr_nl = _sm(osnr_nl_db, l_occ)

    bw_thz = num_ch * 100e-3
    se = (throughput_gbps / (bw_thz * 1e3)) if bw_thz > 0 else 0.0
    variation = float(np.nanmax(gosnr_db[occ]) - np.nanmin(gosnr_db[occ])) if num_ch > 0 else 0.0

    # QAM FEC thresholds
    qam_thresholds = _qam_fec_thresholds()

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY TABLE")
    print("=" * 80)

    print("\n--- Band-Averaged SNR Metrics (dB) ---")
    print(
        f"{'Metric':<15} {'XLRON C':>10} {'Gerard C':>10} {'Delta':>8} {'XLRON L':>10} {'Gerard L':>10} {'Delta':>8}"
    )
    print("-" * 80)
    for label, xc, gc, xl, gl in [
        (
            "GOSNR",
            xlron_c_gosnr,
            GERARD_REF["C_band"]["GOSNR_dB"],
            xlron_l_gosnr,
            GERARD_REF["L_band"]["GOSNR_dB"],
        ),
        (
            "OSNR_ASE",
            xlron_c_osnr_ase,
            GERARD_REF["C_band"]["OSNR_ASE_dB"],
            xlron_l_osnr_ase,
            GERARD_REF["L_band"]["OSNR_ASE_dB"],
        ),
        (
            "OSNR_NL",
            xlron_c_osnr_nl,
            GERARD_REF["C_band"]["OSNR_NL_dB"],
            xlron_l_osnr_nl,
            GERARD_REF["L_band"]["OSNR_NL_dB"],
        ),
    ]:
        dc = xc - gc if not np.isnan(xc) else np.nan
        dl = xl - gl if not np.isnan(xl) else np.nan
        xc_s = f"{xc:.2f}" if not np.isnan(xc) else "N/A"
        xl_s = f"{xl:.2f}" if not np.isnan(xl) else "N/A"
        dc_s = f"{dc:+.2f}" if not np.isnan(dc) else "N/A"
        dl_s = f"{dl:+.2f}" if not np.isnan(dl) else "N/A"
        print(
            f"{label + ' (dB)':<15} {xc_s:>10} {gc:>10.1f} {dc_s:>8} {xl_s:>10} {gl:>10.1f} {dl_s:>8}"
        )

    print("\n--- System-Level Metrics ---")
    print(f"{'Metric':<35} {'XLRON':>12} {'Gerard':>12} {'Delta':>10}")
    print("-" * 72)
    print(
        f"{'Channels loaded':<35} {num_ch:>12d} {GERARD_REF['num_channels']:>12d} {num_ch - GERARD_REF['num_channels']:>+10d}"
    )
    print(f"{'  C-band channels':<35} {int(np.sum(c_occ)):>12d}")
    print(f"{'  L-band channels':<35} {int(np.sum(l_occ)):>12d}")
    print(
        f"{'Total capacity (Tb/s)':<35} {throughput_tbps:>12.1f} {GERARD_REF['total_capacity_tbps']:>12.1f} {throughput_tbps - GERARD_REF['total_capacity_tbps']:>+10.1f}"
    )
    print(
        f"{'Spectral efficiency (b/s/Hz)':<35} {se:>12.2f} {GERARD_REF['spectral_efficiency_bps_hz']:>12.2f} {se - GERARD_REF['spectral_efficiency_bps_hz']:>+10.2f}"
    )
    print(
        f"{'Total bandwidth (THz)':<35} {bw_thz:>12.1f} {GERARD_REF['total_bandwidth_thz']:>12.1f} {bw_thz - GERARD_REF['total_bandwidth_thz']:>+10.1f}"
    )
    print(f"{'GOSNR variation (dB)':<35} {variation:>12.2f} {'< 2.0':>12}")
    print(f"{'GOSNR min (dB)':<35} {float(np.nanmin(gosnr_db[occ])):>12.2f}" if num_ch > 0 else "")
    print(f"{'GOSNR max (dB)':<35} {float(np.nanmax(gosnr_db[occ])):>12.2f}" if num_ch > 0 else "")

    print("\n--- FEC Threshold Analysis ---")
    print(f"{'Modulation':<12} {'Threshold (dB)':>15} {'Channels above':>16} {'Fraction':>10}")
    print("-" * 56)
    for qam, thresh in qam_thresholds.items():
        above = int(np.sum(occ & (gosnr_db > thresh))) if num_ch > 0 else 0
        frac = above / num_ch if num_ch > 0 else 0.0
        print(f"{qam:<12} {thresh:>15.1f} {above:>16d} {frac:>10.1%}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
