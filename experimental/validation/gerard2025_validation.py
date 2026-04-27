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
  Plot 1+2: Combined per-channel SNR metrics vs frequency
  Plot 3: Band-averaged metrics vs Gerard Table I
  Plot 4: Launch power sweep vs average GOSNR
  Plot 5: Ablation study (power sweeps with different model configurations)
  Summary: Combined 3-panel validation figure
  Printed: Comprehensive validation summary table

Usage:
  cd /path/to/XLRON
  python experimental/gerard2025_validation.py
"""

import os

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as speed_of_light

from experimental.plot_style import (
    ACCENT_COLORS,
    PALETTE,
    PRIMARY_COLORS,
    configure_style,
)

from xlron.environments.env_funcs import (
    calculate_throughput_from_active_lightpaths,
    get_launch_power,
    process_path_action,
)
from xlron.environments.gn_model import isrs_gn_model, isrs_gn_model_dra
from xlron.environments.make_env import _gsnr_threshold_db, make, process_config
from xlron.gui.presets import PRESETS
from xlron.heuristics.heuristics import ksp_ff

# Apply publication-quality plot style (smaller fonts for multi-panel figures)
configure_style(font_size=24, axes_label_size=28, tick_size=22, legend_size=20)

# ---------------------------------------------------------------------------
# Data caching
# ---------------------------------------------------------------------------

BASELINE_DATA_FILE = "baseline_data.npz"


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
        gosnr_signalbw_db=np.array(path_d["gosnr_signalbw_db"]),
        osnr_ase_signalbw_db=np.array(path_d["osnr_ase_signalbw_db"]),
        osnr_nl_signalbw_db=np.array(path_d["osnr_nl_signalbw_db"]),
        metric_norm=np.array(["0p1nm"]),
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
    if "metric_norm" not in d.files:
        return None
    path_d = {
        "gosnr_db": d["gosnr_db"],
        "osnr_ase_db": d["osnr_ase_db"],
        "osnr_nl_db": d["osnr_nl_db"],
        "occupied": d["occupied"],
        "ch_power": d["ch_power"],
        "gosnr_signalbw_db": d["gosnr_signalbw_db"],
        "osnr_ase_signalbw_db": d["osnr_ase_signalbw_db"],
        "osnr_nl_signalbw_db": d["osnr_nl_signalbw_db"],
    }
    return d["freqs"], path_d, d["occ"], d["c_mask"], d["l_mask"], float(d["throughput_gbps"])



# Band colors consistent with plot_style palette
C_COLOR = PRIMARY_COLORS[0]   # Teal
L_COLOR = ACCENT_COLORS[1]   # Coral
C_COLOR_LIGHT = PRIMARY_COLORS[2]  # Medium mint
L_COLOR_LIGHT = "#F5B8C4"         # Light coral
GERARD_MARKER = "D"

# ---------------------------------------------------------------------------
# Gerard 2025 reference values (Table I, approximate)
# ---------------------------------------------------------------------------
GERARD_REF = {
    "C_band": {
        "GOSNR_dB": 25.9,
        "OSNR_ASE_dB": 27.2,
        "OSNR_NL_dB": 31.8,
    },
    "L_band": {
        "GOSNR_dB": 25.7,
        "OSNR_ASE_dB": 26.9,
        "OSNR_NL_dB": 31.7,
    },
    "total_capacity_tbps": 72.0,
    "shannon_capacity_tbps": 85.8,
    "spectral_efficiency_bps_hz": 7.29,
    "distance_km": 1504,
    "num_channels": 90,
    "channel_rate_gbps": 800,
    "total_bandwidth_thz": 8.625,
}


def run_simulation(config_overrides=None, quiet=False):
    """Run the simulation and return the final loaded state (before auto-reset).

    Uses the raw env's step_env to avoid auto-reset after max_requests.

    Returns:
        (final_state, env_params, raw_env, config)
    """
    config = {
        **PRESETS["Gerard 2025 recreation"],
        # Use the JAX Raman fitter for validation so the triangular model
        # cutoff is consistent with the DRA GN equations.
        "raman_fit_method": "jax",
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
        _, initial_slot_index = process_path_action(state, env_params, action)
        launch_power = get_launch_power(state, action, action, initial_slot_index, env_params)
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
            span_lumped_loss_db=getattr(params, "span_lumped_loss_db", None),
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
    total_noise_with_trx = total_ase + p_nli_path + trx_path
    total_noise_no_trx = total_ase + p_nli_path

    def safe_db(x):
        return 10 * jnp.log10(jnp.maximum(x, 1e-30))

    # Gerard reports OSNR values in 0.1 nm reference noise bandwidth.
    # Convert model SNR/OSNR values (signal-bandwidth referenced) using:
    #   OSNR_0.1nm = SNR_signalBW + 10*log10(B_signal / B_ref_0.1nm)
    ch_centres_hz = state.channel_centre_freq_array[0] * 1e9
    ch_bw_hz = state.channel_centre_bw_array[0] * 1e9
    abs_freq_hz = speed_of_light / float(params.ref_lambda) + ch_centres_hz
    b_ref_0p1nm_hz = (abs_freq_hz**2 / speed_of_light) * 0.1e-9
    bw_corr_db = safe_db(ch_bw_hz / jnp.maximum(b_ref_0p1nm_hz, 1e-30))

    # GOSNR including transceiver noise (used for actual throughput / Shannon capacity)
    gosnr_with_trx = jnp.where(occupied, ch_power / total_noise_with_trx, jnp.nan)
    # GOSNR excluding transceiver noise (optical-domain metric, matches Gerard Table I)
    gosnr_no_trx = jnp.where(occupied, ch_power / jnp.maximum(total_noise_no_trx, 1e-30), jnp.nan)
    osnr_ase = jnp.where(occupied, ch_power / jnp.maximum(total_ase, 1e-30), jnp.nan)
    osnr_nl = jnp.where(occupied, ch_power / jnp.maximum(p_nli_path, 1e-30), jnp.nan)

    gosnr_signalbw_db = jnp.where(occupied, safe_db(gosnr_no_trx), jnp.nan)
    gosnr_with_trx_signalbw_db = jnp.where(occupied, safe_db(gosnr_with_trx), jnp.nan)
    osnr_ase_signalbw_db = jnp.where(occupied, safe_db(osnr_ase), jnp.nan)
    osnr_nl_signalbw_db = jnp.where(occupied, safe_db(osnr_nl), jnp.nan)

    gosnr_0p1nm_db = jnp.where(occupied, gosnr_signalbw_db + bw_corr_db, jnp.nan)
    gosnr_with_trx_0p1nm_db = jnp.where(occupied, gosnr_with_trx_signalbw_db + bw_corr_db, jnp.nan)
    osnr_ase_0p1nm_db = jnp.where(occupied, osnr_ase_signalbw_db + bw_corr_db, jnp.nan)
    osnr_nl_0p1nm_db = jnp.where(occupied, osnr_nl_signalbw_db + bw_corr_db, jnp.nan)

    return {
        # Primary metrics used for Gerard comparison (0.1 nm referenced).
        "gosnr_db": gosnr_0p1nm_db,  # optical GOSNR (no TRX), 0.1 nm
        "gosnr_with_trx_db": gosnr_with_trx_0p1nm_db,  # full GOSNR, 0.1 nm
        "osnr_ase_db": osnr_ase_0p1nm_db,
        "osnr_nl_db": osnr_nl_0p1nm_db,
        # Raw signal-bandwidth-referenced metrics kept for diagnostics.
        "gosnr_signalbw_db": gosnr_signalbw_db,
        "gosnr_with_trx_signalbw_db": gosnr_with_trx_signalbw_db,
        "osnr_ase_signalbw_db": osnr_ase_signalbw_db,
        "osnr_nl_signalbw_db": osnr_nl_signalbw_db,
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
    "32QAM": ACCENT_COLORS[0],   # Purple
    "64QAM": ACCENT_COLORS[3],   # Orange
    "128QAM": PRIMARY_COLORS[3], # Dark teal
    "256QAM": ACCENT_COLORS[2],  # Seafoam
}


def plot1_2_combined_snr_metrics(freqs, path_d, c_mask, l_mask, occ, out_dir):
    """Combined Plot 1+2: GOSNR (optical, no TRX), OSNR_ASE, OSNR_NL, and received SNR vs frequency."""
    # gosnr_db is optical GOSNR (ASE + NLI only, no transceiver noise)
    gosnr = np.array(path_d["gosnr_db"])
    osnr_ase = np.array(path_d["osnr_ase_db"])
    osnr_nl = np.array(path_d["osnr_nl_db"])
    gosnr_with_trx = np.array(path_d["gosnr_with_trx_db"])

    fig, ax = plt.subplots(figsize=(12, 5))

    # Metrics with Gerard reference lines (3 original metrics)
    metric_defs = [
        ("GOSNR", "GOSNR_dB", gosnr, PRIMARY_COLORS[0], "o"),
        ("OSNR$_{ASE}$", "OSNR_ASE_dB", osnr_ase, ACCENT_COLORS[1], "o"),
        ("OSNR$_{NL}$", "OSNR_NL_dB", osnr_nl, ACCENT_COLORS[0], "o"),
    ]

    metric_handles = {}
    gerard_l_handles = {}
    gerard_c_handles = {}
    for metric_label, metric_key, values, color, marker in metric_defs:
        metric_handles[metric_label] = ax.scatter(
            freqs[occ],
            values[occ],
            c=color,
            s=80,
            marker=marker,
            label=metric_label,
            zorder=3,
            edgecolors="k",
            linewidths=0.5,
        )
        gerard_l_handles[metric_label] = ax.axhline(
            GERARD_REF["L_band"][metric_key],
            color=color,
            ls=":",
            alpha=0.7,
            label=f"Gerard {metric_label} L",
        )
        gerard_c_handles[metric_label] = ax.axhline(
            GERARD_REF["C_band"][metric_key],
            color=color,
            ls="--",
            alpha=0.7,
            label=f"Gerard {metric_label} C",
        )

    # Received SNR (includes transceiver noise) — no Gerard reference lines
    rcv_label = "Received SNR"
    metric_handles[rcv_label] = ax.scatter(
        freqs[occ],
        gosnr_with_trx[occ],
        c="black",
        s=80,
        marker="o",
        label=rcv_label,
        zorder=3,
        edgecolors="k",
        linewidths=0.5,
    )

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("SNR Metric (dB)")

    # Build 3-row × 4-column legend grid explicitly.
    # Columns 1-3: metric scatter | Gerard L ref | Gerard C ref.
    # Column 4: Received SNR (row 1 only), blank in rows 2-3.
    from matplotlib.patches import Patch
    blank = Patch(facecolor="none", edgecolor="none", label="")

    ordered_handles = []
    ordered_labels = []
    for i, (metric_label, _, _, _, _) in enumerate(metric_defs):
        ordered_handles.extend([
            metric_handles[metric_label],
            gerard_l_handles[metric_label],
            gerard_c_handles[metric_label],
            metric_handles[rcv_label] if i == 0 else blank,
        ])
        ordered_labels.extend([
            metric_label,
            gerard_l_handles[metric_label].get_label(),
            gerard_c_handles[metric_label].get_label(),
            rcv_label if i == 0 else " ",
        ])

    # Use two legends: main 3×3 grid + separate Received SNR to the right.
    main_handles = []
    main_labels = []
    for metric_label, _, _, _, _ in metric_defs:
        main_handles.extend([
            metric_handles[metric_label],
            gerard_l_handles[metric_label],
            gerard_c_handles[metric_label],
        ])
        main_labels.extend([
            metric_label,
            gerard_l_handles[metric_label].get_label(),
            gerard_c_handles[metric_label].get_label(),
        ])

    leg1 = ax.legend(
        main_handles,
        main_labels,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        frameon=True,
        borderaxespad=0,
        fontsize=18,
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        [metric_handles[rcv_label]],
        [rcv_label],
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=True,
        borderaxespad=0,
        fontsize=18,
    )

    plt.savefig(
        os.path.join(out_dir, "plot1_2_combined_snr_metrics.png"),
        bbox_extra_artists=[leg1, leg2],
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved plot1_2_combined_snr_metrics.png")


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

    fig, ax = plt.subplots(figsize=(11, 6))
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
                    fontsize=20,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["GOSNR", "OSNR$_{ASE}$", "OSNR$_{NL}$"],
    )
    ax.set_ylabel("Value (dB)")
    ax.set_ylim(bottom=20)

    ax.legend(loc="upper left", fontsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot3_band_comparison.png"))
    plt.close()
    print("  Saved plot3_band_comparison.png")

    return xlron_c, xlron_l



# ---------------------------------------------------------------------------
# Ablation study: power sweep with different configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "Full model (baseline)": {},
    "No Raman": {"use_raman_amp": False},
    "1 subchannel": {"num_subchannels": 1},
    "Incoherent ASE": {"coherent": False},
    "No Raman + 1 subchannel": {"use_raman_amp": False, "num_subchannels": 1},
}

ABLATION_DATA_FILE = "ablation_data.npz"


def run_ablation_sweeps():
    """Run power sweeps for each ablation configuration.

    Uses uniform per-channel power (max_power_per_fibre / num_channels) with
    CSV and per-band overrides disabled, so all channels get equal power.

    Returns dict: {config_name: (power_dbm_values, avg_gosnr_values)}.
    """
    power_dbm_values = np.concatenate([
        np.arange(18.0, 22.0, 0.25),
        np.arange(22.0, 26.0, 0.1),
    ])
    results = {}

    for name, overrides in ABLATION_CONFIGS.items():
        print(f"\n--- Ablation sweep: {name} ---")
        avg_gosnr_values = []

        for pdbm in power_dbm_values:
            print(f"  Power = {pdbm:.2f} dBm...", end="", flush=True)
            try:
                cfg = {
                    "max_power_per_fibre": float(pdbm),
                    "launch_power_csv": None,
                    "power_per_channel_per_band": None,
                }
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
    fig, ax = plt.subplots(figsize=(11, 8))

    colors = PALETTE[:5]
    markers = ["o", "s", "^", "D", "v"]

    for i, (name, (powers, gosnrs)) in enumerate(ablation_results.items()):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        ax.plot(powers, gosnrs, color=c, marker=m, markersize=6, label=name, linewidth=2.0)

        valid = ~np.isnan(gosnrs)
        if np.any(valid):
            opt_idx = int(np.nanargmax(gosnrs))
            ax.scatter(
                [powers[opt_idx]],
                [gosnrs[opt_idx]],
                c=c,
                s=220,
                zorder=5,
                marker="*",
                edgecolors="k",
                linewidths=0.5,
            )

    ax.axvline(21.2, color="red", linestyle=":", linewidth=2.0, label="Gerard optimal (21.2 dBm)")
    ax.set_xlabel("Total Fibre Launch Power (dBm)")
    ax.set_ylabel("Average GOSNR (dB)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.75, 0.0), fontsize=18)

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
    ax1.set_ylabel("GOSNR optical (dB)")

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


def plot_raman_gain_profile(freqs, c_mask, l_mask, occ, params, out_dir):
    """Plot per-channel Raman gain profile vs frequency (ODE target and fitted)."""
    if params is None:
        print("  Skipping Raman gain profile plot (params unavailable).")
        return
    if not getattr(params, "use_raman_amp", False):
        print("  Skipping Raman gain profile plot (Raman amplification disabled).")
        return

    fit_params = np.array(params.raman_fit_params.val)
    if fit_params.ndim != 3 or fit_params.shape[0] < 6:
        print("  Skipping Raman gain profile plot (invalid Raman fit parameters).")
        return

    # Row 5 stores the per-channel ODE-derived Raman gain (linear, first span).
    raman_gain_ode_linear = np.array(fit_params[5, :, 0])
    raman_gain_ode_linear = np.maximum(raman_gain_ode_linear, 1e-30)
    raman_gain_ode_db = 10.0 * np.log10(raman_gain_ode_linear)

    # Reconstruct fitted semi-analytical Raman gain at z=L from rows 0-4:
    # rho(L) = exp(-a_fit*L) * (1 - x_i * delta_f)
    # Raman gain = rho(L) / exp(-a_physical*L)
    #            = exp(-(a_fit - a_physical)*L) * (1 - x_i * delta_f)
    # Note: a_fit (row 4) differs from physical attenuation because the LM fit
    # absorbs part of the Raman gain into the exponential decay term.
    C_f = np.array(fit_params[0, :, 0])
    a_f = np.array(fit_params[1, :, 0])
    C_b = np.array(fit_params[2, :, 0])
    a_b = np.array(fit_params[3, :, 0])
    a_fit = np.array(fit_params[4, :, 0])
    a_physical = float(params.attenuation)

    span_length = float(params.max_span_length)
    l_eff_f = (1.0 - np.exp(-a_f * span_length)) / np.maximum(a_f, 1e-30)
    l_eff_b = (1.0 - np.exp(-a_b * span_length)) / np.maximum(a_b, 1e-30)

    ref_freq_hz = speed_of_light / float(params.ref_lambda)
    rel_ch_freq_hz = np.array(freqs) * 1e12 - ref_freq_hz

    pump_fw = np.array(params.raman_pump_freq_fw.val).reshape(-1)
    pump_bw = np.array(params.raman_pump_freq_bw.val).reshape(-1)
    pump_fw_pow = np.array(params.raman_pump_power_fw.val).reshape(-1)
    active_fw = pump_fw[pump_fw > 0.0]
    active_bw = pump_bw[pump_bw > 0.0]
    all_active = (
        np.concatenate([active_fw, active_bw])
        if (active_fw.size + active_bw.size) > 0
        else np.array([ref_freq_hz])
    )
    f_hat_hz = float(np.mean(all_active) - ref_freq_hz)
    delta_f = rel_ch_freq_hz - f_hat_hz

    # P_f matches the convention used during fitting (sum of channel powers + FW pumps).
    slot_launch_powers = np.array(params.slot_launch_power_array.val).reshape(-1)
    P_f = float(np.sum(slot_launch_powers)) + float(np.sum(pump_fw_pow[pump_fw > 0.0]))
    x_i = C_f * P_f * l_eff_f + C_b * l_eff_b
    # Full rho(L) divided by exp(-a_physical*L) to get Raman gain.
    # a_fit (row 4) differs from a_physical because the LM fit absorbs some of
    # the Raman gain into the exponential decay term.
    rho_at_L = np.exp(-a_fit * span_length) * (1.0 - x_i * delta_f)
    raman_gain_fit_linear = np.maximum(rho_at_L / np.exp(-a_physical * span_length), 1e-30)
    raman_gain_fit_db = 10.0 * np.log10(raman_gain_fit_linear)

    f_occ = np.array(freqs[occ])
    g_ode_occ = np.array(raman_gain_ode_db[occ])
    g_fit_occ = np.array(raman_gain_fit_db[occ])
    if f_occ.size == 0:
        print("  Skipping Raman gain profile plot (no occupied channels).")
        return

    order = np.argsort(f_occ)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        f_occ[order],
        g_ode_occ[order],
        color=PRIMARY_COLORS[3],
        lw=1.2,
        alpha=0.8,
        label=r"ODE Raman gain  $G_\mathrm{ODE} = P(L)\,/\,[P(0)\,e^{-\alpha L}]$",
    )
    ax.plot(
        f_occ[order],
        g_fit_occ[order],
        color=PRIMARY_COLORS[0],
        lw=1.6,
        alpha=0.95,
        label=r"Semi-analytical  $\rho(L)\,/\,e^{-\alpha L}$  (rows 0-4)",
    )
    ax.axhline(0.0, color="k", ls="--", alpha=0.4, linewidth=1.0)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Raman Gain (dB)")
    ax.set_ylim(bottom=-1.0)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_raman_gain_profile.png"))
    plt.close()
    print("  Saved plot_raman_gain_profile.png")


def plot_gain_budget(freqs, state, c_mask, l_mask, occ, params, out_dir):
    """Plot per-channel Raman gain, EDFA gain, and total gain vs frequency.

    The total gain (Raman * EDFA) should restore the transmitted power,
    i.e. equal the fibre span loss (including lumped loss if present).
    """
    if params is None or not getattr(params, "use_raman_amp", False):
        print("  Skipping gain budget plot (Raman amplification unavailable).")
        return

    fit_params = np.array(params.raman_fit_params.val)
    if fit_params.ndim != 3 or fit_params.shape[0] < 6:
        print("  Skipping gain budget plot (invalid fit parameters).")
        return

    # Raman gain from fit_params row 5 (pump-only, linear)
    raman_gain_linear = np.array(fit_params[5, :, 0])
    raman_gain_linear = np.maximum(raman_gain_linear, 1.0)

    # EDFA gain: recompute the same way get_snr_dra does
    span_length = float(params.max_span_length)
    ch_centres_hz = np.array(state.channel_centre_freq_array[0]) * 1e9
    ch_power = np.array(state.channel_power_array[0])

    edfa_gain_no_raman = np.array(
        isrs_gn_model.calculate_amplifier_gain_isrs(
            jnp.array(params.attenuation),
            jnp.array(span_length),
            jnp.array(params.raman_gain_slope),
            jnp.array(ch_power),
            jnp.array(ch_centres_hz),
        )
    )
    lumped_loss_db = getattr(params, "span_lumped_loss_db", None)
    if lumped_loss_db is not None:
        edfa_gain_no_raman = edfa_gain_no_raman * (10 ** (lumped_loss_db / 10))
    edfa_gain_linear = np.maximum(edfa_gain_no_raman / raman_gain_linear, 1.0)

    total_gain_linear = raman_gain_linear * edfa_gain_linear

    # Convert to dB
    raman_gain_db = 10 * np.log10(raman_gain_linear)
    edfa_gain_db = 10 * np.log10(edfa_gain_linear)
    total_gain_db = 10 * np.log10(total_gain_linear)

    # Fibre loss (the target that total gain should match)
    fibre_loss_db = 10 * np.log10(np.exp(float(params.attenuation) * span_length))
    total_target_db = fibre_loss_db
    if lumped_loss_db is not None:
        total_target_db += lumped_loss_db

    # Plot occupied channels only
    f_occ = np.array(freqs[occ])
    if f_occ.size == 0:
        print("  Skipping gain budget plot (no occupied channels).")
        return
    order = np.argsort(f_occ)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        f_occ[order],
        raman_gain_db[occ][order],
        color=PRIMARY_COLORS[0],
        lw=2.5,
        label="Raman Gain",
    )
    ax.plot(
        f_occ[order],
        edfa_gain_db[occ][order],
        color=ACCENT_COLORS[0],
        lw=2.5,
        label="EDFA Gain",
    )
    ax.plot(
        f_occ[order],
        total_gain_db[occ][order],
        color=ACCENT_COLORS[1],
        lw=2.8,
        label="Total Gain",
    )

    # Span-loss reference line (no legend entry)
    ax.axhline(
        total_target_db,
        color="k",
        ls="--",
        lw=1.2,
        alpha=0.6,
    )

    # Band shading + C/L labels positioned just below the total gain line
    # at each band's mean frequency.
    if np.any(c_mask & occ):
        c_f = freqs[c_mask & occ]
        ax.axvspan(c_f.min(), c_f.max(), alpha=0.06, color=C_COLOR, zorder=0)
        c_y = float(np.mean(total_gain_db[c_mask & occ])) - 1.5
        ax.text(
            np.mean(c_f),
            c_y,
            "C",
            ha="center",
            va="top",
            fontsize=22,
            color=C_COLOR,
            alpha=0.85,
            fontweight="bold",
        )
    if np.any(l_mask & occ):
        l_f = freqs[l_mask & occ]
        ax.axvspan(l_f.min(), l_f.max(), alpha=0.06, color=L_COLOR, zorder=0)
        l_y = float(np.mean(total_gain_db[l_mask & occ])) - 1.5
        ax.text(
            np.mean(l_f),
            l_y,
            "L",
            ha="center",
            va="top",
            fontsize=22,
            color=L_COLOR,
            alpha=0.85,
            fontweight="bold",
        )

    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Gain (dB)")
    ax.legend(loc="best", fontsize=18)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_gain_budget.png"))
    plt.close()
    print("  Saved plot_gain_budget.png")


def plot_power_evolution(freqs, state, c_mask, l_mask, occ, params, out_dir):
    """Plot per-channel power evolution in dBm along the span.

    Uses the stored semi-analytical fit parameters (rho(z) from
    ``raman_fit_params[0:5]``) multiplied by each channel's launch power to
    give actual power in dBm vs position.  All occupied channels are shown as
    a bundle, coloured by C/L band, with a pure-attenuation reference.
    """
    if params is None or not getattr(params, "use_raman_amp", False):
        print("  Skipping power evolution plot (Raman amplification unavailable).")
        return

    fit_params_full = np.array(params.raman_fit_params.val)  # (6+, num_ch, max_spans)
    if fit_params_full.ndim != 3 or fit_params_full.shape[0] < 6:
        print("  Skipping power evolution plot (invalid fit parameters shape).")
        return

    # --- Physical parameters for span 0 ---
    L = float(params.max_span_length)
    a = float(params.attenuation)
    ref_freq = speed_of_light / float(params.ref_lambda)

    # Channel data from state (link 0) — occupied channels only
    ch_freq_rel_ghz = np.array(state.channel_centre_freq_array[0])  # GHz, relative
    ch_freq_abs_hz = ref_freq + ch_freq_rel_ghz * 1e9  # Hz, absolute
    ch_power_w = np.array(state.channel_power_array[0])  # W

    # Pump arrays (filter sentinel zeros)
    pump_pow_fw = np.array(params.raman_pump_power_fw.val).reshape(-1)
    pump_freq_bw = np.array(params.raman_pump_freq_bw.val).reshape(-1)
    pump_freq_fw = np.array(params.raman_pump_freq_fw.val).reshape(-1)

    active_bw = pump_freq_bw[pump_freq_bw > 0.0]
    active_fw = pump_freq_fw[pump_freq_fw > 0.0]
    active_fw_pow = pump_pow_fw[pump_freq_fw > 0.0]

    # P_f and f_hat for the semi-analytical formula (same convention as fitting)
    P_signal = float(np.sum(ch_power_w[occ]))
    P_f = P_signal + float(np.sum(active_fw_pow))
    all_pumps = (
        np.concatenate([active_fw, active_bw])
        if (active_fw.size + active_bw.size) > 0
        else np.array([ref_freq])
    )
    f_hat = float(np.mean(all_pumps)) - ref_freq

    # Per-channel frequency offset from mean pump frequency
    delta_f = ch_freq_abs_hz - ref_freq - f_hat  # Hz

    # Fit params for first span (rows 0-4: C_f, a_f, C_b, a_b, a)
    C_f = fit_params_full[0, :, 0]
    a_f = fit_params_full[1, :, 0]
    C_b = fit_params_full[2, :, 0]
    a_b = fit_params_full[3, :, 0]
    a_fit = fit_params_full[4, :, 0]

    n_z = 501
    z_np = np.linspace(0.0, L, n_z)
    z_km = z_np / 1e3

    def _rho(ch_idx):
        """Semi-analytical normalised power profile rho(z) for channel ch_idx."""
        cf, af, cb, ab, af2 = C_f[ch_idx], a_f[ch_idx], C_b[ch_idx], a_b[ch_idx], a_fit[ch_idx]
        l_eff = (1.0 - np.exp(-af * z_np)) / np.maximum(af, 1e-30)
        l_eff_b = (np.exp(-ab * (L - z_np)) - np.exp(-ab * L)) / np.maximum(ab, 1e-30)
        x_i = cf * P_f * l_eff + cb * l_eff_b  # P_b = 1 convention
        return np.exp(-af2 * z_np) * (1.0 - x_i * delta_f[ch_idx])

    occ_idx = np.where(occ)[0]
    if len(occ_idx) == 0:
        print("  Skipping power evolution plot (no occupied channels).")
        return

    c_occ_idx = occ_idx[c_mask[occ_idx]]
    l_occ_idx = occ_idx[l_mask[occ_idx]]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot all C-band channels as a blue bundle
    for i, ch_idx in enumerate(c_occ_idx):
        p_launch_dbm = 10.0 * np.log10(ch_power_w[ch_idx] * 1e3)  # W → dBm
        p_dbm = p_launch_dbm + 10.0 * np.log10(np.maximum(_rho(ch_idx), 1e-30))
        ax.plot(
            z_km,
            p_dbm,
            color=C_COLOR,
            lw=0.8,
            alpha=0.4,
            label="C-band" if i == 0 else None,
        )

    # Plot all L-band channels as a red bundle
    for i, ch_idx in enumerate(l_occ_idx):
        p_launch_dbm = 10.0 * np.log10(ch_power_w[ch_idx] * 1e3)
        p_dbm = p_launch_dbm + 10.0 * np.log10(np.maximum(_rho(ch_idx), 1e-30))
        ax.plot(
            z_km,
            p_dbm,
            color=L_COLOR,
            lw=0.8,
            alpha=0.4,
            label="L-band" if i == 0 else None,
        )

    # Reference: pure fibre attenuation from the mean occupied-channel launch power
    mean_launch_w = float(np.mean(ch_power_w[occ_idx]))
    mean_launch_dbm = 10.0 * np.log10(mean_launch_w * 1e3)
    ax.plot(
        z_km,
        mean_launch_dbm + 10.0 * np.log10(np.exp(-a * z_np)),
        "k--",
        lw=1.8,
        alpha=0.7,
        label=f"Pure attenuation ({mean_launch_dbm:.1f} dBm launch)",
    )

    ax.set_xlabel("Position along span (km)")
    ax.set_ylabel("Power (dBm)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_power_evolution.png"))
    plt.close()
    print("  Saved plot_power_evolution.png")


def plot_power_evolution_3d(freqs, state, c_mask, l_mask, occ, params, out_dir):
    """3-D surface plot of power (dBm) vs distance along span and channel frequency.

    Builds a (num_occupied_channels × n_z) grid of power values from the
    stored semi-analytical DRA fit, then renders it as a ``plot_surface``
    with frequency on the x-axis, distance on the y-axis, and power (dBm)
    on the z-axis.  C-band and L-band faces use distinct colourmaps so the
    inter-band gap and relative power tilt are immediately visible.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection

    if params is None or not getattr(params, "use_raman_amp", False):
        print("  Skipping 3-D power evolution plot (Raman amplification unavailable).")
        return

    fit_params_full = np.array(params.raman_fit_params.val)
    if fit_params_full.ndim != 3 or fit_params_full.shape[0] < 6:
        print("  Skipping 3-D power evolution plot (invalid fit parameters shape).")
        return

    # --- Physical parameters (span 0) ---
    L = float(params.max_span_length)
    a = float(params.attenuation)
    ref_freq = speed_of_light / float(params.ref_lambda)

    ch_freq_rel_ghz = np.array(state.channel_centre_freq_array[0])
    ch_freq_abs_hz = ref_freq + ch_freq_rel_ghz * 1e9
    ch_power_w = np.array(state.channel_power_array[0])

    pump_pow_fw = np.array(params.raman_pump_power_fw.val).reshape(-1)
    pump_freq_bw = np.array(params.raman_pump_freq_bw.val).reshape(-1)
    pump_freq_fw = np.array(params.raman_pump_freq_fw.val).reshape(-1)

    active_bw = pump_freq_bw[pump_freq_bw > 0.0]
    active_fw = pump_freq_fw[pump_freq_fw > 0.0]
    active_fw_pow = pump_pow_fw[pump_freq_fw > 0.0]

    P_signal = float(np.sum(ch_power_w[occ]))
    P_f = P_signal + float(np.sum(active_fw_pow))
    all_pumps = (
        np.concatenate([active_fw, active_bw])
        if (active_fw.size + active_bw.size) > 0
        else np.array([ref_freq])
    )
    f_hat = float(np.mean(all_pumps)) - ref_freq
    delta_f = ch_freq_abs_hz - ref_freq - f_hat

    C_f = fit_params_full[0, :, 0]
    a_f = fit_params_full[1, :, 0]
    C_b = fit_params_full[2, :, 0]
    a_b = fit_params_full[3, :, 0]
    a_fit = fit_params_full[4, :, 0]

    n_z = 200  # coarser grid for 3-D rendering speed
    z_np = np.linspace(0.0, L, n_z)
    z_km = z_np / 1e3

    def _rho(ch_idx):
        cf, af, cb, ab, af2 = C_f[ch_idx], a_f[ch_idx], C_b[ch_idx], a_b[ch_idx], a_fit[ch_idx]
        l_eff = (1.0 - np.exp(-af * z_np)) / np.maximum(af, 1e-30)
        l_eff_b = (np.exp(-ab * (L - z_np)) - np.exp(-ab * L)) / np.maximum(ab, 1e-30)
        x_i = cf * P_f * l_eff + cb * l_eff_b
        return np.exp(-af2 * z_np) * (1.0 - x_i * delta_f[ch_idx])

    occ_idx = np.where(occ)[0]
    if len(occ_idx) == 0:
        print("  Skipping 3-D power evolution plot (no occupied channels).")
        return

    # Sort occupied channels by absolute frequency for a smooth surface
    occ_freqs_thz = ch_freq_abs_hz[occ_idx] / 1e12
    sort_order = np.argsort(occ_freqs_thz)
    occ_idx_sorted = occ_idx[sort_order]
    occ_freqs_sorted = occ_freqs_thz[sort_order]

    n_ch = len(occ_idx_sorted)

    # Build (n_ch × n_z) power grid
    P_grid = np.empty((n_ch, n_z))
    for k, ch_idx in enumerate(occ_idx_sorted):
        p_launch_dbm = 10.0 * np.log10(ch_power_w[ch_idx] * 1e3)
        P_grid[k, :] = p_launch_dbm + 10.0 * np.log10(np.maximum(_rho(ch_idx), 1e-30))

    # Meshgrid: F (frequency, THz) × Z (distance, km)
    F_grid, Z_grid = np.meshgrid(occ_freqs_sorted, z_km, indexing="ij")  # both (n_ch, n_z)

    # Identify C / L band membership for each sorted channel
    c_sorted = c_mask[occ_idx_sorted]
    l_sorted = l_mask[occ_idx_sorted]

    # --- Figure ---
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Build per-face colour arrays from a diverging colourmap keyed to power
    p_min, p_max = float(P_grid.min()), float(P_grid.max())
    p_range = p_max - p_min if p_max > p_min else 1.0

    c_cmap = mcolors.LinearSegmentedColormap.from_list(
        "c_band", [PRIMARY_COLORS[1], PRIMARY_COLORS[0], PRIMARY_COLORS[3]]
    )
    l_cmap = mcolors.LinearSegmentedColormap.from_list(
        "l_band", [L_COLOR_LIGHT, ACCENT_COLORS[1], "#9E2B3A"]
    )

    def _insert_nan_at_gaps(idx, power_jump_threshold_db=1.0):
        """Insert NaN rows where adjacent channels have a large power discontinuity.

        Detects jumps by comparing the mean absolute power difference between
        adjacent channels (across all distance points) to a threshold in dB.
        """
        if len(idx) < 2:
            return F_grid[idx, :], Z_grid[idx, :], P_grid[idx, :]
        F_sub = F_grid[idx, :]
        Z_sub = Z_grid[idx, :]
        P_sub = P_grid[idx, :]
        # Detect large power jumps between adjacent channels at end of span
        # where Raman gain differences are most pronounced
        max_power_diff = np.max(np.abs(np.diff(P_sub, axis=0)), axis=1)
        median_diff = np.median(max_power_diff)
        gap_positions = np.where(max_power_diff > max(power_jump_threshold_db, 3 * median_diff))[0]
        if len(gap_positions) == 0:
            return F_sub, Z_sub, P_sub
        # Insert NaN rows at each gap (work backwards to keep indices valid)
        nan_row_f = np.full((1, F_sub.shape[1]), np.nan)
        nan_row_z = np.full((1, Z_sub.shape[1]), np.nan)
        nan_row_p = np.full((1, P_sub.shape[1]), np.nan)
        for gap_idx in reversed(gap_positions):
            F_sub = np.insert(F_sub, gap_idx + 1, nan_row_f, axis=0)
            Z_sub = np.insert(Z_sub, gap_idx + 1, nan_row_z, axis=0)
            P_sub = np.insert(P_sub, gap_idx + 1, nan_row_p, axis=0)
        return F_sub, Z_sub, P_sub

    # Plot C-band surface patch
    c_idx = np.where(c_sorted)[0]
    if len(c_idx) >= 2:
        F_c, Z_c, P_c = _insert_nan_at_gaps(c_idx)
        norm_c = np.clip((P_c - p_min) / p_range, 0, 1)
        face_colors_c = c_cmap(norm_c)
        # Set alpha to 0 for NaN faces so gaps are fully transparent
        nan_mask = np.isnan(P_c)
        face_colors_c[nan_mask, 3] = 0.0
        ax.plot_surface(
            Z_c,
            F_c,
            np.nan_to_num(P_c, nan=p_min),
            facecolors=face_colors_c,
            alpha=0.85,
            linewidth=0,
            antialiased=True,
            label="C-band",
        )

    # Plot L-band surface patch
    l_idx = np.where(l_sorted)[0]
    if len(l_idx) >= 2:
        F_l, Z_l, P_l = _insert_nan_at_gaps(l_idx)
        norm_l = np.clip((P_l - p_min) / p_range, 0, 1)
        face_colors_l = l_cmap(norm_l)
        nan_mask = np.isnan(P_l)
        face_colors_l[nan_mask, 3] = 0.0
        ax.plot_surface(
            Z_l,
            F_l,
            np.nan_to_num(P_l, nan=p_min),
            facecolors=face_colors_l,
            alpha=0.85,
            linewidth=0,
            antialiased=True,
            label="L-band",
        )

    # Overlay a few highlighted channel traces at the surface edges
    highlight_step = max(1, n_ch // 8)
    for k in range(0, n_ch, highlight_step):
        color = C_COLOR if c_sorted[k] else L_COLOR
        ax.plot(
            z_km,
            np.full(n_z, occ_freqs_sorted[k]),
            P_grid[k, :],
            color=color,
            lw=0.8,
            alpha=0.6,
        )

    # Pure-attenuation reference ribbon at mean launch power
    mean_launch_dbm = float(np.mean([10.0 * np.log10(ch_power_w[i] * 1e3) for i in occ_idx]))
    p_att = mean_launch_dbm + 10.0 * np.log10(np.exp(-a * z_np))
    f_lo, f_hi = float(occ_freqs_sorted[0]), float(occ_freqs_sorted[-1])
    for f_edge in [f_lo, f_hi]:
        ax.plot(
            z_km,
            np.full(n_z, f_edge),
            p_att,
            "k--",
            lw=1.2,
            alpha=0.5,
        )

    ax.set_xlabel("Distance (km)", labelpad=8)
    ax.set_ylabel("Frequency (THz)", labelpad=8)
    ax.set_zlabel("Power (dBm)", labelpad=8)

    # Proxy artists for the legend (plot_surface doesn't auto-register)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=C_COLOR, alpha=0.8, label="C-band"),
        Patch(facecolor=L_COLOR, alpha=0.8, label="L-band"),
        plt.Line2D([0], [0], color="k", ls="--", lw=1.2, alpha=0.6, label="Pure attenuation"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.view_init(elev=25, azim=-50)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_power_evolution_3d.png"), dpi=150)
    plt.close()
    print("  Saved plot_power_evolution_3d.png")




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
    state = None
    params = None
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

        # save_baseline_data(data_dir, freqs, path_d, occ, c_mask, l_mask, throughput_gbps)

    throughput_tbps = throughput_gbps / 1000.0
    fec_oh = params.fec_threshold if params is not None else (1 - 0.838)
    throughput_tbps_shannon = throughput_tbps / (1 - fec_oh)
    gosnr_db = np.array(path_d["gosnr_db"])
    num_ch = int(np.sum(occ))
    print(f"\nChannels loaded: {num_ch}")
    print(f"C-band channels: {int(np.sum(occ & c_mask))}")
    print(f"L-band channels: {int(np.sum(occ & l_mask))}")
    print(
        f"Shannon-Hartley throughput: {throughput_tbps_shannon:.1f} Tb/s "
        f"(Gerard: {GERARD_REF['shannon_capacity_tbps']} Tb/s)"
        f""
    )
    print(f"AIR: {throughput_tbps:.1f} Tb/s (Gerard: {GERARD_REF['total_capacity_tbps']} Tb/s)")
    if num_ch > 0:
        print(f"GOSNR range: {np.nanmin(gosnr_db[occ]):.2f} - {np.nanmax(gosnr_db[occ]):.2f} dB")
        print(f"GOSNR variation: {np.nanmax(gosnr_db[occ]) - np.nanmin(gosnr_db[occ]):.2f} dB")

    # --- Generate core plots ---
    print("\nGenerating plots...")
    plot1_2_combined_snr_metrics(freqs, path_d, c_mask, l_mask, occ, out_dir)
    plot3_band_comparison(path_d, c_mask, l_mask, occ, out_dir)
    plot_summary(freqs, path_d, c_mask, l_mask, occ, out_dir)
    plot_raman_gain_profile(freqs, c_mask, l_mask, occ, params, out_dir)
    if state is not None:
        plot_gain_budget(freqs, state, c_mask, l_mask, occ, params, out_dir)
        plot_power_evolution(freqs, state, c_mask, l_mask, occ, params, out_dir)
        plot_power_evolution_3d(freqs, state, c_mask, l_mask, occ, params, out_dir)
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

    bw_thz = (num_ch * 100e-3) - (config.inter_band_gap_ghz * 1e-3)
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
