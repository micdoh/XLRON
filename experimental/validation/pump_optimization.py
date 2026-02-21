#!/usr/bin/env python3
"""
Raman Pump Power Optimisation
==============================

Optimises backward Raman pump powers (and optionally frequencies and per-channel
launch powers) to maximise total Shannon-Hartley throughput of the loaded
Gerard 2025 network.

Approach
--------
1. Run `run_simulation()` to get the final fully-loaded state (90 channels
   placed by KSP-FF). Freeze this state — lightpath allocations are fixed.
2. Define a fully-differentiable objective:

       throughput(pump_pow_bw, launch_power)

   Inside this function:
     a. Call `fit_dra_params_jax(pump_pow_bw, ...)` — differentiable Raman
        profile fitting via custom_vjp (scipy TNC forward, IFT backward).
     b. Call `get_snr_dra(fit_params, ...)` per link with scaled launch power.
     c. Accumulate Shannon-Hartley throughput over all active lightpaths.

3. Differentiate with `jax.value_and_grad` and run Adam gradient ascent.

Notes
-----
- Lightpath allocations (which path, which slots) are kept fixed from the
  KSP-FF solution. Only pump powers/frequencies/launch powers are optimised.
- Launch power optimisation is controlled by --optimise_launch_power with
  granularity --launch_power_granularity {per_channel, per_band, scalar}.
- A total fibre power budget can be enforced with --max_total_power_dbm.

Usage
-----
  cd /path/to/XLRON
  python validation/pump_optimization.py [--steps 50] [--lr 1e-3]
      [--optimise_launch_power]
      [--launch_power_granularity per_band]
      [--max_total_power_dbm 21.0]
"""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from xlron.environments.gn_model import isrs_gn_model_dra
from xlron.environments.gn_model.isrs_gn_model import from_db

# ---------------------------------------------------------------------------
# Differentiable SNR wrapper (analytical VJP w.r.t. ch_power_w_i)
# ---------------------------------------------------------------------------


def _snr_analytical_grad(ch_pow, p_ase_inline, p_ase_roadm, p_nli, transceiver_noise, g_snr):
    """Analytical gradient of SNR w.r.t. per-channel launch power P.

    SNR = P / N_total,  N_total = ASE + eta*P^3 + P/TRX
    d(SNR)/d(P) = (ASE - 2*eta*P^3) / N_total^2

    This bypasses the NaN-producing reverse-mode through nan_to_num in gn_model_dra.
    Unoccupied channels (P=0) return zero gradient.
    """
    P = jnp.squeeze(ch_pow)
    occupied = P > 0
    # Use safe P to avoid 0^3 = 0 causing 0/0 in eta.
    # Use multiplication by mask instead of jnp.where to prevent NaN leaking
    # through the inactive branch in JAX autodiff.
    P_safe = P + (1.0 - occupied.astype(P.dtype))  # = P if occupied, 1.0 if not
    P3 = P_safe**3
    eta = jnp.nan_to_num(p_nli / P3, nan=0.0, posinf=0.0, neginf=0.0)
    N_total = jnp.maximum(p_ase_inline + p_ase_roadm + p_nli + transceiver_noise, 1e-40)
    ase = p_ase_inline + p_ase_roadm
    d_snr_d_P = (ase - 2.0 * eta * P3) / N_total**2
    # Multiply by mask (not jnp.where) so gradient is exactly 0 for empty channels
    return g_snr * d_snr_d_P * occupied.astype(d_snr_d_P.dtype)


# ---------------------------------------------------------------------------
# Differentiable objective
# ---------------------------------------------------------------------------


def make_throughput_objective(state, params, snr_variance_penalty=0.0):
    """Return a function pump_params -> throughput_gbps.

    Everything from `state` and `params` that is not being optimised is
    captured as a closure constant (no JAX tracing of the params struct,
    which contains non-differentiable HashableArrayWrapper fields).

    Supported keys in pump_params:
      pump_pow_bw   – (num_pumps_bw,) BW pump powers [W]
      pump_pow_fw   – (num_pumps_fw,) FW pump powers [W]  (optional)
      launch_power  – (num_channels,) per-channel power multipliers, relative
                      to the baseline from state.channel_power_array.
                      1.0 = no change. (optional)

    The full gradient path is:
        pump_pow_bw   →  fit_dra_params_jax  →  raman_fit_params
                      →  get_snr_dra (per link)  →  path SNR  →  throughput
        launch_power  →  ch_power_w_i  →  get_snr_dra  →  throughput

    Note: pump frequency optimisation is not supported — the Raman fitting
    uses int() casts on frequency-derived masks and scipy inside custom_vjp,
    making it non-differentiable w.r.t. frequencies.
    """
    # --- Static values captured from params ---
    num_channels = params.link_resources
    num_links = params.num_links
    max_spans = params.max_spans
    max_span_length = params.max_span_length
    ref_lambda = float(params.ref_lambda)
    raman_gain_slope = float(params.raman_gain_slope)
    attenuation = float(params.attenuation)
    attenuation_bar = float(params.attenuation_bar)
    nonlinear_coeff = float(params.nonlinear_coeff)
    dispersion_c = float(params.dispersion_coeff)
    dispersion_s = float(params.dispersion_slope)
    coherent = params.coherent
    amp_nf = params.amplifier_noise_figure.val  # (N,)
    trx_snr = params.transceiver_snr.val  # (N,)
    num_subchannels = params.num_subchannels
    slot_size_hz = params.slot_size * 1e9
    fec_threshold = params.fec_threshold
    min_snr = params.min_snr
    pack_bits = params.pack_path_bits
    link_resources = params.link_resources
    link_lengths_all = params.link_length_array.val  # (num_links, max_spans) metres
    ch_centre_freq_ghz = params.slot_centre_freq_array.val  # (N,) relative GHz
    active_lps = state.active_lightpaths_array  # (M, 3)
    path_link_arr = params.path_link_array.val

    # Baseline per-channel power per link, from the loaded state
    baseline_ch_pow = jnp.array(state.channel_power_array)  # (num_links, N) W

    # Static mask: slots that are zero on ALL links are permanent gap/guard slots.
    # Multiplying launch_multiplier by this mask prevents NaN gradients from
    # flowing through permanently-empty slots (e.g. inter-band gap at slot 45).
    _non_gap_mask = jnp.array(
        (np.array(state.channel_power_array).max(axis=0) > 0).astype(np.float32)
    )  # (N,)

    # Nominal pump freqs/powers — take row [0] since pumps are uniform per-span.
    # pump arrays in params have shape (num_spans, num_pumps); we optimise the
    # single representative row and replicate to all spans inside get_snr_dra.
    nominal_pump_freq_bw = jnp.array(params.raman_pump_freq_bw.val[0], dtype=jnp.float32)
    nominal_pump_freq_fw = jnp.array(params.raman_pump_freq_fw.val[0], dtype=jnp.float32)
    nominal_pump_pow_fw = jnp.array(params.raman_pump_power_fw.val[0], dtype=jnp.float32)

    # Static pump counts from nominal freqs — passed to fit_dra_params_jax to
    # avoid int(jnp.sum(traced_mask)) errors when throughput_fn is JIT-compiled.
    _static_num_bw = int(np.sum(np.array(params.raman_pump_freq_bw.val[0]) > 0))
    _static_num_fw = int(np.sum(np.array(params.raman_pump_freq_fw.val[0]) > 0))

    def throughput_fn(pump_params):
        # Optimised params are 1D (num_pumps,); get_snr_dra expects (max_spans, num_pumps).
        # Broadcast by tiling to all spans.
        pump_pow_bw_1d = pump_params["pump_pow_bw"]  # (num_pumps_bw,)
        pump_pow_fw_1d = pump_params.get("pump_pow_fw", nominal_pump_pow_fw)
        pump_pow_bw = jnp.broadcast_to(pump_pow_bw_1d[None, :], (max_spans, len(pump_pow_bw_1d)))
        pump_pow_fw = jnp.broadcast_to(pump_pow_fw_1d[None, :], (max_spans, len(pump_pow_fw_1d)))
        pump_freq_bw = jnp.broadcast_to(
            nominal_pump_freq_bw[None, :], (max_spans, len(nominal_pump_freq_bw))
        )
        pump_freq_fw = jnp.broadcast_to(
            nominal_pump_freq_fw[None, :], (max_spans, len(nominal_pump_freq_fw))
        )
        # launch_power: (num_channels,) multiplier, 1.0 = baseline
        launch_multiplier = pump_params.get(
            "launch_power", jnp.ones(num_channels, dtype=jnp.float32)
        )

        # --- Recompute raman_fit_params from current pump params ---
        span_length = float(max_span_length)  # uniform spans
        fit_params = isrs_gn_model_dra.fit_dra_params_jax(
            raman_gain_slope=raman_gain_slope,
            attenuation=attenuation,
            num_channels=num_channels,
            max_spans=max_spans,
            span_length=span_length,
            ref_lambda=ref_lambda,
            pump_pow_fw=pump_pow_fw_1d,
            pump_pow_bw=pump_pow_bw_1d,
            pump_freq_fw=nominal_pump_freq_fw,
            pump_freq_bw=nominal_pump_freq_bw,
            ch_centre_freq_ghz=ch_centre_freq_ghz,
            ch_power_w_i=baseline_ch_pow[0],
            num_bw=_static_num_bw,
            num_fw=_static_num_fw,
        )  # (6, num_channels, max_spans)

        # --- Compute per-link SNR ---
        def snr_for_link(link_index):
            link_lengths = link_lengths_all[link_index]  # (max_spans,) metres
            num_spans = jnp.ceil(jnp.sum(link_lengths) / max_span_length).astype(jnp.int32)
            bw_link = state.channel_centre_bw_array[link_index]  # (N,) GHz
            # Apply launch power multiplier to baseline channel powers.
            # _non_gap_mask zeroes permanently-empty gap slots so their gradient
            # is exactly zero and cannot produce NaN in the backward pass.
            ch_pow = baseline_ch_pow[link_index] * (launch_multiplier * _non_gap_mask)  # (N,) W
            ch_ctrs = state.channel_centre_freq_array[link_index]  # (N,) GHz

            snr, aux = isrs_gn_model_dra.get_snr_dra(
                num_channels=num_channels,
                num_spans=num_spans,
                max_spans=max_spans,
                ref_lambda=ref_lambda,
                length=link_lengths,
                attenuation_i=jnp.array(attenuation),
                attenuation_bar_i=jnp.array(attenuation_bar),
                nonlinear_coeff=jnp.array(nonlinear_coeff),
                raman_gain_slope_i=jnp.array(raman_gain_slope),
                dispersion_coeff=jnp.array(dispersion_c),
                dispersion_slope=jnp.array(dispersion_s),
                coherent=coherent,
                amplifier_noise_figure=amp_nf,
                transceiver_snr=trx_snr,
                ch_power_w_i=jax.lax.stop_gradient(ch_pow),
                ch_centre_i=ch_ctrs * 1e9,
                ch_bandwidth_i=bw_link * 1e9,
                excess_kurtosis_i=jnp.zeros(num_channels),
                uniform_spans=True,
                num_subchannels=num_subchannels,
                fit_params_ij=fit_params,
                raman_pump_power_fw=pump_pow_fw,
                raman_pump_power_bw=pump_pow_bw,
                raman_pump_freq_fw=pump_freq_fw,
                raman_pump_freq_bw=pump_freq_bw,
            )
            _, _, _, p_ase_inline, p_ase_roadm, p_nli, transceiver_noise = aux
            # Guard aux against NaN (occurs for zero-power channels in gn_model_dra)
            p_ase_inline = jnp.nan_to_num(p_ase_inline, nan=0.0)
            p_ase_roadm = jnp.nan_to_num(p_ase_roadm, nan=0.0)
            p_nli = jnp.nan_to_num(p_nli, nan=0.0)
            transceiver_noise = jnp.nan_to_num(transceiver_noise, nan=0.0)

            # Straight-through gradient for launch power:
            # SNR is computed with stop_gradient(ch_pow) to avoid NaN from
            # gn_model_dra's nan_to_num in reverse mode.  We add a zero-valued
            # phantom term whose gradient w.r.t. ch_pow equals d(SNR)/d(P),
            # so the overall gradient is analytically correct.
            #   value:    snr_sg + (ch_pow - sg(ch_pow)) * sg(d_snr_d_P)
            #           = snr_sg + 0 * sg(...)  = snr_sg  (correct forward)
            #   gradient: sg(d_snr_d_P)         (correct backward)
            d_snr_d_P = _snr_analytical_grad(
                ch_pow,
                p_ase_inline,
                p_ase_roadm,
                p_nli,
                transceiver_noise,
                g_snr=jnp.ones_like(snr),  # unit upstream grad; scaled by real g later
            )
            d_snr_d_P = jnp.nan_to_num(d_snr_d_P, nan=0.0, posinf=0.0, neginf=0.0)
            # Zero out the gap slot(s) explicitly — they have ch_pow=0 permanently
            # (inter-band guard band) and must not contribute any gradient.
            gap_mask = jnp.squeeze(ch_pow) > 0
            d_snr_d_P = d_snr_d_P * gap_mask
            phantom = (ch_pow - jax.lax.stop_gradient(ch_pow)) * jax.lax.stop_gradient(d_snr_d_P)
            # snr already has stop_gradient on ch_pow, so fit_params gradient
            # flows through snr normally. We only add phantom for launch power.
            # Zero out gap slots completely (stop_gradient prevents NaN leaking
            # back through snr into launch_multiplier for gap channels).
            snr_with_grad = jnp.where(
                _non_gap_mask > 0,
                snr + phantom,
                jax.lax.stop_gradient(jnp.zeros_like(snr)),
            )
            return snr_with_grad

        link_snr = jax.vmap(snr_for_link)(jnp.arange(num_links))  # (num_links, N)
        link_snr = jnp.nan_to_num(link_snr, nan=1e-5)

        # --- Accumulate Shannon throughput + SNR stats over active lightpaths ---
        slot_indices = jnp.arange(link_resources)
        one = jnp.ones((), dtype=link_snr.dtype)
        zero = jnp.zeros((), dtype=link_snr.dtype)

        def get_tp_iter(i, carry):
            throughput, snr_sum, snr_min, count = carry
            path_index, initial_slot, num_slots = active_lps[i]
            path_packed = path_link_arr[path_index]
            path = jnp.unpackbits(path_packed)[:num_links] if pack_bits else path_packed
            path = path.reshape((num_links, 1))

            # End-to-end SNR: sum 1/SNR across links on this path
            inv_snr_sum = jnp.sum(
                jnp.where(path, 1.0 / jnp.maximum(link_snr, 1e-20), 0.0), axis=0
            )  # (N,)
            # Add transceiver noise once at path level
            trx_snr_linear = 10.0 ** (trx_snr / 10.0)
            inv_snr_sum = inv_snr_sum + jnp.where(trx_snr_linear > 1.0, 1.0 / trx_snr_linear, 0.0)
            path_snr_db = jnp.where(
                inv_snr_sum > 0,
                -10.0 * jnp.log10(jnp.maximum(inv_snr_sum, 1e-20)),
                -50.0,
            )

            path_indices_on_slots = jnp.max(jnp.where(path, state.path_index_array, -1), axis=0)
            slot_mask = jnp.where(path_indices_on_slots == path_index, one, zero)
            cond = jnp.logical_and(
                slot_indices >= initial_slot,
                slot_indices < initial_slot + num_slots,
            )
            slot_mask = jnp.where(cond, slot_mask, zero)

            # Accumulate SNR dB stats for flatness penalty BEFORE the -50 dB
            # clamping, using only active slots above min_snr.  This avoids the
            # -50 dB sentinel dragging snr_min to an extreme value and creating
            # a ~63 dB discontinuous penalty that causes oscillation.
            above_threshold = jnp.logical_and(slot_mask > 0, path_snr_db >= min_snr)
            active_snr_db = jnp.where(above_threshold, path_snr_db, 1e9)
            snr_sum += jnp.sum(jnp.where(above_threshold, path_snr_db, 0.0))
            snr_min = jnp.minimum(snr_min, jnp.min(active_snr_db))
            count += jnp.sum(above_threshold.astype(path_snr_db.dtype))

            path_snr_db = jnp.where(path_snr_db < min_snr, -50.0, path_snr_db)
            snr_lin = from_db(path_snr_db) * slot_mask
            datarate = jnp.log2(1.0 + snr_lin) * slot_size_hz * 2.0 * (1.0 - fec_threshold)
            throughput += jnp.sum(datarate)
            return throughput, snr_sum, snr_min, count

        init_carry = (
            jnp.zeros((1,), dtype=link_snr.dtype),  # throughput
            jnp.zeros((1,), dtype=link_snr.dtype),  # snr_sum
            jnp.full((1,), 1e9, dtype=link_snr.dtype),  # snr_min (large sentinel)
            jnp.zeros((1,), dtype=link_snr.dtype),  # count
        )
        total, snr_sum, snr_min, count = jax.lax.fori_loop(
            0, active_lps.shape[0], get_tp_iter, init_carry
        )
        throughput_gbps = total[0] / 1e9  # bits/s → Gb/s

        if snr_variance_penalty > 0.0:
            # Penalise the worst low-SNR outlier: max(0, mean - min) in dB.
            # Only low-SNR outliers are penalised (high SNR outliers are fine).
            n = jnp.maximum(count[0], 1.0)
            snr_mean = snr_sum[0] / n
            snr_flatness_penalty = jnp.maximum(0.0, snr_mean - snr_min[0])
            throughput_gbps = throughput_gbps - snr_variance_penalty * snr_flatness_penalty

        return throughput_gbps

    return throughput_fn


# ---------------------------------------------------------------------------
# Launch power utilities
# ---------------------------------------------------------------------------


def _get_band_boundaries(ch_centre_freq_ghz):
    """Return list of (start_idx, end_idx) tuples for each contiguous band.

    Bands are identified by gaps > 200 GHz between adjacent slot centres.
    For 91 channels at 100 GHz spacing this typically gives one band.
    For C+L (43+48 channels) with a gap it gives two bands.
    """
    freqs = np.array(ch_centre_freq_ghz)
    gaps = np.diff(freqs)
    band_breaks = np.where(gaps > 200.0)[0] + 1  # index of first channel in new band
    boundaries = []
    prev = 0
    for b in band_breaks:
        boundaries.append((prev, b))
        prev = b
    boundaries.append((prev, len(freqs)))
    return boundaries


def make_launch_power_multiplier(band_scalars, ch_centre_freq_ghz):
    """Expand per-band scalars to per-channel multipliers via static band map.

    Args:
        band_scalars: (num_bands,) array of per-band launch power multipliers.
        ch_centre_freq_ghz: (num_channels,) static channel centres.

    Returns:
        (num_channels,) per-channel multipliers.
    """
    boundaries = _get_band_boundaries(np.array(ch_centre_freq_ghz))
    assert len(band_scalars) == len(boundaries), (
        f"Expected {len(boundaries)} band scalars, got {len(band_scalars)}"
    )
    result = jnp.zeros(len(ch_centre_freq_ghz))
    for i, (start, end) in enumerate(boundaries):
        result = result.at[start:end].set(band_scalars[i])
    return result


# ---------------------------------------------------------------------------
# Optimisation loop
# ---------------------------------------------------------------------------


def optimise_pump_powers(
    state,
    params,
    steps=200,
    lr=1e-3,
    lr_launch_power=None,
    optimise_launch_power=False,
    launch_power_granularity="per_band",
    max_total_power_dbm=None,
    max_total_pump_power_mw=None,
    num_starts=1,
    noise_scale=0.3,
    seed=0,
    snr_variance_penalty=0.0,
):
    print("\n" + "=" * 70)
    print("PUMP PARAMETER OPTIMISATION")
    print("=" * 70)

    ch_centre_freq_ghz = np.array(params.slot_centre_freq_array.val)
    num_channels = params.link_resources

    # Optimise in log-space: log_pump_pow_bw = log(pump_pow_bw), so pump powers
    # are always positive by construction (no clamping needed, gradients scale-invariant).
    pump_params = {
        "log_pump_pow_bw": jnp.log(jnp.array(params.raman_pump_power_bw.val[0], dtype=jnp.float32)),
        "log_pump_pow_fw": jnp.log(jnp.array(params.raman_pump_power_fw.val[0], dtype=jnp.float32)),
    }

    # --- Launch power setup ---
    band_boundaries = _get_band_boundaries(ch_centre_freq_ghz)
    num_bands = len(band_boundaries)

    # Baseline per-channel powers on link 0 (W), used for initial/final reporting
    baseline_ch_pow_w = np.array(state.channel_power_array[0])

    if optimise_launch_power:
        if launch_power_granularity == "per_channel":
            pump_params["log_launch_power"] = jnp.zeros(num_channels, dtype=jnp.float32)
            print(f"\nLaunch power: per-channel ({num_channels} params)")
            print(
                f"  Initial per-channel power (dBm): "
                f"{10 * np.log10(np.maximum(baseline_ch_pow_w, 1e-30) * 1e3)}"
            )
        elif launch_power_granularity == "per_band":
            pump_params["log_launch_power_band"] = jnp.zeros(num_bands, dtype=jnp.float32)
            print(f"\nLaunch power: per-band ({num_bands} bands)")
            for i, (s, e) in enumerate(band_boundaries):
                bw = (ch_centre_freq_ghz[e - 1] - ch_centre_freq_ghz[s]) + 100.0
                band_pow_dbm = 10 * np.log10(float(np.mean(baseline_ch_pow_w[s:e])) * 1e3)
                print(
                    f"  Band {i}: ch[{s}:{e}]  ({ch_centre_freq_ghz[s]:.0f} to "
                    f"{ch_centre_freq_ghz[e - 1]:.0f} GHz,  BW≈{bw:.0f} GHz, "
                    f"init={band_pow_dbm:.2f} dBm/ch)"
                )
        else:  # scalar
            pump_params["log_launch_power_scalar"] = jnp.zeros(1, dtype=jnp.float32)
            mean_pow_dbm = 10 * np.log10(float(np.mean(baseline_ch_pow_w)) * 1e3)
            print(f"\nLaunch power: scalar, init={mean_pow_dbm:.2f} dBm/ch (uniform)")

    print(
        f"\nInitial BW pump powers (mW): {np.exp(np.array(pump_params['log_pump_pow_bw'])) * 1e3}"
    )
    print(f"Initial BW pump freqs (THz): {np.array(params.raman_pump_freq_bw.val[0]) / 1e12}")
    _lr_lp = lr_launch_power if lr_launch_power is not None else lr
    print(
        f"Steps: {steps}, LR (pump): {lr}, LR (launch power): {_lr_lp}, "
        f"optimise_launch_power: {optimise_launch_power}"
    )
    if max_total_power_dbm is not None:
        print(f"Max total fibre launch power: {max_total_power_dbm:.1f} dBm")
    if max_total_pump_power_mw is not None:
        initial_pump_total_mw = float(jnp.sum(jnp.exp(pump_params["log_pump_pow_bw"]))) * 1e3
        print(
            f"Max total BW pump power: {max_total_pump_power_mw:.1f} mW (initial: {initial_pump_total_mw:.1f} mW)"
        )

    # --- Build the objective ---
    # For per-band/scalar, wrap throughput_fn to expand band scalars → per-channel
    if snr_variance_penalty > 0.0:
        print(f"SNR flatness penalty weight: {snr_variance_penalty} Gb/s per dB (mean-min)")
    raw_fn = make_throughput_objective(state, params, snr_variance_penalty=snr_variance_penalty)

    def throughput_fn(pp):
        # Convert from log-space back to linear before calling objective
        expanded = {
            "pump_pow_bw": jnp.exp(pp["log_pump_pow_bw"]),
            "pump_pow_fw": jnp.exp(pp["log_pump_pow_fw"]),
        }
        # Expand launch power (log-space multipliers → per-channel linear)
        if "log_launch_power_band" in pp:
            expanded["launch_power"] = make_launch_power_multiplier(
                jnp.exp(pp["log_launch_power_band"]), ch_centre_freq_ghz
            )
        elif "log_launch_power_scalar" in pp:
            expanded["launch_power"] = jnp.full(
                num_channels, jnp.exp(pp["log_launch_power_scalar"][0])
            )
        elif "log_launch_power" in pp:
            expanded["launch_power"] = jnp.exp(pp["log_launch_power"])
        return raw_fn(expanded)

    print("\nComputing baseline...")
    baseline_gbps = float(throughput_fn(pump_params))
    print(f"Baseline: {baseline_gbps:.3f} Gb/s  ({baseline_gbps / 1e3:.4f} Tb/s)")

    print("\nComputing initial gradients...")
    val, grads = jax.value_and_grad(throughput_fn)(pump_params)
    for k, g in grads.items():
        g_np = np.nan_to_num(np.array(g), nan=0.0)
        print(f"  d(T)/d({k}): mean={g_np.mean():.3e}, |max|={np.abs(g_np).max():.3e}")

    if np.all(np.abs(np.nan_to_num(np.array(grads["log_pump_pow_bw"]), nan=0.0)) < 1e-30):
        print("\nWARNING: Zero gradient. Check that channels are active.")
        return pump_params, [(0, baseline_gbps)]

    # Cosine decay schedule with linear warmup (5% of steps)
    warmup_steps = max(1, steps // 20)

    def _make_schedule(peak_lr):
        return optax.join_schedules(
            [
                optax.linear_schedule(0.0, peak_lr, warmup_steps),
                optax.cosine_decay_schedule(peak_lr, steps - warmup_steps, alpha=0.1),
            ],
            [warmup_steps],
        )

    _lr_lp = lr_launch_power if lr_launch_power is not None else lr
    _log_lp_keys = {"log_launch_power", "log_launch_power_band", "log_launch_power_scalar"}
    if optimise_launch_power and lr_launch_power is not None:
        param_labels = {k: "launch_power" if k in _log_lp_keys else "pump" for k in pump_params}
        optimizer = optax.multi_transform(
            {
                "pump": optax.adam(_make_schedule(lr)),
                "launch_power": optax.adam(_make_schedule(_lr_lp)),
            },
            param_labels,
        )
    else:
        optimizer = optax.adam(_make_schedule(lr))
    # ---- Build scan-based inner loop (vmappable over starting points) ----
    max_pump_w = (max_total_pump_power_mw * 1e-3) if max_total_pump_power_mw is not None else None
    max_launch_pow_w = (
        (10 ** ((max_total_power_dbm - 30) / 10)) if max_total_power_dbm is not None else None
    )
    _lp_keys_present = [
        k
        for k in ("log_launch_power", "log_launch_power_band", "log_launch_power_scalar")
        if k in pump_params
    ]

    def _apply_constraints(pp):
        if max_pump_w is not None:
            pump_lin = jnp.exp(pp["log_pump_pow_bw"])
            scale = jnp.minimum(1.0, max_pump_w / jnp.maximum(jnp.sum(pump_lin), 1e-20))
            pp = {**pp, "log_pump_pow_bw": jnp.log(pump_lin * scale)}
        if max_launch_pow_w is not None:
            for lp_key in _lp_keys_present:
                lp_lin = jnp.exp(pp[lp_key])
                if lp_key == "log_launch_power":
                    mult = lp_lin
                elif lp_key == "log_launch_power_band":
                    mult = make_launch_power_multiplier(lp_lin, ch_centre_freq_ghz)
                else:
                    mult = jnp.full(num_channels, lp_lin[0])
                total_pow_w = jnp.sum(jnp.array(state.channel_power_array[0]) * mult)
                scale = jnp.minimum(1.0, max_launch_pow_w / jnp.maximum(total_pow_w, 1e-20))
                pp = {**pp, lp_key: jnp.log(lp_lin * scale)}
        return pp

    # ---- Generate starting points ----
    rng = np.random.default_rng(seed)
    if num_starts == 1:
        all_init_params = [pump_params]
    else:
        print(f"\nMulti-start: {num_starts} starting points, noise_scale={noise_scale} (log-space)")

        def _perturb(base_pp):
            starts = [base_pp]
            for _ in range(num_starts - 1):
                noise_pp = {
                    k: v + jnp.array(rng.normal(0.0, noise_scale, v.shape).astype(np.float32))
                    for k, v in base_pp.items()
                }
                starts.append(noise_pp)
            return starts

        all_init_params = _perturb(pump_params)

    # ---- Run all starts sequentially ----
    print(f"\nRunning {num_starts} start(s) × {steps} steps...")
    # NOTE: fit_dra_params_jax uses scipy inside a custom_vjp, so throughput_fn
    # cannot be wrapped in jax.jit at the outer level.
    step_fn = jax.value_and_grad(throughput_fn)

    # Unpenalised evaluator for per-step SNR std reporting (built once, not per step).
    if snr_variance_penalty > 0.0:
        _raw_fn_no_penalty = make_throughput_objective(state, params, snr_variance_penalty=0.0)

        def _raw_tp_step_fn(pp):
            """Evaluate raw throughput (no penalty) by expanding log-params."""
            expanded = {
                "pump_pow_bw": jnp.exp(pp["log_pump_pow_bw"]),
                "pump_pow_fw": jnp.exp(pp["log_pump_pow_fw"]),
            }
            if "log_launch_power_band" in pp:
                expanded["launch_power"] = make_launch_power_multiplier(
                    jnp.exp(pp["log_launch_power_band"]), ch_centre_freq_ghz
                )
            elif "log_launch_power_scalar" in pp:
                expanded["launch_power"] = jnp.full(
                    num_channels, jnp.exp(pp["log_launch_power_scalar"][0])
                )
            elif "log_launch_power" in pp:
                expanded["launch_power"] = jnp.exp(pp["log_launch_power"])
            return float(_raw_fn_no_penalty(expanded))

    best_params = pump_params
    best_tp = baseline_gbps
    best_history = [(0, baseline_gbps)]
    all_best_tps = []

    for start_i, init_pp in enumerate(all_init_params):
        if num_starts > 1:
            print(f"\n--- Start {start_i} ---")
        pp = init_pp
        opt_state = optimizer.init(pp)
        tp_history = []
        for s in range(steps):
            tp, grads = step_fn(pp)
            # Compute SNR penalty at the SAME params that tp was evaluated at
            # (before the update), so the back-calculation is consistent.
            if snr_variance_penalty > 0.0:
                _raw_tp_at_current = _raw_tp_step_fn(pp)
            neg_grads = jax.tree.map(lambda g: -jnp.nan_to_num(g, nan=0.0), grads)
            updates, opt_state = optimizer.update(neg_grads, opt_state)
            pp = optax.apply_updates(pp, updates)
            pp = _apply_constraints(pp)
            tp_f = float(tp)
            tp_history.append(tp_f)
            pump_mw = np.exp(np.array(pp["log_pump_pow_bw"])) * 1e3
            pump_str = "  ".join(f"{v:6.2f}" for v in pump_mw)
            # Launch power field — fixed width per granularity mode
            if "log_launch_power" in pp:
                mult = np.exp(np.array(pp["log_launch_power"]))
                new_pow_w = baseline_ch_pow_w * mult
                active = new_pow_w[new_pow_w > 0]
                mean_dbm = float(10 * np.log10(np.mean(active) * 1e3)) if len(active) else 0.0
                std_db = (
                    float(np.std(10 * np.log10(np.maximum(active, 1e-30) * 1e3)))
                    if len(active) > 1
                    else 0.0
                )
                lp_str = f"  LP mean={mean_dbm:+6.2f} dBm  std={std_db:.3f} dB"
            elif "log_launch_power_band" in pp:
                mult = np.exp(np.array(pp["log_launch_power_band"]))
                band_strs = []
                for bi, (bs, be) in enumerate(band_boundaries):
                    init_dbm = 10 * np.log10(float(np.mean(baseline_ch_pow_w[bs:be])) * 1e3)
                    band_strs.append(f"B{bi}={init_dbm + 10 * np.log10(mult[bi]):+6.2f}")
                lp_str = "  LP " + "  ".join(band_strs) + " dBm/ch"
            elif "log_launch_power_scalar" in pp:
                m = float(np.exp(np.array(pp["log_launch_power_scalar"])[0]))
                init_dbm = 10 * np.log10(float(np.mean(baseline_ch_pow_w)) * 1e3)
                lp_str = f"  LP={init_dbm + 10 * np.log10(m):+6.2f} dBm/ch"
            else:
                lp_str = ""
            # SNR std field — only when variance penalty is active
            if snr_variance_penalty > 0.0:
                _snr_penalty_step = max(0.0, (_raw_tp_at_current - tp_f) / snr_variance_penalty)
                snr_str = f"  SNR(mean-min)={_snr_penalty_step:5.3f} dB"
            else:
                snr_str = ""
            print(
                f"  Step {s + 1:>{len(str(steps))}}/{steps}:"
                f"  {tp_f:11.3f} Gb/s  delta={tp_f - baseline_gbps:+10.2f} Gb/s"
                f"  pumps=[{pump_str}] mW{lp_str}{snr_str}"
            )
        start_best = max(tp_history)
        all_best_tps.append(start_best)
        if start_best > best_tp:
            best_tp = start_best
            best_params = pp
            best_history = [(0, baseline_gbps)] + [(s + 1, t) for s, t in enumerate(tp_history)]

    if not best_history or len(best_history) == 1:
        # fallback: use last start's history
        best_history = [(0, baseline_gbps)] + [(s + 1, t) for s, t in enumerate(tp_history)]

    # ---- Print per-start summary ----
    if num_starts > 1:
        print(f"\n{'Start':>6}  {'Best Tp (Tb/s)':>16}  {'Delta (Gb/s)':>14}")
        print("-" * 42)
        for i, tp_i in enumerate(all_best_tps):
            marker = " <-- best" if tp_i == best_tp else ""
            print(f"{i:>6}  {tp_i / 1e3:>16.4f}  {tp_i - baseline_gbps:>+14.2f}{marker}")

    history = best_history

    if snr_variance_penalty > 0.0:
        # Compute raw throughput (no penalty) at best params to report SNR std.
        # raw_fn is the penalised objective; swap it out temporarily.
        raw_tp = float(
            make_throughput_objective(state, params, snr_variance_penalty=0.0)(
                {
                    "pump_pow_bw": jnp.exp(best_params["log_pump_pow_bw"]),
                    "pump_pow_fw": jnp.exp(best_params["log_pump_pow_fw"]),
                    **(
                        {
                            "launch_power": make_launch_power_multiplier(
                                jnp.exp(best_params["log_launch_power_band"]), ch_centre_freq_ghz
                            )
                        }
                        if "log_launch_power_band" in best_params
                        else {}
                    ),
                    **(
                        {
                            "launch_power": jnp.full(
                                num_channels, jnp.exp(best_params["log_launch_power_scalar"][0])
                            )
                        }
                        if "log_launch_power_scalar" in best_params
                        else {}
                    ),
                    **(
                        {"launch_power": jnp.exp(best_params["log_launch_power"])}
                        if "log_launch_power" in best_params
                        else {}
                    ),
                }
            )
        )
        snr_mean_min = max(0.0, (raw_tp - best_tp) / snr_variance_penalty)
        print(
            f"\nBest (penalised obj): {best_tp:.3f} Gb/s ({best_tp / 1e3:.4f} Tb/s), "
            f"delta={best_tp - baseline_gbps:+.2f} Gb/s"
        )
        print(
            f"Raw throughput: {raw_tp:.3f} Gb/s ({raw_tp / 1e3:.4f} Tb/s), "
            f"SNR (mean-min): {snr_mean_min:.3f} dB"
        )
    else:
        print(
            f"\nBest: {best_tp:.3f} Gb/s ({best_tp / 1e3:.4f} Tb/s), "
            f"delta={best_tp - baseline_gbps:+.2f} Gb/s"
        )
    # --- Pump powers ---
    opt_pump_pow_bw_w = np.exp(np.array(best_params["log_pump_pow_bw"]))
    opt_pump_pow_bw_mw = opt_pump_pow_bw_w * 1e3
    print(f"Optimised BW pump powers (mW): {opt_pump_pow_bw_mw}")
    pump_csv_str = ",".join(f"{v:.6g}" for v in opt_pump_pow_bw_w)
    print(f"  Copy-paste (--raman_pump_power_bw): {pump_csv_str}")

    # --- Launch power ---
    if optimise_launch_power:
        if "log_launch_power" in best_params:
            # Per-channel: print human-readable summary + CSV for --launch_power_csv
            mult = np.exp(np.array(best_params["log_launch_power"]))
            new_pow_w = baseline_ch_pow_w * mult
            new_pow_dbm = 10 * np.log10(np.maximum(new_pow_w, 1e-30) * 1e3)
            init_pow_dbm = 10 * np.log10(np.maximum(baseline_ch_pow_w, 1e-30) * 1e3)
            print(f"Optimised per-channel power (dBm): {new_pow_dbm}")
            print(f"  (was: {init_pow_dbm})")
            # CSV output for --launch_power_csv
            slot_freqs_ghz = np.array(params.slot_centre_freq_array.val)
            print("\n--- Copy-paste CSV for --launch_power_csv ---")
            print("slot_index,freq_ghz,power_dbm")
            for si in range(num_channels):
                if baseline_ch_pow_w[si] > 0:
                    print(f"{si},{slot_freqs_ghz[si]:.3f},{new_pow_dbm[si]:.4f}")
            print("--- End CSV ---")
        elif "log_launch_power_band" in best_params:
            mult = np.exp(np.array(best_params["log_launch_power_band"]))
            print("Optimised per-band launch power:")
            band_dbm_vals = []
            for i, (s, e) in enumerate(band_boundaries):
                init_dbm = 10 * np.log10(float(np.mean(baseline_ch_pow_w[s:e])) * 1e3)
                new_dbm = init_dbm + 10 * np.log10(mult[i])
                band_dbm_vals.append(new_dbm)
                print(f"  Band {i}: {new_dbm:.3f} dBm/ch  (was {init_dbm:.3f}, ×{mult[i]:.4f})")
            band_csv_str = ",".join(f"{v:.4f}" for v in band_dbm_vals)
            print(f"  Copy-paste (--power_per_channel_per_band): {band_csv_str}")
        elif "log_launch_power_scalar" in best_params:
            mult = float(np.exp(np.array(best_params["log_launch_power_scalar"])[0]))
            init_dbm = 10 * np.log10(float(np.mean(baseline_ch_pow_w)) * 1e3)
            new_dbm = init_dbm + 10 * np.log10(mult)
            print(
                f"Optimised scalar launch power: {new_dbm:.3f} dBm/ch  (was {init_dbm:.3f}, ×{mult:.4f})"
            )
            print(f"  Copy-paste (--power_per_channel): {new_dbm:.4f}")

    return best_params, history


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_optimisation_history(history, out_dir):
    try:
        import matplotlib.pyplot as plt

        steps, tps = zip(*history)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, tps, "b-", lw=1.5)
        ax.axhline(tps[0], color="gray", ls="--", alpha=0.7, label=f"Baseline {tps[0]:.2f} Gb/s")
        ax.set_xlabel("Optimisation step")
        ax.set_ylabel("Shannon-Hartley throughput (Gb/s)")
        ax.legend()
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "pump_optimisation.png")
        plt.savefig(path)
        plt.close()
        print(f"  Saved {path}")
    except Exception as e:
        print(f"  Plot failed: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Gradient ascent steps (default 50; each step runs fit+SNR)",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr_launch_power",
        type=float,
        default=None,
        help="Learning rate for launch power (defaults to --lr if not set)",
    )
    parser.add_argument(
        "--optimise_launch_power",
        action="store_true",
        help="Co-optimise per-channel/band launch powers",
    )
    parser.add_argument(
        "--launch_power_granularity",
        default="per_band",
        choices=["per_channel", "per_band", "scalar"],
        help="Granularity of launch power optimisation (default: per_band)",
    )
    parser.add_argument(
        "--max_total_power_dbm",
        type=float,
        default=None,
        help="Optional total fibre launch power budget in dBm (e.g. 21.0)",
    )
    parser.add_argument(
        "--max_total_pump_power_mw",
        type=float,
        default=None,
        help="Optional total BW pump power budget in mW (e.g. 350.0)",
    )
    parser.add_argument(
        "--num_starts",
        type=int,
        default=1,
        help="Number of random starting points for multi-start optimisation (default: 1)",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.3,
        help="Std-dev of log-space noise added to initial params for extra starts (default: 0.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generating starting points (default: 0)",
    )
    parser.add_argument(
        "--snr_variance_penalty",
        type=float,
        default=0.0,
        help="Weight λ for SNR flatness penalty: objective = throughput - λ·max(0, mean(SNR)-min(SNR)). "
        "Penalises the worst low-SNR channel relative to the mean. "
        "Higher values enforce a flatter gain spectrum (default: 0.0 = disabled)",
    )
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gerard2025_results")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gerard2025_validation import run_simulation

    print("Running baseline simulation (Gerard 2025 preset)...")
    state, params, env, config = run_simulation(quiet=False)

    if not params.use_raman_amp:
        print("ERROR: use_raman_amp must be True.")
        sys.exit(1)

    best_params, history = optimise_pump_powers(
        state,
        params,
        steps=args.steps,
        lr=args.lr,
        lr_launch_power=args.lr_launch_power,
        optimise_launch_power=args.optimise_launch_power,
        launch_power_granularity=args.launch_power_granularity,
        max_total_power_dbm=args.max_total_power_dbm,
        max_total_pump_power_mw=args.max_total_pump_power_mw,
        num_starts=args.num_starts,
        noise_scale=args.noise_scale,
        seed=args.seed,
        snr_variance_penalty=args.snr_variance_penalty,
    )

    print("\nGenerating plots...")
    plot_optimisation_history(history, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
