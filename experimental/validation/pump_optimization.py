#!/usr/bin/env python3
"""
Raman Pump Power Optimisation
==============================

Optimises backward Raman pump powers (and optionally per-channel launch powers)
to maximise total Shannon-Hartley throughput of the loaded Gerard 2025 network.

Pump powers can be optimised independently (default), constrained to lie on a
straight line in log-space via --pump_power_mode=slope (2 params: intercept + slope
vs normalised pump frequency), or with fixed total power and slope-only optimisation
via --pump_power_mode=fixed_total_slope (1 param: slope; intercept derived
analytically to maintain exact total pump power).

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

import jax
import jax.numpy as jnp
import numpy as np
import optax

from xlron.environments.gn_model import isrs_gn_model_dra
from xlron.environments.gn_model.isrs_gn_model import calculate_roadm_ase, from_db

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
    span_lumped_loss_db = params.span_lumped_loss_db
    roadm_express_loss = params.roadm_express_loss.val  # (N,) dB
    roadm_add_drop_loss = params.roadm_add_drop_loss.val  # (N,) dB
    roadm_noise_figure = params.roadm_noise_figure.val  # (N,) dB

    # Baseline per-channel power per link, from the loaded state
    baseline_ch_pow = jnp.array(state.channel_power_array)  # (num_links, N) W

    # Initial per-slot launch powers used at env creation for the Raman fit.
    # Using the same array ensures the baseline fit matches the cached params.
    _initial_launch_pow = jnp.array(params.slot_launch_power_array.val)  # (N,)

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
            ch_power_w_i=_initial_launch_pow,
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
                span_lumped_loss_db=span_lumped_loss_db,
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
            # Add ROADM ASE noise at path level
            num_links_on_path = jnp.sum(path.astype(jnp.float32))
            num_express = jnp.maximum(num_links_on_path - 1, 0)
            first_link_idx = jnp.argmax(path.squeeze())
            ch_pow_first = baseline_ch_pow[first_link_idx]  # (N,) W
            ch_bw_hz = state.channel_centre_bw_array[first_link_idx] * 1e9
            ch_ctrs_hz = state.channel_centre_freq_array[first_link_idx] * 1e9
            p_ase_roadm = calculate_roadm_ase(
                roadm_express_loss=roadm_express_loss,
                roadm_add_drop_loss=roadm_add_drop_loss,
                roadm_noise_figure=roadm_noise_figure,
                num_roadm_express=num_express,
                ref_lambda=ref_lambda,
                ch_centre_i=ch_ctrs_hz,
                ch_bandwidth_i=ch_bw_hz,
            )
            nsr_roadm = jnp.where(ch_pow_first > 0, p_ase_roadm / ch_pow_first, 0.0)
            inv_snr_sum = inv_snr_sum + nsr_roadm
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


def _make_pump_pow_from_slope(log_intercept, slope, norm_freqs):
    """Compute per-pump log-powers from a linear slope parameterisation.

    log_power[i] = log_intercept + slope * norm_freq[i]

    where norm_freq is in [-1, 1] (normalised from the active pump frequencies).
    Returns log-space powers (apply jnp.exp to get linear).
    """
    return log_intercept + slope * norm_freqs


def _intercept_for_fixed_total(slope, norm_freqs, total_power_w):
    """Compute the log-intercept that yields a given total pump power.

    Given:  P_total = sum_i exp(intercept + slope * norm_freq_i)
            P_total = exp(intercept) * sum_i exp(slope * norm_freq_i)
            intercept = log(P_total) - log(sum_i exp(slope * norm_freq_i))

    Uses log-sum-exp for numerical stability.
    """
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(slope * norm_freqs)))
    return jnp.log(total_power_w) - log_sum_exp


def optimise_pump_powers(
    state,
    params,
    steps=200,
    lr=1e-3,
    lr_launch_power=None,
    optimise_launch_power=False,
    launch_power_granularity="per_band",
    pump_power_mode="independent",
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

    # --- Pump power parameterisation ---
    initial_log_pump_bw = jnp.log(jnp.array(params.raman_pump_power_bw.val[0], dtype=jnp.float32))
    initial_log_pump_fw = jnp.log(jnp.array(params.raman_pump_power_fw.val[0], dtype=jnp.float32))

    # For slope mode: normalise active BW pump frequencies to [-1, 1]
    pump_freq_bw_np = np.array(params.raman_pump_freq_bw.val[0])
    active_bw_mask = pump_freq_bw_np > 0
    num_active_bw = int(np.sum(active_bw_mask))
    if num_active_bw > 0:
        active_freqs = pump_freq_bw_np[active_bw_mask]
        freq_min, freq_max = float(active_freqs.min()), float(active_freqs.max())
        freq_range = max(freq_max - freq_min, 1.0)  # avoid div-by-zero for single pump
        # Normalise all pumps (inactive ones get 0, but won't be used)
        norm_freqs_bw = np.where(
            active_bw_mask,
            2.0 * (pump_freq_bw_np - freq_min) / freq_range - 1.0,
            0.0,
        ).astype(np.float32)
    else:
        norm_freqs_bw = np.zeros_like(pump_freq_bw_np, dtype=np.float32)
    norm_freqs_bw_jnp = jnp.array(norm_freqs_bw)

    # Default: not used unless pump_power_mode == "fixed_total_slope"
    fixed_total_pump_w = None

    # Compute initial slope from least-squares fit (shared by slope and fixed_total_slope)
    if pump_power_mode in ("slope", "fixed_total_slope"):
        if num_active_bw > 1:
            active_log_pow = np.array(initial_log_pump_bw)[active_bw_mask]
            active_nf = norm_freqs_bw[active_bw_mask]
            A = np.stack([np.ones_like(active_nf), active_nf], axis=1)
            lstsq_result = np.linalg.lstsq(A, active_log_pow, rcond=None)
            init_intercept, init_slope = float(lstsq_result[0][0]), float(lstsq_result[0][1])
        else:
            init_intercept = float(initial_log_pump_bw[0]) if num_active_bw == 1 else 0.0
            init_slope = 0.0

    if pump_power_mode == "fixed_total_slope":
        # Fixed total power: only the slope is optimised (1 param).
        # The intercept is computed analytically so sum(pump_pow) = target.
        initial_total_pump_w = float(jnp.sum(jnp.exp(initial_log_pump_bw)))
        fixed_total_pump_w = (
            (max_total_pump_power_mw * 1e-3) if max_total_pump_power_mw is not None
            else initial_total_pump_w
        )
        # Recompute intercept for the target total at the initial slope
        init_intercept = float(_intercept_for_fixed_total(
            jnp.float32(init_slope), norm_freqs_bw_jnp, fixed_total_pump_w
        ))
        pump_params = {
            "pump_slope_bw": jnp.array([init_slope], dtype=jnp.float32),
            "log_pump_pow_fw": initial_log_pump_fw,
        }
        fitted_log_pow = init_intercept + init_slope * norm_freqs_bw
        fitted_pow_mw = np.exp(fitted_log_pow[active_bw_mask]) * 1e3
        print(f"\nPump power mode: fixed_total_slope (1 param: slope only)")
        print(f"  Fixed total BW pump power: {fixed_total_pump_w * 1e3:.1f} mW")
        print(f"  Initial intercept (log, derived): {init_intercept:.4f}, slope: {init_slope:.4f}")
        print(f"  Fitted BW pump powers (mW): {fitted_pow_mw}")
    elif pump_power_mode == "slope":
        pump_params = {
            "log_pump_intercept_bw": jnp.array([init_intercept], dtype=jnp.float32),
            "pump_slope_bw": jnp.array([init_slope], dtype=jnp.float32),
            "log_pump_pow_fw": initial_log_pump_fw,
        }
        fitted_log_pow = init_intercept + init_slope * norm_freqs_bw
        fitted_pow_mw = np.exp(fitted_log_pow[active_bw_mask]) * 1e3
        print(f"\nPump power mode: slope (2 params: intercept + slope)")
        print(f"  Initial intercept (log): {init_intercept:.4f}, slope: {init_slope:.4f}")
        print(f"  Fitted BW pump powers (mW): {fitted_pow_mw}")
    else:
        # Optimise in log-space: log_pump_pow_bw = log(pump_pow_bw), so pump powers
        # are always positive by construction (no clamping needed, gradients scale-invariant).
        pump_params = {
            "log_pump_pow_bw": initial_log_pump_bw,
            "log_pump_pow_fw": initial_log_pump_fw,
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

    if "log_pump_pow_bw" in pump_params:
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
        if pump_power_mode == "fixed_total_slope":
            print(
                f"Total BW pump power fixed at: {fixed_total_pump_w * 1e3:.1f} mW (from --max_total_pump_power_mw)"
            )
        elif "log_pump_intercept_bw" in pump_params:
            _init_log = _make_pump_pow_from_slope(
                float(pump_params["log_pump_intercept_bw"][0]),
                float(pump_params["pump_slope_bw"][0]),
                norm_freqs_bw_jnp,
            )
            initial_pump_total_mw = float(jnp.sum(jnp.exp(_init_log))) * 1e3
            print(
                f"Max total BW pump power: {max_total_pump_power_mw:.1f} mW (initial: {initial_pump_total_mw:.1f} mW)"
            )
        else:
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
        # Expand pump powers based on mode
        if "pump_slope_bw" in pp and "log_pump_intercept_bw" not in pp:
            # fixed_total_slope: intercept derived analytically from slope + total power
            ic = _intercept_for_fixed_total(
                pp["pump_slope_bw"][0], norm_freqs_bw_jnp, fixed_total_pump_w
            )
            log_pump_bw = _make_pump_pow_from_slope(ic, pp["pump_slope_bw"][0], norm_freqs_bw_jnp)
            pump_pow_bw = jnp.exp(log_pump_bw)
        elif "log_pump_intercept_bw" in pp:
            log_pump_bw = _make_pump_pow_from_slope(
                pp["log_pump_intercept_bw"][0], pp["pump_slope_bw"][0], norm_freqs_bw_jnp
            )
            pump_pow_bw = jnp.exp(log_pump_bw)
        else:
            pump_pow_bw = jnp.exp(pp["log_pump_pow_bw"])
        expanded = {
            "pump_pow_bw": pump_pow_bw,
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

    pump_grad_key = (
        "pump_slope_bw" if ("pump_slope_bw" in grads and "log_pump_intercept_bw" not in grads)
        else "log_pump_intercept_bw" if "log_pump_intercept_bw" in grads
        else "log_pump_pow_bw"
    )
    if np.all(np.abs(np.nan_to_num(np.array(grads[pump_grad_key]), nan=0.0)) < 1e-30):
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
            if "pump_slope_bw" in pp and "log_pump_intercept_bw" not in pp:
                # fixed_total_slope: total power is exact by construction, skip
                pass
            elif "log_pump_intercept_bw" in pp:
                # Slope mode: compute per-pump powers, scale intercept to satisfy budget
                log_pow = _make_pump_pow_from_slope(
                    pp["log_pump_intercept_bw"][0], pp["pump_slope_bw"][0], norm_freqs_bw_jnp
                )
                pump_lin = jnp.exp(log_pow)
                scale = jnp.minimum(1.0, max_pump_w / jnp.maximum(jnp.sum(pump_lin), 1e-20))
                # Shift intercept by log(scale) to uniformly scale all pumps
                pp = {**pp, "log_pump_intercept_bw": pp["log_pump_intercept_bw"] + jnp.log(scale)}
            else:
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
    _is_slope_mode = pump_power_mode in ("slope", "fixed_total_slope")
    if _is_slope_mode and num_starts > 1:
        # For slope modes: sweep slopes from "all power at low-freq pump" to
        # "all power at high-freq pump".  With norm_freqs in [-1, 1], a slope
        # of S gives a highest/lowest pump ratio of exp(2*S).  We use a max
        # ratio of ~20 dB (100x), i.e. |S_max| ≈ ln(100)/2 ≈ 2.3.
        s_max = np.log(100.0) / 2.0  # ≈ 2.3
        sweep_slopes = np.linspace(-s_max, s_max, num_starts).astype(np.float32)
        all_init_params = []
        print(f"\nSlope sweep: {num_starts} starting slopes from {-s_max:.2f} to {+s_max:.2f}")
        for i, sl in enumerate(sweep_slopes):
            pp_i = {**pump_params, "pump_slope_bw": jnp.array([sl], dtype=jnp.float32)}
            if "log_pump_intercept_bw" in pp_i:
                # slope mode: recompute intercept to preserve initial total power
                _init_total = float(jnp.sum(jnp.exp(initial_log_pump_bw)))
                ic = float(_intercept_for_fixed_total(
                    jnp.float32(sl), norm_freqs_bw_jnp, _init_total
                ))
                pp_i = {**pp_i, "log_pump_intercept_bw": jnp.array([ic], dtype=jnp.float32)}
            all_init_params.append(pp_i)
            # Show the resulting pump powers for this start
            if pump_power_mode == "fixed_total_slope":
                _ic = float(_intercept_for_fixed_total(
                    jnp.float32(sl), norm_freqs_bw_jnp, fixed_total_pump_w
                ))
            else:
                _ic = float(pp_i["log_pump_intercept_bw"][0])
            _pow = np.exp((_ic + sl * norm_freqs_bw)[active_bw_mask]) * 1e3
            print(f"  Start {i}: slope={sl:+.4f}  pumps(mW)=[{', '.join(f'{v:.1f}' for v in _pow)}]")
    elif num_starts == 1:
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
            if "pump_slope_bw" in pp and "log_pump_intercept_bw" not in pp:
                ic = _intercept_for_fixed_total(
                    pp["pump_slope_bw"][0], norm_freqs_bw_jnp, fixed_total_pump_w
                )
                pump_bw = jnp.exp(_make_pump_pow_from_slope(ic, pp["pump_slope_bw"][0], norm_freqs_bw_jnp))
            elif "log_pump_intercept_bw" in pp:
                log_pow = _make_pump_pow_from_slope(
                    pp["log_pump_intercept_bw"][0], pp["pump_slope_bw"][0], norm_freqs_bw_jnp
                )
                pump_bw = jnp.exp(log_pow)
            else:
                pump_bw = jnp.exp(pp["log_pump_pow_bw"])
            expanded = {
                "pump_pow_bw": pump_bw,
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
        start_best_tp = -float("inf")
        start_best_pp = pp  # track best params within this start
        for s in range(steps):
            tp, grads = step_fn(pp)
            tp_f = float(tp)
            # tp was evaluated at pp BEFORE the update — snapshot if it's best
            if tp_f > start_best_tp:
                start_best_tp = tp_f
                start_best_pp = jax.tree.map(lambda x: x.copy(), pp)
            # Compute SNR penalty at the SAME params that tp was evaluated at
            # (before the update), so the back-calculation is consistent.
            if snr_variance_penalty > 0.0:
                _raw_tp_at_current = _raw_tp_step_fn(pp)
            neg_grads = jax.tree.map(lambda g: -jnp.nan_to_num(g, nan=0.0), grads)
            updates, opt_state = optimizer.update(neg_grads, opt_state)
            pp = optax.apply_updates(pp, updates)
            pp = _apply_constraints(pp)
            tp_history.append(tp_f)
            if "pump_slope_bw" in pp and "log_pump_intercept_bw" not in pp:
                # fixed_total_slope mode
                _sl = float(pp["pump_slope_bw"][0])
                _ic = float(_intercept_for_fixed_total(
                    jnp.float32(_sl), norm_freqs_bw_jnp, fixed_total_pump_w
                ))
                log_pow = _make_pump_pow_from_slope(_ic, _sl, norm_freqs_bw)
                pump_mw = np.exp(log_pow[active_bw_mask]) * 1e3
                pump_str = "  ".join(f"{v:6.2f}" for v in pump_mw)
                pump_str += f"  (sl={_sl:.4f} total={np.sum(pump_mw):.1f}mW)"
            elif "log_pump_intercept_bw" in pp:
                _ic = float(pp["log_pump_intercept_bw"][0])
                _sl = float(pp["pump_slope_bw"][0])
                log_pow = _make_pump_pow_from_slope(_ic, _sl, norm_freqs_bw)
                pump_mw = np.exp(log_pow[active_bw_mask]) * 1e3
                pump_str = "  ".join(f"{v:6.2f}" for v in pump_mw)
                pump_str += f"  (ic={_ic:.4f} sl={_sl:.4f})"
            else:
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
        all_best_tps.append(start_best_tp)
        if start_best_tp > best_tp:
            best_tp = start_best_tp
            best_params = start_best_pp
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
        if "pump_slope_bw" in best_params and "log_pump_intercept_bw" not in best_params:
            _sl = best_params["pump_slope_bw"][0]
            _ic = _intercept_for_fixed_total(_sl, norm_freqs_bw_jnp, fixed_total_pump_w)
            _best_pump_bw = jnp.exp(_make_pump_pow_from_slope(_ic, _sl, norm_freqs_bw_jnp))
        elif "log_pump_intercept_bw" in best_params:
            _best_log_pow = _make_pump_pow_from_slope(
                float(best_params["log_pump_intercept_bw"][0]),
                float(best_params["pump_slope_bw"][0]),
                norm_freqs_bw_jnp,
            )
            _best_pump_bw = jnp.exp(_best_log_pow)
        else:
            _best_pump_bw = jnp.exp(best_params["log_pump_pow_bw"])
        raw_tp = float(
            make_throughput_objective(state, params, snr_variance_penalty=0.0)(
                {
                    "pump_pow_bw": _best_pump_bw,
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
    if "pump_slope_bw" in best_params and "log_pump_intercept_bw" not in best_params:
        # fixed_total_slope mode
        opt_slope = float(best_params["pump_slope_bw"][0])
        opt_intercept = float(_intercept_for_fixed_total(
            jnp.float32(opt_slope), norm_freqs_bw_jnp, fixed_total_pump_w
        ))
        opt_log_pow = _make_pump_pow_from_slope(opt_intercept, opt_slope, norm_freqs_bw)
        opt_pump_pow_bw_w = np.exp(opt_log_pow)
        opt_pump_pow_bw_mw = opt_pump_pow_bw_w[active_bw_mask] * 1e3
        print(f"Optimised BW pump powers (mW, fixed_total_slope): {opt_pump_pow_bw_mw}")
        print(f"  slope: {opt_slope:.6f}, intercept (derived): {opt_intercept:.6f}")
        print(f"  Total: {np.sum(opt_pump_pow_bw_mw):.1f} mW (target: {fixed_total_pump_w * 1e3:.1f} mW)")
        pump_csv_str = ",".join(f"{v:.6g}" for v in opt_pump_pow_bw_w)
        print(f"  Copy-paste (--raman_pump_power_bw): {pump_csv_str}")
    elif "log_pump_intercept_bw" in best_params:
        opt_intercept = float(best_params["log_pump_intercept_bw"][0])
        opt_slope = float(best_params["pump_slope_bw"][0])
        opt_log_pow = _make_pump_pow_from_slope(opt_intercept, opt_slope, norm_freqs_bw)
        opt_pump_pow_bw_w = np.exp(opt_log_pow)
        opt_pump_pow_bw_mw = opt_pump_pow_bw_w[active_bw_mask] * 1e3
        print(f"Optimised BW pump powers (mW, slope mode): {opt_pump_pow_bw_mw}")
        print(f"  intercept (log): {opt_intercept:.6f}, slope: {opt_slope:.6f}")
        pump_csv_str = ",".join(f"{v:.6g}" for v in opt_pump_pow_bw_w)
        print(f"  Copy-paste (--raman_pump_power_bw): {pump_csv_str}")
    else:
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


def save_optimisation_history(history, out_dir, tag=None):
    """Persist the (step, throughput) history to JSON so it can be re-plotted
    without re-running the optimisation.
    """
    import json
    os.makedirs(out_dir, exist_ok=True)
    fname = "pump_history.json" if tag is None else f"pump_history_{tag}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(
            {"history": [(int(s), float(t)) for s, t in history]},
            f,
        )
    print(f"  Saved history to {path}")
    return path


def load_optimisation_history(path):
    import json
    with open(path) as f:
        data = json.load(f)
    return [tuple(p) for p in data["history"]]


def plot_optimisation_history(history, out_dir, fname="pump_optimisation.png"):
    try:
        import matplotlib.pyplot as plt

        from experimental.plot_style import ACCENT_COLORS, PRIMARY_COLORS, configure_style

        configure_style(font_size=24, axes_label_size=28, tick_size=22, legend_size=20)

        steps, tps = zip(*history)
        steps = np.array(steps)
        tps = np.array(tps)

        # Truncate to first 30 optimisation steps
        max_steps = 30
        keep = steps <= max_steps
        steps = steps[keep]
        tps = tps[keep]

        # Running best (cumulative max)
        running_best = np.maximum.accumulate(tps)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(steps, tps, color=PRIMARY_COLORS[2], lw=1.2, alpha=0.5, label="Per-step")
        ax.plot(steps, running_best, color=PRIMARY_COLORS[0], lw=2.8, label="Running best")
        ax.axhline(tps[0], color=ACCENT_COLORS[1], ls="--", alpha=0.7, lw=2.5,
                   label=f"Baseline {tps[0]:.2f} Gb/s")
        ax.set_xlim(0, max_steps)
        # Choose a sensible y-axis lower bound: show the baseline with a small
        # margin below it.  Cap at 72,400 to keep zoomed-in views from
        # extending too low when the optimisation is already near-optimal.
        y_min_data = float(min(np.min(tps), np.min(running_best)))
        y_lower = min(y_min_data - 50.0, 72400.0)
        ax.set_ylim(bottom=y_lower)
        ax.set_xlabel("Optimisation step")
        ax.set_ylabel("Throughput (Gbps)")
        ax.legend()
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, fname)
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
        "--pump_power_mode",
        default="independent",
        choices=["independent", "slope", "fixed_total_slope"],
        help="Pump power parameterisation: 'independent' optimises each pump power "
        "separately (default); 'slope' constrains all pump powers to lie on a "
        "straight line in log-space (2 params: intercept + slope vs frequency); "
        "'fixed_total_slope' fixes the total pump power (from initial config or "
        "--max_total_pump_power_mw) and optimises only the slope (1 param)",
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
        "--equal_pump_power",
        action="store_true",
        help="Override preset pump powers with equal power per pump (total preserved)",
    )
    parser.add_argument(
        "--pump_tilt",
        type=float,
        default=None,
        help="Ratio of highest to lowest pump power (e.g. 2.0 means first pump is 2x last). "
        "Total power is preserved. Implies --equal_pump_power base.",
    )
    parser.add_argument(
        "--snr_variance_penalty",
        type=float,
        default=0.0,
        help="Weight λ for SNR flatness penalty: objective = throughput - λ·max(0, mean(SNR)-min(SNR)). "
        "Penalises the worst low-SNR channel relative to the mean. "
        "Higher values enforce a flatter gain spectrum (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--launch_power_csv",
        type=str,
        default=None,
        help="Path to CSV with per-slot launch powers (columns: slot_index, freq_ghz, power_dbm)",
    )
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gerard2025_results")

    from experimental.validation.gerard2025_validation import run_simulation

    sim_overrides = {}
    if args.launch_power_csv:
        sim_overrides["launch_power_csv"] = args.launch_power_csv
    print("Running baseline simulation (Gerard 2025 preset)...")
    state, params, env, config = run_simulation(sim_overrides or None, quiet=False)

    if not params.use_raman_amp:
        print("ERROR: use_raman_amp must be True.")
        sys.exit(1)

    if args.equal_pump_power or args.pump_tilt is not None:
        pump_bw = np.array(params.raman_pump_power_bw.val)  # shape (num_links, num_pumps)
        num_links, num_pumps = pump_bw.shape
        # Work per-link: apply tilt/equal across the num_pumps dimension
        # Use first link as reference (all links have same pump config)
        link0 = pump_bw[0]
        active_mask_1d = link0 > 0
        num_active = int(np.sum(active_mask_1d))
        if num_active > 0:
            total_power_per_link = float(np.sum(link0[active_mask_1d]))
            if args.pump_tilt is not None and num_active > 1:
                weights = np.linspace(args.pump_tilt, 1.0, num_active)
                weights = weights / weights.sum() * num_active
                equal_power = total_power_per_link / num_active
                new_link = np.zeros_like(link0)
                new_link[active_mask_1d] = (equal_power * weights).astype(pump_bw.dtype)
                print(f"\nOverriding pump powers with tilt {args.pump_tilt:.1f}:1 "
                      f"(total {total_power_per_link*1e3:.1f} mW per link preserved)")
                print(f"  Powers per link (mW): {new_link[active_mask_1d]*1e3}")
            else:
                equal_power = total_power_per_link / num_active
                new_link = np.where(active_mask_1d, equal_power, 0.0).astype(pump_bw.dtype)
                print(f"\nOverriding pump powers to equal: {equal_power*1e3:.2f} mW each "
                      f"(total {total_power_per_link*1e3:.1f} mW per link preserved)")
            # Broadcast to all links
            new_powers = np.tile(new_link, (num_links, 1))
            from xlron.environments.dataclasses import HashableArrayWrapper
            params = params.replace(raman_pump_power_bw=HashableArrayWrapper(new_powers))

    best_params, history = optimise_pump_powers(
        state,
        params,
        steps=args.steps,
        lr=args.lr,
        lr_launch_power=args.lr_launch_power,
        optimise_launch_power=args.optimise_launch_power,
        launch_power_granularity=args.launch_power_granularity,
        pump_power_mode=args.pump_power_mode,
        max_total_power_dbm=args.max_total_power_dbm,
        max_total_pump_power_mw=args.max_total_pump_power_mw,
        num_starts=args.num_starts,
        noise_scale=args.noise_scale,
        seed=args.seed,
        snr_variance_penalty=args.snr_variance_penalty,
    )

    print("\nSaving history and generating plots...")
    tag = (
        f"lr{args.lr:g}_steps{args.steps}_starts{args.num_starts}"
        f"_mode-{args.pump_power_mode}_seed{args.seed}"
    )
    save_optimisation_history(history, out_dir, tag=tag)
    plot_optimisation_history(history, out_dir, fname=f"pump_optimisation_{tag}.png")
    plot_optimisation_history(history, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
