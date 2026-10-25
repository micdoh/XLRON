"""Candidate endpoint GN screen with an in-kernel channel insertion.

The two endpoints of the proposed channel have the same centre, bandwidth and
power. Their XPM/SPM and ISRS amplifier gain can be calculated once per changed
link, retaining every interfering slot. Per-slot amplifier noise figures remain
distinct. This is only the conservative screen: original admission checks still
decide acceptance, including every affected existing path and fibre power.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from scipy.constants import c, h, pi

from xlron.environments.gn_model import isrs_gn_model as gn
from xlron.kernels.gn_xpm import _xpm_pairs


def endpoint_snr_pallas(state, params, links, start, end, power, centre, *, interpret=False):
    """Return exact-model optical SNR for both endpoints on each changed link.

    All reductions and arithmetic are float32. The original GN broadcast
    convention and deferred dispersion underflow behavior are preserved.
    No candidate-by-links-by-spectrum channel arrays are materialized.
    """
    if not params.uniform_spans or params.mod_format_correction or params.use_raman_amp:
        raise ValueError("Fused endpoint screening requires uncorrected uniform GN")
    if any(
        x.dtype != jnp.float32
        for x in (
            state.channel_power_array,
            state.channel_centre_freq_array,
            state.channel_centre_bw_array,
        )
    ):
        raise ValueError("Fused endpoint screening requires float32 channel arrays")
    d = jnp.asarray(params.dispersion_coeff, jnp.float32)
    slope = jnp.asarray(params.dispersion_slope, jnp.float32)
    wavelength = params.ref_lambda
    beta2 = -d * wavelength**2 / (2 * pi * c)
    beta3 = wavelength**2 / (2 * pi * c) ** 2 * (wavelength**2 * slope + 2 * wavelength * d)
    beta4 = (
        -(wavelength**4)
        / (2 * pi * c) ** 3
        * (6 * wavelength * slope + 6 * d + slope * wavelength**2)
    )
    physics = jnp.stack(
        [
            jnp.asarray(x, jnp.float32)
            for x in (
                params.attenuation,
                params.attenuation_bar,
                params.nonlinear_coeff,
                params.raman_gain_slope,
                beta2,
                beta3,
                beta4,
            )
        ]
    )
    lengths = jnp.sum(params.link_length_array.val, axis=1)
    spans = jnp.ceil(lengths / params.max_span_length).astype(jnp.int32)
    span_length = lengths / spans
    n = params.link_resources
    columns = pl.next_power_of_2(n)
    insertion = jnp.stack((start, end, power, centre)).astype(jnp.float32)
    if interpret:
        # Native programs load these buffers at runtime. Prevent the CPU
        # interpreter from propagating caller constants through that boundary.
        insertion, physics = jax.lax.optimization_barrier((insertion, physics))

    def kernel(
        p_ref,
        f_ref,
        b_ref,
        links_ref,
        insertion_ref,
        physics_ref,
        spans_ref,
        length_ref,
        nf_ref,
        out_ref,
    ):
        row = pl.program_id(0)
        link = links_ref[row]

        def calculate():
            start, end, new_power, new_centre = (insertion_ref[i] for i in range(4))
            k = jnp.arange(columns)
            inside = (k >= start) & (k <= end)
            p = plt.load(p_ref.at[link, k], mask=k < n, other=0.0)
            f = plt.load(f_ref.at[link, k], mask=k < n, other=0.0)
            b = plt.load(b_ref.at[link, k], mask=k < n, other=0.0)
            p = jnp.where(inside, new_power, p)
            f = jnp.where(inside, new_centre, f) * 1e9
            b = jnp.where(inside, params.slot_size, b) * 1e9
            fi = new_centre * 1e9
            bi = jnp.float32(params.slot_size) * 1e9
            a, ab, gamma, cr, b2, b3, b4 = (physics_ref[i] for i in range(7))
            total = jnp.sum(p)
            xpm = jnp.sum(
                _xpm_pairs(
                    p,
                    fi,
                    b,
                    p,
                    f,
                    b,
                    total,
                    a=a,
                    a_bar=ab,
                    gamma=gamma,
                    cr=cr,
                    beta2=b2,
                    beta3=b3,
                    beta4=b4,
                )
            )
            phi = 3 / 2 * pi**2 * (b2 + pi * b3 * (fi + fi) + 2 * pi**2 * b4 * fi**2)
            tilt = (a + ab - fi * total * cr) ** 2
            effective_bw = bi / params.num_subchannels
            spm = gn._spm(phi, tilt, effective_bw, a, ab, gamma)
            eps = gn._eps(effective_bw, fi, a, length_ref[link], b2, b3) * params.coherent
            count = spans_ref[link]
            spm = spm * count ** (1 + eps) / params.num_subchannels**2
            nli = new_power**3 * (spm + xpm * count)
            leff = (1 - jnp.exp(-a * length_ref[link])) / a
            cr_leff_total = cr * leff * total
            exp_tilt = jnp.exp(-f * cr_leff_total)
            psd_sum = jnp.maximum(jnp.sum(p * exp_tilt), gn.EPS)
            gsrs = total * jnp.exp(-fi * cr_leff_total) / psd_sum
            loss = jnp.exp(length_ref[link] * a)
            gain = jnp.where(new_power > 0, loss / gsrs, loss)
            if params.span_lumped_loss_db is not None:
                gain = gain * 10 ** (params.span_lumped_loss_db / 10)
            gain_m1 = gain - 1
            # Triton cannot stack scalar IR values; build the two-lane vector
            # directly so both endpoint noise figures remain independent.
            endpoints = jnp.where(jnp.arange(2) == 0, start, end).astype(jnp.int32)
            nf = plt.load(nf_ref.at[endpoints])
            nsp = (10 ** (nf / 10) * gain) / (2 * gain_m1)
            ase = count * (2 * nsp * gain_m1 * h * (c / wavelength + fi) * bi)
            snr = jnp.where(new_power > 0, new_power / jnp.maximum(ase + nli, gn.EPS), -1e5)
            return snr

        result = jax.lax.cond(
            link < params.num_links, calculate, lambda: jnp.ones((2,), jnp.float32)
        )
        plt.store(out_ref.at[row, jnp.arange(2)], result)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((links.size, 2), jnp.float32),
        grid=(links.size,),
        compiler_params=plt.CompilerParams(num_warps=4),
        interpret=interpret,
        name="gn_endpoint_snr",
    )(
        state.channel_power_array,
        state.channel_centre_freq_array,
        state.channel_centre_bw_array,
        links,
        insertion,
        physics,
        spans,
        span_length,
        params.amplifier_noise_figure.val,
    )
    return jnp.nan_to_num(result, nan=1e-5)
