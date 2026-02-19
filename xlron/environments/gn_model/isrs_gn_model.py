"""ISRS GN model implementation
Source: https://github.com/dsemrau/ISRSGNmodel/blob/master/Functions/Python/ISRSGNmodel.py

This module implements the function that returns the nonlinear interference
power and coefficient for each WDM channel. This function implements the
ISRS GN model in closed-form published in:

D. Semrau, R. I. Killey, P. Bayvel, "A Closed-Form Approximation of the
Gaussian Noise Model in the Presence of Inter-Channel Stimulated Raman
Scattering, " J. Lighw. Technol., Early Access, Jan. 2019

Author: Daniel Semrau, Eric Sillekens, R. I. Killey, P. Bayvel, Jan 2019.
"""

import chex
import jax
from jax import numpy as jnp
from scipy.constants import c, h, pi

# Small constant to avoid division by zero
EPS = 1e-12


def isrs_gn_model(
    num_channels: int = 420,
    num_spans: int = 1,
    max_spans: int = 10,
    ref_lambda: float = 1577.5e-9,
    length: chex.Array | None = None,
    attenuation_i: chex.Array | None = None,
    attenuation_bar_i: chex.Array | None = None,
    ch_power_W_i: chex.Array | None = None,
    nonlinear_coeff: chex.Array | None = None,
    ch_centre_i: chex.Array | None = None,
    ch_bandwidth_i: chex.Array | None = None,
    raman_gain_slope_i: chex.Array | None = None,
    dispersion_coeff: chex.Array | None = None,
    dispersion_slope: chex.Array | None = None,
    coherent: bool = True,
    excess_kurtosis_i: chex.Array | None = None,
    mod_format_correction: bool = False,
    num_subchannels: int = 1,
):
    """
    This function implements the ISRS GN model in closed-form from [1].
    It's an approximation with rectangular Raman slope and valid for C+L bands

    Mostly copied from:
    https://github.com/dsemrau/ISRSGNmodel/blob/master/Functions/Python/ISRSGNmodel.py

    References
    ----------
        D. Semrau, R. I. Killey, P. Bayvel, "A Closed-Form Approximation of the
        Gaussian Noise Model in the Presence of Inter-Channel Stimulated Raman
        Scattering, " J. Lighw. Technol., Early Access, Jan. 2019

        Format:

    - channel dependent quantities have the format of a N_ch x n matrix,
      where N_ch is the number of channels slots and n is the number of spans.
    - channel independent quantities have the format of a 1 x n matrix
    - channel and span independent quantities are scalars

    INPUTS:
        num_channels: number of channels
        num_spans: number of spans
        length_j: fiber length per span [m]
        attenuation_ij: attenuation coefficient [1/m]
            format: N_ch x n matrix
        attenuation_bar_ij: attenuation coefficient [1/m]
            format: N_ch x n matrix
        ch_power_W_ij: channel power [W]
            format: N_ch x n matrix
        nonlinear_coeff_j: Nonlinear coefficient [1/W^2]
            format: 1 x n matrix
        ch_centre_ij: channel center frequency [Hz]
            format: N_ch x n matrix
        ch_bandwidth_ij: channel bandwidth [Hz]
            format: N_ch x n matrix
        raman_gain_slope_ij: Raman gain slope [1/(W*m)]
            format: N_ch x n matrix
        ref_lambda: reference wavelength [m]
        dispersion_coeff: dispersion coefficient [s/m^2]
            format: 1 x n matrix
        dispersion_slope: dispersion slope [s/m^3]
        coherent: NLI is added coherently across multiple spans

    RETURNS:
        NLI: Nonlinear Interference Power[W],
            format: N_ch x 1 vector
        eta_n: Nonlinear Interference coefficient [1/W^2],
            format: N_ch x 1 matrix
    """
    # set default values
    length = length if length is not None else 100 * 1e3 * jnp.ones(1)
    attenuation_i = (
        attenuation_i if attenuation_i is not None else 0.2 / 4.343 / 1e3 * jnp.ones(num_channels)
    )
    attenuation_bar_i = (
        attenuation_bar_i
        if attenuation_bar_i is not None
        else 0.2 / 4.343 / 1e3 * jnp.ones(num_channels)
    )
    ch_power_W_i = (
        ch_power_W_i
        if ch_power_W_i is not None
        else 10 ** (0 / 10) * 0.001 * jnp.ones(num_channels)
    )
    nonlinear_coeff = nonlinear_coeff if nonlinear_coeff is not None else 1.2 / 1e3 * jnp.ones(1)
    ch_centre_i = ch_centre_i if ch_centre_i is not None else jnp.ones(num_channels)
    ch_bandwidth_i = (
        ch_bandwidth_i if ch_bandwidth_i is not None else 40.004e9 * jnp.ones(num_channels)
    )
    raman_gain_slope_i = (
        raman_gain_slope_i
        if raman_gain_slope_i is not None
        else 0.028 / 1e3 / 1e12 * jnp.ones(num_channels)
    )
    dispersion = (
        dispersion_coeff if dispersion_coeff is not None else 17 * 1e-12 / 1e-9 / 1e3 * jnp.ones(1)
    )
    dispersion_slope = (
        dispersion_slope
        if dispersion_slope is not None
        else 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9 * jnp.ones(1)
    )
    a = attenuation_i if attenuation_i is not None else 0.2 / 4.343 / 1e3 * jnp.ones(1)
    a_bar = attenuation_bar_i if attenuation_bar_i is not None else 0.2 / 4.343 / 1e3 * jnp.ones(1)

    l_mean = jnp.sum(length) / num_spans
    ch_pow = ch_power_W_i
    power = jnp.sum(ch_pow, axis=0)
    gamma = nonlinear_coeff
    f = ch_centre_i
    ch_bw = ch_bandwidth_i
    cr = raman_gain_slope_i

    beta2 = -dispersion * ref_lambda**2 / (2 * pi * c)
    beta3 = ref_lambda**3 / (2 * pi * c) ** 2 * (ref_lambda * dispersion_slope + 2 * dispersion)
    beta4 = (
        -(ref_lambda**4)
        / (2 * pi * c) ** 3
        * (6 * ref_lambda * dispersion_slope + 6 * dispersion + dispersion_slope * ref_lambda**2)
    )

    Phi = excess_kurtosis_i
    tx1_i = jnp.ones((num_channels, 1))
    tx2_i = jnp.ones((num_channels, 1))
    T_0 = (a + a_bar - f * power * cr) ** 2

    f_i = f[:, None]
    f_k = f[None, :]
    f_i_plus_f_k = f_i + f_k

    phi_ik_corr = (
        2 * pi**2 * (f_k - f_i) * (beta2 + pi * beta3 * f_i_plus_f_k + 2 * pi**2 * beta4 * f_i**2)
    )
    eta_xpm_corr = _xpm_corr(
        ch_pow, ch_pow.T, phi_ik_corr, T_0.T, ch_bw, ch_bw.T, a.T, a_bar.T, gamma, Phi.T, tx1_i
    )

    # Precompute all scan-invariant quantities
    ch_bw_k = ch_bw.T
    ch_pow_k = ch_pow.T
    df = jnp.abs(f_k - f_i)
    phi_i = 3 / 2 * pi**2 * (beta2 + pi * beta3 * (f + f) + 2 * pi**2 * beta4 * f**2)
    phi = jnp.abs(4 * pi**2 * (beta2 + pi * beta3 * f_i_plus_f_k))
    phi_ik = (
        2
        * pi**2
        * (f_i - f_k)
        * (
            beta2
            + pi * beta3 * f_i_plus_f_k
            + 2 / 3 * pi**2 * beta4 * (f_i**2 + f_i * f_k + f_k**2)
        )
    )
    T_i = T_0
    T_k = T_i.T

    # Nyquist subchannels: use effective bandwidth B_eff = B / N_sub for SPM
    B_eff = ch_bw / num_subchannels
    spm_single = jnp.squeeze(_spm(phi_i, T_i, B_eff, a, a_bar, gamma)) / (num_subchannels**2)
    xpm_single = _xpm(ch_pow, ch_pow_k, phi_ik, T_k, ch_bw, ch_bw_k, a.T, a_bar.T, gamma)

    def scan_fun(carry, l_span):
        _eta_spm = carry[0] + spm_single
        _eta_xpm = carry[1] + xpm_single
        _eta_xpm_corr_asymp = carry[2] + _xpm_corr_asymp(
            ch_pow,
            ch_pow_k,
            phi_ik,
            phi,
            T_k,
            ch_bw_k,
            a.T,
            a_bar.T,
            gamma,
            df,
            Phi.T,
            tx2_i,
            l_span,
        )
        return (_eta_spm, _eta_xpm, _eta_xpm_corr_asymp), None

    init = (jnp.zeros(num_channels), jnp.zeros(num_channels), jnp.zeros(num_channels))
    final_state, _ = jax.lax.scan(scan_fun, init, length, length=max_spans)

    eps = _eps(B_eff, f, a, l_mean, beta2, beta3)
    eta_spm = final_state[0] * num_spans ** (eps * coherent)
    eta_xpm = final_state[1]
    eta_xpm_corr_asymp = final_state[2]

    if not mod_format_correction:
        eta_xpm_corr = jnp.zeros((num_channels,))
        eta_xpm_corr_asymp = jnp.zeros((num_channels,))

    eta_n = (eta_spm + eta_xpm + eta_xpm_corr_asymp).T + eta_xpm_corr
    NLI = jnp.squeeze(ch_pow) ** 3 * eta_n
    return NLI, eta_n, eta_spm, eta_xpm


def isrs_gn_model_uniform(
    num_channels: int = 420,
    num_spans: int = 1,
    ref_lambda: float = 1577.5e-9,
    length: float = 100e3,
    attenuation_i: chex.Array | None = None,
    attenuation_bar_i: chex.Array | None = None,
    ch_power_W_i: chex.Array | None = None,
    nonlinear_coeff: float = 1.2e-3,
    ch_centre_i: chex.Array | None = None,
    ch_bandwidth_i: chex.Array | None = None,
    raman_gain_slope_i: chex.Array | None = None,
    dispersion_coeff: float = 17e-12 / 1e-9 / 1e3,
    dispersion_slope: float = 0.067e-12 / 1e-9 / 1e3 / 1e-9,
    coherent: bool = True,
    excess_kurtosis_i: chex.Array | None = None,
    mod_format_correction: bool = False,
    max_spans: int | None = None,
    num_subchannels: int = 1,
):
    """
    Simplified ISRS GN model for uniform spans with identical parameters.
    Computes single span and scales appropriately.
    """
    a = attenuation_i if attenuation_i is not None else 0.2 / 4.343 / 1e3 * jnp.ones(num_channels)
    a_bar = (
        attenuation_bar_i
        if attenuation_bar_i is not None
        else 0.2 / 4.343 / 1e3 * jnp.ones(num_channels)
    )
    ch_pow = (
        ch_power_W_i
        if ch_power_W_i is not None
        else 10 ** (0 / 10) * 0.001 * jnp.ones(num_channels)
    )
    gamma = nonlinear_coeff
    f = ch_centre_i if ch_centre_i is not None else jnp.ones(num_channels)
    ch_bw = ch_bandwidth_i if ch_bandwidth_i is not None else 40.004e9 * jnp.ones(num_channels)
    cr = (
        raman_gain_slope_i
        if raman_gain_slope_i is not None
        else 0.028 / 1e3 / 1e12 * jnp.ones(num_channels)
    )

    beta2 = -dispersion_coeff * ref_lambda**2 / (2 * pi * c)
    beta3 = (
        ref_lambda**2
        / (2 * pi * c) ** 2
        * (ref_lambda**2 * dispersion_slope + 2 * ref_lambda * dispersion_coeff)
    )
    beta4 = (
        -(ref_lambda**4)
        / (2 * pi * c) ** 3
        * (
            6 * ref_lambda * dispersion_slope
            + 6 * dispersion_coeff
            + dispersion_slope * ref_lambda**2
        )
    )

    Phi = excess_kurtosis_i if excess_kurtosis_i is not None else jnp.zeros(num_channels)

    total_power = jnp.sum(ch_pow)

    # -- Fused SPM computation --
    # Nyquist subchannels: use effective bandwidth B_eff = B / N_sub for SPM
    B_eff = ch_bw / num_subchannels
    # phi_i for SPM
    phi_i = 3 / 2 * pi**2 * (beta2 + pi * beta3 * (f + f) + 2 * pi**2 * beta4 * f**2)
    # T term
    T = (a + a_bar - f * total_power * cr) ** 2
    # SPM (inlined _spm)
    B2 = jnp.maximum(B_eff**2, EPS)
    a_plus_abar = a + a_bar
    eta_spm = 4 / 9 * gamma**2 / B2 * pi / (phi_i * a_bar * (2 * a + a_bar)) * (
        T - a**2
    ) / a * jnp.arcsinh(phi_i * B2 / a / pi) + (a_plus_abar**2 - T) / a_plus_abar * jnp.arcsinh(
        phi_i * B2 / a_plus_abar / pi
    )
    # Coherence scaling (uses B_eff for subchannel coherence)
    eps = _eps(B_eff, f, a, length, beta2, beta3) * coherent
    eta_spm = eta_spm * num_spans ** (1 + eps)
    # Scale: total NLI = N_sub * (P/N_sub)^3 * eta(B_eff) = P^3 * eta(B_eff) / N_sub^2
    eta_spm = eta_spm / (num_subchannels**2)

    # -- XPM computation --
    f_i = f[:, None]
    f_k = f[None, :]
    f_i_plus_f_k = f_i + f_k
    phi_ik = (
        2
        * pi**2
        * (f_i - f_k)
        * (
            beta2
            + pi * beta3 * f_i_plus_f_k
            + 2 / 3 * pi**2 * beta4 * (f_i**2 + f_i * f_k + f_k**2)
        )
    )
    T_k = T[None, :]
    # .T is a no-op on 1D arrays — matches original _xpm broadcasting conventions
    xpm_single = _xpm(ch_pow, ch_pow.T, phi_ik, T_k, ch_bw, ch_bw.T, a.T, a_bar.T, gamma)
    eta_xpm = xpm_single * num_spans

    # XPM corrections (if enabled)
    if mod_format_correction:
        tx1_i = jnp.ones((num_channels, 1))
        eta_xpm_corr = _xpm_corr(
            ch_pow, ch_pow.T, phi_ik, T_k, ch_bw, ch_bw.T, a.T, a_bar.T, gamma, Phi.T, tx1_i
        )
        df = jnp.abs(f_k - f_i)
        phi = jnp.abs(4 * pi**2 * (beta2 + pi * beta3 * f_i_plus_f_k))
        tx2_i = jnp.ones((num_channels, 1))
        eta_xpm_corr_asymp = (
            _xpm_corr_asymp(
                ch_pow,
                ch_pow.T,
                phi_ik,
                phi,
                T_k,
                ch_bw.T,
                a.T,
                a_bar.T,
                gamma,
                df,
                Phi.T,
                tx2_i,
                length,
            )
            * num_spans
        )
    else:
        eta_xpm_corr = jnp.zeros((num_channels,))
        eta_xpm_corr_asymp = jnp.zeros((num_channels,))

    eta_n = (eta_spm + eta_xpm + eta_xpm_corr_asymp).T + eta_xpm_corr
    NLI = jnp.squeeze(ch_pow) ** 3 * eta_n

    return NLI, eta_n, eta_spm, eta_xpm


def _eps(B_i, f_i, a_i, mean_l, beta2, beta3):
    """Coherence factor, cf. Ref. [2, Eq. (22)]"""
    return (
        3
        / 10
        * jnp.log(
            1
            + (6 / a_i)
            / (
                mean_l
                * jnp.arcsinh(
                    pi**2
                    / 2
                    * jnp.abs(jnp.mean(beta2) + 2 * pi * jnp.mean(beta3) * f_i)
                    / a_i
                    * B_i**2
                )
            )
        )
    )


def _spm(phi_i, t_i, B_i, a_i, a_bar_i, gamma):
    """SPM contribution, see Ref. [1, Eq. (9-10)]"""
    B2 = jnp.maximum(B_i**2, EPS)
    a_plus_abar = a_i + a_bar_i
    return 4 / 9 * gamma**2 / B2 * pi / (phi_i * a_bar_i * (2 * a_i + a_bar_i)) * (
        t_i - a_i**2
    ) / a_i * jnp.arcsinh(phi_i * B2 / a_i / pi) + (
        a_plus_abar**2 - t_i
    ) / a_plus_abar * jnp.arcsinh(phi_i * B2 / a_plus_abar / pi)


def _xpm(p_i, p_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k, gamma):
    """XPM contribution, see Ref. [1, Eq. (11)]"""
    p_i_safe = jnp.where(p_i != 0.0, p_i, 1.0)
    power_ratio_sq = (p_k / p_i_safe) ** 2
    denom = B_k * phi_ik * a_bar_k * (2 * a_k + a_bar_k)
    denom_mask = denom != 0
    denom_safe = jnp.where(denom_mask, denom, 1.0)
    a_k_plus_abar_k = a_k + a_bar_k
    xpm_ik = (
        32
        / 27
        * power_ratio_sq
        * gamma**2
        / denom_safe
        * (
            (T_k - a_k**2) / a_k * jnp.arctan(phi_ik * B_i / a_k)
            + (a_k_plus_abar_k**2 - T_k)
            / a_k_plus_abar_k
            * jnp.arctan(phi_ik * B_i / a_k_plus_abar_k)
        )
    )
    return jnp.sum(xpm_ik * denom_mask, axis=1)


def _xpm_corr(p_i, p_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k, gamma, Phi, TX1):
    """XPM correction, see Ref. 1"""
    p_i = jnp.where(p_i > 0.0, p_i, 1.0)
    B_k = jnp.where(B_k > 0.0, B_k, 1.0)
    a = Phi * TX1.T * jnp.where(p_i > 1.0, p_k / p_i, 0.0) ** 2
    b = gamma ** jnp.where(B_k > 1.0, 2 / B_k, 0.0)
    return (
        5
        / 6
        * 32
        / 27
        * jnp.sum(
            a
            * b
            / (phi_ik * a_bar_k * (2 * a_k + a_bar_k))
            * (T_k - a_k**2)
            / a_k
            * jnp.arctan(phi_ik * B_i / a_k)
            + ((a_k + a_bar_k) ** 2 - T_k)
            / (a_k + a_bar_k)
            * jnp.arctan(phi_ik * B_i / (a_k + a_bar_k)),
            axis=1,
        )
    )


def _xpm_corr_asymp(p_i, p_k, phi_ik, phi, T_k, B_k, a, a_bar, gamma, df, Phi, TX2, L):
    """Asymptotic XPM correction, see Ref. 1"""
    p_i = jnp.where(p_i > 0.0, p_i, 1)
    B_k = jnp.where(B_k > 0.0, B_k, 1)
    a0 = jnp.where(p_i > 1.0, p_k / p_i, 0.0) ** 2
    a1 = jnp.where(B_k > 1, T_k / jnp.where(B_k > 1.0, phi / B_k, 1.0), 0.0) ** 3
    a2 = jnp.where(
        B_k > 1, jnp.log(jnp.clip((2 * df - B_k) / (2 * df + B_k), min=EPS) + 2 * B_k), 0
    )
    return (
        5
        / 3
        * 32
        / 27
        * jnp.sum(
            (phi_ik != 0)
            * a0
            * TX2
            * gamma**2
            / L
            * Phi
            * pi
            * a1
            / a**2
            / (a + a_bar) ** 2
            * (2.0 * df - B_k)
            * a2,
            axis=1,
        )
    )


def calculate_amplifier_gain_isrs(attenuation, length, raman_slope, ch_power, ch_centre_freq):
    """
    Calculate amplifier gain compensating for fiber loss and ISRS.
    All inputs in SI units. ch_power and ch_centre_freq are 1D arrays of shape (N,).
    """
    a = attenuation * 1000
    L = length / 1000
    cr = raman_slope * 1e12
    f_ch = jnp.squeeze(ch_centre_freq).flatten() / 1e12
    P = jnp.squeeze(ch_power).flatten()

    Leff = (1 - jnp.exp(-a * L)) / a
    Ptot = jnp.sum(P)
    cr_Leff_Ptot = cr * Leff * Ptot

    # N x N Raman transfer matrix
    raman_transfer = jnp.exp(-(f_ch[:, None] - f_ch[None, :]) * cr_Leff_Ptot)
    psd_sum = jnp.maximum(jnp.sum(P[None, :] * raman_transfer, axis=1), EPS)

    gsrs_tilt = Ptot * jnp.exp(-f_ch * cr_Leff_Ptot) / psd_sum
    total_loss_compensation = jnp.exp(L * a)
    gain = jnp.where(P > 0, total_loss_compensation / gsrs_tilt, total_loss_compensation)

    return jnp.squeeze(gain)


def _calculate_ase_with_isrs_gain(
    noise_figure,
    attenuation_i,
    length,
    ref_lambda,
    ch_centre_i,
    ch_bandwidth,
    ch_power_i,
    raman_gain_slope_i,
    span_lumped_loss_db: float | None = None,
):
    """Compute ASE power using ISRS-aware gain. Fused for fewer XLA ops."""
    a_mean = jnp.mean(attenuation_i)
    cr_mean = jnp.mean(raman_gain_slope_i)
    gain = calculate_amplifier_gain_isrs(a_mean, length, cr_mean, ch_power_i, ch_centre_i)
    if span_lumped_loss_db is not None:
        gain = gain * (10 ** (span_lumped_loss_db / 10))
    gain = jnp.squeeze(gain)
    gain_minus_1 = gain - 1
    N_sp = (10 ** (noise_figure / 10) * gain) / (2.0 * gain_minus_1)
    return jnp.squeeze(2 * N_sp * gain_minus_1 * h * (c / ref_lambda + ch_centre_i) * ch_bandwidth)


def _calculate_ase_with_fixed_gain(
    noise_figure,
    ref_lambda,
    ch_centre_i,
    ch_bandwidth,
    gain,
):
    """Compute ASE power with a pre-computed scalar gain. Fused for fewer XLA ops."""
    gain_minus_1 = gain - 1
    N_sp = (10 ** (noise_figure / 10) * gain) / (2.0 * gain_minus_1)
    return jnp.squeeze(2 * N_sp * gain_minus_1 * h * (c / ref_lambda + ch_centre_i) * ch_bandwidth)


def get_ase_power(
    noise_figure,
    attenuation_i,
    length,
    ref_lambda,
    ch_centre_i,
    ch_bandwidth,
    gain=None,
    ch_power_i=None,
    raman_gain_slope_i=None,
    roadm_loss_db=0,
):
    """
    Modified to use ISRS-aware gain calculation with ROADM loss compensation.

    Additional parameters:
        ch_power_i: channel powers [Nch] in W (needed for ISRS calculation)
        raman_gain_slope_i: Raman gain slope in 1/(W*m) (needed for ISRS calculation)
        roadm_loss_db: ROADM loss in dB (default=0)
    """
    if gain is None:
        if ch_power_i is not None and raman_gain_slope_i is not None:
            a = jnp.mean(attenuation_i)
            cr = jnp.mean(raman_gain_slope_i)
            gain = calculate_amplifier_gain_isrs(a, length, cr, ch_power_i, ch_centre_i)
        else:
            a = jnp.mean(attenuation_i)
            roadm_loss_linear = 10 ** (roadm_loss_db / 10)
            gain = jnp.exp(a * length) * roadm_loss_linear

    gain = jnp.squeeze(gain)
    gain_minus_1 = gain - 1
    N_sp = (10 ** (noise_figure / 10) * gain) / (2.0 * gain_minus_1)
    p_ASE = 2 * N_sp * gain_minus_1 * h * (c / ref_lambda + ch_centre_i) * ch_bandwidth
    return jnp.squeeze(p_ASE)


def calculate_roadm_ase(
    roadm_express_loss,
    roadm_add_drop_loss,
    roadm_noise_figure,
    num_roadm_express,
    ref_lambda,
    ch_centre_i,
    ch_bandwidth_i,
):
    """Compute total ROADM ASE noise power for a path.

    Models ROADMs with separate express (pass-through) and add/drop losses.
    Each ROADM has a booster amplifier that compensates for the ROADM loss
    and adds ASE noise.

    A path traverses:
      - 1 add ROADM at the source (loss = roadm_add_drop_loss)
      - num_roadm_express express ROADMs at intermediate nodes (loss = roadm_express_loss each)
      - 1 drop ROADM at the destination (loss = roadm_add_drop_loss)

    Args:
        roadm_express_loss: Express ROADM loss in dB
        roadm_add_drop_loss: Add/drop ROADM loss in dB
        roadm_noise_figure: ROADM booster amplifier noise figure in dB
        num_roadm_express: Number of express (intermediate) ROADMs on the path
        ref_lambda: Reference wavelength in metres
        ch_centre_i: Channel centre frequencies in Hz, shape (N,)
        ch_bandwidth_i: Channel bandwidths in Hz, shape (N,)

    Returns:
        Total ROADM ASE noise power per channel, shape (N,)
    """
    freq_abs = c / ref_lambda + ch_centre_i
    nf_lin = 10 ** (roadm_noise_figure / 10)

    # Express ROADM booster amplifiers
    express_gain = 10 ** (roadm_express_loss / 10)
    express_gain_m1 = express_gain - 1
    nsp_express = (nf_lin * express_gain) / (2.0 * express_gain_m1)
    p_ase_express = (
        num_roadm_express * 2 * nsp_express * express_gain_m1 * h * freq_abs * ch_bandwidth_i
    )

    # Add/drop ROADM booster amplifiers (2: one at source, one at destination)
    ad_gain = 10 ** (roadm_add_drop_loss / 10)
    ad_gain_m1 = ad_gain - 1
    nsp_ad = (nf_lin * ad_gain) / (2.0 * ad_gain_m1)
    p_ase_ad = 2 * 2 * nsp_ad * ad_gain_m1 * h * freq_abs * ch_bandwidth_i

    return jnp.squeeze(p_ase_express + p_ase_ad)


def get_snr(
    num_channels: int = 420,
    max_spans: int = 20,
    ref_lambda: float = 1577.5e-9,
    attenuation_i: chex.Array = 0.2 / 4.343 / 1e3 * jnp.ones(420),
    attenuation_bar_i: chex.Array = 0.2 / 4.343 / 1e3 * jnp.ones(420),
    nonlinear_coeff: chex.Array = 1.2 / 1e3 * jnp.ones(1),
    coherent: bool = True,
    mod_format_correction: bool = False,
    raman_gain_slope_i: chex.Array = 0.028 / 1e3 / 1e12 * jnp.ones(1),
    dispersion_coeff: chex.Array = 17 * 1e-12 / 1e-9 / 1e3 * jnp.ones(1),
    dispersion_slope: chex.Array = 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9 * jnp.ones(1),
    roadm_loss: float = 6,
    num_roadms: int = 1,
    length: chex.Array = 100 * 1e3 * jnp.ones(20),
    num_spans: int = 20,
    ch_power_w_i: chex.Array = 10 ** (0 / 10) * 0.001 * jnp.ones(420),
    ch_centre_i: chex.Array = ((jnp.arange(420) - (420 - 1) / 2) * 25e-9),
    ch_bandwidth_i: chex.Array = 25e9 * jnp.ones((420, 1)),
    excess_kurtosis_i: chex.Array = jnp.zeros((420, 1)),
    amplifier_noise_figure: chex.Array = jnp.zeros((420, 1)),
    transceiver_snr: chex.Array = jnp.zeros((420, 1)),
    uniform_spans: bool = True,
    num_subchannels: int = 1,
    span_lumped_loss_db: float | None = None,
):
    """
    Compute the signal-to-noise ratio (SNR) of a WDM system.

    Returns:
        snr: signal-to-noise ratio (linear units)
    """
    gn_model = isrs_gn_model_uniform if uniform_spans else isrs_gn_model
    span_length = jnp.sum(length) / num_spans

    # ASE noise - inline amplifiers (ISRS-aware gain)
    p_ase_inline = num_spans * _calculate_ase_with_isrs_gain(
        amplifier_noise_figure,
        attenuation_i,
        span_length,
        ref_lambda,
        ch_centre_i,
        ch_bandwidth_i,
        ch_power_w_i,
        raman_gain_slope_i,
        span_lumped_loss_db,
    )

    # ROADM ASE is now computed at path level (see calculate_roadm_ase)
    p_ase_roadm = jnp.zeros_like(p_ase_inline)

    # NLI
    p_nli, eta_nli, eta_spm, eta_xpm = gn_model(
        num_channels=num_channels,
        num_spans=num_spans,
        max_spans=max_spans,
        ref_lambda=ref_lambda,
        length=span_length if uniform_spans else length,
        attenuation_i=attenuation_i,
        attenuation_bar_i=attenuation_bar_i,
        ch_power_W_i=ch_power_w_i,
        nonlinear_coeff=nonlinear_coeff,
        ch_centre_i=ch_centre_i,
        ch_bandwidth_i=ch_bandwidth_i,
        raman_gain_slope_i=raman_gain_slope_i,
        dispersion_coeff=dispersion_coeff,
        dispersion_slope=dispersion_slope,
        coherent=coherent,
        mod_format_correction=mod_format_correction,
        excess_kurtosis_i=excess_kurtosis_i,
        num_subchannels=num_subchannels,
    )

    # SNR (optical only — transceiver noise is added once at path level in get_snr_for_path)
    transceiver_noise = ch_power_w_i / from_db(transceiver_snr)
    noise_power = p_ase_inline + p_ase_roadm + p_nli
    noise_power = jnp.where(noise_power > 0, noise_power, EPS)
    ch_power_squeezed = jnp.squeeze(ch_power_w_i)
    snr = jnp.where(ch_power_squeezed > 0, ch_power_squeezed / noise_power, -1e5)
    return snr, (eta_nli, eta_spm, eta_xpm, p_ase_inline, p_ase_roadm, p_nli, transceiver_noise)


def get_snr_fused(
    ch_power_w_i: chex.Array,
    ch_centre_i: chex.Array,
    ch_bandwidth_i: chex.Array,
    num_spans: int,
    span_length: float,
    num_channels: int,
    ref_lambda: float,
    attenuation: float,
    attenuation_bar: float,
    nonlinear_coeff: float,
    raman_gain_slope: float,
    dispersion_coeff: float,
    dispersion_slope: float,
    amplifier_noise_figure: chex.Array,
    transceiver_snr: chex.Array,
    roadm_loss: float = 6.0,
    num_roadms: int = 1,
    coherent: bool = True,
    num_subchannels: int = 1,
    span_lumped_loss_db: float | None = None,
):
    """Fully fused SNR computation for a single link (uniform spans, no mod_format_correction).

    This function computes the full SNR in a single pass with minimal intermediate
    allocations, optimized for GPU execution where kernel launch overhead dominates.

    Dynamic inputs (change per call / per link):
        ch_power_w_i: channel powers [W], shape (N,)
        ch_centre_i: channel centre frequencies [Hz], shape (N,)
        ch_bandwidth_i: channel bandwidths [Hz], shape (N,)
        num_spans: number of spans (can be traced)
        span_length: single span length [m] (can be traced)

    Static inputs (same for all links, from params):
        Everything else.
        num_subchannels: number of Nyquist subchannels per slot for SPM calculation.
            Divides slot bandwidth into N subchannels with B_eff = B / N, reducing SPM.

    Returns:
        snr: shape (N,) — linear SNR per channel
    """
    a = attenuation
    a_bar = attenuation_bar
    gamma = nonlinear_coeff
    cr = raman_gain_slope

    # Dispersion coefficients (scalars)
    beta2 = -dispersion_coeff * ref_lambda**2 / (2 * pi * c)
    beta3 = (
        ref_lambda**2
        / (2 * pi * c) ** 2
        * (ref_lambda**2 * dispersion_slope + 2 * ref_lambda * dispersion_coeff)
    )
    beta4 = (
        -(ref_lambda**4)
        / (2 * pi * c) ** 3
        * (
            6 * ref_lambda * dispersion_slope
            + 6 * dispersion_coeff
            + dispersion_slope * ref_lambda**2
        )
    )

    f = ch_centre_i  # (N,)
    ch_bw = ch_bandwidth_i  # (N,)
    ch_pow = ch_power_w_i  # (N,)
    total_power = jnp.sum(ch_pow)

    # === SPM (inlined, no function call overhead) ===
    # Nyquist subchannels: use effective bandwidth B_eff = B / N_sub for SPM
    B_eff = ch_bw / num_subchannels
    phi_i = 3 / 2 * pi**2 * (beta2 + pi * beta3 * (f + f) + 2 * pi**2 * beta4 * f**2)
    T = (a + a_bar - f * total_power * cr) ** 2
    B2 = jnp.maximum(B_eff**2, EPS)
    a_plus_abar = a + a_bar
    eta_spm = 4 / 9 * gamma**2 / B2 * pi / (phi_i * a_bar * (2 * a + a_bar)) * (
        T - a**2
    ) / a * jnp.arcsinh(phi_i * B2 / a / pi) + (a_plus_abar**2 - T) / a_plus_abar * jnp.arcsinh(
        phi_i * B2 / a_plus_abar / pi
    )
    # Coherence factor (uses B_eff for subchannel coherence)
    eps = _eps(B_eff, f, a, span_length, beta2, beta3) * coherent
    eta_spm = eta_spm * num_spans ** (1 + eps)
    # Scale: total NLI = N_sub * (P/N_sub)^3 * eta(B_eff) = P^3 * eta(B_eff) / N_sub^2
    eta_spm = eta_spm / (num_subchannels**2)

    # === XPM (inlined, fused N x N computation) ===
    f_i = f[:, None]  # (N, 1)
    f_k = f[None, :]  # (1, N)
    f_sum = f_i + f_k
    phi_ik = (
        2
        * pi**2
        * (f_i - f_k)
        * (beta2 + pi * beta3 * f_sum + 2 / 3 * pi**2 * beta4 * (f_i**2 + f_i * f_k + f_k**2))
    )
    T_k = T[None, :]  # (1, N)
    a_k = a  # scalar (uniform fiber)
    a_bar_k = a_bar  # scalar
    a_k_plus_abar_k = a_k + a_bar_k

    p_i_safe = jnp.where(ch_pow[:, None] != 0.0, ch_pow[:, None], 1.0)
    power_ratio_sq = (ch_pow[None, :] / p_i_safe) ** 2
    denom = ch_bw[None, :] * phi_ik * a_bar_k * (2 * a_k + a_bar_k)
    denom_mask = denom != 0
    denom_safe = jnp.where(denom_mask, denom, 1.0)

    xpm_ik = (
        32
        / 27
        * power_ratio_sq
        * gamma**2
        / denom_safe
        * (
            (T_k - a_k**2) / a_k * jnp.arctan(phi_ik * ch_bw[:, None] / a_k)
            + (a_k_plus_abar_k**2 - T_k)
            / a_k_plus_abar_k
            * jnp.arctan(phi_ik * ch_bw[:, None] / a_k_plus_abar_k)
        )
    )
    eta_xpm = jnp.sum(xpm_ik * denom_mask, axis=1) * num_spans

    # === NLI power ===
    eta_n = jnp.squeeze(eta_spm + eta_xpm)
    p_nli = jnp.squeeze(ch_pow) ** 3 * eta_n

    # === ASE inline (ISRS-aware gain) ===
    a_si = a * 1000  # convert to 1/km
    L_km = span_length / 1000
    cr_scaled = cr * 1e12
    f_THz = jnp.squeeze(ch_centre_i).flatten() / 1e12
    P_flat = jnp.squeeze(ch_pow).flatten()

    Leff = (1 - jnp.exp(-a_si * L_km)) / a_si
    Ptot = jnp.sum(P_flat)
    cr_Leff_Ptot = cr_scaled * Leff * Ptot

    raman_transfer = jnp.exp(-(f_THz[:, None] - f_THz[None, :]) * cr_Leff_Ptot)
    psd_sum = jnp.maximum(jnp.sum(P_flat[None, :] * raman_transfer, axis=1), EPS)
    gsrs_tilt = Ptot * jnp.exp(-f_THz * cr_Leff_Ptot) / psd_sum
    total_loss = jnp.exp(L_km * a_si)
    if span_lumped_loss_db is not None:
        total_loss = total_loss * (10 ** (span_lumped_loss_db / 10))
    gain_inline = jnp.where(P_flat > 0, total_loss / gsrs_tilt, total_loss)
    gain_inline = jnp.squeeze(gain_inline)

    gain_m1 = gain_inline - 1
    N_sp_inline = (10 ** (amplifier_noise_figure / 10) * gain_inline) / (2.0 * gain_m1)
    freq_abs = c / ref_lambda + ch_centre_i
    p_ase_inline = num_spans * jnp.squeeze(
        2 * N_sp_inline * gain_m1 * h * freq_abs * ch_bandwidth_i
    )

    # ROADM ASE is now computed at path level (see calculate_roadm_ase)
    p_ase_roadm = jnp.zeros_like(p_ase_inline)

    # === Final SNR (optical only — TRX noise added at path level) ===
    noise_power = p_ase_inline + p_ase_roadm + p_nli
    noise_power = jnp.where(noise_power > 0, noise_power, EPS)
    ch_pow_sq = jnp.squeeze(ch_pow)
    snr = jnp.where(ch_pow_sq > 0, ch_pow_sq / noise_power, -1e5)

    return snr


def to_db(x):
    return 10 * jnp.log10(x)


def to_dbm(x):
    return 10 * jnp.log10(x / 0.001)


def from_dbm(x):
    return 10 ** (x / 10) * 0.001


def from_db(x):
    return 10 ** (x / 10)
