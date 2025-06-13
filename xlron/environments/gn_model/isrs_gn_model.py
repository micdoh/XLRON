
""" ISRS GN model implementation
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
import numpy as np
from functools import partial
from scipy.constants import pi, h, c
from jax import numpy as jnp, jit
import jax

# Small constant to avoid division by zero
EPS = 1e-12

def isrs_gn_model(
        num_channels: int = 420,
        num_spans: int = 1,
        max_spans: int = 10,
        ref_lambda: float = 1577.5e-9,
        length: chex.Array = None,
        attenuation_i: chex.Array = None,
        attenuation_bar_i: chex.Array = None,
        ch_power_W_i: chex.Array = None,
        nonlinear_coeff: chex.Array = None,
        ch_centre_i: chex.Array = None,
        ch_bandwidth_i: chex.Array = None,
        raman_gain_slope_i: chex.Array = None,
        dispersion_coeff: chex.Array = None,
        dispersion_slope: chex.Array = None,
        coherent: bool = 1,
        excess_kurtosis_i: chex.Array = None,
        mod_format_correction: bool = 0,

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
    attenuation_i = attenuation_i if attenuation_i is not None else 0.2 / 4.343 / 1e3 * jnp.ones(num_channels)
    attenuation_bar_i = attenuation_bar_i if attenuation_bar_i is not None else 0.2 / 4.343 / 1e3 * jnp.ones(
        num_channels)
    ch_power_W_i = ch_power_W_i if ch_power_W_i is not None else 10 ** (0 / 10) * 0.001 * jnp.ones(num_channels)
    nonlinear_coeff = nonlinear_coeff if nonlinear_coeff is not None else 1.2 / 1e3 * jnp.ones(1)
    ch_centre_i = ch_centre_i if ch_centre_i is not None else jnp.ones(num_channels)
    ch_bandwidth_i = ch_bandwidth_i if ch_bandwidth_i is not None else 40.004e9 * jnp.ones(num_channels)
    raman_gain_slope_i = raman_gain_slope_i if raman_gain_slope_i is not None else 0.028 / 1e3 / 1e12 * jnp.ones(
        num_channels)
    dispersion = dispersion_coeff if dispersion_coeff is not None else 17 * 1e-12 / 1e-9 / 1e3 * jnp.ones(1)
    dispersion_slope = dispersion_slope if dispersion_slope is not None else 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9 * jnp.ones(1)
    a = attenuation_i if attenuation_i is not None else 0.2 / 4.343 / 1e3 * jnp.ones(1)
    a_bar = attenuation_bar_i if attenuation_bar_i is not None else 0.2 / 4.343 / 1e3 * jnp.ones(1)

    l_mean = jnp.sum(length) / num_spans
    ch_pow = ch_power_W_i
    power = jnp.sum(ch_pow, axis=0)
    gamma = nonlinear_coeff
    f = ch_centre_i
    ch_bw = ch_bandwidth_i
    cr = raman_gain_slope_i

    beta2 = -dispersion * ref_lambda ** 2 / (2 * pi * c)
    beta3 = ref_lambda ** 3 / (2 * pi * c) ** 2 * (ref_lambda * dispersion_slope + 2 * dispersion)
    beta4 = -ref_lambda ** 4 / (2 * pi * c) ** 3 * (6 * ref_lambda * dispersion_slope + 6 * dispersion + dispersion_slope * ref_lambda ** 2)

    Phi = excess_kurtosis_i
    # indicates this channels first transmission span is span j=1 of channel i
    tx1_i = jnp.ones((num_channels, 1))
    # indicates this channels first two transmission spans are span j=1 and j=2 of channel i
    tx2_i = jnp.ones((num_channels, 1))
    T_0 = (a + a_bar - f * power * cr) ** 2

    f_i = f[:, None]  # Shape: (N, 1) - column vector
    f_k = f[None, :]  # Shape: (1, N) - row vector
    phi_ik = 2 * pi ** 2 * (f_k - f_i) * (beta2 + pi * beta3 * (f_i + f_k) + 2 * pi ** 2 * beta4 * f_i ** 2)
    eta_xpm_corr = _xpm_corr(ch_pow, ch_pow.T, phi_ik, T_0.T, ch_bw, ch_bw.T, a.T, a_bar.T, gamma, Phi.T, tx1_i)

    def scan_fun(carry, l_span):
        """ Compute the NLI of each COI """
        # TODO - replace zeros with i if using different heterogeneous spans
        a_i = a_k = a
        a_bar_i = a_bar_k = a_bar
        cr_k = cr_i = cr
        f_i = f[:, None]  # Shape: (N, 1) - column vector
        f_k = f[None, :]  # Shape: (1, N) - row vector
        ch_bw_i = ch_bw # B_i of COI in fiber span j
        ch_bw_k = ch_bw_i.T
        ch_pow_i = ch_pow # P_i of COI in fiber span j
        ch_pow_k = ch_pow_i.T
        Phi_k = Phi  # excess kurtosis of mod. format of INT in fiber span j
        df = jnp.abs(f_k - f_i)

        # Phase term for SPM
        phi_i = 3 / 2 * pi ** 2 * (beta2 + pi * beta3 * (f + f) + 2 * pi ** 2 * beta4 * f ** 2)

        # Frequency differences and phase terms for XPM
        phi = jnp.abs(4 * pi ** 2 * (beta2 + pi * beta3 * (f_i + f_k)))
        phi_ik = 2 * pi ** 2 * (f_i - f_k) * (beta2 + pi * beta3 * (f_i + f_k) + 2 / 3 * pi ** 2 * beta4 * (f_i ** 2 + f_i * f_k + f_k ** 2))

        T_i = (a + a_bar - f * power * cr) ** 2  # T_i of COI in fiber span j
        T_k = T_i.T  # T_k of INT in fiber span j

        # computation of SPM contribution in fiber span j
        spm = _spm(phi_i, T_i, ch_bw_i, a_i, a_bar_i, gamma)
        _eta_spm = carry[0] + jnp.squeeze(spm)

        # computation of XPM contribution in fiber span j
        _eta_xpm = carry[1] + _xpm(ch_pow_i, ch_pow_k, phi_ik, T_k, ch_bw_i, ch_bw_k, a_k, a_bar_k, gamma)

        # Asymptotic correction for non-Gaussian modulation format
        _eta_xpm_corr_asymp = carry[2] + _xpm_corr_asymp(ch_pow_i, ch_pow_k, phi_ik, phi, T_k, ch_bw_k, a_k, a_bar_k, gamma, df, Phi_k, tx2_i, l_span)

        return (_eta_spm, _eta_xpm, _eta_xpm_corr_asymp), (_eta_spm, _eta_xpm, _eta_xpm_corr_asymp)

    final_state, traj = jax.lax.scan(
        scan_fun,
        (jnp.zeros((num_channels,)), jnp.zeros((num_channels,)), jnp.zeros((num_channels,))),
        length,
        length=max_spans
    )
    # Apply coherence factor to SPM
    eps = _eps(ch_bw, f, a, l_mean, beta2, beta3)
    eta_spm = final_state[0] * num_spans ** (eps * coherent)
    eta_xpm = final_state[1]
    eta_xpm_corr_asymp = final_state[2]

    if not mod_format_correction:
        eta_xpm_corr = jnp.zeros((num_channels,))
        eta_xpm_corr_asymp = jnp.zeros((num_channels,))

    # computation of NLI - see Ref. [1, Eq. (5)]
    eta_n = (eta_spm + eta_xpm + eta_xpm_corr_asymp).T + eta_xpm_corr
    NLI = jnp.squeeze(ch_pow) ** 3 * eta_n  # Ref. [1, Eq. (1)]
    return NLI, eta_n, eta_spm, eta_xpm


def isrs_gn_model_uniform(
        num_channels: int = 420,
        num_spans: int = 1,
        ref_lambda: float = 1577.5e-9,
        length: float = 100e3,  # Single span length
        attenuation_i: chex.Array = None,
        attenuation_bar_i: chex.Array = None,
        ch_power_W_i: chex.Array = None,
        nonlinear_coeff: float = 1.2e-3,
        ch_centre_i: chex.Array = None,
        ch_bandwidth_i: chex.Array = None,
        raman_gain_slope_i: chex.Array = None,
        dispersion_coeff: float = 17e-12 / 1e-9 / 1e3,
        dispersion_slope: float = 0.067e-12 / 1e-9 / 1e3 / 1e-9,
        coherent: bool = True,
        excess_kurtosis_i: chex.Array = None,
        mod_format_correction: bool = False,
        max_spans: int = None,  # Keep to match signature of isrs_gn_model
):
    """
    Simplified ISRS GN model for uniform spans with identical parameters.
    Computes single span and scales appropriately.
    Assumptions:
    All spans have the same length
    All spans have the same fiber parameters (loss, dispersion, etc.)
    Power is restored to the same level after each span
    """
    # Set defaults (keeping your original defaults)
    a = attenuation_i if attenuation_i is not None else 0.2 / 4.343 / 1e3 * jnp.ones(num_channels)
    a_bar = attenuation_bar_i if attenuation_bar_i is not None else 0.2 / 4.343 / 1e3 * jnp.ones(num_channels)
    ch_pow = ch_power_W_i if ch_power_W_i is not None else 10 ** (0 / 10) * 0.001 * jnp.ones(num_channels)
    gamma = nonlinear_coeff
    f = ch_centre_i if ch_centre_i is not None else jnp.ones(num_channels)
    ch_bw = ch_bandwidth_i if ch_bandwidth_i is not None else 40.004e9 * jnp.ones(num_channels)
    cr = raman_gain_slope_i if raman_gain_slope_i is not None else 0.028 / 1e3 / 1e12 * jnp.ones(num_channels)

    beta2 = -dispersion_coeff * ref_lambda ** 2 / (2 * pi * c)
    beta3 = ref_lambda ** 2 / (2 * pi * c) ** 2 * (
                ref_lambda ** 2 * dispersion_slope + 2 * ref_lambda * dispersion_coeff)
    beta4 = -ref_lambda ** 4 / (2 * pi * c) ** 3 * (6 * ref_lambda * dispersion_slope + 6 * dispersion_coeff + dispersion_slope * ref_lambda ** 2)

    Phi = excess_kurtosis_i if excess_kurtosis_i is not None else jnp.zeros(num_channels)

    # Total power
    total_power = jnp.sum(ch_pow)

    # Phase term for SPM
    phi_i = 3 / 2 * pi ** 2 * (beta2 + pi * beta3 * (f + f) + 2 * pi ** 2 * beta4 * f ** 2)

    # Frequency difference and phase terms for XPM
    f_i = f[:, None]  # Shape: (N, 1) - column vector
    f_k = f[None, :]  # Shape: (1, N) - row vector
    phi_ik = 2 * pi ** 2 * (f_i - f_k) * (
                beta2 + pi * beta3 * (f_i + f_k) + 2 / 3 * pi ** 2 * beta4 * (f_i ** 2 + f_i * f_k + f_k ** 2))


    # T terms
    T = (a + a_bar - f * total_power * cr) ** 2
    T_i = T[:, None]
    T_k = T[None, :]

    # Single span SPM
    spm_single = _spm(phi_i, T, ch_bw, a, a_bar, gamma)

    # Apply coherence factor for total SPM across all spans
    eps = _eps(ch_bw, f, a, length, beta2, beta3) * coherent
    eta_spm = spm_single * num_spans ** (1 + eps)

    # Single span XPM (linear accumulation)
    ch_pow_k = ch_pow.T
    ch_bw_k = ch_bw.T
    a_k = a.T
    a_bar_k = a_bar.T
    xpm_single = _xpm(ch_pow, ch_pow_k, phi_ik, T_k, ch_bw, ch_bw_k, a_k, a_bar_k, gamma)
    eta_xpm = xpm_single * num_spans

    # XPM corrections (if enabled)
    if mod_format_correction:
        # First-order correction (single span)
        tx1_i = jnp.ones((num_channels, 1))
        eta_xpm_corr = _xpm_corr(ch_pow, ch_pow_k, phi_ik, T_k, ch_bw, ch_bw_k, a_k, a_bar_k, gamma, Phi.T, tx1_i)

        # Asymptotic correction - this one is more complex
        # For uniform spans, we can still simplify
        df = jnp.abs(f_k - f_i)
        phi = jnp.abs(4 * pi ** 2 * (beta2 + pi * beta3 * (f_i + f_k)))
        Phi_k = Phi.T
        tx2_i = jnp.ones((num_channels, 1))

        eta_xpm_corr_asymp_single = _xpm_corr_asymp(
            ch_pow, ch_pow_k, phi_ik, phi, T_k, ch_bw_k,
            a_k, a_bar_k, gamma, df, Phi_k, tx2_i, length
        )
        eta_xpm_corr_asymp = eta_xpm_corr_asymp_single * num_spans
    else:
        eta_xpm_corr = jnp.zeros((num_channels,))
        eta_xpm_corr_asymp = jnp.zeros((num_channels,))

    # Total NLI
    eta_n = (eta_spm + eta_xpm + eta_xpm_corr_asymp).T + eta_xpm_corr
    NLI = jnp.squeeze(ch_pow) ** 3 * eta_n

    return NLI, eta_n, eta_spm, eta_xpm


def _eps(B_i, f_i, a_i, mean_l, beta2, beta3):
    """
    Closed-for formula for average coherence factor extended by dispersion slope, cf. Ref. [2, Eq. (22)]
    """
    # closed-form formula for average coherence factor extended by dispersion slope, cf. Ref. [2, Eq. (22)]
    eps = (3 / 10 * jnp.log(
        1 + (6 / a_i) / (
                mean_l * jnp.arcsinh(pi ** 2 / 2 * jnp.abs(jnp.mean(beta2) + 2 * pi * jnp.mean(beta3) * f_i) / a_i * B_i ** 2)
        )))
    return eps


def _spm(phi_i, t_i, B_i, a_i, a_bar_i, gamma):
    """
    Closed-form formula for SPM contribution, see Ref. [1, Eq. (9-10)]
    """
    a = gamma ** 2 / jnp.where(B_i != 0, B_i ** 2, 0.)
    b = jnp.arcsinh(phi_i * B_i ** 2 / a_i / pi)
    c = jnp.arcsinh(phi_i * B_i ** 2 / (a_i + a_bar_i) / pi)
    return (4 / 9 * a * pi / (phi_i * a_bar_i * (2 * a_i + a_bar_i)) * (t_i - a_i ** 2) / a_i * b +
        ((a_i + a_bar_i) ** 2 - t_i) / (a_i + a_bar_i) * c)


def _xpm(p_i, p_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k, gamma):
    """
    Closed-form formula for XPM contribution, see Ref. [1, Eq. (11)]
    """
    p_i = jnp.where(p_i != 0., p_i, 1.)
    denom = B_k * phi_ik * a_bar_k * (2 * a_k + a_bar_k)
    xpm = jnp.sum(jnp.where(
        denom != 0,
        (32 / 27 * (p_k / p_i) ** 2 * gamma ** 2 / denom
         * ((T_k - a_k ** 2) / a_k * jnp.arctan(phi_ik * B_i / a_k)
            + ((a_k + a_bar_k) ** 2 - T_k) / (a_k + a_bar_k) * jnp.arctan(phi_ik * B_i / (a_k + a_bar_k)))),
        0.
    ), axis=1)
    return xpm


def _xpm_corr(p_i, p_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k, gamma, Phi, TX1):
    """
    Closed-form formula for XPM correction, see Ref. 1
    """
    p_i = jnp.where(p_i > 0., p_i, 1.)
    B_k = jnp.where(B_k > 0., B_k, 1.)
    a = Phi * TX1.T * jnp.where(p_i > 1., p_k / p_i, 0.) ** 2
    b = gamma ** jnp.where(B_k > 1., 2 / B_k, 0.)
    return 5 / 6 * 32 / 27 * jnp.sum(
        a * b / (phi_ik * a_bar_k * (2 * a_k + a_bar_k)) * (T_k - a_k ** 2) / a_k * jnp.arctan(phi_ik * B_i / a_k) +
        ((a_k + a_bar_k) ** 2 - T_k) / (a_k + a_bar_k) * jnp.arctan(phi_ik * B_i / (a_k + a_bar_k)),
        axis=1)


def _xpm_corr_asymp(p_i, p_k, phi_ik, phi, T_k, B_k, a, a_bar, gamma, df, Phi, TX2, L):
    """
    Closed-form formula for asymptotic XPM correction, see Ref. 1
    """
    p_i = jnp.where(p_i > 0., p_i, 1)
    B_k = jnp.where(B_k > 0., B_k, 1)
    a0 = jnp.where(p_i > 1., p_k / p_i, 0.) ** 2
    a1 = jnp.where(B_k > 1, T_k / jnp.where(B_k > 1., phi / B_k, 1.), 0.) ** 3
    a2 = jnp.where(B_k > 1, jnp.log(jnp.clip((2 * df - B_k) / (2 * df + B_k), min=EPS) + 2 * B_k), 0)
    return 5 / 3 * 32 / 27 * jnp.sum(
        (phi_ik != 0) * a0 * TX2 * gamma ** 2 / L * Phi * pi * a1 / a ** 2 / (a + a_bar) ** 2 *
        (2. * df - B_k) * a2,
        axis=1)


def calculate_amplifier_gain_isrs(attenuation, length, raman_slope, ch_power, ch_centre_freq):
    """
    Calculate amplifier gain compensating for fiber loss, ROADM loss, and ISRS.
    """
    # Convert units
    a = attenuation * 1000
    L = length / 1000
    cr = raman_slope * 1e12
    f_ch = jnp.squeeze(ch_centre_freq).flatten() / 1e12
    P = jnp.squeeze(ch_power).flatten()

    # Effective fiber length [km]
    Leff = (1 - jnp.exp(-a * L)) / a

    # Total power across all active channels
    Ptot = jnp.sum(P)

    # Calculate ISRS gain tilt factor for each channel
    f_k = f_ch[:, None]  # [Nch x 1]
    f_l = f_ch[None, :]  # [1 x Nch]

    # Raman-induced power transfer
    raman_transfer = jnp.exp(-(f_k - f_l) * cr * Leff * Ptot)  # [Nch x Nch]

    # Sum of power-weighted Raman contributions
    psd_sum = jnp.sum(P[None, :] * raman_transfer, axis=1)  # [Nch]

    # Prevent division by zero
    psd_sum = jnp.where(psd_sum > 0, psd_sum, 1.0)

    # ISRS gain tilt factor
    gsrs_tilt = Ptot * jnp.exp(-f_ch * cr * Leff * Ptot) / psd_sum

    # Total gain: fiber loss compensation / ISRS tilt
    total_loss_compensation = jnp.exp(L * a)
    gain = total_loss_compensation / gsrs_tilt

    # Set gain to total loss compensation only for inactive channels
    # Make sure total_loss_compensation is broadcast to same shape as gain
    gain = jnp.where(P > 0, gain, jnp.full_like(gain, total_loss_compensation))

    return jnp.squeeze(gain)  # Ensure 1D output


def get_ase_power(noise_figure, attenuation_i, length, ref_lambda, ch_centre_i, ch_bandwidth,
                  gain=None, ch_power_i=None, raman_gain_slope_i=None, roadm_loss_db=0):
    """
    Modified to use ISRS-aware gain calculation with ROADM loss compensation.

    Additional parameters:
        ch_power_i: channel powers [Nch] in W (needed for ISRS calculation)
        raman_gain_slope_i: Raman gain slope in 1/(W*m) (needed for ISRS calculation)
        roadm_loss_db: ROADM loss in dB (default=0)
    """
    if gain is None:
        if ch_power_i is not None and raman_gain_slope_i is not None:
            # Use ISRS-aware gain calculation with ROADM loss
            a = jnp.mean(attenuation_i)
            cr = jnp.mean(raman_gain_slope_i)
            gain = calculate_amplifier_gain_isrs(a, length, cr, ch_power_i, ch_centre_i)
        else:
            # Fallback to simple gain (no ISRS compensation)
            a = jnp.mean(attenuation_i)
            roadm_loss_linear = 10 ** (roadm_loss_db / 10)
            gain = jnp.exp(a * length) * roadm_loss_linear

    # Ensure gain is properly shaped (same shape as ch_bandwidth)
    gain = jnp.squeeze(gain)
    # Calculate ASE power for each channel
    N_sp = (10 ** (noise_figure / 10) * gain) / (2. * (gain - 1))
    p_ASE = 2 * N_sp * (gain - 1) * h * (c / ref_lambda + ch_centre_i) * ch_bandwidth
    return jnp.squeeze(p_ASE)


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
        ch_centre_i: chex.Array =((jnp.arange(420) - (420 - 1) / 2) * 25e-9),
        ch_bandwidth_i: chex.Array = 25e9 * jnp.ones((420, 1)),
        excess_kurtosis_i: chex.Array = jnp.zeros((420, 1)),
        amplifier_noise_figure: chex.Array = jnp.zeros((420, 1)),
        transceiver_snr: chex.Array = jnp.zeros((420, 1)),
        uniform_spans: bool = True,
):
    """
    Compute the signal-to-noise ratio (SNR) of a WDM system.
    The SNR is defined as the ratio of the signal power to the sum of the ASE and NLI powers.

    Args:
        num_channels: number of channels
        num_spans: number of spans
        max_spans: maximum number of spans
        ref_lambda: reference wavelength [m]
        length: fiber length per span [m]
        attenuation_i: attenuation coefficient [1/m]
        attenuation_bar_i: attenuation coefficient [1/m]
        ch_power_w_i: channel power [W]
        nonlinear_coeff: Nonlinear coefficient [1/W^2]
        ch_centre_i: channel center frequency [Hz]
        ch_bandwidth_i: channel bandwidth [Hz]
        raman_gain_slope_i: Raman gain slope [1/(W*m)]
        dispersion_coeff: dispersion coefficient [s/m^2]
        dispersion_slope: dispersion slope [s/m^3]
        coherent: NLI is added coherently across multiple spans
        noise_figure: amplifier noise figure [dB]
        mod_format_correction: apply modulation format correction
        excess_kurtosis_i: excess kurtosis of modulation format
        roadm_loss: ROADM loss [dB]
        num_roadms: number of ROADMs per link (i.e. one at receive end or none)
        uniform_spans: uniform spans (simplified caluclations)

    Returns:
        snr: signal-to-noise ratio (linear units)
    """
    gn_model = isrs_gn_model_uniform if uniform_spans else isrs_gn_model
    span_length = jnp.sum(length) / num_spans
    roadm_noise_figure = amplifier_noise_figure + 1  # ROADM noise figure is 1 dB higher than amplifier noise figure
    p_ase_inline = num_spans * get_ase_power(
        amplifier_noise_figure, attenuation_i, span_length, ref_lambda, ch_centre_i, ch_bandwidth_i,
        ch_power_i=ch_power_w_i, raman_gain_slope_i=raman_gain_slope_i)
    p_ase_roadm = num_roadms * get_ase_power(roadm_noise_figure, attenuation_i, span_length, ref_lambda, ch_centre_i, ch_bandwidth_i, gain=10**(roadm_loss/10))
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
    )
    transceiver_noise = ch_power_w_i / from_db(transceiver_snr)
    noise_power = p_ase_inline + p_ase_roadm + p_nli + transceiver_noise
    noise_power = jnp.where(noise_power > 0, noise_power, EPS)
    ch_power_W_i = jnp.squeeze(ch_power_w_i)
    snr = jnp.where(ch_power_W_i > 0, ch_power_W_i / noise_power, -1e5)
    return snr, (eta_nli, eta_spm, eta_xpm, p_ase_inline, p_ase_roadm, p_nli, transceiver_noise)


def to_db(x):
    return 10*jnp.log10(x)

def to_dbm(x):
    return 10*jnp.log10(x/0.001)

def from_dbm(x):
    return jnp.round(10**(x/10) * 0.001, decimals=6)

def from_db(x):
    return jnp.round(10**(x/10), decimals=6)
