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
EPS = 1e-10

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
    beta3 = ref_lambda ** 2 / (2 * pi * c) ** 2 * (ref_lambda ** 2 * dispersion_slope + 2 * ref_lambda * dispersion)

    Phi = excess_kurtosis_i
    # indicates this channels first transmission span is span j=1 of channel i
    tx1_i = jnp.ones((num_channels, 1))
    # indicates this channels first two transmission spans are span j=1 and j=2 of channel i
    tx2_i = jnp.ones((num_channels, 1))
    T_0 = (a + a_bar - f * power * cr) ** 2
    phi_ik = 2 * pi ** 2 * (f.T - f) * (beta2 + pi * beta3 * (f + f.T))
    eta_xpm_corr = _xpm_corr(ch_pow, ch_pow.T, phi_ik, T_0.T, ch_bw, ch_bw.T, a.T, a_bar.T, gamma, Phi.T, tx1_i)

    # @jax.jit
    # def _fun(i, val):
    #     """ Compute the NLI of each COI """
    #     # TODO - replace zeros with i if using different heterogeneous spans
    #     a_i = a_k = a
    #     a_bar_i = a_bar_k = a_bar
    #     cr_k = cr_i = cr
    #     f_i = jnp.vstack(f[:, 0])  # f_i of COI in fiber span j
    #     f_k = f_i.T  # f_k of INT in fiber span j
    #     ch_bw_i = jnp.vstack(ch_bw[:, 0])  # B_i of COI in fiber span j
    #     ch_bw_k = ch_bw_i.T
    #     ch_pow_i = jnp.vstack(ch_pow[:, 0])  # P_i of COI in fiber span j
    #     ch_pow_k = ch_pow_i.T
    #     Phi_k = jnp.vstack(Phi[:, 0])  # excess kurtosis of mod. format of INT in fiber span j
    #     df = jnp.abs(f_k - f_i)
    #
    #     phi = jnp.abs(4 * pi ** 2 * (beta2 + pi * beta3 * (f_i + f_k)))
    #     phi_i = 3 / 2 * pi ** 2 * (beta2 + pi * beta3 * (f_i + f_i))  # \phi_i  of COI in fiber span j
    #     phi_ik = 2 * pi ** 2 * (f_k - f_i) * (beta2 + pi * beta3 * (f_i + f_k))  # \phi_ik of COI-INT pair in fiber span j
    #
    #     T_i = (a + a_bar - f * power * cr) ** 2  # T_i of COI in fiber span j
    #     T_k = T_i.T  # T_k of INT in fiber span j
    #
    #     # computation of SPM contribution in fiber span j
    #      _eta_spm = val[0] + jnp.squeeze(jnp.where(i < num_spans, _spm(phi_i, T_i, ch_bw_i, a_i, a_bar_i, gamma) * num_spans ** (1 + eps(ch_bw_i, f_i, a_i, l_mean) * coherent), 0))
    #
    #     # computation of XPM contribution in fiber span j
    #     _eta_xpm = val[1] + jnp.where(i < num_spans, _xpm(ch_pow_i, ch_pow_k, phi_ik, T_k, ch_bw_i, ch_bw_k, a_k, a_bar_k, gamma), 0)
    #
    #     # Asymptotic correction for non-Gaussian modulation format
    #     # TODO - should this use length or L_mean?
    #     _eta_xpm_corr_asymp = val[2] + jnp.where(i < num_spans, _xpm_corr_asymp(ch_pow_i, ch_pow_k, phi_ik, phi, T_k, ch_bw_k, a_k, a_bar_k, gamma, df, Phi_k, tx2_i, l_mean), 0)
    #
    #     return _eta_spm, _eta_xpm, _eta_xpm_corr_asymp
    #
    # eta_spm, eta_xpm, eta_xpm_corr_asymp = jax.lax.fori_loop(0, max_spans, _fun,
    #                                                          (
    #                                                              jnp.zeros((num_channels,)),
    #                                                              jnp.zeros((num_channels,)),
    #                                                              jnp.zeros((num_channels,))
    #                                                           ), unroll=True)

    # TODO - try the for loop without the where conditions and with/without unroll

    # a_i = a_k = a
    # a_bar_i = a_bar_k = a_bar
    # cr_k = cr_i = cr
    # f_i = jnp.vstack(f[:, 0])  # f_i of COI in fiber span j
    # f_k = f_i.T  # f_k of INT in fiber span j
    # ch_bw_i = jnp.vstack(ch_bw[:, 0])  # B_i of COI in fiber span j
    # ch_bw_k = ch_bw_i.T
    # ch_pow_i = jnp.vstack(ch_pow[:, 0])  # P_i of COI in fiber span j
    # ch_pow_k = ch_pow_i.T
    # Phi_k = jnp.vstack(Phi[:, 0])  # excess kurtosis of mod. format of INT in fiber span j
    # df = jnp.abs(f_k - f_i)
    # phi = jnp.abs(4 * pi ** 2 * (beta2 + pi * beta3 * (f_i + f_k)))
    # phi_i = 3 / 2 * pi ** 2 * (beta2 + pi * beta3 * (f_i + f_i))  # \phi_i  of COI in fiber span j
    # phi_ik = 2 * pi ** 2 * (f_k - f_i) * (beta2 + pi * beta3 * (f_i + f_k))  # \phi_ik of COI-INT pair in fiber span j
    # T_i = (a + a_bar - f * power * cr) ** 2  # T_i of COI in fiber span j
    # T_k = T_i.T  # T_k of INT in fiber span j
    # # computation of SPM contribution in fiber span j
    # eta_spm = jnp.squeeze(_spm(phi_i, T_i, ch_bw_i, a_i, a_bar_i, gamma) * num_spans ** (1 + eps(ch_bw_i,f_i,a_i,l_mean) * coherent)) * num_spans
    # # computation of XPM contribution in fiber span j
    # eta_xpm = _xpm(ch_pow_i, ch_pow_k, phi_ik, T_k, ch_bw_i, ch_bw_k, a_k, a_bar_k, gamma) * num_spans
    # # Asymptotic correction for non-Gaussian modulation format
    # eta_xpm_corr_asymp = _xpm_corr_asymp(ch_pow_i, ch_pow_k, phi_ik, phi, T_k, ch_bw_k, a_k, a_bar_k, gamma, df, Phi_k, tx2_i, l_mean) * num_spans

    def scan_fun(carry, l_span):
        """ Compute the NLI of each COI """
        # TODO - replace zeros with i if using different heterogeneous spans
        a_i = a_k = a
        a_bar_i = a_bar_k = a_bar
        cr_k = cr_i = cr
        f_i = jnp.vstack(f[:, 0])  # f_i of COI in fiber span j
        f_k = f_i.T  # f_k of INT in fiber span j
        ch_bw_i = jnp.vstack(ch_bw[:, 0])  # B_i of COI in fiber span j
        ch_bw_k = ch_bw_i.T
        ch_pow_i = jnp.vstack(ch_pow[:, 0])  # P_i of COI in fiber span j
        ch_pow_k = ch_pow_i.T
        Phi_k = jnp.vstack(Phi[:, 0])  # excess kurtosis of mod. format of INT in fiber span j
        df = jnp.abs(f_k - f_i)

        phi = jnp.abs(4 * pi ** 2 * (beta2 + pi * beta3 * (f_i + f_k)))
        phi_i = 3 / 2 * pi ** 2 * (beta2 + pi * beta3 * (f_i + f_i))  # \phi_i  of COI in fiber span j
        phi_ik = 2 * pi ** 2 * (f_k - f_i) * (beta2 + pi * beta3 * (f_i + f_k))  # \phi_ik of COI-INT pair in fiber span j

        T_i = (a + a_bar - f * power * cr) ** 2  # T_i of COI in fiber span j
        T_k = T_i.T  # T_k of INT in fiber span j

        # computation of SPM contribution in fiber span j
        spm = _spm(phi_i, T_i, ch_bw_i, a_i, a_bar_i, gamma)
        eps = _eps(ch_bw_i, f_i, a_i, l_mean, beta2, beta3)
        # jax.debug.print("ch_bw_i {}", ch_bw_i.T, ordered=True)
        # jax.debug.print("spm {}", spm.T, ordered=True)
        # jax.debug.print("eps {}", eps.T, ordered=True)
        _eta_spm = carry[0] + jnp.squeeze(spm * num_spans ** (1 + (eps * coherent)))

        # computation of XPM contribution in fiber span j
        _eta_xpm = carry[1] + _xpm(ch_pow_i, ch_pow_k, phi_ik, T_k, ch_bw_i, ch_bw_k, a_k, a_bar_k, gamma)

        # Asymptotic correction for non-Gaussian modulation format
        # TODO - should this use length or L_mean?
        _eta_xpm_corr_asymp = carry[2] + _xpm_corr_asymp(ch_pow_i, ch_pow_k, phi_ik, phi, T_k, ch_bw_k, a_k, a_bar_k, gamma, df, Phi_k, tx2_i, l_span)

        return (_eta_spm, _eta_xpm, _eta_xpm_corr_asymp), (_eta_spm, _eta_xpm, _eta_xpm_corr_asymp)

    final_state, traj = jax.lax.scan(
        scan_fun,
        (jnp.zeros((num_channels,)), jnp.zeros((num_channels,)), jnp.zeros((num_channels,))),
        length,
        length=max_spans
    )
    eta_spm = final_state[0]
    eta_xpm = final_state[1]
    eta_xpm_corr_asymp = final_state[2]

    if not mod_format_correction:
        eta_xpm_corr = jnp.zeros((num_channels,))
        eta_xpm_corr_asymp = jnp.zeros((num_channels,))

    # computation of NLI - see Ref. [1, Eq. (5)]
    eta_n = (eta_spm + eta_xpm + eta_xpm_corr_asymp).T + eta_xpm_corr
    # jax.debug.print("eta_n {}", eta_n, ordered=True)
    # jax.debug.print("eta_spm {}", eta_spm, ordered=True)
    # jax.debug.print("eta_xpm {}", eta_xpm, ordered=True)
    # jax.debug.print("eta_xpm_corr_asymp {}", eta_xpm_corr_asymp, ordered=True)
    # jax.debug.print("eta_xpm_corr {}", eta_xpm_corr, ordered=True)
    NLI = jnp.squeeze(ch_pow) ** 3 * eta_n  # Ref. [1, Eq. (1)]
    return NLI, eta_n


def _eps(B_i, f_i, a_i, l_i, beta2, beta3):
    """
    Closed-for formula for average coherence factor extended by dispersion slope, cf. Ref. [2, Eq. (22)]
    """
    # closed-form formula for average coherence factor extended by dispersion slope, cf. Ref. [2, Eq. (22)]
    eps = (3 / 10 * jnp.log(
        jnp.clip(
        1 + (6 / a_i) / (
                l_i * jnp.arcsinh(pi ** 2 / 2 * jnp.abs(jnp.mean(beta2) + 2 * pi * jnp.mean(beta3) * f_i / a_i * B_i ** 2))
        ), min=EPS)))
    return eps


def _spm(phi_i, t_i, B_i, a_i, a_bar_i, gamma):
    """
    Closed-form formula for SPM contribution, see Ref. [1, Eq. (9-10)]
    """
    B_i = jnp.where(B_i > 0., B_i, EPS)
    a = gamma ** 2 / jnp.where(B_i > EPS, B_i ** 2, 0.)
    b = jnp.arcsinh(phi_i * B_i ** 2 / a_i / pi)
    b = jnp.where(B_i > EPS, b, 0.)
    c = jnp.arcsinh(phi_i * B_i ** 2 / (a_i + a_bar_i) / pi)
    c = jnp.where(B_i > EPS, c, 0.)
    return (4 / 9 * a * pi / (phi_i * a_bar_i * (2 * a_i + a_bar_i)) * (t_i - a_i ** 2) / a_i * b +
        ((a_i + a_bar_i) ** 2 - t_i) / (a_i + a_bar_i) * c)


def _xpm(p_i, p_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k, gamma):
    """
    Closed-form formula for XPM contribution, see Ref. [1, Eq. (11)]
    """
    p_i = jnp.where(p_i > 0., p_i, 1.)
    B_k = jnp.where(B_k > 0., B_k, 1.)
    a1 = jnp.where(p_i > 1., p_k / p_i, 0.) ** 2
    a2 = (B_k * phi_ik * a_bar_k * (2 * a_k + a_bar_k))
    a2 = gamma ** jnp.where(a2 > EPS, 2 / a2, 0.)
    a = a1 * a2
    b = (T_k - a_k ** 2) / a_k * jnp.arctan(phi_ik * B_i / a_k)
    c = ((a_k + a_bar_k) ** 2 - T_k) / (a_k + a_bar_k) * jnp.arctan(phi_ik * B_i / (a_k + a_bar_k))
    r = a * (b + c)
    return 32 / 27 * jnp.sum(r, axis=1 if r.ndim > 1 else 0)


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


def get_ase_power(noise_figure, attenuation_i, length, ref_lambda, ch_centre_i, ch_bandwidth, gain=None):
    if gain is None:
        a = jnp.mean(attenuation_i)
        gain = 10 ** (a * length / 10)
    N_sp = (10 ** (noise_figure / 10) * gain) / (2. * (gain - 1))
    p_ASE = 2 * N_sp * (gain - 1) * h * (c / ref_lambda + ch_centre_i) * ch_bandwidth
    return jnp.squeeze(p_ASE)


def get_snr(
        num_channels: int = 420,
        max_spans: int = 20,
        ref_lambda: float = 1577.5e-9,
        attenuation_i: chex.Array = 0.2 / 4.343 / 1e3 * jnp.ones((420, 1)),
        attenuation_bar_i: chex.Array = 0.2 / 4.343 / 1e3 * jnp.ones((420, 1)),
        nonlinear_coeff: chex.Array = 1.2 / 1e3 * jnp.ones(1),
        coherent: bool = True,
        noise_figure: float = 4,
        mod_format_correction: bool = False,
        raman_gain_slope_i: chex.Array = 0.028 / 1e3 / 1e12 * jnp.ones(1),
        dispersion_coeff: chex.Array = 17 * 1e-12 / 1e-9 / 1e3 * jnp.ones(1),
        dispersion_slope: chex.Array = 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9 * jnp.ones(1),
        roadm_loss: float = 18,
        num_roadms: int = 1,
        length: chex.Array = 100 * 1e3 * jnp.ones(20),
        num_spans: int = 20,
        ch_power_w_i: chex.Array = 10 ** (0 / 10) * 0.001 * jnp.ones((420, 1)),
        ch_centre_i: chex.Array = ((jnp.arange(420) - (420 - 1) / 2) * 25e-9).reshape((420, 1)),
        ch_bandwidth_i: chex.Array = 25e9 * jnp.ones((420, 1)),
        excess_kurtosis_i: chex.Array = jnp.zeros((420, 1)),
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

    Returns:
        snr: signal-to-noise ratio (linear units)
    """
    span_length = jnp.sum(length) / num_spans
    p_ase = get_ase_power(noise_figure, attenuation_i, span_length, ref_lambda, ch_centre_i, ch_bandwidth_i) * num_spans
    p_ase_roadm = num_roadms * get_ase_power(noise_figure, roadm_loss, span_length, ref_lambda, ch_centre_i, ch_bandwidth_i, gain=10**(roadm_loss/10))
    p_nli, eta_nli = isrs_gn_model(
        num_channels=num_channels,
        num_spans=num_spans,
        max_spans=max_spans,
        ref_lambda=ref_lambda,
        length=length,
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
    noise_power = p_ase + p_ase_roadm + p_nli
    noise_power = jnp.where(noise_power > 0, noise_power, EPS)
    ch_power_W_i = jnp.squeeze(ch_power_w_i)
    snr = jnp.where(ch_power_W_i > 0, ch_power_W_i / noise_power, -1e5)
    # jax.debug.print("p_ASE {}", p_ASE)
    # jax.debug.print("p_NLI {}", p_NLI)
    # jax.debug.print("p_ASE_ROADM {}", p_ASE_ROADM)
    # jax.debug.print("ch_power_W_i {}", ch_power_W_i.T)
    # jax.debug.print("noise_power {}", noise_power, ordered=True)
    # jax.debug.print("snr {}", snr, ordered=True)
    return snr, eta_nli


def to_db(x):
    return 10*jnp.log10(x)

def to_dbm(x):
    return 10*jnp.log10(x/0.001)


def from_dbm(x):
    return jnp.round(10**(x/10) * 0.001, decimals=6)


def from_db(x):
    return jnp.round(10**(x/10), decimals=6)


if __name__ == "__main__":
    n = 30  # number of spans
    b_ch = 25e9  # WDM channel bandwidth
    channels = 420  # number of channels
    spacing = 25e9  # WDM channel spacing

    P = {
        'num_channels': channels,  # number of channels
        'num_spans': n,  # number of spans
        'max_spans': n,
        'ref_lambda': 1550e-9,  # reference wavelength
        'length': 100 * 1e3 * jnp.ones(n),  # fiber length (same for each span)
        'attenuation_i': 0.2 / 4.343 / 1e3 * jnp.ones(1),  # attenuation coefficient (same for each channel and span)
        'attenuation_bar_i': 0.2 / 4.343 / 1e3 * jnp.ones(1),  # attenuation coefficient (same for each channel and span)
        'ch_power_W_i': 10 ** (-7 / 10) * 0.001 * jnp.ones((channels, 1)),  # launch power per channel (same) for each channel and span
        'nonlinear_coeff': 1.2 / 1e3 * jnp.ones(1),  # nonlinearity coefficient    (same) for each span
        'ch_centre_i': ((jnp.arange(channels) - (channels - 1) / 2) * spacing).reshape((channels, 1)),  # center frequencies of WDM channels (relative to reference frequency)
        'ch_bandwidth_i': jnp.tile(b_ch, (channels, 1)),  # channel bandwith
        'raman_gain_slope_i': 0.028 / 1e3 / 1e12 * jnp.ones(1),  # Raman gain spectrum slope   (same) for each channel and span
        'dispersion_coeff': 17 * 1e-12 / 1e-9 / 1e3 * jnp.ones(1),  # dispersion coefficient      (same) for each span
        'dispersion_slope': 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9 * jnp.ones(1),  # dispersion slope            (same) for each span
        'coherent': True,  # NLI is added coherently across multiple spans
        'mod_format_correction': True,
        'consider_roadm': True,
        'roadm_loss': 18,
        'excess_kurtosis_i': jnp.full((channels, 1), -1),  # excess kurtosis of modulation format
    }
    # Set last 50 channel power to zero
    #P['ch_power_W_i'] = P['ch_power_W_i'].at[-100:,].set(0.000001)
    # # Set every odd channel to zero
    # P["ch_power_W_i"] = jnp.zeros(int(channels/10)).repeat(10).reshape((channels, 1))
    # P['ch_power_W_i'] = P['ch_power_W_i'].at[0].set(0.001)#.at[::10].set(0.001)
    # P["ch_bandwidth_i"] = jnp.zeros(int(channels/10)).repeat(10).reshape((channels, 1))
    # P['ch_bandwidth_i'] = P['ch_bandwidth_i'].at[0].set(b_ch*4)#.at[::10].set(b_ch*4)
    #P["ch_centre_i"] = P['ch_centre_i'].at[0].set(b_ch*4)#((jnp.arange(channels/10).repeat(10) - (channels/10 - 1) / 2) * spacing*10).reshape((channels, 1))

    # print(P['ch_power_W_i'])
    # print(P['ch_centre_i'])
    # print(P['ch_bandwidth_i'])
    print(P["length"])

    # # set launch power: 0 dBm/ch.
    #P['ch_power_W_i'] = 10 ** (0 / 10) * 0.001 * jnp.ones((channels, 1))  # launch power per channel (same) for each channel and span
    # p_nli_1span_0dbm, eta_1span_0dBm = isrs_gn_model(**P)  # compute NLI
    # print(f"P_launch_0dbm: {to_db(P['ch_power_W_i'])[0]}")
    # print(f"P_NLI_0dbm: {to_db(p_nli_1span_0dbm)[0]}")
    #
    # # # set launch power: 2 dBm/ch.
    # P['ch_power_W_i'] = 10 ** (2 / 10) * 0.001 * jnp.ones(channels)  # launch power per channel (same) for each channel and span
    # p_nli_1span_2dBm, eta_1span_2dBm = isrs_gn_model(**P)  # compute NLI
    # print(f"P_launch_2dbm: {to_db(P['ch_power_W_i'])[0]}")
    # print(f"P_NLI_2dbm: {to_db(p_nli_1span_2dBm)[0]}")
    #
    # # no ISRS
    # P['raman_gain_slope_i'] = jnp.zeros(channels)  # set Raman gain spectrum to zero (switch off ISRS)
    # p_nli_1span_noISRSm, eta_1span_noISRSm = isrs_gn_model(**P)  # compute NLI
    # print(f"P_launch_no_ISRS: {to_db(P['ch_power_W_i'])[0]}")
    # print(f"P_NLI_no_ISRS: {to_db(p_nli_1span_noISRSm)[0]}")

    noise_figure_span = 4  # dB
    p_ase = get_ase_power(noise_figure_span, P['attenuation_i'], P['length'], P['ref_lambda'], P['ch_centre_i'], P['ch_bandwidth_i'])
    print(f"ASE power: {to_db(p_ase)[0]}")

    P['noise_figure'] = noise_figure_span
    snr, eta_nli = get_snr(**P)
    print(f"SNR: {to_db(snr)}")
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.plot(P["ch_centre_i"], pd.Series(to_db(snr)).ffill())
    plt.title("SNR vs Channel")
    plt.show()
    plt.plot(P["ch_centre_i"], pd.Series(to_db(eta_nli)).ffill())
    plt.title("eta_NLI vs Channel")
    plt.show()
