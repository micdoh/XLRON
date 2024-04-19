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
from scipy.constants import pi, h, c
from jax import numpy as jnp, jit
import jax


def isrs_gn_model(
        num_channels: int = 320,
        num_spans: int = 1,
        ref_lambda: float = 1550e-9,
        length_j: chex.Array = 100 * 1e3 * jnp.ones(1),
        attenuation_ij: chex.Array = 0.2 / 4.343 / 1e3 * jnp.ones([320, 1]),
        attenuation_bar_ij: chex.Array = 0.2 / 4.343 / 1e3 * jnp.ones([320, 1]),
        ch_power_W_ij: chex.Array = 10 ** (0 / 10) * 0.001 * jnp.ones([320, 1]),
        nonlinear_coeff_j: chex.Array = 1.2 / 1e3 * jnp.ones(1),
        ch_centre_ij: chex.Array = jnp.ones([320, 1]),
        ch_bandwidth_ij: chex.Array = 40.004e9 * jnp.ones([320, 1]),
        raman_gain_slope_ij: chex.Array = 0.028 / 1e3 / 1e12 * jnp.ones([320, 1]),
        dispersion_ij: chex.Array = 17 * 1e-12 / 1e-9 / 1e3 * jnp.ones(1),
        coherent: bool = 1,
):
    """
    This function implements the ISRS GN model in closed-form from [1]. It's an approximation with rectangular
    Raman slope and valid for C+L bands, and has no support for modulation correction

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
        dispersion_ij: dispersion coefficient [s/m^2]
            format: 1 x n matrix
        coherent: NLI is added coherently across multiple spans

    RETURNS:
        NLI: Nonlinear Interference Power[W],
            format: N_ch x 1 vector
        eta_n: Nonlinear Interference coefficient [1/W^2],
            format: N_ch x 1 matrix
    """


    # dB/m => Np/m
    a = attenuation_ij * (jnp.log(10) / 10)
    a_bar = attenuation_bar_ij * (jnp.log(10) / 10)

    l_mean = length_j.mean()
    ch_pow = ch_power_W_ij
    ch_pow0 = jnp.vstack(ch_pow[:, 0])
    power = jnp.sum(ch_pow, axis=0)
    gamma = nonlinear_coeff_j
    fi = ch_centre_ij
    ch_bw = ch_bandwidth_ij
    cr = raman_gain_slope_ij

    beta2 = -dispersion_ij * ref_lambda ** 2 / (2 * pi * c)
    beta3 = ref_lambda ** 2 / (2 * pi * c) ** 2 * (ref_lambda ** 2 * length_j + 2 * ref_lambda * dispersion_ij)

    # closed-form formula for average coherence factor extended by dispersion slope, cf. Ref. [2, Eq. (22)]
    eps = lambda B_i, f_i, a_i: (3 / 10) * jnp.log(1 + (6 / a_i) / (
                l_mean * jnp.arcsinh(pi ** 2 / 2 * abs(jnp.mean(beta2) + 2 * pi * jnp.mean(beta3) * f_i) / a_i * B_i ** 2)))

    #@jax.jit
    def _fun(j):
        """ Compute the NLI of each COI """
        a_i = jnp.vstack(a[:, j])  # \alpha of COI in fiber span j
        a_k = a_i.T  # \alpha of INT in fiber span j
        a_bar_i = jnp.vstack(a_bar[:, j])  # \bar{\alpha} of COI in fiber span j
        a_bar_k = a_bar_i.T  # \bar{\alpha} of INT in fiber span j
        f_i = jnp.vstack(fi[:, j])  # f_i of COI in fiber span j
        f_k = f_i.T  # f_k of INT in fiber span j
        B_i = jnp.vstack(ch_bw[:, j])  # B_i of COI in fiber span j
        B_k = B_i.T  # B_k of INT in fiber span j
        Cr_i = jnp.vstack(cr[:, j])  # Cr  of COI in fiber span j
        P_i = jnp.vstack(ch_pow[:, j])  # P_i of COI in fiber span j
        P_k = P_i.T  # P_k of INT in fiber span j

        phi_i = 3 / 2 * pi ** 2 * (beta2[j] + pi * beta3[j] * (f_i + f_i))  # \phi_i  of COI in fiber span j
        phi_ik = 2 * pi ** 2 * (f_k - f_i) * (
                    beta2[j] + pi * beta3[j] * (f_i + f_k))  # \phi_ik of COI-INT pair in fiber span j

        T_i = (a_i + a_bar_i - f_i * power[j] * Cr_i) ** 2  # T_i of COI in fiber span j
        T_k = T_i.T  # T_k of INT in fiber span j

        # computation of SPM contribution in fiber span j
        eta_SPM = _spm(phi_i, T_i, B_i, a_i, a_bar_i, gamma[j]) * num_spans ** eps(B_i, f_i, l_mean)*coherent

        # computation of XPM contribution in fiber span j
        eta_XPM = _xpm(P_i, P_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k, gamma[j])
        return eta_SPM, eta_XPM

    eta_spm, eta_xpm = jax.lax.map(_fun, jnp.arange(num_spans))

    # computation of NLI normalized to transmitter power, see Ref. [1, Eq. (5)]
    #eta_n = jnp.sum((P_ij / P_ij[:, 0, None]) ** 2 * (eta_SPM[0] + eta_XPM).T, axis=1)
    #eta_n = jnp.sum((ch_pow / ch_pow0) ** 2 * (eta_spm[:, :, :, 0] + eta_xpm).T, axis=1)
    #NLI = P_ij[:, 0] ** 3 * eta_n  # Ref. [1, Eq. (1)]
    #NLI = jnp.squeeze(ch_pow0) ** 3 * jnp.sum(eta_n, axis=1)

    eta_n = jnp.sum((ch_pow / ch_pow0) ** 2 * (jnp.squeeze(eta_spm) + eta_xpm).T, axis=1)
    NLI = jnp.squeeze(ch_pow0) ** 3 * eta_n  # Ref. [1, Eq. (1)]
    print(f"NLI.shape {NLI.shape}")
    print(f"eta_n.shape {eta_n.shape}")
    print(f"ch_pow0.shape {ch_pow0.shape}")

    return NLI, eta_n


#@jit
def _eps(b_i, f_i, a_i, l_mean, beta2, beta3, beta4):
    """
    Closed-for formula for average coherence factor extended by dispersion slope, cf. Ref. [2, Eq. (22)]
    """
    return (3 / 10) * jnp.log(1 + (6 / a_i) / (l_mean * jnp.arcsinh(
        pi ** 2 / 2 * jnp.abs(jnp.mean(beta2) + 2 * pi * jnp.mean(beta3) * f_i + 2 * pi ** 2 * jnp.mean(beta4) * f_i ** 2) / a_i * b_i ** 2)))


#@jit
def _spm(phi_i, t_i, b_i, a_i, a_bar_i, gamma):
    """
    Closed-form formula for SPM contribution, see Ref. [1, Eq. (9-10)]
    """
    print(f"phi_i.shape {phi_i.shape}")
    print(f"t_i.shape {t_i.shape}")
    print(f"b_i.shape {b_i.shape}")
    print(f"a_i.shape {a_i.shape}")
    print(f"a_bar_i.shape {a_bar_i.shape}")
    print(f"gamma.shape {gamma.shape}")
    return (
        4 / 9 * gamma ** 2 / b_i ** 2 * pi / (phi_i * a_bar_i * (2 * a_i + a_bar_i)) * (
            (t_i - a_i ** 2) / a_i * jnp.arcsinh(phi_i * b_i ** 2 / a_i / pi) +
            ((a_i + a_bar_i) ** 2 - t_i) / (a_i + a_bar_i) * jnp.arcsinh(phi_i * b_i ** 2 / (a_i + a_bar_i) / pi)
        )
    )


#@jit
def _xpm(p_i, p_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k, gamma):
    """
    Closed-form formula for XPM contribution, see Ref. [1, Eq. (11)]
    """
    a = (p_k / p_i) ** 2 * gamma ** 2 / (B_k * phi_ik * a_bar_k * (2 * a_k + a_bar_k))
    print(f"T_k.shape {T_k.shape}")
    print(f"a_k.shape {a_k.shape}")
    print(f"phi_ik.shape {phi_ik.shape}")
    print(f"B_i.shape {B_i.shape}")
    t1 = (T_k - a_k ** 2) / a_k
    t2 = jnp.arctan(phi_ik * B_i / a_k)
    print(f"t1.shape {t1.shape}")
    print(f"t2.shape {t2.shape}")
    b = (T_k - a_k ** 2) / a_k * jnp.arctan(phi_ik * B_i / a_k)
    c = ((a_k + a_bar_k) ** 2 - T_k) / (a_k + a_bar_k) * jnp.arctan(phi_ik * B_i / (a_k + a_bar_k))
    r = a * (b + c)
    print(f"a.shape {a.shape}")
    print(f"b.shape {b.shape}")
    print(f"r.shape {r.shape}")
    return 32 / 27 * jnp.nansum(r, axis=1 if r.ndim > 1 else 0)


@jit
def _xpm_corr(p_i, p_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k, gamma, Phi, TX1):
    return 5 / 6 * 32 / 27 * jnp.sum(
        jnp.nan_to_num(
            Phi * TX1.T * (p_k / p_i) ** 2 * gamma ** 2 / B_k / (phi_ik * a_bar_k * (2 * a_k + a_bar_k)) * (
                    (T_k - a_k ** 2) / a_k * jnp.arctan(phi_ik * B_i / a_k) +
                    ((a_k + a_bar_k) ** 2 - T_k) / (a_k + a_bar_k) * jnp.arctan(phi_ik * B_i / (a_k + a_bar_k))
            ),
            copy=False),
        axis=1)


@jit
def _xpm_corr_asymp(p_i, p_k, phi_ik, phi, T_k, B_k, a, a_bar, gamma, df, Phi, TX2, L):
    return 5 / 3 * 32 / 27 * jnp.sum(
        jnp.nan_to_num(
            (phi_ik != 0) * (p_k / p_i) ** 2 * TX2 * gamma ** 2 / L * Phi * pi * T_k / phi / B_k ** 3 / a ** 2 /
            (a + a_bar) ** 2 * ((2. * df - B_k) * jnp.log((2 * df - B_k) / (2 * df + B_k)) + 2 * B_k),
            copy=False),
        axis=1)


#@jit
def get_ASE_power(noise_figure, attenuation_ij, length_j, ref_lambda, ch_centre_ij, ch_bandwidth, gain=None):
    if gain is None:
        a = jnp.mean(attenuation_ij)
        gain = 10 ** (a * length_j[0] / 10)
    N_sp = (10 ** (noise_figure / 10) * gain) / (2. * (gain - 1))
    p_ASE = 2 * N_sp * (gain - 1) * h * (c / ref_lambda + ch_centre_ij[:, 0]) * ch_bandwidth[:, 0]
    print(f"p_ASE.shape {p_ASE.shape}")
    return p_ASE


def get_lightpath_snr(
        num_channels: int = 320,
        num_spans: int = 1,
        ref_lambda: float = 1550e-9,
        length_j: chex.Array = 100 * 1e3 * jnp.ones(1),
        attenuation_ij: chex.Array = 0.2 / 4.343 / 1e3 * jnp.ones([320, 1]),
        attenuation_bar_ij: chex.Array = 0.2 / 4.343 / 1e3 * jnp.ones([320, 1]),
        ch_power_W_ij: chex.Array = 10 ** (0 / 10) * 0.001 * jnp.ones([320, 1]),
        nonlinear_coeff_j: chex.Array = 1.2 / 1e3 * jnp.ones(1),
        ch_centre_ij: chex.Array = jnp.ones([320, 1]),
        ch_bandwidth_ij: chex.Array = 40.004e9 * jnp.ones([320, 1]),
        raman_gain_slope_ij: chex.Array = 0.028 / 1e3 / 1e12 * jnp.ones([320, 1]),
        dispersion_ij: chex.Array = 17 * 1e-12 / 1e-9 / 1e3 * jnp.ones(1),
        coherent: bool = True,
        noise_figure: float = 4,
        gain: float = None,
):
    p_ASE = get_ASE_power(noise_figure, attenuation_ij, length_j, ref_lambda, ch_centre_ij, ch_bandwidth_ij, gain)
    p_NLI = isrs_gn_model(
        num_channels=num_channels,
        num_spans=num_spans,
        ref_lambda=ref_lambda,
        length_j=length_j,
        attenuation_ij=attenuation_ij,
        attenuation_bar_ij=attenuation_bar_ij,
        ch_power_W_ij=ch_power_W_ij,
        nonlinear_coeff_j=nonlinear_coeff_j,
        ch_centre_ij=ch_centre_ij,
        ch_bandwidth_ij=ch_bandwidth_ij,
        raman_gain_slope_ij=raman_gain_slope_ij,
        dispersion_ij=dispersion_ij,
        coherent=coherent
    )[0]
    print(f"p_NLI.shape {p_NLI.shape}")
    print(f"p_ASE.shape {p_ASE.shape}")
    return to_db(ch_power_W_ij[:, 0] / (p_ASE + p_NLI))


def to_db(x):
    return 10*jnp.log(x)/jnp.log(10)


if __name__ == "__main__":
    n = 17  # number of spans
    b_ch = 40.004e9  # WDM channel bandwidth
    channels = 320  # number of channels
    spacing = 40.005e9  # WDM channel spacing

    P = {
        'num_channels': channels,  # number of channels
        'num_spans': n,  # number of spans
        'ref_lambda': 1550e-9,  # reference wavelength
        'length_j': 100 * 1e3 * jnp.ones(n),  # fiber length (same for each span)
        'attenuation_ij': 0.2 / 4.343 / 1e3 * jnp.ones([channels, n]),  # attenuation coefficient (same for each channel and span)
        'attenuation_bar_ij': 0.2 / 4.343 / 1e3 * jnp.ones([channels, n]),  # attenuation coefficient (same for each channel and span)
        'ch_power_W_ij': 10 ** (0 / 10) * 0.001 * jnp.ones([channels, n]),  # launch power per channel (same) for each channel and span
        'nonlinear_coeff_j': 1.2 / 1e3 * jnp.ones(n),  # nonlinearity coefficient    (same) for each span
        'ch_centre_ij': jnp.repeat(np.reshape(
            (jnp.arange(channels) - (channels - 1) / 2) * spacing
            , [-1, 1]), n, axis=1),  # center frequencies of WDM channels (relative to reference frequency)
        'ch_bandwidth_ij': jnp.tile(b_ch, [channels, n]),  # channel bandwith
        'raman_gain_slope_ij': 0.028 / 1e3 / 1e12 * jnp.ones([channels, n]),  # Raman gain spectrum slope   (same) for each channel and span
        'dispersion_ij': 17 * 1e-12 / 1e-9 / 1e3 * jnp.ones(n),  # dispersion coefficient      (same) for each span
        'coherent': True  # NLI is added coherently across multiple spans
    }

    # set launch power: 0 dBm/ch.
    P['ch_power_W_ij'] = 10 ** (0 / 10) * 0.001 * jnp.ones([channels, n])  # launch power per channel (same) for each channel and span
    p_nli_1span_0dbm, eta_1span_0dBm = isrs_gn_model(**P)  # compute NLI
    print(f"P_launch_0dbm: {to_db(P['ch_power_W_ij'])[0][0]}")
    print(f"P_NLI_0dbm: {to_db(p_nli_1span_0dbm)[0]}")

    # # set launch power: 2 dBm/ch.
    P['ch_power_W_ij'] = 10 ** (2 / 10) * 0.001 * jnp.ones([channels, n])  # launch power per channel (same) for each channel and span
    p_nli_1span_2dBm, eta_1span_2dBm = isrs_gn_model(**P)  # compute NLI
    print(f"P_launch_2dbm: {to_db(P['ch_power_W_ij'])[0][0]}")
    print(f"P_NLI_2dbm: {to_db(p_nli_1span_2dBm)[0]}")

    # no ISRS
    P['raman_gain_slope_ij'] = jnp.zeros([channels, n])  # set Raman gain spectrum to zero (switch off ISRS)
    p_nli_1span_noISRSm, eta_1span_noISRSm = isrs_gn_model(**P)  # compute NLI
    print(f"P_launch_no_ISRS: {to_db(P['ch_power_W_ij'])[0][0]}")
    print(f"P_NLI_no_ISRS: {to_db(p_nli_1span_noISRSm)[0]}")

    noise_figure_span = 2  # dB
    p_ase = get_ASE_power(noise_figure_span, P['attenuation_ij'], P['length_j'], P['ref_lambda'], P['ch_centre_ij'], P['ch_bandwidth_ij'])
    print(f"ASE power: {to_db(p_ase)[0]}")

    P['noise_figure'] = noise_figure_span
    snr = get_lightpath_snr(**P)
    print(f"SNR: {snr}")
