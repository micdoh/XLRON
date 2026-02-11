"""Benchmark suite for the ISRS GN model.

Measures JIT compilation time and steady-state execution time for each function
at various channel counts. Records numerical reference values for correctness
verification after optimizations.

Usage:
    python xlron/environments/gn_model/benchmark_gn_model.py
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
from scipy.constants import c, pi

from xlron.environments.gn_model.isrs_gn_model import (
    EPS,
    calculate_amplifier_gain_isrs,
    get_ase_power,
    get_snr,
    isrs_gn_model,
    isrs_gn_model_uniform,
)
from xlron.environments.wrappers import Profiler


def make_test_params(num_channels, num_spans=10, ref_lambda=1564e-9):
    """Create realistic test parameters for a given channel count."""
    slot_size_hz = 50e9  # 50 GHz slot size
    f_centre = c / ref_lambda  # centre frequency in Hz

    # Channel centre frequencies (evenly spaced around ref_lambda)
    ch_indices = jnp.arange(num_channels) - (num_channels - 1) / 2.0
    ch_centre_i = ch_indices * slot_size_hz  # relative to f_centre, in Hz

    # Channel bandwidths (all same)
    ch_bandwidth_i = slot_size_hz * jnp.ones(num_channels)

    # Channel powers: mix of active and inactive channels
    # ~80% active at 0 dBm, ~20% inactive
    key = jax.random.PRNGKey(42)
    active_mask = jax.random.uniform(key, (num_channels,)) < 0.8
    ch_power_w_i = jnp.where(active_mask, 1e-3, 0.0)  # 0 dBm = 1 mW

    # Fiber parameters (typical SMF-28)
    attenuation_i = 0.2 / 4.343 / 1e3 * jnp.ones(num_channels)
    attenuation_bar_i = 0.2 / 4.343 / 1e3 * jnp.ones(num_channels)
    raman_gain_slope_i = 0.028 / 1e3 / 1e12 * jnp.ones(num_channels)
    nonlinear_coeff = jnp.array([1.2e-3])
    dispersion_coeff = jnp.array([17e-12 / 1e-9 / 1e3])
    dispersion_slope = jnp.array([0.067e-12 / 1e-9 / 1e3 / 1e-9])

    # Span lengths
    span_length = 100e3  # 100 km per span
    length = span_length * jnp.ones(num_spans)

    # Noise parameters
    amplifier_noise_figure = 5.5 * jnp.ones(num_channels)
    transceiver_snr = 21.25 * jnp.ones(num_channels)
    excess_kurtosis_i = jnp.zeros(num_channels)

    return dict(
        num_channels=num_channels,
        num_spans=num_spans,
        max_spans=num_spans,
        ref_lambda=ref_lambda,
        length=length,
        span_length=span_length,
        attenuation_i=attenuation_i,
        attenuation_bar_i=attenuation_bar_i,
        ch_power_w_i=ch_power_w_i,
        nonlinear_coeff=nonlinear_coeff,
        ch_centre_i=ch_centre_i,
        ch_bandwidth_i=ch_bandwidth_i,
        raman_gain_slope_i=raman_gain_slope_i,
        dispersion_coeff=dispersion_coeff,
        dispersion_slope=dispersion_slope,
        coherent=True,
        mod_format_correction=False,
        excess_kurtosis_i=excess_kurtosis_i,
        amplifier_noise_figure=amplifier_noise_figure,
        transceiver_snr=transceiver_snr,
    )


def benchmark_isrs_gn_model_uniform(params, n_warmup=3, n_iters=50):
    """Benchmark isrs_gn_model_uniform."""
    fn = jax.jit(
        partial(
            isrs_gn_model_uniform,
            num_channels=params["num_channels"],
            coherent=params["coherent"],
            mod_format_correction=params["mod_format_correction"],
        )
    )

    call_args = dict(
        num_spans=params["num_spans"],
        ref_lambda=params["ref_lambda"],
        length=params["span_length"],
        attenuation_i=params["attenuation_i"],
        attenuation_bar_i=params["attenuation_bar_i"],
        ch_power_W_i=params["ch_power_w_i"],
        nonlinear_coeff=params["nonlinear_coeff"],
        ch_centre_i=params["ch_centre_i"],
        ch_bandwidth_i=params["ch_bandwidth_i"],
        raman_gain_slope_i=params["raman_gain_slope_i"],
        dispersion_coeff=params["dispersion_coeff"],
        dispersion_slope=params["dispersion_slope"],
        excess_kurtosis_i=params["excess_kurtosis_i"],
    )

    # Compile + first call
    t0 = time.perf_counter()
    result = fn(**call_args)
    jax.block_until_ready(result)
    compile_time = time.perf_counter() - t0

    # Warmup
    for _ in range(n_warmup):
        result = fn(**call_args)
        jax.block_until_ready(result)

    # Steady-state
    t0 = time.perf_counter()
    for _ in range(n_iters):
        result = fn(**call_args)
        jax.block_until_ready(result)
    exec_time = (time.perf_counter() - t0) / n_iters

    NLI, eta_n, eta_spm, eta_xpm = result
    return compile_time, exec_time, NLI, eta_n, eta_spm, eta_xpm


def benchmark_isrs_gn_model_hetero(params, n_warmup=3, n_iters=50):
    """Benchmark isrs_gn_model (heterogeneous spans)."""
    fn = jax.jit(
        partial(
            isrs_gn_model,
            num_channels=params["num_channels"],
            max_spans=params["max_spans"],
            coherent=params["coherent"],
            mod_format_correction=params["mod_format_correction"],
        )
    )

    call_args = dict(
        num_spans=params["num_spans"],
        ref_lambda=params["ref_lambda"],
        length=params["length"],
        attenuation_i=params["attenuation_i"],
        attenuation_bar_i=params["attenuation_bar_i"],
        ch_power_W_i=params["ch_power_w_i"],
        nonlinear_coeff=params["nonlinear_coeff"],
        ch_centre_i=params["ch_centre_i"],
        ch_bandwidth_i=params["ch_bandwidth_i"],
        raman_gain_slope_i=params["raman_gain_slope_i"],
        dispersion_coeff=params["dispersion_coeff"],
        dispersion_slope=params["dispersion_slope"],
        excess_kurtosis_i=params["excess_kurtosis_i"],
    )

    # Compile + first call
    t0 = time.perf_counter()
    result = fn(**call_args)
    jax.block_until_ready(result)
    compile_time = time.perf_counter() - t0

    # Warmup
    for _ in range(n_warmup):
        result = fn(**call_args)
        jax.block_until_ready(result)

    # Steady-state
    t0 = time.perf_counter()
    for _ in range(n_iters):
        result = fn(**call_args)
        jax.block_until_ready(result)
    exec_time = (time.perf_counter() - t0) / n_iters

    NLI, eta_n, eta_spm, eta_xpm = result
    return compile_time, exec_time, NLI, eta_n, eta_spm, eta_xpm


def benchmark_get_snr(params, uniform_spans=True, n_warmup=3, n_iters=50):
    """Benchmark get_snr (the main entry point)."""
    fn = jax.jit(
        partial(
            get_snr,
            num_channels=params["num_channels"],
            max_spans=params["max_spans"],
            coherent=params["coherent"],
            mod_format_correction=params["mod_format_correction"],
            uniform_spans=uniform_spans,
        )
    )

    call_args = dict(
        num_spans=params["num_spans"],
        ref_lambda=params["ref_lambda"],
        length=params["length"],
        attenuation_i=params["attenuation_i"],
        attenuation_bar_i=params["attenuation_bar_i"],
        nonlinear_coeff=params["nonlinear_coeff"],
        raman_gain_slope_i=params["raman_gain_slope_i"],
        dispersion_coeff=params["dispersion_coeff"],
        dispersion_slope=params["dispersion_slope"],
        ch_power_w_i=params["ch_power_w_i"],
        ch_centre_i=params["ch_centre_i"],
        ch_bandwidth_i=params["ch_bandwidth_i"],
        excess_kurtosis_i=params["excess_kurtosis_i"],
        amplifier_noise_figure=params["amplifier_noise_figure"],
        transceiver_snr=params["transceiver_snr"],
    )

    # Compile + first call
    t0 = time.perf_counter()
    result = fn(**call_args)
    jax.block_until_ready(result)
    compile_time = time.perf_counter() - t0

    # Warmup
    for _ in range(n_warmup):
        result = fn(**call_args)
        jax.block_until_ready(result)

    # Steady-state
    t0 = time.perf_counter()
    for _ in range(n_iters):
        result = fn(**call_args)
        jax.block_until_ready(result)
    exec_time = (time.perf_counter() - t0) / n_iters

    snr, aux = result
    return compile_time, exec_time, snr, aux


def benchmark_calculate_amplifier_gain(params, n_warmup=3, n_iters=200):
    """Benchmark calculate_amplifier_gain_isrs."""
    a = jnp.mean(params["attenuation_i"])
    cr = jnp.mean(params["raman_gain_slope_i"])

    fn = jax.jit(calculate_amplifier_gain_isrs)

    call_args = (
        a,
        params["span_length"],
        cr,
        params["ch_power_w_i"],
        params["ch_centre_i"],
    )

    # Compile + first call
    t0 = time.perf_counter()
    result = fn(*call_args)
    jax.block_until_ready(result)
    compile_time = time.perf_counter() - t0

    # Warmup
    for _ in range(n_warmup):
        result = fn(*call_args)
        jax.block_until_ready(result)

    # Steady-state
    t0 = time.perf_counter()
    for _ in range(n_iters):
        result = fn(*call_args)
        jax.block_until_ready(result)
    exec_time = (time.perf_counter() - t0) / n_iters

    return compile_time, exec_time, result


def print_reference_values(label, **arrays):
    """Print summary statistics for numerical reference."""
    print(f"\n  Reference values ({label}):")
    for name, arr in arrays.items():
        arr = jnp.asarray(arr).flatten()
        active = arr[arr != 0]
        if active.size > 0:
            print(
                f"    {name:20s}: sum={float(jnp.sum(arr)):.8e}  "
                f"mean(active)={float(jnp.mean(active)):.8e}  "
                f"min={float(jnp.min(arr)):.8e}  max={float(jnp.max(arr)):.8e}"
            )
        else:
            print(f"    {name:20s}: all zeros")


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    channel_counts = [4, 100, 420]
    num_spans = 10

    # Table header
    header = f"{'Function':<35} {'N_ch':>5} {'Compile (s)':>12} {'Exec (us)':>12} {'Exec (ms)':>12}"
    sep = "-" * len(header)

    print("=" * len(header))
    print("ISRS GN MODEL BENCHMARK")
    print("=" * len(header))
    print(header)
    print(sep)

    for n_ch in channel_counts:
        params = make_test_params(n_ch, num_spans=num_spans)

        # --- isrs_gn_model_uniform ---
        ct, et, NLI, eta_n, eta_spm, eta_xpm = benchmark_isrs_gn_model_uniform(params)
        print(
            f"{'isrs_gn_model_uniform':<35} {n_ch:>5} {ct:>12.4f} "
            f"{et * 1e6:>12.1f} {et * 1e3:>12.4f}"
        )
        print_reference_values(
            f"uniform N={n_ch}", NLI=NLI, eta_n=eta_n, eta_spm=eta_spm, eta_xpm=eta_xpm
        )

        # --- isrs_gn_model (heterogeneous) ---
        ct, et, NLI_h, eta_n_h, eta_spm_h, eta_xpm_h = benchmark_isrs_gn_model_hetero(params)
        print(
            f"{'isrs_gn_model (hetero)':<35} {n_ch:>5} {ct:>12.4f} "
            f"{et * 1e6:>12.1f} {et * 1e3:>12.4f}"
        )
        print_reference_values(
            f"hetero N={n_ch}",
            NLI=NLI_h,
            eta_n=eta_n_h,
            eta_spm=eta_spm_h,
            eta_xpm=eta_xpm_h,
        )

        # --- get_snr (uniform) ---
        ct, et, snr, aux = benchmark_get_snr(params, uniform_spans=True)
        print(
            f"{'get_snr (uniform)':<35} {n_ch:>5} {ct:>12.4f} {et * 1e6:>12.1f} {et * 1e3:>12.4f}"
        )
        print_reference_values(f"get_snr uniform N={n_ch}", snr=snr)

        # --- get_snr (heterogeneous) ---
        ct, et, snr_h, aux_h = benchmark_get_snr(params, uniform_spans=False)
        print(f"{'get_snr (hetero)':<35} {n_ch:>5} {ct:>12.4f} {et * 1e6:>12.1f} {et * 1e3:>12.4f}")
        print_reference_values(f"get_snr hetero N={n_ch}", snr=snr_h)

        # --- calculate_amplifier_gain_isrs ---
        ct, et, gain = benchmark_calculate_amplifier_gain(params)
        print(
            f"{'calculate_amplifier_gain_isrs':<35} {n_ch:>5} {ct:>12.4f} "
            f"{et * 1e6:>12.1f} {et * 1e3:>12.4f}"
        )
        print_reference_values(f"amp_gain N={n_ch}", gain=gain)

        print(sep)

    print()


if __name__ == "__main__":
    main()
