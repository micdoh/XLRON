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
    get_snr_fused,
    isrs_gn_model,
    isrs_gn_model_uniform,
)
from xlron.environments.wrappers import Profiler


def make_test_params(num_channels, num_spans=10, ref_lambda=1564e-9):
    """Create realistic test parameters for a given channel count."""
    slot_size_hz = 50e9  # 50 GHz slot size

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
    attenuation = 0.2 / 4.343 / 1e3
    attenuation_bar = 0.2 / 4.343 / 1e3
    raman_gain_slope = 0.028 / 1e3 / 1e12
    nonlinear_coeff = 1.2e-3
    dispersion_coeff = 17e-12 / 1e-9 / 1e3
    dispersion_slope_val = 0.067e-12 / 1e-9 / 1e3 / 1e-9

    attenuation_i = attenuation * jnp.ones(num_channels)
    attenuation_bar_i = attenuation_bar * jnp.ones(num_channels)
    raman_gain_slope_i = raman_gain_slope * jnp.ones(num_channels)

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
        attenuation=attenuation,
        attenuation_bar=attenuation_bar,
        attenuation_i=attenuation_i,
        attenuation_bar_i=attenuation_bar_i,
        ch_power_w_i=ch_power_w_i,
        nonlinear_coeff=nonlinear_coeff,
        nonlinear_coeff_arr=jnp.array([nonlinear_coeff]),
        ch_centre_i=ch_centre_i,
        ch_bandwidth_i=ch_bandwidth_i,
        raman_gain_slope=raman_gain_slope,
        raman_gain_slope_i=raman_gain_slope_i,
        dispersion_coeff=dispersion_coeff,
        dispersion_coeff_arr=jnp.array([dispersion_coeff]),
        dispersion_slope=dispersion_slope_val,
        dispersion_slope_arr=jnp.array([dispersion_slope_val]),
        coherent=True,
        mod_format_correction=False,
        excess_kurtosis_i=excess_kurtosis_i,
        amplifier_noise_figure=amplifier_noise_figure,
        transceiver_snr=transceiver_snr,
    )


def _bench(fn, args, kwargs, n_warmup=3, n_iters=50):
    """Generic benchmark: compile, warmup, measure."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    jax.block_until_ready(result)
    compile_time = time.perf_counter() - t0

    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        result = fn(*args, **kwargs)
        jax.block_until_ready(result)
    exec_time = (time.perf_counter() - t0) / n_iters

    return compile_time, exec_time, result


def benchmark_isrs_gn_model_uniform(params, **kw):
    fn = jax.jit(
        partial(
            isrs_gn_model_uniform,
            num_channels=params["num_channels"],
            coherent=params["coherent"],
            mod_format_correction=params["mod_format_correction"],
        )
    )
    kwargs = dict(
        num_spans=params["num_spans"],
        ref_lambda=params["ref_lambda"],
        length=params["span_length"],
        attenuation_i=params["attenuation_i"],
        attenuation_bar_i=params["attenuation_bar_i"],
        ch_power_W_i=params["ch_power_w_i"],
        nonlinear_coeff=params["nonlinear_coeff_arr"],
        ch_centre_i=params["ch_centre_i"],
        ch_bandwidth_i=params["ch_bandwidth_i"],
        raman_gain_slope_i=params["raman_gain_slope_i"],
        dispersion_coeff=params["dispersion_coeff_arr"],
        dispersion_slope=params["dispersion_slope_arr"],
        excess_kurtosis_i=params["excess_kurtosis_i"],
    )
    ct, et, result = _bench(fn, (), kwargs, **kw)
    return ct, et, *result


def benchmark_isrs_gn_model_hetero(params, **kw):
    fn = jax.jit(
        partial(
            isrs_gn_model,
            num_channels=params["num_channels"],
            max_spans=params["max_spans"],
            coherent=params["coherent"],
            mod_format_correction=params["mod_format_correction"],
        )
    )
    kwargs = dict(
        num_spans=params["num_spans"],
        ref_lambda=params["ref_lambda"],
        length=params["length"],
        attenuation_i=params["attenuation_i"],
        attenuation_bar_i=params["attenuation_bar_i"],
        ch_power_W_i=params["ch_power_w_i"],
        nonlinear_coeff=params["nonlinear_coeff_arr"],
        ch_centre_i=params["ch_centre_i"],
        ch_bandwidth_i=params["ch_bandwidth_i"],
        raman_gain_slope_i=params["raman_gain_slope_i"],
        dispersion_coeff=params["dispersion_coeff_arr"],
        dispersion_slope=params["dispersion_slope_arr"],
        excess_kurtosis_i=params["excess_kurtosis_i"],
    )
    ct, et, result = _bench(fn, (), kwargs, **kw)
    return ct, et, *result


def benchmark_get_snr(params, uniform_spans=True, **kw):
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
    kwargs = dict(
        num_spans=params["num_spans"],
        ref_lambda=params["ref_lambda"],
        length=params["length"],
        attenuation_i=params["attenuation_i"],
        attenuation_bar_i=params["attenuation_bar_i"],
        nonlinear_coeff=params["nonlinear_coeff_arr"],
        raman_gain_slope_i=params["raman_gain_slope_i"],
        dispersion_coeff=params["dispersion_coeff_arr"],
        dispersion_slope=params["dispersion_slope_arr"],
        ch_power_w_i=params["ch_power_w_i"],
        ch_centre_i=params["ch_centre_i"],
        ch_bandwidth_i=params["ch_bandwidth_i"],
        excess_kurtosis_i=params["excess_kurtosis_i"],
        amplifier_noise_figure=params["amplifier_noise_figure"],
        transceiver_snr=params["transceiver_snr"],
    )
    ct, et, result = _bench(fn, (), kwargs, **kw)
    snr, aux = result
    return ct, et, snr, aux


def benchmark_get_snr_fused(params, **kw):
    """Benchmark the fused SNR computation (uniform spans, no mod_format_correction)."""
    fn = jax.jit(
        partial(
            get_snr_fused,
            num_channels=params["num_channels"],
            ref_lambda=params["ref_lambda"],
            attenuation=params["attenuation"],
            attenuation_bar=params["attenuation_bar"],
            nonlinear_coeff=params["nonlinear_coeff"],
            raman_gain_slope=params["raman_gain_slope"],
            dispersion_coeff=params["dispersion_coeff"],
            dispersion_slope=params["dispersion_slope"],
            roadm_loss=6.0,
            num_roadms=1,
            coherent=params["coherent"],
        )
    )
    kwargs = dict(
        ch_power_w_i=params["ch_power_w_i"],
        ch_centre_i=params["ch_centre_i"],
        ch_bandwidth_i=params["ch_bandwidth_i"],
        num_spans=params["num_spans"],
        span_length=params["span_length"],
        amplifier_noise_figure=params["amplifier_noise_figure"],
        transceiver_snr=params["transceiver_snr"],
    )
    ct, et, snr = _bench(fn, (), kwargs, **kw)
    return ct, et, snr


def benchmark_get_snr_fused_vmapped(params, num_links=44, **kw):
    """Benchmark vmapped fused SNR over multiple links (simulates get_snr_link_array)."""

    # Wrap to separate vmapped (positional) from static (closure) args
    def _link_snr(ch_pow, ch_centre, ch_bw, n_spans, s_length, amp_nf, trx_snr):
        return get_snr_fused(
            ch_power_w_i=ch_pow,
            ch_centre_i=ch_centre,
            ch_bandwidth_i=ch_bw,
            num_spans=n_spans,
            span_length=s_length,
            num_channels=params["num_channels"],
            ref_lambda=params["ref_lambda"],
            attenuation=params["attenuation"],
            attenuation_bar=params["attenuation_bar"],
            nonlinear_coeff=params["nonlinear_coeff"],
            raman_gain_slope=params["raman_gain_slope"],
            dispersion_coeff=params["dispersion_coeff"],
            dispersion_slope=params["dispersion_slope"],
            amplifier_noise_figure=amp_nf,
            transceiver_snr=trx_snr,
            roadm_loss=6.0,
            num_roadms=1,
            coherent=params["coherent"],
        )

    fn_vmapped = jax.jit(jax.vmap(_link_snr))

    # Create batched inputs (num_links, num_channels)
    key = jax.random.PRNGKey(123)
    ch_power_batch = jnp.tile(params["ch_power_w_i"], (num_links, 1))
    # Add some variation per link
    noise = jax.random.uniform(key, (num_links, params["num_channels"])) * 0.2e-3
    ch_power_batch = ch_power_batch + noise
    ch_power_batch = jnp.where(ch_power_batch > 0.1e-3, ch_power_batch, 0.0)

    ch_centre_batch = jnp.tile(params["ch_centre_i"], (num_links, 1))
    ch_bw_batch = jnp.tile(params["ch_bandwidth_i"], (num_links, 1))
    num_spans_batch = jnp.full(num_links, params["num_spans"])
    span_length_batch = jnp.full(num_links, params["span_length"])
    amp_nf_batch = jnp.tile(params["amplifier_noise_figure"], (num_links, 1))
    trx_snr_batch = jnp.tile(params["transceiver_snr"], (num_links, 1))

    args = (
        ch_power_batch,
        ch_centre_batch,
        ch_bw_batch,
        num_spans_batch,
        span_length_batch,
        amp_nf_batch,
        trx_snr_batch,
    )

    ct, et, snr_batch = _bench(fn_vmapped, args, {}, **kw)
    return ct, et, snr_batch


def benchmark_calculate_amplifier_gain(params, **kw):
    a = jnp.mean(params["attenuation_i"])
    cr = jnp.mean(params["raman_gain_slope_i"])
    fn = jax.jit(calculate_amplifier_gain_isrs)
    args = (a, params["span_length"], cr, params["ch_power_w_i"], params["ch_centre_i"])
    ct, et, result = _bench(fn, args, {}, n_iters=200, **kw)
    return ct, et, result


def print_reference_values(label, **arrays):
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

    header = f"{'Function':<40} {'N_ch':>5} {'Compile (s)':>12} {'Exec (us)':>12} {'Exec (ms)':>12}"
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
            f"{'isrs_gn_model_uniform':<40} {n_ch:>5} {ct:>12.4f} {et * 1e6:>12.1f} {et * 1e3:>12.4f}"
        )
        print_reference_values(
            f"uniform N={n_ch}", NLI=NLI, eta_n=eta_n, eta_spm=eta_spm, eta_xpm=eta_xpm
        )

        # --- isrs_gn_model (heterogeneous) ---
        ct, et, NLI_h, eta_n_h, eta_spm_h, eta_xpm_h = benchmark_isrs_gn_model_hetero(params)
        print(
            f"{'isrs_gn_model (hetero)':<40} {n_ch:>5} {ct:>12.4f} {et * 1e6:>12.1f} {et * 1e3:>12.4f}"
        )
        print_reference_values(
            f"hetero N={n_ch}", NLI=NLI_h, eta_n=eta_n_h, eta_spm=eta_spm_h, eta_xpm=eta_xpm_h
        )

        # --- get_snr (uniform) ---
        ct, et, snr, aux = benchmark_get_snr(params, uniform_spans=True)
        print(
            f"{'get_snr (uniform)':<40} {n_ch:>5} {ct:>12.4f} {et * 1e6:>12.1f} {et * 1e3:>12.4f}"
        )
        print_reference_values(f"get_snr uniform N={n_ch}", snr=snr)

        # --- get_snr_fused (single link) ---
        ct, et, snr_f = benchmark_get_snr_fused(params)
        print(
            f"{'get_snr_fused (single link)':<40} {n_ch:>5} {ct:>12.4f} {et * 1e6:>12.1f} {et * 1e3:>12.4f}"
        )
        print_reference_values(f"get_snr_fused N={n_ch}", snr=snr_f)

        # --- get_snr_fused vmapped (44 links, simulating NSFNET) ---
        ct, et, snr_batch = benchmark_get_snr_fused_vmapped(params, num_links=44)
        print(
            f"{'get_snr_fused vmap(44 links)':<40} {n_ch:>5} {ct:>12.4f} {et * 1e6:>12.1f} {et * 1e3:>12.4f}"
        )
        print_reference_values(f"get_snr_fused vmap44 N={n_ch}", snr=snr_batch[0])

        # --- get_snr (heterogeneous) ---
        ct, et, snr_h, aux_h = benchmark_get_snr(params, uniform_spans=False)
        print(f"{'get_snr (hetero)':<40} {n_ch:>5} {ct:>12.4f} {et * 1e6:>12.1f} {et * 1e3:>12.4f}")
        print_reference_values(f"get_snr hetero N={n_ch}", snr=snr_h)

        # --- calculate_amplifier_gain_isrs ---
        ct, et, gain = benchmark_calculate_amplifier_gain(params)
        print(
            f"{'calculate_amplifier_gain_isrs':<40} {n_ch:>5} {ct:>12.4f} {et * 1e6:>12.1f} {et * 1e3:>12.4f}"
        )
        print_reference_values(f"amp_gain N={n_ch}", gain=gain)

        print(sep)

    # =====================================================================
    # End-to-end benchmark: get_snr_link_array vs get_snr_link_array_fused
    # Uses a real environment setup (NSFNET undirected, 22 links)
    # =====================================================================
    print()
    print("=" * len(header))
    print("END-TO-END: get_snr_link_array vs get_snr_link_array_fused")
    print("=" * len(header))

    from xlron.environments.env_funcs import get_snr_link_array, get_snr_link_array_fused
    from xlron.environments.make_env import make

    for n_ch in [4, 100]:
        env_settings = dict(
            k=5,
            topology_name="nsfnet_deeprmsa_undirected",
            link_resources=n_ch,
            max_requests=10,
            values_bw=[100],
            incremental_loading=True,
            env_type="rsa_gn_model",
            interband_gap=0,
            slot_size=12.5 if n_ch >= 100 else 25,
            mod_format_correction=False,
            launch_power=0.0,
        )
        env, env_params = make(env_settings, log_wrapper=False)
        state = env.initial_state

        # Populate ~30% of channels with active traffic
        key = jax.random.PRNGKey(42)
        active_mask = jax.random.uniform(key, state.channel_power_array.shape) < 0.3
        ch_pow = jnp.where(active_mask, 1e-3, 0.0)
        bw_val = 12.5 if n_ch >= 100 else 25.0
        bw_arr = jnp.where(active_mask, bw_val, 0.0)
        state = state.replace(channel_power_array=ch_pow, channel_centre_bw_array=bw_arr)

        num_links = env_params.num_links
        print(f"\n  N_ch={n_ch}, num_links={num_links}")

        # --- Original get_snr_link_array ---
        fn_orig = jax.jit(get_snr_link_array, static_argnums=(1,))
        ct_o, et_o, snr_orig = _bench(fn_orig, (state, env_params), {}, n_warmup=2, n_iters=20)
        print(f"  {'get_snr_link_array':<40} compile={ct_o:.3f}s  exec={et_o * 1e3:.3f}ms")

        # --- Fused get_snr_link_array_fused ---
        fn_fused = jax.jit(get_snr_link_array_fused, static_argnums=(1,))
        ct_f, et_f, snr_fused = _bench(fn_fused, (state, env_params), {}, n_warmup=2, n_iters=20)
        print(f"  {'get_snr_link_array_fused':<40} compile={ct_f:.3f}s  exec={et_f * 1e3:.3f}ms")

        # Verify correctness
        diff = jnp.abs(snr_orig - snr_fused)
        max_diff = float(jnp.max(diff))
        mask = snr_orig > -1e4
        if jnp.any(mask):
            rel_diff = float(
                jnp.max(
                    jnp.abs(snr_orig[mask] - snr_fused[mask])
                    / jnp.maximum(jnp.abs(snr_orig[mask]), 1e-10)
                )
            )
        else:
            rel_diff = 0.0
        speedup = et_o / et_f if et_f > 0 else float("inf")
        print(f"  max_rel_diff={rel_diff:.2e}  speedup={speedup:.2f}x")

    print()


if __name__ == "__main__":
    main()
