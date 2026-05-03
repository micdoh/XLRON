"""Benchmark suite for the ISRS GN model.

Measures performance in contexts that reflect real usage:
1. Isolated function benchmarks (for numerical verification)
2. Inside-JIT benchmarks via jax.lax.scan (simulates training loop)
3. XLA HLO operation counts (measures compilation-graph complexity)
4. Full env.step comparison: rsa_gn_model vs plain rsa

Usage:
    python xlron/environments/gn_model/benchmark_gn_model.py
"""

import time
from functools import partial

import jax
import jax.numpy as jnp

from xlron.environments.gn_model.isrs_gn_model import (
    get_snr,
    get_snr_fused,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_params(num_channels, num_spans=10, ref_lambda=1564e-9):
    """Create realistic test parameters for a given channel count."""
    slot_size_hz = 50e9

    ch_indices = jnp.arange(num_channels) - (num_channels - 1) / 2.0
    ch_centre_i = ch_indices * slot_size_hz

    ch_bandwidth_i = slot_size_hz * jnp.ones(num_channels)

    key = jax.random.PRNGKey(42)
    active_mask = jax.random.uniform(key, (num_channels,)) < 0.8
    ch_power_w_i = jnp.where(active_mask, 1e-3, 0.0)

    attenuation = 0.2 / 4.343 / 1e3
    attenuation_bar = 0.2 / 4.343 / 1e3
    raman_gain_slope = 0.028 / 1e3 / 1e12
    nonlinear_coeff = 1.2e-3
    dispersion_coeff = 17e-12 / 1e-9 / 1e3
    dispersion_slope_val = 0.067e-12 / 1e-9 / 1e3 / 1e-9

    attenuation_i = attenuation * jnp.ones(num_channels)
    attenuation_bar_i = attenuation_bar * jnp.ones(num_channels)
    raman_gain_slope_i = raman_gain_slope * jnp.ones(num_channels)

    span_length = 100e3
    length = span_length * jnp.ones(num_spans)

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


def count_hlo_ops(fn, *args, **kwargs):
    """Count HLO operations in the compiled XLA graph."""
    lowered = jax.jit(fn).lower(*args, **kwargs)
    compiled = lowered.compile()
    hlo_text = compiled.as_text() or ""
    # Count lines that look like HLO ops (indented, containing '=')
    op_count = sum(1 for line in hlo_text.split("\n") if "=" in line and line.startswith("  "))
    return op_count, len(hlo_text)


# ---------------------------------------------------------------------------
# Section 1: Numerical verification (isolated calls)
# ---------------------------------------------------------------------------


def run_numerical_verification():
    """Quick numerical check that get_snr and get_snr_fused agree."""
    print("=" * 80)
    print("NUMERICAL VERIFICATION: get_snr vs get_snr_fused")
    print("=" * 80)

    for n_ch in [4, 100, 420]:
        p = make_test_params(n_ch)

        snr_orig, _ = get_snr(
            num_channels=n_ch,
            max_spans=p["max_spans"],
            ref_lambda=p["ref_lambda"],
            attenuation_i=p["attenuation_i"],
            attenuation_bar_i=p["attenuation_bar_i"],
            nonlinear_coeff=p["nonlinear_coeff_arr"],
            raman_gain_slope_i=p["raman_gain_slope_i"],
            dispersion_coeff=p["dispersion_coeff_arr"],
            dispersion_slope=p["dispersion_slope_arr"],
            length=p["length"],
            num_spans=p["num_spans"],
            ch_power_w_i=p["ch_power_w_i"],
            ch_centre_i=p["ch_centre_i"],
            ch_bandwidth_i=p["ch_bandwidth_i"],
            excess_kurtosis_i=p["excess_kurtosis_i"],
            amplifier_noise_figure=p["amplifier_noise_figure"],
            transceiver_snr=p["transceiver_snr"],
            uniform_spans=True,
        )

        snr_fused = get_snr_fused(
            ch_power_w_i=p["ch_power_w_i"],
            ch_centre_i=p["ch_centre_i"],
            ch_bandwidth_i=p["ch_bandwidth_i"],
            num_spans=p["num_spans"],
            span_length=p["span_length"],
            num_channels=n_ch,
            ref_lambda=p["ref_lambda"],
            attenuation=p["attenuation"],
            attenuation_bar=p["attenuation_bar"],
            nonlinear_coeff=p["nonlinear_coeff"],
            raman_gain_slope=p["raman_gain_slope"],
            dispersion_coeff=p["dispersion_coeff"],
            dispersion_slope=p["dispersion_slope"],
            amplifier_noise_figure=p["amplifier_noise_figure"],
            transceiver_snr=p["transceiver_snr"],
        )

        mask = jnp.abs(snr_orig) > 1e-5
        if jnp.any(mask):
            rel_diff = float(
                jnp.max(
                    jnp.abs(snr_orig[mask] - snr_fused[mask])
                    / jnp.maximum(jnp.abs(snr_orig[mask]), 1e-10)
                )
            )
        else:
            rel_diff = 0.0
        status = "PASS" if rel_diff < 1e-5 else "FAIL"
        print(f"  N={n_ch:>3}: max_rel_diff={rel_diff:.2e}  [{status}]")

    print()


# ---------------------------------------------------------------------------
# Section 2: XLA HLO op count comparison
# ---------------------------------------------------------------------------


def run_hlo_comparison():
    """Compare XLA graph complexity between get_snr and get_snr_fused."""
    print("=" * 80)
    print("XLA HLO GRAPH COMPLEXITY")
    print("=" * 80)
    print(f"  {'Function':<45} {'N_ch':>5} {'HLO ops':>10} {'HLO bytes':>12}")
    print("  " + "-" * 74)

    for n_ch in [4, 100]:
        p = make_test_params(n_ch)

        # get_snr
        fn_orig = partial(
            get_snr,
            num_channels=n_ch,
            max_spans=p["max_spans"],
            coherent=True,
            mod_format_correction=False,
            uniform_spans=True,
        )
        ops_o, size_o = count_hlo_ops(
            fn_orig,
            num_spans=p["num_spans"],
            ref_lambda=p["ref_lambda"],
            length=p["length"],
            attenuation_i=p["attenuation_i"],
            attenuation_bar_i=p["attenuation_bar_i"],
            nonlinear_coeff=p["nonlinear_coeff_arr"],
            raman_gain_slope_i=p["raman_gain_slope_i"],
            dispersion_coeff=p["dispersion_coeff_arr"],
            dispersion_slope=p["dispersion_slope_arr"],
            ch_power_w_i=p["ch_power_w_i"],
            ch_centre_i=p["ch_centre_i"],
            ch_bandwidth_i=p["ch_bandwidth_i"],
            excess_kurtosis_i=p["excess_kurtosis_i"],
            amplifier_noise_figure=p["amplifier_noise_figure"],
            transceiver_snr=p["transceiver_snr"],
        )

        # get_snr_fused
        fn_fused = partial(
            get_snr_fused,
            num_channels=n_ch,
            ref_lambda=p["ref_lambda"],
            attenuation=p["attenuation"],
            attenuation_bar=p["attenuation_bar"],
            nonlinear_coeff=p["nonlinear_coeff"],
            raman_gain_slope=p["raman_gain_slope"],
            dispersion_coeff=p["dispersion_coeff"],
            dispersion_slope=p["dispersion_slope"],
            roadm_loss=6.0,
            num_roadms=1,
            coherent=True,
        )
        ops_f, size_f = count_hlo_ops(
            fn_fused,
            ch_power_w_i=p["ch_power_w_i"],
            ch_centre_i=p["ch_centre_i"],
            ch_bandwidth_i=p["ch_bandwidth_i"],
            num_spans=p["num_spans"],
            span_length=p["span_length"],
            amplifier_noise_figure=p["amplifier_noise_figure"],
            transceiver_snr=p["transceiver_snr"],
        )

        print(f"  {'get_snr':<45} {n_ch:>5} {ops_o:>10} {size_o:>12}")
        print(f"  {'get_snr_fused':<45} {n_ch:>5} {ops_f:>10} {size_f:>12}")
        if ops_o > 0:
            print(f"  {'':45s} {'':>5} reduction: {(1 - ops_f / ops_o) * 100:.1f}%")

    print()


# ---------------------------------------------------------------------------
# Section 3: Inside-JIT scan benchmark (simulates training loop)
# ---------------------------------------------------------------------------


def run_scan_benchmark():
    """Measure per-iteration cost when called inside jax.lax.scan.

    This is the realistic scenario: during training, get_snr_link_array is
    called inside env.step() which is inside a scan over ROLLOUT_LENGTH steps.
    The entire scan is one JIT-compiled XLA graph, so there's no per-call
    kernel dispatch overhead.
    """
    print("=" * 80)
    print("INSIDE-JIT BENCHMARK (jax.lax.scan, amortised per-iteration cost)")
    print("=" * 80)

    scan_lengths = [10, 50, 200]

    for n_ch in [4, 100]:
        p = make_test_params(n_ch)

        # --- get_snr inside scan ---
        def snr_scan_body_orig(carry, _):
            ch_pow = carry
            snr, _ = get_snr(
                num_channels=n_ch,
                max_spans=p["max_spans"],
                ref_lambda=p["ref_lambda"],
                attenuation_i=p["attenuation_i"],
                attenuation_bar_i=p["attenuation_bar_i"],
                nonlinear_coeff=p["nonlinear_coeff_arr"],
                raman_gain_slope_i=p["raman_gain_slope_i"],
                dispersion_coeff=p["dispersion_coeff_arr"],
                dispersion_slope=p["dispersion_slope_arr"],
                length=p["length"],
                num_spans=p["num_spans"],
                ch_power_w_i=ch_pow,
                ch_centre_i=p["ch_centre_i"],
                ch_bandwidth_i=p["ch_bandwidth_i"],
                excess_kurtosis_i=p["excess_kurtosis_i"],
                amplifier_noise_figure=p["amplifier_noise_figure"],
                transceiver_snr=p["transceiver_snr"],
                uniform_spans=True,
                coherent=True,
                mod_format_correction=False,
            )
            # Slightly perturb power each iteration to prevent dead-code elimination
            new_pow = ch_pow + snr * 1e-15
            return new_pow, snr

        # --- get_snr_fused inside scan ---
        def snr_scan_body_fused(carry, _):
            ch_pow = carry
            snr = get_snr_fused(
                ch_power_w_i=ch_pow,
                ch_centre_i=p["ch_centre_i"],
                ch_bandwidth_i=p["ch_bandwidth_i"],
                num_spans=p["num_spans"],
                span_length=p["span_length"],
                num_channels=n_ch,
                ref_lambda=p["ref_lambda"],
                attenuation=p["attenuation"],
                attenuation_bar=p["attenuation_bar"],
                nonlinear_coeff=p["nonlinear_coeff"],
                raman_gain_slope=p["raman_gain_slope"],
                dispersion_coeff=p["dispersion_coeff"],
                dispersion_slope=p["dispersion_slope"],
                amplifier_noise_figure=p["amplifier_noise_figure"],
                transceiver_snr=p["transceiver_snr"],
                roadm_loss=6.0,
                num_roadms=1,
                coherent=True,
            )
            new_pow = ch_pow + snr * 1e-15
            return new_pow, snr

        print(f"\n  N_ch={n_ch}")
        print(
            f"  {'Function':<30} {'scan_len':>9} {'compile':>10} {'total(ms)':>10} {'per_iter(us)':>13}"
        )
        print("  " + "-" * 74)

        for scan_len in scan_lengths:
            for label, body_fn in [
                ("get_snr", snr_scan_body_orig),
                ("get_snr_fused", snr_scan_body_fused),
            ]:

                @jax.jit
                def run_scan(init_pow, body=body_fn, length=scan_len):
                    return jax.lax.scan(body, init_pow, None, length=length)

                ct, et, _ = _bench(run_scan, (p["ch_power_w_i"],), {}, n_warmup=2, n_iters=20)
                per_iter_us = et * 1e6 / scan_len
                print(
                    f"  {label:<30} {scan_len:>9} {ct:>10.3f} {et * 1e3:>10.3f} {per_iter_us:>13.1f}"
                )

    print()


# ---------------------------------------------------------------------------
# Section 4: Full env.step comparison (rsa_gn_model vs plain rsa)
# ---------------------------------------------------------------------------


def run_env_step_comparison():
    """Compare env.step() time with and without GN model to see its overhead."""
    print("=" * 80)
    print("ENV.STEP COMPARISON: rsa vs rsa_gn_model")
    print("=" * 80)

    from xlron.environments.make_env import make

    for n_ch in [4, 100]:
        print(f"\n  link_resources={n_ch}")

        results = {}
        for env_type in ["rsa", "rsa_gn_model"]:
            env_settings = dict(
                k=5,
                topology_name="nsfnet_deeprmsa_undirected",
                link_resources=n_ch,
                max_requests=10,
                values_bw=[100],
                incremental_loading=True,
                env_type=env_type,
                slot_size=12.5 if n_ch >= 100 else 25,
            )
            if env_type == "rsa_gn_model":
                env_settings.update(
                    mod_format_correction=False,
                    launch_power=0.0,
                    interband_gap=0,
                )

            try:
                env, env_params = make(env_settings, log_wrapper=False)
            except Exception as e:
                print(f"  {env_type:<20} SKIPPED ({e})")
                continue

            state = env.initial_state
            n_actions = env.num_actions(env_params)

            # Build a scan that runs N steps with random actions
            num_steps = 50

            @jax.jit
            def run_steps(key, state, env=env, params=env_params, _n_act=n_actions):
                def step_body(carry, _):
                    k, s = carry
                    k, subkey = jax.random.split(k)
                    action = jax.random.randint(subkey, (), 0, _n_act)
                    obs, new_s, reward, done, truncated, info = env.step_env(
                        subkey, s, action, params
                    )
                    # Squeeze scalar state fields that step_env may return as (1,)
                    new_s = new_s.replace(
                        current_time=jnp.squeeze(new_s.current_time),
                        holding_time=jnp.squeeze(new_s.holding_time),
                        arrival_time=jnp.squeeze(new_s.arrival_time),
                    )
                    return (k, new_s), jnp.squeeze(reward)

                return jax.lax.scan(step_body, (key, state), None, length=num_steps)

            key = jax.random.PRNGKey(0)

            try:
                # Compile
                t0 = time.perf_counter()
                result = run_steps(key, state)
                jax.block_until_ready(result)
                compile_time = time.perf_counter() - t0

                # Warmup
                for _ in range(3):
                    result = run_steps(key, state)
                    jax.block_until_ready(result)

                # Measure
                n_iters = 10
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    result = run_steps(key, state)
                    jax.block_until_ready(result)
                exec_time = (time.perf_counter() - t0) / n_iters
                per_step_us = exec_time * 1e6 / num_steps
            except Exception as e:
                print(f"  {env_type:<20} FAILED ({type(e).__name__}: {e})")
                continue

            results[env_type] = per_step_us
            print(
                f"  {env_type:<20} compile={compile_time:.2f}s  "
                f"total({num_steps} steps)={exec_time * 1e3:.2f}ms  "
                f"per_step={per_step_us:.1f}us"
            )

        if "rsa" in results and "rsa_gn_model" in results:
            overhead = results["rsa_gn_model"] - results["rsa"]
            ratio = results["rsa_gn_model"] / results["rsa"] if results["rsa"] > 0 else float("inf")
            print(f"  GN model overhead: {overhead:.1f}us/step ({ratio:.2f}x)")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    run_numerical_verification()
    run_hlo_comparison()
    run_scan_benchmark()

    try:
        run_env_step_comparison()
    except Exception as e:
        print(f"  env.step comparison skipped: {e}")
        print()


if __name__ == "__main__":
    main()
