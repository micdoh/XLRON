"""Reproducible GN performance/accuracy measurements, without training or logging.

Run from a checkout, e.g.:
  CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --no-sync python -m experimental.benchmarks.gn_scaling \
    --slots 420 --k 5 --output /tmp/gn-420.json --snapshot /tmp/gn-420.npz

Use the same script, arguments, JAX version and device in both checkouts. Compilation
is timed separately; every measurement synchronizes the device. Memory is XLA's
compiled buffer estimate (not process peak VRAM). Synthetic loaded states contain
whole, non-overlapping lightpaths; rollout measurements exercise real env steps.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from xlron.environments.env_funcs import (
    calculate_throughput_from_active_lightpaths,
    get_lightpath_snr,
    get_snr_link_array,
    get_snr_link_array_fused,
)
from xlron.environments.make_env import make


def loaded_state(state, params, seed=7, target_occupancy=0.25):
    """Install coherent synthetic lightpaths, identically on each traversed link."""
    if not 0 < target_occupancy <= 1:
        raise ValueError("Synthetic occupancy target must be in (0, 1]")
    rng = np.random.default_rng(seed)
    paths = np.asarray(params.path_link_array.val)
    if params.pack_path_bits:
        paths = np.unpackbits(paths, axis=1)[:, : params.num_links]
    arrays = {
        name: np.array(getattr(state, name))
        for name in (
            "link_slot_array",
            "link_slot_departure_array",
            "path_index_array",
            "channel_power_array",
            "channel_centre_bw_array",
            "channel_centre_freq_array",
            "modulation_format_index_array",
        )
        if hasattr(state, name)
    }
    centres = np.asarray(params.slot_centre_freq_array.val)
    placed = 0
    registry = []
    for _ in range(params.link_resources * 15):
        pid = int(rng.integers(len(paths) - 1))
        links = np.flatnonzero(paths[pid])
        width = int(rng.integers(1, min(8, params.link_resources) + 1))
        start = int(rng.integers(params.link_resources - width + 1))
        ix = np.ix_(links, np.arange(start, start + width))
        if len(links) == 0 or np.any(arrays["link_slot_array"][ix] != 0):
            continue
        arrays["link_slot_array"][ix] = -1
        arrays["link_slot_departure_array"][ix] = 10000
        arrays["path_index_array"][ix] = pid
        arrays["channel_power_array"][ix] = 1e-4
        arrays["channel_centre_bw_array"][ix] = params.slot_size
        arrays["channel_centre_freq_array"][ix] = (centres[start] + centres[start + width - 1]) / 2
        if "modulation_format_index_array" in arrays:
            arrays["modulation_format_index_array"][ix] = 0
        registry.append((pid, start, width))
        placed += 1
        if np.mean(arrays["channel_power_array"] > 0) >= target_occupancy:
            break
    if hasattr(state, "active_lightpaths_array"):
        records = np.full_like(np.asarray(state.active_lightpaths_array), -1)
        count = min(len(registry), len(records))
        records[:count] = registry[:count]
        arrays["active_lightpaths_array"] = records
        departures = np.zeros_like(np.asarray(state.active_lightpaths_array_departure))
        departures[:count] = 10000
        arrays["active_lightpaths_array_departure"] = departures
    return state.replace(**{n: jnp.asarray(a) for n, a in arrays.items()}), placed


def measure(fn, arg, iterations):
    jax.block_until_ready(arg)
    start = time.perf_counter()
    executable = jax.jit(fn).lower(arg).compile()
    compile_s = time.perf_counter() - start
    for _ in range(3):
        result = jax.block_until_ready(executable(arg))
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = jax.block_until_ready(executable(arg))
        samples.append(time.perf_counter() - start)
    mem = executable.memory_analysis()
    metrics = {
        "compile_s": compile_s,
        "median_ms": 1000 * float(np.median(samples)),
        "min_ms": 1000 * min(samples),
        "argument_bytes": mem.argument_size_in_bytes,
        "output_bytes": mem.output_size_in_bytes,
        "temp_bytes": mem.temp_size_in_bytes,
        "alias_bytes": mem.alias_size_in_bytes,
    }
    return metrics, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=100)
    parser.add_argument(
        "--env-type", choices=["rmsa_gn_model", "rsa_gn_model"], default="rmsa_gn_model"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--occupancy", type=float, default=0.25, help="Synthetic occupancy target")
    parser.add_argument(
        "--independent-streams",
        action="store_true",
        help="Use independent reset/traffic keys per environment, including real warmup",
    )
    parser.add_argument("--topology", default="nsfnet_deeprmsa_directed")
    parser.add_argument("--mode", choices=["uniform", "corrected", "nonuniform"], default="uniform")
    parser.add_argument("--kernels", nargs="+", default=["link", "lightpath", "mask"])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Build a physically admitted state with real steps instead of synthetic occupancy",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--reference", type=Path, help="Compare outputs with a baseline .npz")
    parser.add_argument(
        "--matmul-precision", default="highest", choices=["default", "high", "highest"]
    )
    args = parser.parse_args()
    # Compare FP32 accumulation, not GPU TF32 rounding in the all-path matmul.
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    config = {
        "env_type": args.env_type,
        "topology_name": args.topology,
        "link_resources": args.slots,
        "k": args.k,
        "values_bw": [100, 200, 400],
        "max_requests": 10000,
        "slot_size": 12.5,
        "guardband": 0,
        "max_power_per_fibre": 21.0,
        "power_per_channel": -10.0,
        "coherent": False,
        "mod_format_correction": args.mode == "corrected",
        "uniform_spans": args.mode != "nonuniform",
        "include_no_op": False,
        "continuous_operation": True,
        "TOTAL_TIMESTEPS": 10000,
    }
    env, params = make(config, log_wrapper=False)
    key = jax.random.PRNGKey(3)
    _, empty = env.reset(key, params)

    def rollout(s, steps, rng=key):
        def step(carry, unused):
            rng, current = carry
            rng, step_key = jax.random.split(rng)
            if args.env_type == "rmsa_gn_model":
                mask, full_mask, mods = env.action_mask(current, params)
                current = current.replace(
                    link_slot_mask=mask, full_link_slot_mask=full_mask, mod_format_mask=mods
                )
                action = jnp.array([jnp.argmax(mask), 0], dtype=jnp.int32)
            else:
                mask, full_mask = env.action_mask(current, params)
                current = current.replace(link_slot_mask=mask, full_link_slot_mask=full_mask)
                action = jnp.argmax(mask)
            _, current, reward, *_ = env.step(step_key, current, action, params)
            return (rng, current), reward

        return jax.lax.scan(step, (rng, s), None, length=steps)

    link_fn = get_snr_link_array_fused if args.mode == "uniform" else get_snr_link_array
    keys = jax.random.split(key, args.envs * 2).reshape(args.envs, 2, 2)
    reset_keys, env_keys = keys[:, 0], keys[:, 1]
    if args.independent_streams and args.warmup_steps:
        _, batched = jax.vmap(lambda k: env.reset(k, params))(reset_keys)
        (env_keys, batched), _ = jax.jit(jax.vmap(lambda s, k: rollout(s, args.warmup_steps, k)))(
            batched, env_keys
        )
        placed = int(batched.accepted_services[0])
    else:
        if args.warmup_steps:
            (_, state), _ = jax.jit(lambda s: rollout(s, args.warmup_steps))(empty)
            placed = int(state.accepted_services)
        else:
            state, placed = loaded_state(empty, params, target_occupancy=args.occupancy)
        batched = jax.tree.map(lambda x: jnp.stack([x] * args.envs), state)
    batched = batched.replace(link_snr_array=jax.vmap(lambda s: link_fn(s, params))(batched))
    jax.block_until_ready(batched)
    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "platform": platform.platform(),
        "links": params.num_links,
        "paths": params.path_link_array.val.shape[0],
        "modulations": params.modulations_array.val.shape[0]
        if hasattr(params, "modulations_array")
        else None,
        "lightpaths": placed,
        "occupancy": float(jnp.mean(batched.channel_power_array > 0)),
        "initial_accepted_services": np.asarray(batched.accepted_services).tolist(),
        "metrics": {},
    }
    snapshots = {}
    for name in args.kernels:
        print(f"Benchmarking {name}, slots={args.slots}, envs={args.envs}", flush=True)
        if name == "link":
            fn = lambda s: link_fn(s, params)
        elif name == "lightpath":
            fn = lambda s: get_lightpath_snr(s, params)
        elif name == "mask":
            fn = lambda s: env.action_mask(s, params)
        elif name == "throughput":
            fn = lambda s: calculate_throughput_from_active_lightpaths(s, params)
        elif name == "rollout":
            fn = lambda s: rollout(s, args.steps)
        else:
            raise ValueError(name)
        if name == "rollout" and args.independent_streams:
            metrics, result = measure(
                lambda sk: jax.vmap(lambda s, k: rollout(s, args.steps, k))(*sk),
                (batched, env_keys),
                args.iterations,
            )
        else:
            metrics, result = measure(jax.vmap(fn), batched, args.iterations)
        report["metrics"][name] = metrics
        if name == "rollout":
            snapshots["rollout_rewards"] = np.asarray(result[1])
            final = result[0][1]
            snapshots["rollout_occupancy"] = np.asarray(final.link_slot_array)
            metrics["steps_per_s"] = args.envs * args.steps * 1000 / metrics["median_ms"]
            metrics["accepted_services"] = np.asarray(final.accepted_services).tolist()
            metrics["distinct_final_requests"] = len(
                np.unique(np.asarray(final.request_array), axis=0)
            )
        else:
            for i, leaf in enumerate(jax.tree.leaves(result)):
                snapshots[f"{name}_{i}"] = np.asarray(leaf)
        print(json.dumps({name: metrics}), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.snapshot, **snapshots)
        del result
        jax.clear_caches()
    if args.reference:
        report["accuracy"] = {}
        with np.load(args.reference) as reference:
            for name, actual in snapshots.items():
                expected = reference[name]
                if name.startswith(("mask_", "rollout_")):
                    np.testing.assert_array_equal(actual, expected, err_msg=name)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4, err_msg=name)
                report["accuracy"][name] = {
                    "exact": bool(np.array_equal(actual, expected)),
                    "max_abs_error": float(np.max(np.abs(actual.astype(float) - expected))),
                }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.snapshot, **snapshots)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
