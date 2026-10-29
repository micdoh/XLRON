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
import ctypes
import json
import os
import platform
import sysconfig
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from xlron.environments.env_funcs import held_lightpath_mask, lightpath_representatives
from xlron.heuristics.heuristics import FIRST_FIT_MASKS
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
    paths = np.asarray(params.path_link_array)
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
    centres = np.asarray(params.slot_centre_freq_array)
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


def measure(fn, arg, iterations, profile=False, warm_executions=3):
    jax.block_until_ready(arg)
    start = time.perf_counter()
    executable = jax.jit(fn).lower(arg).compile()
    compile_s = time.perf_counter() - start
    if dump_path := os.environ.get("XLRON_DUMP_HLO"):
        path = Path(dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(executable.as_text())
    print(f"Compiled in {compile_s:.3f}s; executing", flush=True)
    for _ in range(warm_executions):
        result = jax.block_until_ready(executable(arg))
    runtime = None
    if profile:
        library = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime/lib/libcudart.so.12"
        runtime = ctypes.CDLL(str(library) if library.is_file() else "libcudart.so.12")
        if runtime.cudaProfilerStart():
            raise RuntimeError("cudaProfilerStart failed")
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = jax.block_until_ready(executable(arg))
        samples.append(time.perf_counter() - start)
    if runtime is not None and runtime.cudaProfilerStop():
        raise RuntimeError("cudaProfilerStop failed")
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
    parser.add_argument(
        "--gn-mask-mode", choices=["ff_lf", "all_slots", "selected_slot"], default="ff_lf"
    )
    parser.add_argument("--gn-mask-batch-size", type=int, default=8, choices=range(1, 513))
    parser.add_argument("--gn-mask-env-batch-size", type=int, default=0, choices=range(33))
    parser.add_argument("--gn-kernel-backend", choices=["jax", "pallas"], default="jax")
    parser.add_argument(
        "--departure-policy", choices=["immediate", "safe_release"], default="safe_release"
    )
    parser.add_argument("--heuristic", help="Select actions with this heuristic (RMSA-GN)")
    parser.add_argument("--heuristic-recompute-mask", action="store_true")
    parser.add_argument("--gn-mask-screen-batch-size", type=int, default=512)
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument(
        "--env-batch-size",
        type=int,
        choices=(0, 1, 2, 4, 8, 16, 32),
        default=0,
        help="0 vmaps all envs; positive values stream independent env chunks",
    )
    parser.add_argument("--occupancy", type=float, default=0.25, help="Synthetic occupancy target")
    parser.add_argument(
        "--independent-streams",
        action="store_true",
        help="Use independent reset/traffic keys per environment, including real warmup",
    )
    parser.add_argument("--topology", default="nsfnet_deeprmsa_directed")
    parser.add_argument("--band-preference", help="Use the full named optical bands, e.g. C,L,S")
    parser.add_argument("--load", type=float, help="Override the default offered load")
    parser.add_argument(
        "--launch-power-type", choices=["fixed", "scaled"], help="Override the launch power type"
    )
    parser.add_argument("--coherent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mode", choices=["uniform", "corrected", "nonuniform"], default="uniform")
    parser.add_argument("--kernels", nargs="+", default=["link", "lightpath", "mask"])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warm-executions", type=int, default=3)
    parser.add_argument(
        "--profile", action="store_true", help="Capture timed executions with CUDA profiler API"
    )
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Build a physically admitted state with real steps instead of synthetic occupancy",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--state-input", type=Path, help="Reuse a complete admitted batched warmup state"
    )
    parser.add_argument(
        "--state-output", type=Path, help="Save the complete batched state and traffic keys"
    )
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
        "coherent": args.coherent,
        "mod_format_correction": args.mode == "corrected",
        "uniform_spans": args.mode != "nonuniform",
        "include_no_op": False,
        "continuous_operation": True,
        "TOTAL_TIMESTEPS": 10000,
    }
    if args.band_preference:
        config["band_preference"] = args.band_preference
    if args.load is not None:
        config["load"] = args.load
    if args.launch_power_type:
        config["launch_power_type"] = args.launch_power_type
    # The kernel backend changes how, not what, is computed: keep it out of the
    # workload key so saved states can be replayed with either backend.
    workload_key = json.dumps(config, sort_keys=True)
    config["gn_kernel_backend"] = args.gn_kernel_backend
    config["departure_policy"] = args.departure_policy
    env, params = make(config, log_wrapper=False)
    if hasattr(params, "gn_mask_mode"):
        params = params.replace(
            gn_mask_mode=args.gn_mask_mode,
            gn_mask_batch_size=args.gn_mask_batch_size,
            gn_mask_env_batch_size=args.gn_mask_env_batch_size,
            gn_mask_screen_batch_size=args.gn_mask_screen_batch_size,
        )
    key = jax.random.PRNGKey(3)
    _, empty = env.reset(key, params)

    def rollout(s, steps, rng=key):
        def step(carry, unused):
            rng, current = carry
            rng, step_key = jax.random.split(rng)
            if args.env_type == "rmsa_gn_model":
                # As select_action_eval: first-fit heuristics need only first fits.
                mask_params = params.replace(
                    gn_mask_first_fit=FIRST_FIT_MASKS.get(args.heuristic or "", "")
                )
                mask, full_mask, mods = env.action_mask(current, mask_params)
                current = current.replace(
                    link_slot_mask=mask, full_link_slot_mask=full_mask, mod_format_mask=mods
                )
                action = jnp.array([jnp.argmax(mask), 0], dtype=jnp.int32)
                if args.heuristic:
                    from xlron.heuristics import heuristics

                    # As select_action_eval: reuse the mask just computed, unless
                    # measuring the previous recompute-in-heuristic behaviour.
                    hp = params.replace(heuristic_mask_from_state=not args.heuristic_recompute_mask)
                    action = getattr(heuristics, args.heuristic)(current, hp)
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
    warmup_timing = {}
    if args.state_input:
        _, template = jax.vmap(lambda k: env.reset(k, params))(reset_keys)
        leaves, definition = jax.tree.flatten(template)
        with np.load(args.state_input) as archive:
            if str(archive["workload_key"]) != workload_key:
                raise ValueError("Saved state belongs to a different workload configuration")
            restored = []
            for index, leaf in enumerate(leaves):
                value = archive[f"state_{index}"]
                if value.shape[1:] == leaf.shape[1:] and value.shape[0] >= leaf.shape[0]:
                    value = value[: leaf.shape[0]]
                if value.shape != leaf.shape or value.dtype != leaf.dtype:
                    raise ValueError(f"Saved state leaf {index} has a different shape or dtype")
                restored.append(jnp.asarray(value))
            batched = jax.tree.unflatten(definition, restored)
            env_keys = jnp.asarray(archive["env_keys"][: args.envs])
        placed = int(batched.accepted_services[0])
        print(f"Loaded identical warmup state from {args.state_input}", flush=True)
    elif args.independent_streams and args.warmup_steps:
        _, batched = jax.vmap(lambda k: env.reset(k, params))(reset_keys)
        print(
            f"Compiling {args.warmup_steps}-request warmup for {params.link_resources} slots / {args.envs} envs",
            flush=True,
        )
        start = time.perf_counter()
        warmup = (
            jax.jit(jax.vmap(lambda s, k: rollout(s, args.warmup_steps, k)))
            .lower(batched, env_keys)
            .compile()
        )
        warmup_timing["trace_compile_s"] = time.perf_counter() - start
        print(
            f"Warmup compile/load: {warmup_timing['trace_compile_s']:.3f}s; executing warmup",
            flush=True,
        )
        start = time.perf_counter()
        (env_keys, batched), _ = jax.block_until_ready(warmup(batched, env_keys))
        warmup_timing["execute_s"] = time.perf_counter() - start
        print(f"Warmup execution: {warmup_timing['execute_s']:.3f}s", flush=True)
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
    if args.state_output:
        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.state_output,
            workload_key=workload_key,
            env_keys=np.asarray(env_keys),
            **{f"state_{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(batched))},
        )
        print(f"Saved complete warmup state to {args.state_output}", flush=True)
    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "platform": platform.platform(),
        "links": params.num_links,
        "resolved_slots": params.link_resources,
        "paths": params.path_link_array.shape[0],
        "modulations": params.modulations_array.shape[0]
        if hasattr(params, "modulations_array")
        else None,
        "lightpaths": placed,
        "occupancy": float(jnp.mean(batched.channel_power_array > 0)),
        "initial_accepted_services": np.asarray(batched.accepted_services).tolist(),
        "warmup_timing": warmup_timing,
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

            def independent_rollouts(sk):
                one = lambda pair: rollout(pair[0], args.steps, pair[1])
                if args.env_batch_size:
                    return jax.lax.map(one, sk, batch_size=args.env_batch_size)
                return jax.vmap(lambda s, k: rollout(s, args.steps, k))(*sk)

            metrics, result = measure(
                independent_rollouts,
                (batched, env_keys),
                args.iterations,
                args.profile,
                args.warm_executions,
            )
        else:
            batch_fn = (
                (lambda states, fn=fn: jax.lax.map(fn, states, batch_size=args.env_batch_size))
                if args.env_batch_size
                else jax.vmap(fn)
            )
            metrics, result = measure(
                batch_fn, batched, args.iterations, args.profile, args.warm_executions
            )
        report["metrics"][name] = metrics
        if name == "rollout":
            snapshots["rollout_rewards"] = np.asarray(result[1])
            final = result[0][1]
            snapshots["rollout_occupancy"] = np.asarray(final.link_slot_array)
            metrics["steps_per_s"] = args.envs * args.steps * 1000 / metrics["median_ms"]
            metrics["accepted_services"] = np.asarray(final.accepted_services).tolist()
            if args.env_type == "rmsa_gn_model":
                held = jax.vmap(
                    lambda st: jnp.sum(
                        held_lightpath_mask(st, params) & lightpath_representatives(st, params)
                    )
                )(final)
                metrics["held_lightpaths"] = np.asarray(held).tolist()
                # Under safe_release the link-SNR cache should equal a full re-solve.
                fresh = jax.jit(jax.vmap(lambda st: get_snr_link_array(st, params)))(final)
                cached = np.asarray(final.link_snr_array)
                live = np.asarray(final.channel_power_array) > 0
                metrics["link_snr_cache_max_abs_error_live"] = float(
                    np.max(np.abs(cached - np.asarray(fresh))[live], initial=0.0)
                )
                metrics["link_snr_cache_bitwise_live"] = bool(
                    np.array_equal(cached[live], np.asarray(fresh)[live])
                )
            metrics["distinct_final_requests"] = len(
                np.unique(np.asarray(final.request_array), axis=0)
            )
        else:
            if name == "mask":
                metrics["valid_actions_per_env"] = np.asarray(result[0]).sum(axis=-1).tolist()
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
