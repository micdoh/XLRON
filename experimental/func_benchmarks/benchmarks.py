"""Benchmark script for XLRON environment functions.
Usage:
    python -m xlron.environments.benchmarks --env_type=rsa --topology_name=nsfnet_chen --link_resources=100
    # With specific settings:
    python -m xlron.environments.benchmarks --env_type=rmsa --topology_name=4node --k_paths=5
"""

import sys
import time

import jax
import jax.numpy as jnp
from absl import app, flags

from xlron import dtype_config
from xlron.environments.env_funcs import (
    get_path_index_array,
    get_paths,
    get_paths_se,
    mask_slots,
    read_rsa_request,
    required_slots,
)
from xlron.environments.make_env import process_config
from xlron.parameter_flags import *  # noqa: F403,F401
from xlron.train.train_utils import experiment_data_setup

FLAGS = flags.FLAGS


def benchmark_cpu(state, params, request_array, N=100):
    """CPU benchmarks with proper warmup."""
    print("=== CPU Benchmarks ===\n")

    # Warmup - run multiple times to ensure JIT compilation is done
    print("Warming up...")
    for _ in range(5):
        mask, full_mask = mask_slots(state, params)
        mask.block_until_ready()
        nodes_sd, _ = read_rsa_request(state.request_array)
        paths = get_paths(params, nodes_sd)
        paths.block_until_ready()
        paths_se = get_paths_se(params, nodes_sd)
        paths_se.block_until_ready()

    print(f"Running {N} iterations per benchmark...\n")

    nodes_sd, _ = read_rsa_request(request_array)

    t0 = time.time()
    for _ in range(N):
        nodes_sd, _ = read_rsa_request(request_array)
        paths = get_paths(params, nodes_sd)
        paths.block_until_ready()
    print(f"get_paths: {(time.time() - t0) / N * 1000:.2f}ms")

    t0 = time.time()
    for _ in range(N):
        paths_se = get_paths_se(params, nodes_sd)
        paths_se.block_until_ready()
    print(f"get_paths_se: {(time.time() - t0) / N * 1000:.2f}ms")

    t0 = time.time()
    for _ in range(N):
        mask, full_mask = mask_slots(state, params)
        mask.block_until_ready()
    print(f"full mask_slots: {(time.time() - t0) / N * 1000:.2f}ms")


def benchmark_gpu(state, params, request_array, N=100):
    """GPU benchmarks with proper synchronization."""
    print("\n=== GPU Benchmarks ===\n")

    # Force compilation with multiple warmup runs
    print("Warming up and compiling...")
    for _ in range(5):
        mask, full_mask = mask_slots(state, params)
        mask.block_until_ready()
        nodes_sd, _ = read_rsa_request(request_array)
        _ = get_paths(params, nodes_sd).block_until_ready()
        _ = get_paths_se(params, nodes_sd).block_until_ready()

    nodes_sd, _ = read_rsa_request(request_array)

    print(f"Running {N} iterations per benchmark...\n")

    # Benchmark get_paths
    # Sync before timing starts
    jax.block_until_ready(state.link_slot_array)
    t0 = time.perf_counter()
    for _ in range(N):
        paths = get_paths(params, nodes_sd)
    paths.block_until_ready()  # Single sync after all iterations
    print(f"get_paths: {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # Benchmark get_paths_se
    jax.block_until_ready(paths)
    t0 = time.perf_counter()
    for _ in range(N):
        paths_se = get_paths_se(params, nodes_sd)
    paths_se.block_until_ready()
    print(f"get_paths_se: {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # Benchmark full mask_slots
    jax.block_until_ready(paths_se)
    t0 = time.perf_counter()
    for _ in range(N):
        mask, full_mask = mask_slots(state, params)
    mask.block_until_ready()
    print(f"full mask_slots: {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # Detailed breakdown
    print("\n--- mask_slots breakdown ---\n")

    # 1. Path index computation
    jax.block_until_ready(mask)
    t0 = time.perf_counter()
    for _ in range(N):
        path_indices = get_path_index_array(params, nodes_sd).astype(jnp.int32)
    path_indices.block_until_ready()
    print(f"  get_path_index_array: {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # 2. Path retrieval (take)
    jax.block_until_ready(path_indices)
    t0 = time.perf_counter()
    for _ in range(N):
        paths = jnp.take(params.path_link_array.val, path_indices, axis=0)
    paths.block_until_ready()
    print(f"  jnp.take (paths): {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # 3. SE retrieval
    jax.block_until_ready(paths)
    t0 = time.perf_counter()
    for _ in range(N):
        paths_se = jnp.take(params.path_se_array.val, path_indices, axis=0)
    paths_se.block_until_ready()
    print(f"  jnp.take (SE): {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # 4. Occupied computation
    jax.block_until_ready(paths_se)
    t0 = time.perf_counter()
    for _ in range(N):
        masked_slots = jnp.where(paths[:, :, None], jnp.abs(state.link_slot_array), 0)
        occupied = (jnp.max(masked_slots, axis=1) != 0).astype(jnp.float32)
    occupied.block_until_ready()
    print(f"  occupied calc: {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # 5. Cumsum + padding
    jax.block_until_ready(occupied)
    t0 = time.perf_counter()
    for _ in range(N):
        padded = jnp.concatenate(
            [occupied, jnp.ones((params.k_paths, params.max_slots - 1), dtype=jnp.float32)], axis=1
        )
        cumsum = jnp.cumsum(padded, axis=1)
        cumsum = jnp.concatenate(
            [jnp.zeros((params.k_paths, 1), dtype=jnp.float32), cumsum], axis=1
        )
    cumsum.block_until_ready()
    print(f"  cumsum + padding: {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # 6. Window sum computation (if using cumsum approach)
    all_se_values = params.unique_se_values.val
    slot_indices = jnp.arange(params.link_resources)
    _, requested_datarate = read_rsa_request(request_array)

    # Precompute req_slots
    all_req_slots = jax.vmap(
        lambda se: jnp.ceil(requested_datarate / (se * params.slot_size) + params.guardband).astype(
            jnp.int32
        )
    )(all_se_values)
    all_req_slots.block_until_ready()

    jax.block_until_ready(cumsum)
    t0 = time.perf_counter()
    for _ in range(N):
        end_indices = slot_indices[None, :] + all_req_slots[:, None]
        cumsum_at_end = cumsum[:, end_indices]
        cumsum_at_start = cumsum[:, slot_indices][:, None, :]  # (k, 1, link_resources)
        # window_sums = cumsum_at_end - cumsum_at_start  # broadcasts correctly
        window_sums = cumsum_at_end - cumsum_at_start[:, None, :]
        all_masks = (window_sums == 0).astype(jnp.float32)
    all_masks.block_until_ready()
    print(f"  window sums + masks: {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # 7. Final selection
    jax.block_until_ready(all_masks)
    t0 = time.perf_counter()
    for _ in range(N):
        mod_indices = jnp.argmax(paths_se[:, None] == all_se_values[None, :], axis=1)
        final_masks = all_masks[jnp.arange(params.k_paths), mod_indices, :]
        link_slot_mask = final_masks.reshape(-1)
    link_slot_mask.block_until_ready()
    print(f"  final selection: {(time.perf_counter() - t0) / N * 1000:.3f}ms")


def benchmark_gpu_jitted(state, params, request_array, N=100):
    """GPU benchmarks with JIT-compiled components."""
    print("\n=== GPU Benchmarks (JIT-compiled components) ===\n")

    nodes_sd, requested_datarate = read_rsa_request(request_array)

    # Define JIT-compiled component functions
    @jax.jit
    def bench_path_index(nodes):
        return get_path_index_array(params, nodes).astype(jnp.int32)

    @jax.jit
    def bench_take_paths(indices):
        return jnp.take(params.path_link_array.val, indices, axis=0)

    @jax.jit
    def bench_take_se(indices):
        return jnp.take(params.path_se_array.val, indices, axis=0)

    @jax.jit
    def bench_occupied(paths, link_slot_array):
        masked_slots = jnp.where(paths[:, :, None], jnp.abs(link_slot_array), 0)
        return (jnp.max(masked_slots, axis=1) != 0).astype(jnp.float32)

    @jax.jit
    def bench_cumsum(occupied):
        padded = jnp.concatenate(
            [occupied, jnp.ones((params.k_paths, params.max_slots - 1), dtype=jnp.float32)], axis=1
        )
        cumsum = jnp.cumsum(padded, axis=1)
        return jnp.concatenate([jnp.zeros((params.k_paths, 1), dtype=jnp.float32), cumsum], axis=1)

    @jax.jit
    def bench_window_sums(cumsum, all_req_slots):
        slot_indices = jnp.arange(params.link_resources)
        end_indices = slot_indices[None, :] + all_req_slots[:, None]
        cumsum_at_end = cumsum[:, end_indices]
        cumsum_at_start = cumsum[:, slot_indices][:, None, :]
        window_sums = cumsum_at_end - cumsum_at_start
        return (window_sums == 0).astype(jnp.float32)

    @jax.jit
    def bench_final_select(all_masks, paths_se, all_se_values):
        mod_indices = jnp.argmax(paths_se[:, None] == all_se_values[None, :], axis=1)
        final_masks = all_masks[jnp.arange(params.k_paths), mod_indices, :]
        return final_masks.reshape(-1)

    # Warmup all JIT functions
    print("Warming up JIT-compiled components...")
    path_indices = bench_path_index(nodes_sd).block_until_ready()
    paths = bench_take_paths(path_indices).block_until_ready()
    paths_se = bench_take_se(path_indices).block_until_ready()
    occupied = bench_occupied(paths, state.link_slot_array).block_until_ready()
    cumsum = bench_cumsum(occupied).block_until_ready()

    all_se_values = params.unique_se_values.val
    all_req_slots = jax.vmap(
        lambda se: required_slots(
            requested_datarate,
            se,
            params.slot_size,
            guardband=params.guardband,
            temperature=params.temperature,
        )
    )(all_se_values).astype(jnp.int32)
    all_req_slots.block_until_ready()

    all_masks = bench_window_sums(cumsum, all_req_slots).block_until_ready()
    _ = bench_final_select(all_masks, paths_se, all_se_values).block_until_ready()

    # Also warmup full mask_slots
    for _ in range(3):
        mask, full_mask = mask_slots(state, params)
        mask.block_until_ready()

    print(f"Running {N} iterations per benchmark...\n")

    # Benchmark full mask_slots first
    t0 = time.perf_counter()
    for _ in range(N):
        mask, full_mask = mask_slots(state, params)
    mask.block_until_ready()
    full_time = (time.perf_counter() - t0) / N * 1000
    print(f"full mask_slots: {full_time:.3f}ms")

    print("\n--- Component breakdown (JIT-compiled) ---\n")

    # 1. Path index
    t0 = time.perf_counter()
    for _ in range(N):
        path_indices = bench_path_index(nodes_sd)
    path_indices.block_until_ready()
    t1 = (time.perf_counter() - t0) / N * 1000
    print(f"  get_path_index_array: {t1:.3f}ms")

    # 2. Take paths
    t0 = time.perf_counter()
    for _ in range(N):
        paths = bench_take_paths(path_indices)
    paths.block_until_ready()
    t2 = (time.perf_counter() - t0) / N * 1000
    print(f"  jnp.take (paths): {t2:.3f}ms")

    # 3. Take SE
    t0 = time.perf_counter()
    for _ in range(N):
        paths_se = bench_take_se(path_indices)
    paths_se.block_until_ready()
    t3 = (time.perf_counter() - t0) / N * 1000
    print(f"  jnp.take (SE): {t3:.3f}ms")

    # 4. Occupied
    t0 = time.perf_counter()
    for _ in range(N):
        occupied = bench_occupied(paths, state.link_slot_array)
    occupied.block_until_ready()
    t4 = (time.perf_counter() - t0) / N * 1000
    print(f"  occupied calc: {t4:.3f}ms")

    # 5. Cumsum
    t0 = time.perf_counter()
    for _ in range(N):
        cumsum = bench_cumsum(occupied)
    cumsum.block_until_ready()
    t5 = (time.perf_counter() - t0) / N * 1000
    print(f"  cumsum + padding: {t5:.3f}ms")

    # 6. Window sums
    t0 = time.perf_counter()
    for _ in range(N):
        all_masks = bench_window_sums(cumsum, all_req_slots)
    all_masks.block_until_ready()
    t6 = (time.perf_counter() - t0) / N * 1000
    print(f"  window sums + masks: {t6:.3f}ms")

    # 7. Final select
    t0 = time.perf_counter()
    for _ in range(N):
        link_slot_mask = bench_final_select(all_masks, paths_se, all_se_values)
    link_slot_mask.block_until_ready()
    t7 = (time.perf_counter() - t0) / N * 1000
    print(f"  final selection: {t7:.3f}ms")

    total_components = t1 + t2 + t3 + t4 + t5 + t6 + t7
    print(f"\n  Sum of components: {total_components:.3f}ms")
    print(f"  Full function: {full_time:.3f}ms")
    print(
        f"  XLA fusion savings: {total_components - full_time:.3f}ms ({(1 - full_time / total_components) * 100:.1f}%)"
    )


def benchmark_gpu_detailed(state, params, request_array, N=100):
    """Detailed breakdown including all operations."""
    print("\n=== Detailed GPU Benchmarks ===\n")

    nodes_sd, requested_datarate = read_rsa_request(request_array)

    # Precompute for benchmarks
    path_indices = get_path_index_array(params, nodes_sd).astype(jnp.int32)
    paths = jnp.take(params.path_link_array.val, path_indices, axis=0)
    paths_se = jnp.take(params.path_se_array.val, path_indices, axis=0)

    all_se_values = params.unique_se_values.val
    all_req_slots = jax.vmap(
        lambda se: required_slots(
            requested_datarate,
            se,
            params.slot_size,
            guardband=params.guardband,
            temperature=params.temperature,
        )
    )(all_se_values).astype(jnp.int32)

    masked_slots = jnp.where(paths[:, :, None], jnp.abs(state.link_slot_array), 0)
    occupied = (jnp.max(masked_slots, axis=1) != 0).astype(jnp.float32)

    padded = jnp.concatenate(
        [occupied, jnp.ones((params.k_paths, params.max_slots - 1), dtype=jnp.float32)], axis=1
    )
    cumsum = jnp.concatenate(
        [jnp.zeros((params.k_paths, 1), dtype=jnp.float32), jnp.cumsum(padded, axis=1)], axis=1
    )

    slot_indices = jnp.arange(params.link_resources)
    end_indices = slot_indices[None, :] + all_req_slots[:, None]
    cumsum_at_end = cumsum[:, end_indices]
    cumsum_at_start = cumsum[:, slot_indices][:, None, :]
    all_masks = ((cumsum_at_end - cumsum_at_start) == 0).astype(jnp.float32)

    mod_indices = jnp.argmax(paths_se[:, None] == all_se_values[None, :], axis=1)

    # Benchmark different final selection approaches
    print("--- Final selection variants ---\n")

    # Current approach: advanced indexing
    @jax.jit
    def select_v1(all_masks, mod_indices):
        final_masks = all_masks[jnp.arange(params.k_paths), mod_indices, :]
        return final_masks.reshape(-1)

    # Alternative: take_along_axis
    @jax.jit
    def select_v2(all_masks, mod_indices):
        # all_masks: (k, num_mods, link_resources)
        all_masks_t = jnp.moveaxis(all_masks, 0, 1)  # (num_mods, k, link_resources)
        all_masks_t2 = jnp.moveaxis(all_masks_t, 0, 1)  # back to (k, num_mods, link_resources)
        final_masks = jnp.take_along_axis(all_masks_t2, mod_indices[:, None, None], axis=1)[:, 0, :]
        return final_masks.reshape(-1)

    # Alternative: einsum gather
    @jax.jit
    def select_v3(all_masks, mod_indices):
        # One-hot encode mod_indices: (k, num_mods)
        num_mods = all_masks.shape[1]
        one_hot = (jnp.arange(num_mods)[None, :] == mod_indices[:, None]).astype(jnp.float32)
        # all_masks: (k, num_mods, link_resources)
        # one_hot: (k, num_mods)
        # Result: (k, link_resources)
        final_masks = jnp.einsum("kmr,km->kr", all_masks, one_hot)
        return final_masks.reshape(-1)

    # Warmup
    _ = select_v1(all_masks, mod_indices).block_until_ready()
    _ = select_v2(all_masks, mod_indices).block_until_ready()
    _ = select_v3(all_masks, mod_indices).block_until_ready()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(N):
        r1 = select_v1(all_masks, mod_indices)
    r1.block_until_ready()
    print(f"  v1 (advanced indexing): {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    t0 = time.perf_counter()
    for _ in range(N):
        r2 = select_v2(all_masks, mod_indices)
    r2.block_until_ready()
    print(f"  v2 (take_along_axis): {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    t0 = time.perf_counter()
    for _ in range(N):
        r3 = select_v3(all_masks, mod_indices)
    r3.block_until_ready()
    print(f"  v3 (einsum one-hot): {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # Verify correctness
    assert jnp.allclose(r1, r2), "v2 mismatch!"
    assert jnp.allclose(r1, r3), "v3 mismatch!"
    print("\n  All variants produce identical results ✓")

    # Benchmark state.replace overhead
    print("\n--- state.replace overhead ---\n")

    link_slot_mask = r1
    if params.include_no_op:
        link_slot_mask_with_noop = jnp.concatenate([link_slot_mask, jnp.ones(1)])
    else:
        link_slot_mask_with_noop = link_slot_mask

    @jax.jit
    def bench_replace(state, mask):
        return state.replace(link_slot_mask=mask)

    _ = bench_replace(state, link_slot_mask_with_noop).link_slot_mask.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(N):
        state = bench_replace(state, link_slot_mask_with_noop)
    state.link_slot_mask.block_until_ready()
    print(f"  state.replace: {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # Benchmark concatenate for no_op
    @jax.jit
    def bench_concat(mask):
        return jnp.concatenate([mask, jnp.ones(1)])

    _ = bench_concat(link_slot_mask).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(N):
        mask_with_noop = bench_concat(link_slot_mask)
    mask_with_noop.block_until_ready()
    print(f"  jnp.concatenate (no_op): {(time.perf_counter() - t0) / N * 1000:.3f}ms")


def benchmark_original_vs_new(state, params, request_array, N=100):
    """Compare original mask_slots implementation vs optimized."""
    print("\n=== Original vs Optimized Comparison ===\n")

    # You'll need to save the original implementation somewhere
    # For now, let's just check if there's room for improvement

    nodes_sd, requested_datarate = read_rsa_request(request_array)
    paths = get_paths(params, nodes_sd)

    # Profile the expensive broadcast operation in isolation
    @jax.jit
    def bench_broadcast_max(paths, link_slot_array):
        masked_slots = jnp.where(paths[:, :, None], jnp.abs(link_slot_array), 0)
        return jnp.max(masked_slots, axis=1)

    # Alternative: einsum approach
    @jax.jit
    def bench_einsum_max(paths, link_slot_array):
        # paths: (k, num_links) binary
        # link_slot_array: (num_links, link_resources)
        # Want: for each (k, slot), max over links where path[k,l]=1
        abs_slots = jnp.abs(link_slot_array)
        # Multiply paths by slots, take max
        # This creates (k, num_links, link_resources), then max over axis 1
        return jnp.max(paths[:, :, None] * abs_slots[None, :, :], axis=1)

    # Alternative: matmul for "any non-zero" check
    @jax.jit
    def bench_matmul_occupied(paths, link_slot_array):
        # If we only need binary occupied (not the actual values)
        # paths: (k, num_links), slots_occupied: (num_links, link_resources)
        slots_occupied = (link_slot_array != 0).astype(jnp.float32)
        # Matrix multiply gives count of occupied slots per path
        # If count > 0, slot is occupied on that path
        return (paths.astype(jnp.float32) @ slots_occupied) > 0

    # Warmup
    _ = bench_broadcast_max(paths, state.link_slot_array).block_until_ready()
    _ = bench_einsum_max(paths, state.link_slot_array).block_until_ready()
    _ = bench_matmul_occupied(paths, state.link_slot_array).block_until_ready()

    # Verify correctness
    r1 = bench_broadcast_max(paths, state.link_slot_array)
    r2 = bench_einsum_max(paths, state.link_slot_array)
    r3 = bench_matmul_occupied(paths, state.link_slot_array)

    occupied_v1 = r1 != 0
    occupied_v2 = r2 != 0
    occupied_v3 = r3

    assert jnp.allclose(occupied_v1, occupied_v2), "einsum mismatch!"
    assert jnp.allclose(occupied_v1, occupied_v3), "matmul mismatch!"
    print("All variants produce identical results ✓\n")

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(N):
        r = bench_broadcast_max(paths, state.link_slot_array)
    r.block_until_ready()
    print(f"broadcast + max: {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    t0 = time.perf_counter()
    for _ in range(N):
        r = bench_einsum_max(paths, state.link_slot_array)
    r.block_until_ready()
    print(f"einsum + max: {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    t0 = time.perf_counter()
    for _ in range(N):
        r = bench_matmul_occupied(paths, state.link_slot_array)
    r.block_until_ready()
    print(f"matmul (binary): {(time.perf_counter() - t0) / N * 1000:.3f}ms")

    # Check memory bandwidth bound
    print(f"\n--- Memory analysis ---")
    print(f"paths shape: {paths.shape}")
    print(f"link_slot_array shape: {state.link_slot_array.shape}")
    print(
        f"Intermediate tensor: {params.k_paths} × {params.num_links} × {params.link_resources} = {params.k_paths * params.num_links * params.link_resources:,} elements"
    )
    print(
        f"At float32: {params.k_paths * params.num_links * params.link_resources * 4 / 1024:.1f} KB"
    )


def benchmark(argv):
    dtype_config.initialize_dtypes(FLAGS)
    config = process_config(FLAGS)
    rng = jax.random.PRNGKey(config.SEED)

    print(f"Environment type: {config.get('env_type', 'rmsa')}")
    print(f"Topology: {config.get('topology_name', 'nsfnet_deeprmsa_directed')}")
    print(f"Link resources: {config.get('link_resources', 100)}")
    print(f"K paths: {config.get('k', 5)}")
    print(f"Device: {jax.devices()[0]}")
    print()

    benchmark_funcs = [
        ("cpu", benchmark_cpu),
        ("gpu", benchmark_gpu),
        ("gpu_jitted", benchmark_gpu_jitted),
        ("gpu_detailed", benchmark_gpu_detailed),
        ("original_vs_new", benchmark_original_vs_new),
    ]

    N = 100

    for func_name, func in benchmark_funcs:
        experiment_input, env, env_params = experiment_data_setup(config, rng)

        # Extract state from experiment input
        _, env_state, _, _, _ = experiment_input
        state = env_state.env_state
        params = env_params
        request_array = state.request_array.copy()

        # Run the benchmark function
        func(state, params, request_array, N)


if __name__ == "__main__":
    FLAGS(sys.argv)
    app.run(benchmark)
