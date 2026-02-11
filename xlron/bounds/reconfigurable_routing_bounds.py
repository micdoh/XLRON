import copy
import sys
from functools import partial
from typing import Callable

import absl.app as app
import jax
import jax.numpy as jnp
from absl import flags
from tqdm import tqdm

from xlron.environments.dataclasses import EnvState, EnvParams
from xlron.environments.env_funcs import (
    generate_request_rsa,
    get_paths,
    get_paths_se,
    required_slots,
)
from xlron.environments.make_env import make
from xlron.environments.wrappers import JitProfiler, TimeIt
from xlron.heuristics.heuristics import ff_ksp, ksp_ff
from xlron.train.train import identify_default_device

FLAGS = flags.FLAGS


@partial(jax.jit, static_argnums=(1, 3))
def generate_request_list(
    rng: jnp.ndarray, num_requests: int, state: EnvState, params: EnvParams
) -> jnp.ndarray:
    """Generates a list of requests for the given state and environment parameters.

    Args:
        rng: A PRNGKey.
        num_requests: The number of requests to generate.
        state: The current environment state.
        params: The environment parameters.

    Returns:
        Request array, with columns [source, bitrate, destination, arrival_time, holding_time].
    """

    def get_request(carry, _rng):
        _state, _params = carry
        new_state = generate_request_rsa(_rng, _state, _params)
        source, bitrate, dest = new_state.request_array
        source = source.astype(jnp.int32)
        bitrate = bitrate.astype(jnp.float32)
        dest = dest.astype(jnp.int32)
        holding_time = jnp.squeeze(new_state.holding_time)
        arrival_time = jnp.squeeze(new_state.current_time - _state.current_time)
        current_time = jnp.squeeze(new_state.current_time)
        return (new_state, params), jnp.array(
            [source, bitrate, dest, arrival_time, holding_time, current_time]
        )

    rngs = jax.random.split(rng, int(num_requests))
    initial_state = (state.env_state, params)
    requests = jax.lax.scan(get_request, initial_state, rngs)[1]
    return requests


@partial(jax.jit, static_argnums=(1,))
def sort_requests(requests: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
    # Sort by required_slots x number of hops for shortest path
    def get_required_resources_sp(request):
        nodes_sd = jnp.concatenate([request[0].reshape(1), request[2].reshape(1)])
        spectral_efficiency = (
            get_paths_se(params, nodes_sd)[0] if params.consider_modulation_format else 1
        )
        requested_slots = required_slots(
            request[1], spectral_efficiency, params.slot_size, guardband=params.guardband
        )
        path = get_paths(params, nodes_sd)[0]
        return requested_slots * jnp.sum(path)

    # Sort by sum of required_slots x number of hops for k-shortest paths, weighted by 1/k
    # We weight the required resources for each k-path by the inverse of the index,
    # such that the shorter more-likely-to-be-selected paths have higher weighting
    def get_required_resources_weighted_sp(request):
        nodes_sd = jnp.concatenate([request[0].reshape(1), request[2].reshape(1)])
        spectral_efficiency = (
            get_paths_se(params, nodes_sd)
            if params.consider_modulation_format
            else jnp.ones(params.k_paths)
        )
        requested_slots = jax.vmap(required_slots, in_axes=(None, 0, None, None))(
            request[1], spectral_efficiency, params.slot_size, params.guardband
        )
        paths = get_paths(params, nodes_sd)
        required_resources_per_path = requested_slots * jnp.sum(paths, axis=1)
        inverse_k_indices = 1 / jnp.arange(1, params.k_paths + 1)
        weighted_required_resources = jnp.sum(required_resources_per_path * inverse_k_indices)
        return weighted_required_resources

    # Sort by required resources
    sort_resources = jax.vmap(get_required_resources_weighted_sp)(requests)
    sort_indices_resources = jnp.argsort(sort_resources, descending=True)
    sorted_requests = requests[sort_indices_resources]
    # We want to return the indices of the original positions of the sorted requests
    unsorted_current_times = sorted_requests[:, 5]
    sort_indices_time = jnp.argsort(unsorted_current_times)
    return sorted_requests, sort_indices_time


@jax.jit
def get_active_requests(requests: jnp.ndarray, i: int) -> jnp.ndarray:
    """Returns requests with inactive rows having bitrate zeroed, and timing
    columns replaced so that all requests coexist during the defrag episode.

    Args:
        requests: Request array with columns [source, bitrate, dest, arrival, holding, current_time].
        i: Index into requests for the current timestep.

    Returns:
        Request array with inactive bitrates zeroed and synthetic timing columns.
    """
    current_time = requests[i][5]

    def ignore_expired_and_future_request(request):
        expired_condition = request[5] + request[4] < current_time
        future_condition = request[5] > current_time
        condition = jnp.logical_or(expired_condition, future_condition)
        inactive_request = request.at[1].set(0)  # Zero bitrate only
        return jnp.where(condition, inactive_request, request)

    requests = jax.vmap(ignore_expired_and_future_request)(requests)

    # Replace timing columns so all requests coexist for the full episode:
    # - arrival_time = 1 (arrive 1 unit apart)
    # - holding_time = N (never expire)
    # - current_time = 0, 1, 2, ... (sequential)
    n = requests.shape[0]
    arrival_times = jnp.ones((n, 1))
    holding_times = jnp.full((n, 1), n)
    current_times = jnp.arange(n).reshape(-1, 1)
    requests = jnp.concatenate(
        [requests[:, :3], arrival_times, holding_times, current_times], axis=1
    )
    return requests


def get_active_requests_filtered(requests: jnp.ndarray, i: int) -> jnp.ndarray:
    """Returns only the active rows (variable-length, not JIT-compatible).

    Calls the JIT-compiled get_active_requests then filters out rows
    where bitrate (column 1) is zero.
    """
    processed = get_active_requests(requests, i)
    mask = processed[:, 1] != 0
    return processed[mask]


jit_profiler = JitProfiler()


def get_eval_fn(config, env, env_params, compile_defrag=False) -> Callable:
    # Use the raw env (unwrap LogWrapper) for step_env which does not auto-reset.
    # Auto-reset causes dtype/shape mismatches when list_of_requests differs
    # between the injected requests and the default reset state.
    raw_env = env._env if hasattr(env, "_env") else env
    profile = bool(config.PROFILE)
    select_action = ksp_ff if config.path_heuristic == "ksp_ff" else ff_ksp

    if compile_defrag:
        # --- Compiled path: full main loop as jax.lax.scan ---
        max_active = max(1, int(config.load * 3))

        total_ts = int(config.TOTAL_TIMESTEPS)

        def trim_active_requests(active_requests):
            """Sort by bitrate descending (active rows first), keep top max_active.

            Always returns shape (max_active, 6). If TOTAL_TIMESTEPS < max_active,
            pad with zero rows.
            """
            sort_idx = jnp.argsort(active_requests[:, 1], descending=True)
            sorted_reqs = active_requests[sort_idx]
            if total_ts >= max_active:
                return sorted_reqs[:max_active]
            else:
                # Pad to max_active rows
                pad_rows = max_active - total_ts
                padding = jnp.zeros(
                    (pad_rows, active_requests.shape[1]), dtype=active_requests.dtype
                )
                return jnp.concatenate([sorted_reqs, padding], axis=0)

        def fix_timing_after_trim(trimmed):
            """Re-number timing columns for the trimmed request set."""
            n = max_active
            dtype = trimmed.dtype
            arrival_times = jnp.ones((n, 1), dtype=dtype)
            holding_times = jnp.full((n, 1), n, dtype=dtype)
            current_times = jnp.arange(n, dtype=dtype).reshape(-1, 1)
            return jnp.concatenate(
                [trimmed[:, :3], arrival_times, holding_times, current_times], axis=1
            )

        def _env_episode_defrag(runner_state):
            """Defrag episode: scan for max_active steps."""

            def _env_step(runner_state, unused):
                env_state, last_obs, rng = runner_state
                rng, action_key, step_key = jax.random.split(rng, 3)
                inner_state = env_state.env_state
                action = jit_profiler.call(
                    profile, select_action, inner_state, env_params, name="select_action_defrag"
                )
                obsv, new_inner_state, reward, terminal, truncated, info = jit_profiler.call(
                    profile,
                    raw_env.step_env,
                    step_key,
                    inner_state,
                    action,
                    env_params,
                    name="step_env_defrag",
                )
                env_state = env_state.replace(env_state=new_inner_state)
                blocking = reward < 0
                return (env_state, obsv, rng), blocking

            runner_state, blocking_events = jax.lax.scan(_env_step, runner_state, None, max_active)
            return runner_state[0], blocking_events

        def run_defrag_trimmed(rng, sorted_requests, sort_index, init_obs, defrag_initial_state):
            """Run defrag with trimmed active requests (max_active steps)."""
            active_requests = get_active_requests(sorted_requests, sort_index)
            trimmed = fix_timing_after_trim(trim_active_requests(active_requests))

            defrag_state = defrag_initial_state.replace(
                env_state=defrag_initial_state.env_state.replace(
                    list_of_requests=trimmed,
                    total_requests=jnp.array(0),
                )
            )
            runner_state = (defrag_state, init_obs, rng)
            final_state, blocking_events = _env_episode_defrag(runner_state)
            blocking = jnp.any(blocking_events)

            val = jnp.where(blocking, 0, sorted_requests[sort_index, 1])
            sorted_requests = sorted_requests.at[sort_index, 1].set(val)

            lsa = final_state.env_state.link_slot_array
            lsda = final_state.env_state.link_slot_departure_array
            return sorted_requests, lsa, lsda, blocking

        def compiled_main_loop(
            rng,
            env_state,
            sorted_requests,
            init_obs,
            request_array,
            defrag_initial_state,
            sort_indices,
        ):
            """Full compiled main loop: jax.lax.scan over TOTAL_TIMESTEPS."""

            def scan_body(carry, sort_index):
                env_state, sorted_requests, rng, block_count, fix_count = carry

                # 1. Step env
                rng, step_key = jax.random.split(rng)
                inner_state = env_state.env_state
                action = select_action(inner_state, env_params)
                obsv, new_inner_state, reward, terminal, truncated, info = raw_env.step_env(
                    step_key, inner_state, action, env_params
                )
                env_state = env_state.replace(env_state=new_inner_state)
                step_blocking = reward < 0

                # 2. Conditional defrag
                def do_defrag(args):
                    sorted_reqs, rng_in, env_st = args
                    rng_d, defrag_key = jax.random.split(rng_in)
                    sr_new, lsa, lsda, defrag_blocked = run_defrag_trimmed(
                        defrag_key, sorted_reqs, sort_index, init_obs, defrag_initial_state
                    )
                    lsa_out = jnp.where(defrag_blocked, env_st.env_state.link_slot_array, lsa)
                    lsda_out = jnp.where(
                        defrag_blocked, env_st.env_state.link_slot_departure_array, lsda
                    )
                    fix_inc = (~defrag_blocked).astype(jnp.int32)
                    return sr_new, lsa_out, lsda_out, fix_inc

                def no_defrag(args):
                    sorted_reqs, rng_in, env_st = args
                    lsa = env_st.env_state.link_slot_array
                    lsda = env_st.env_state.link_slot_departure_array
                    return sorted_reqs, lsa, lsda, jnp.int32(0)

                sorted_requests, lsa, lsda, fix_inc = jax.lax.cond(
                    step_blocking,
                    do_defrag,
                    no_defrag,
                    (sorted_requests, rng, env_state),
                )

                # 3. Update env_state: transplant link arrays, restore request_array
                inner = env_state.env_state.replace(
                    link_slot_array=lsa,
                    link_slot_departure_array=lsda,
                    list_of_requests=request_array,
                )
                env_state = env_state.replace(env_state=inner)

                # 4. Update counters
                block_count = block_count + step_blocking.astype(jnp.int32)
                fix_count = fix_count + fix_inc

                # 5. Final blocking: blocked if step blocked AND defrag didn't fix it
                final_blocking = step_blocking & (fix_inc == 0)

                new_carry = (env_state, sorted_requests, rng, block_count, fix_count)
                return new_carry, final_blocking

            init_carry = (env_state, sorted_requests, rng, jnp.int32(0), jnp.int32(0))
            (final_env_state, final_sorted_requests, _, block_count, fix_count), blocking_events = (
                jax.lax.scan(scan_body, init_carry, sort_indices)
            )
            return final_env_state, final_sorted_requests, blocking_events, block_count, fix_count
            
        return compiled_main_loop, run_defrag_trimmed

    else:
        # --- Non-compiled path: Python loop over only actual active requests ---

        def run_defragmentation(rng, sorted_requests, sort_index, init_obs, env_state):
            active_requests = get_active_requests_filtered(sorted_requests, sort_index)
            num_active = active_requests.shape[0]

            # Re-number timing columns for the compacted active requests:
            # arrival=1, holding=num_active, current_time=0..num_active-1
            arrival_times = jnp.ones((num_active, 1))
            holding_times = jnp.full((num_active, 1), num_active)
            current_times = jnp.arange(num_active).reshape(-1, 1)
            active_requests = jnp.concatenate(
                [active_requests[:, :3], arrival_times, holding_times, current_times], axis=1
            )

            # Reset env state and inject active requests as the request list
            # Pad back to TOTAL_TIMESTEPS so the env state shape is consistent
            total_timesteps = int(config.TOTAL_TIMESTEPS)
            padded = jnp.zeros((total_timesteps, active_requests.shape[1]))
            padded = padded.at[:num_active].set(active_requests)
            inner_state = env_state.env_state.replace(
                list_of_requests=padded,
                total_requests=jnp.array(0),  # Reset so defrag reads from index 0
            )
            inner_state = generate_request_rsa(rng, inner_state, env_params)
            env_state = env_state.replace(env_state=inner_state)

            blocking = False
            for j in range(num_active):
                rng, step_key = jax.random.split(rng)
                inner_state = env_state.env_state
                action = select_action(inner_state, env_params)
                obsv, new_inner_state, reward, terminal, truncated, info = raw_env.step_env(
                    step_key, inner_state, action, env_params
                )
                env_state = env_state.replace(env_state=new_inner_state)
                if reward < 0:
                    blocking = True
                    break

            blocking = jnp.bool_(blocking)
            val = jnp.where(blocking, 0, sorted_requests[sort_index].at[1].get())
            blocked_request = sorted_requests[sort_index].at[1].set(val)
            sorted_requests = sorted_requests.at[sort_index].set(blocked_request)
            return sorted_requests, env_state, blocking

        return None, run_defragmentation


@partial(jax.jit, static_argnums=(1, 3, 4))
def step_env(rng, raw_env, env_state, env_params, profile=False):
    """Step through the environment using raw step_env (no auto-reset).

    Uses raw_env.step_env to avoid auto-reset which causes dtype/shape
    mismatches when list_of_requests has been injected.
    """
    rng, action_key, step_key = jax.random.split(rng, 3)
    # SELECT ACTION
    inner_state = env_state.env_state
    select_action = ksp_ff if FLAGS.path_heuristic == "ksp_ff" else ff_ksp
    action = jit_profiler.call(
        profile, select_action, inner_state, env_params, name="main_select_action"
    )
    # STEP ENV (use raw step_env to avoid auto-reset on done)
    obsv, new_inner_state, reward, terminal, truncated, info = jit_profiler.call(
        profile, raw_env.step_env, step_key, inner_state, action, env_params, name="main_step_env"
    )
    env_state = env_state.replace(env_state=new_inner_state)
    return obsv, env_state, reward, terminal, truncated, info


def main(argv):
    # Bounds code requires absolute arrival times to track request lifetimes
    FLAGS.__setattr__("relative_arrival_times", False)
    FLAGS.__setattr__("max_requests", FLAGS.TOTAL_TIMESTEPS)
    if FLAGS.VISIBLE_DEVICES:
        gpu_indices = [int(x) for x in FLAGS.VISIBLE_DEVICES.split(",")]
        default_device = identify_default_device(gpu_index=gpu_indices[0], auto_select=False)
    else:
        default_device = identify_default_device(auto_select=True)
    print(f"Default device set to: {default_device}")
    jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays
    jax.numpy.set_printoptions(linewidth=220)
    profile = bool(FLAGS.PROFILE)

    all_blocking_probs = []
    all_block_counts = []
    all_fix_counts = []

    num_seeds = 1 if profile else 10
    for seed in range(num_seeds):
        print(f"  Seed {seed + 1}/{num_seeds}: setting up environment...", flush=True)

        # Define environment
        FLAGS.__setattr__("deterministic_requests", False)
        env, env_params = make(FLAGS)

        # Generate requests for parallel envs
        rng = jax.random.PRNGKey(seed)
        setup_key = jax.random.split(rng, 1)[0]
        init_obs, env_state = env.reset(setup_key, env_params)
        request_array = generate_request_list(
            setup_key, FLAGS.TOTAL_TIMESTEPS, env_state, env_params
        )
        # Define env again but this time with deterministic requests
        FLAGS.__setattr__("deterministic_requests", True)
        env, env_params = make(FLAGS)
        raw_env = env._env if hasattr(env, "_env") else env
        # Set the requests arrays for each state
        inner_state = env_state.env_state.replace(list_of_requests=request_array)
        inner_state = generate_request_rsa(setup_key, inner_state, env_params)
        env_state = env_state.replace(env_state=inner_state)
        initial_state = copy.deepcopy(env_state)

        sorted_requests, sort_indices = sort_requests(request_array, env_params)

        # Define the heuristic evaluation function
        env_key = jax.random.split(rng, 1)[0]
        compile_defrag = bool(FLAGS.COMPILE_RR_BOUNDS)
        print(
            f"  Seed {seed + 1}/{num_seeds}: compiling (compile_defrag={compile_defrag})...",
            flush=True,
        )
        main_loop_fn, defrag_fn = get_eval_fn(FLAGS, env, env_params, compile_defrag=compile_defrag)
        if compile_defrag:
            max_active = max(1, int(FLAGS.load * 3))
            defrag_list = jnp.zeros((max_active, 6), dtype=request_array.dtype)
            defrag_initial_state = initial_state.replace(
                env_state=initial_state.env_state.replace(list_of_requests=defrag_list)
            )
            with TimeIt(tag="MAIN LOOP COMPILATION"):
                compiled_main = (
                    jax.jit(main_loop_fn)
                    .lower(
                        env_key,
                        env_state,
                        sorted_requests,
                        init_obs,
                        request_array,
                        defrag_initial_state,
                        sort_indices,
                    )
                    .compile()
                )

            print(
                f"  Seed {seed + 1}/{num_seeds}: running {int(FLAGS.TOTAL_TIMESTEPS)} timesteps (compiled)...",
                flush=True,
            )
            with TimeIt(tag="EXECUTION", frames=FLAGS.TOTAL_TIMESTEPS):
                _, _, blocking_events_arr, block_count_arr, fix_count_arr = compiled_main(
                    env_key,
                    env_state,
                    sorted_requests,
                    init_obs,
                    request_array,
                    defrag_initial_state,
                    sort_indices,
                )
                jax.block_until_ready(blocking_events_arr)

            blocking_events = blocking_events_arr.tolist()
            block_count = int(block_count_arr)
            fix_count = int(fix_count_arr)
            print(
                f"  Seed {seed + 1}/{num_seeds}: blocking={sum(blocking_events) / FLAGS.TOTAL_TIMESTEPS:.5f}, blocks={block_count}, fixes={fix_count}",
                flush=True,
            )
            blocking_prob = sum(blocking_events) / FLAGS.TOTAL_TIMESTEPS
            all_blocking_probs.append(blocking_prob)
            all_block_counts.append(block_count)
            all_fix_counts.append(fix_count)
        else:
            run_defrag = defrag_fn
            with TimeIt(tag="STEP ENV COMPILATION"):
                step_env.lower(env_key, raw_env, env_state, env_params, profile).compile()

            jit_profiler.reset()
            import time as _time

            step_time_total = 0.0
            defrag_time_total = 0.0
            defrag_call_count = 0

            with TimeIt(tag="EXECUTION", frames=FLAGS.TOTAL_TIMESTEPS):
                blocking_events = []
                block_count = 0
                fix_count = 0
                pbar = tqdm(
                    range(int(FLAGS.TOTAL_TIMESTEPS)),
                    desc=f"  Seed {seed + 1}/{num_seeds}",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] blocks={postfix[0]} fixes={postfix[1]}",
                    postfix=[0, 0],
                )
                for i in pbar:
                    t0 = _time.time()
                    obsv, env_state, reward, terminal, truncated, info = step_env(
                        env_key, raw_env, env_state, env_params, profile
                    )
                    step_time_total += _time.time() - t0
                    blocking = 1 if reward < 0 else 0
                    if blocking:
                        block_count += 1
                        sort_index = sort_indices[i]
                        t0 = _time.time()
                        sorted_requests, new_env_state, blocking = run_defrag(
                            env_key, sorted_requests, sort_index, init_obs, initial_state
                        )
                        defrag_time_total += _time.time() - t0
                        defrag_call_count += 1
                        if not blocking:
                            new_link_slot_array = new_env_state.env_state.link_slot_array
                            new_link_slot_departure_array = (
                                new_env_state.env_state.link_slot_departure_array
                            )
                            inner = env_state.env_state.replace(
                                link_slot_array=new_link_slot_array,
                                link_slot_departure_array=new_link_slot_departure_array,
                            )
                            env_state = env_state.replace(env_state=inner)
                            fix_count += 1
                        pbar.postfix[0] = block_count
                        pbar.postfix[1] = fix_count
                    # Replace env_state requests with unsorted ones
                    env_state = env_state.replace(
                        env_state=env_state.env_state.replace(list_of_requests=request_array)
                    )
                    blocking_events.append(blocking)
                print(
                    f"  Seed {seed + 1}/{num_seeds}: blocking={sum(blocking_events) / FLAGS.TOTAL_TIMESTEPS:.5f}, blocks={block_count}, fixes={fix_count}",
                    flush=True,
                )
                blocking_prob = sum(blocking_events) / FLAGS.TOTAL_TIMESTEPS
                all_blocking_probs.append(blocking_prob)
                all_block_counts.append(block_count)
                all_fix_counts.append(fix_count)

            # Python-level timing summary
            print(f"\n--- Python-level timing (seed {seed}) ---")
            print(
                f"  step_env total:  {step_time_total:.4f}s  ({int(FLAGS.TOTAL_TIMESTEPS)} calls, "
                f"{1e6 * step_time_total / FLAGS.TOTAL_TIMESTEPS:.1f} us/call)"
            )
            print(
                f"  run_defrag total: {defrag_time_total:.4f}s  ({defrag_call_count} calls"
                + (
                    f", {1e6 * defrag_time_total / defrag_call_count:.1f} us/call"
                    if defrag_call_count
                    else ""
                )
                + ")"
            )
            print(
                f"  step_env fraction:  {100 * step_time_total / (step_time_total + defrag_time_total):.1f}%"
            )
            print(
                f"  run_defrag fraction: {100 * defrag_time_total / (step_time_total + defrag_time_total):.1f}%"
            )

        if profile:
            jit_profiler.summary()

    blocking_probs = jnp.array(all_blocking_probs)
    block_counts = jnp.array(all_block_counts)
    fix_counts = jnp.array(all_fix_counts)
    blocking_prob_mean = jnp.mean(blocking_probs)
    blocking_prob_std = jnp.std(blocking_probs)
    blocking_prob_iqr_lower = jnp.percentile(blocking_probs, 25)
    blocking_prob_iqr_upper = jnp.percentile(blocking_probs, 75)
    block_count_mean = jnp.mean(block_counts)
    block_count_std = jnp.std(block_counts)
    block_count_iqr_lower = jnp.percentile(block_counts, 25)
    block_count_iqr_upper = jnp.percentile(block_counts, 75)
    fix_count_mean = jnp.mean(fix_counts)
    fix_count_std = jnp.std(fix_counts)
    fix_count_iqr_lower = jnp.percentile(fix_counts, 25)
    fix_count_iqr_upper = jnp.percentile(fix_counts, 75)
    fix_ratio_mean = jnp.nan_to_num(fix_count_mean / block_count_mean, nan=1)
    fix_ratio_std = jnp.nan_to_num(jnp.std(fix_counts / block_counts), nan=0)
    fix_ratio_iqr_lower = jnp.nan_to_num(
        jnp.percentile(fix_counts / block_counts, 25), nan=fix_ratio_mean
    )
    fix_ratio_iqr_upper = jnp.nan_to_num(
        jnp.percentile(fix_counts / block_counts, 75), nan=fix_ratio_mean
    )
    print(f"Blocking Probability: {blocking_prob_mean:.5f} ± {blocking_prob_std:.5f}")
    print(f"Blocking Probability mean: {blocking_prob_mean:.5f}")
    print(f"Blocking Probability std: {blocking_prob_std:.5f}")
    print(f"Blocking Probability IQR lower: {blocking_prob_iqr_lower:.5f}")
    print(f"Blocking Probability IQR upper: {blocking_prob_iqr_upper:.5f}")
    print(f"Block Count mean: {block_count_mean:.5f}")
    print(f"Block Count std: {block_count_std:.5f}")
    print(f"Block Count IQR lower: {block_count_iqr_lower:.5f}")
    print(f"Block Count IQR upper: {block_count_iqr_upper:.5f}")
    print(f"Fix Count mean: {fix_count_mean:.5f}")
    print(f"Fix Count std: {fix_count_std:.5f}")
    print(f"Fix Count IQR lower: {fix_count_iqr_lower:.5f}")
    print(f"Fix Count IQR upper: {fix_count_iqr_upper:.5f}")
    print(f"Fix Ratio mean: {fix_ratio_mean:.5f}")
    print(f"Fix Ratio std: {fix_ratio_std:.5f}")
    print(f"Fix Ratio IQR lower: {fix_ratio_iqr_lower:.5f}")
    print(f"Fix Ratio IQR upper: {fix_ratio_iqr_upper:.5f}")


if __name__ == "__main__":
    app.run(main)
