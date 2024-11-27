from itertools import combinations, islice
from functools import partial
from typing import Callable
import sys
import absl.app as app
from absl import flags
import copy
import jax.numpy as jnp
import chex
import jax
import xlron.train.parameter_flags
from xlron.environments.dataclasses import *
from xlron.environments import isrs_gn_model
from xlron.environments.env_funcs import generate_request_rsa, get_paths_se, required_slots, get_paths, \
    implement_action_rsa, finalise_action_rsa, check_action_rsa
from xlron.environments.wrappers import TimeIt
from xlron.train.train_utils import define_env
from xlron.heuristics.heuristics import ksp_ff
from reconfigurable_routing_bounds import generate_request_list, sort_requests

FLAGS = flags.FLAGS


@jax.jit
def get_active_requests(requests: jnp.ndarray, i: int) -> jnp.ndarray:
    """Returns the subset of requests that are active at the current time.

    Args:
        requests: Request array, with columns [source, bitrate, destination, arrival_time, holding_time].
        current_time: The current time.

    Returns:
        Active request array, with columns [source, bitrate, destination, arrival_time, holding_time].
    """
    current_time = requests[i][5]

    def ignore_expired_and_future_request(request):
        expired_condition = request[5] + request[4] < current_time
        future_condition = request[5] > current_time
        condition = jnp.logical_or(expired_condition, future_condition)
        inactive_request = jnp.zeros(request.shape)
        return jnp.where(condition, inactive_request, request)

    requests = jax.vmap(ignore_expired_and_future_request)(requests)

    return requests


def get_eval_fn(config, env, env_params) -> Callable:

    def _env_episode(runner_state):

        def _env_step(runner_state, unused):

            env_state, last_obs, rng = runner_state
            rng, action_key, step_key = jax.random.split(rng, 3)

            # SELECT ACTION
            action = ksp_ff(env_state.env_state, env_params)
            state = implement_action_rsa(env_state.env_state, action, env_params)
            blocking = check_action_rsa(state)
            state = finalise_action_rsa(state, env_params)
            state = generate_request_rsa(step_key, state, env_params)
            env_state = env_state.replace(env_state=state)
            runner_state = (env_state, last_obs, rng)

            return runner_state, blocking

        runner_state, blocking_events = jax.lax.scan(
            _env_step, runner_state, None, config.TOTAL_TIMESTEPS
        )
        if config.PROFILE:
            jax.profiler.save_device_memory_profile("memory_scan.prof")

        return runner_state[0], blocking_events


    @jax.jit
    def run_defragmentation(
            rng, sorted_requests, sort_index, init_obs, env_state
    ):
        active_requests = get_active_requests(sorted_requests, sort_index)
        env_state = env_state.replace(env_state=env_state.env_state.replace(list_of_requests=active_requests))
        runner_state = (env_state, init_obs, rng)
        env_state, blocking_events = _env_episode(runner_state)
        blocking = jnp.any(blocking_events)
        # After eval, set current request bitrate to 0 if blocking_prob > 0
        # If blocking is True then val should be 0, otherwise val should be requests[i].at[1]
        val = jnp.where(blocking, 0, sorted_requests[sort_index].at[1].get())
        blocked_request = sorted_requests[sort_index].at[1].set(val)
        sorted_requests = sorted_requests.at[sort_index].set(blocked_request)
        return sorted_requests, env_state, blocking

    return run_defragmentation


@partial(jax.jit, static_argnums=(1, 3))
def step_env(rng, env, env_state, env_params):
    # Step through the environment
    rng, action_key, step_key = jax.random.split(rng, 3)
    # SELECT ACTION
    action = ksp_ff(env_state.env_state, env_params)
    # STEP ENV
    obsv, env_state, reward, done, info = env.step(step_key, env_state, action, env_params)
    return obsv, env_state, reward, done, info


def main(argv):

    FLAGS.__setattr__("max_requests", FLAGS.TOTAL_TIMESTEPS)
    FLAGS.__setattr__("max_timesteps", FLAGS.TOTAL_TIMESTEPS)
    print(f"Using device {jax.devices()[int(FLAGS.VISIBLE_DEVICES)]}")
    jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays
    jax.numpy.set_printoptions(linewidth=220)

    all_blocking_probs = []
    all_block_counts = []
    all_fix_counts = []

    for seed in range(10):

        # Define environment
        FLAGS.__setattr__("deterministic_requests", False)
        env, env_params = define_env(FLAGS)

        # Generate requests for parallel envs
        rng = jax.random.PRNGKey(seed)
        setup_key = jax.random.split(rng, 1)[0]
        init_obs, env_state = env.reset(setup_key, env_params)
        request_array = generate_request_list(setup_key, FLAGS.TOTAL_TIMESTEPS, env_state, env_params)
        # Define env again but this time with deterministic requests
        FLAGS.__setattr__("deterministic_requests", True)
        env, env_params = define_env(FLAGS)
        # Set the requests arrays for each state
        inner_state = env_state.env_state.replace(list_of_requests=request_array)
        inner_state = generate_request_rsa(setup_key, inner_state, env_params)
        env_state = env_state.replace(env_state=inner_state)
        initial_state = copy.deepcopy(env_state)

        sorted_requests, sort_indices = sort_requests(request_array, env_params)

        # Define the heuristic evaluation function
        env_key = jax.random.split(rng, 1)[0]
        with TimeIt(tag='COMPILATION'):
            defrag_fn = get_eval_fn(FLAGS, env, env_params)
            run_defrag = jax.jit(defrag_fn).lower(
                env_key, sorted_requests, sort_indices[0], init_obs, env_state
            ).compile()

        with (TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS)):
            blocking_events = []
            block_count = 0
            fix_count = 0
            for i in range(int(FLAGS.TOTAL_TIMESTEPS)):
                # Step through the environment
                obsv, env_state, reward, done, info = step_env(env_key, env, env_state, env_params)
                blocking = 1 if reward < 0 else 0
                if blocking:
                    block_count += 1
                    sort_index = sort_indices[i]
                    sorted_requests, new_env_state, blocking = run_defrag(env_key, sorted_requests, sort_index, init_obs, initial_state)
                    if not blocking:
                        new_link_slot_array = new_env_state.env_state.link_slot_array
                        new_link_slot_departure_array = new_env_state.env_state.link_slot_departure_array
                        inner = env_state.env_state.replace(
                            link_slot_array=new_link_slot_array,
                            link_slot_departure_array=new_link_slot_departure_array
                        )
                        inner = finalise_action_rsa(inner, env_params)
                        env_state = env_state.replace(env_state=inner)
                        fix_count += 1
                # Replace env_state requests with unsorted ones
                env_state = env_state.replace(env_state=env_state.env_state.replace(list_of_requests=request_array))
                blocking_events.append(blocking)
            print(f"Blocking prob. : {sum(blocking_events) / FLAGS.TOTAL_TIMESTEPS}")
            blocking_prob = sum(blocking_events) / FLAGS.TOTAL_TIMESTEPS
            all_blocking_probs.append(blocking_prob)
            all_block_counts.append(block_count)
            all_fix_counts.append(fix_count)

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
    fix_ratio = fix_count_mean / block_count_mean
    fix_ratio_std = jnp.std(fix_counts / block_counts)
    fix_ratio_iqr_lower = jnp.percentile(fix_counts / block_counts, 25)
    fix_ratio_iqr_upper = jnp.percentile(fix_counts / block_counts, 75)
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
    print(f"Fix Ratio mean: {fix_ratio:.5f}")
    print(f"Fix Ratio std: {fix_ratio_std:.5f}")
    print(f"Fix Ratio IQR lower: {fix_ratio_iqr_lower:.5f}")
    print(f"Fix Ratio IQR upper: {fix_ratio_iqr_upper:.5f}")

if __name__ == "__main__":
    app.run(main)
