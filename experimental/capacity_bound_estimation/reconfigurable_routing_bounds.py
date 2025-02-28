from itertools import combinations, islice
from functools import partial
from typing import Callable
import sys
import absl.app as app
from absl import flags
import jax.numpy as jnp
import chex
import jax
import xlron.train.parameter_flags
from xlron.environments.dataclasses import *
from xlron.environments.gn_model import isrs_gn_model
from xlron.environments.env_funcs import generate_request_rsa, get_paths_se, required_slots, get_paths
from xlron.environments.wrappers import TimeIt
from xlron.train.train_utils import define_env
from xlron.heuristics.heuristics import ksp_ff

FLAGS = flags.FLAGS


@partial(jax.jit, static_argnums=(1, 3))
def generate_request_list(rng: jnp.ndarray, num_requests: int, state: EnvState, params: EnvParams) -> jnp.ndarray:
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
        return (new_state, params), jnp.array([source, bitrate, dest, arrival_time, holding_time, current_time])

    rngs = jax.random.split(rng, int(num_requests))
    initial_state = (state.env_state, params)
    requests = jax.lax.scan(get_request, initial_state, rngs)[1]
    return requests


@partial(jax.jit, static_argnums=(1,))
def sort_requests(requests: jnp.ndarray, params: EnvParams) -> jnp.ndarray:

    # Sort by required_slots x number of hops for shortest path
    def get_required_resources_sp(request):
        nodes_sd = jnp.concatenate([request[0].reshape(1), request[2].reshape(1)])
        spectral_efficiency = get_paths_se(params, nodes_sd)[0] if params.consider_modulation_format else 1
        requested_slots = required_slots(request[1], spectral_efficiency, params.slot_size,
                                         guardband=params.guardband)
        path = get_paths(params, nodes_sd)[0]
        return requested_slots * jnp.sum(path)

    # Sort by sum of required_slots x number of hops for k-shortest paths, weighted by 1/k
    # We weight the required resources for each k-path by the inverse of the index,
    # such that the shorter more-likely-to-be-selected paths have higher weighting
    def get_required_resources_weighted_sp(request):
        nodes_sd = jnp.concatenate([request[0].reshape(1), request[2].reshape(1)])
        spectral_efficiency = get_paths_se(params, nodes_sd) if params.consider_modulation_format else jnp.ones(params.k_paths)
        requested_slots = (jax.vmap(required_slots, in_axes=(None, 0, None, None))
                           (request[1], spectral_efficiency, params.slot_size, params.guardband))
        paths = get_paths(params, nodes_sd)
        required_resources_per_path = requested_slots * jnp.sum(paths, axis=1)
        inverse_k_indices = 1/jnp.arange(1, params.k_paths+1)
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
        inactive_request = request.at[1].set(0)
        return jnp.where(condition, inactive_request, request)

    requests = jax.vmap(ignore_expired_and_future_request)(requests)

    # We add in new arrival and holding times such that the active requests at each step remain active for the full step-episode
    arrival_times = jnp.ones((requests.shape[0], 1))
    holding_times = jnp.full((requests.shape[0], 1), requests.shape[0])
    current_times = jnp.arange(requests.shape[0]).reshape(-1, 1)
    requests = jnp.concatenate([requests[:, :3], arrival_times, holding_times, current_times], axis=1)
    return requests


def get_eval_fn(config, env, env_params) -> Callable:

    # COLLECT TRAJECTORIES
    def _env_episode(runner_state):

        def _env_step(runner_state, unused):

            env_state, last_obs, rng = runner_state
            rng, action_key, step_key = jax.random.split(rng, 3)

            # SELECT ACTION
            action = ksp_ff(env_state.env_state, env_params)

            # STEP ENV
            obsv, env_state, reward, done, info = env.step(step_key, env_state, action, env_params)
            if config.PROFILE:
                jax.profiler.save_device_memory_profile("memory_step.prof")
            transition = Transition(
                done, action, reward, last_obs, info
            )
            runner_state = (env_state, obsv, rng)

            return runner_state, transition

        runner_state, traj_episode = jax.lax.scan(
            _env_step, runner_state, None, config.TOTAL_TIMESTEPS
        )
        if config.PROFILE:
            jax.profiler.save_device_memory_profile("memory_scan.prof")

        return traj_episode.reward


    @partial(jax.jit, static_argnums=(5,))
    def run_reconfigurable_routing_bound(
            rng, sorted_requests, sort_indices, init_obs, env_state, env_params
    ):
        """This function runs the evaluate function for the given requests and returns the blocking probability.
        The requests array must be modified first to identify only active requests, which have been sorted in
        descending order of required resources. Then, all of the requests are allocated. If any are blocked, then
        the bitrate of the blocked request is set to 0 and the process is repeated.

        Args:
            rng: A PRNGKey.
            sorted_requests: Request array, with columns [source, bitrate, destination, arrival_time, holding_time].
            sort_indices: The indices of the original positions of the sorted requests.
            init_obs: Initial observation.
            env_state: The current environment state.
            env_params: The environment parameters.

        Returns:
            Tuple containing the (updated requests, state, parameters), and the blocking probability.
        """

        def estimate_blocking_step(carry, i):
            _sorted_requests, initial_state, params = carry
            active_requests = get_active_requests(_sorted_requests, i)
            jax.debug.print("active_requests {}", active_requests, ordered=FLAGS.ORDERED)
            state = initial_state.replace(env_state=initial_state.env_state.replace(
                list_of_requests=active_requests
            ))
            state = state.replace(env_state=generate_request_rsa(rng, state.env_state, env_params))
            runner_state = (state, init_obs, rng)
            returns = _env_episode(runner_state)
            blocking = jnp.any(returns < 0)
            jax.debug.print("returns {}", returns, ordered=FLAGS.ORDERED)
            # After eval, set current request bitrate to 0 if blocking_prob > 0
            # If blocking is True then val should be 0, otherwise val should be requests[i].at[1]
            val = jnp.where(blocking, 0, _sorted_requests[i].at[1].get())
            blocked_request = _sorted_requests[i].at[1].set(val)
            _sorted_requests = _sorted_requests.at[i].set(blocked_request)
            jax.debug.print("blocking {}", blocking, ordered=FLAGS.ORDERED)
            jax.debug.print("{} active_requests {}", i, active_requests, ordered=FLAGS.ORDERED)
            jax.debug.print("{} _sorted_requests {}", i, _sorted_requests, ordered=FLAGS.ORDERED)
            return (_sorted_requests, initial_state, params), blocking

        # Scan through step
        _, blocking_events = \
        jax.lax.scan(estimate_blocking_step, (sorted_requests, env_state, env_params), sort_indices)

        return blocking_events

    return run_reconfigurable_routing_bound


def main(argv):

    # Define environment
    FLAGS.__setattr__("deterministic_requests", False)
    FLAGS.__setattr__("max_requests", FLAGS.TOTAL_TIMESTEPS)
    print(f"Using device {jax.devices()[int(FLAGS.VISIBLE_DEVICES)]}")
    jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays
    jax.numpy.set_printoptions(linewidth=220)
    env, env_params = define_env(FLAGS)

    # Generate requests for parallel envs
    rng = jax.random.PRNGKey(FLAGS.SEED)
    setup_keys = jax.random.split(rng, FLAGS.NUM_ENVS)
    init_obs, env_states = jax.vmap(env.reset, in_axes=(0, None))(setup_keys, env_params)
    request_arrays = jax.vmap(generate_request_list, in_axes=(0, None, 0, None))(setup_keys, FLAGS.TOTAL_TIMESTEPS, env_states, env_params)
    # Define env again but this time with deterministic requests
    FLAGS.__setattr__("deterministic_requests", True)
    env, env_params = define_env(FLAGS)
    # Set the requests arrays for each state
    inner_states = jax.vmap(lambda x, y: x.replace(list_of_requests=y), in_axes=(0, 0))(env_states.env_state, request_arrays)
    env_states = env_states.replace(env_state=inner_states)

    print(f"Sort requests: {FLAGS.sort_requests}")
    sorted_requests, sort_indices = jax.vmap(sort_requests, in_axes=(0, None))(request_arrays, env_params) if FLAGS.sort_requests\
        else jax.vmap(lambda x: (request_arrays, jnp.arange(request_arrays.shape[0])), in_axes=(0,))(setup_keys)

    # Define the heuristic evaluation function
    env_keys = jax.random.split(rng, FLAGS.NUM_ENVS)
    with TimeIt(tag='COMPILATION'):
        eval_fn = get_eval_fn(FLAGS, env, env_params)
        run_experiment = jax.jit(
            jax.vmap(
                eval_fn, in_axes=(0, 0, 0, 0, 0, None)
            ), static_argnums=(5,)
        ).lower(
            env_keys, sorted_requests, sort_indices, init_obs, env_states, env_params
        ).compile()

    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS ** 2 * FLAGS.NUM_ENVS):
        blocking_events = run_experiment(
            env_keys, sorted_requests, sort_indices, init_obs, env_states
        )
        blocking_events = blocking_events.block_until_ready()

    jax.debug.print("blocking_events {}", blocking_events, ordered=FLAGS.ORDERED)
    jax.debug.print("1 - blocking_events {}", 1 - blocking_events, ordered=FLAGS.ORDERED)
    blocking_probs = jnp.sum(blocking_events, axis=1) / FLAGS.TOTAL_TIMESTEPS
    # jax.debug.print("blocking_probs {}", blocking_probs, ordered=True)
    blocking_prob_mean = jnp.mean(blocking_probs)
    blocking_prob_std = jnp.std(blocking_probs)
    blocking_prob_iqr_lower = jnp.percentile(blocking_probs, 25)
    blocking_prob_iqr_upper = jnp.percentile(blocking_probs, 75)
    print(f"Blocking Probability: {blocking_prob_mean:.5f} ± {blocking_prob_std:.5f}")
    print(f"Blocking Probability IQR: {blocking_prob_iqr_lower:.5f} - {blocking_prob_iqr_upper:.5f}")

if __name__ == "__main__":
    app.run(main)
