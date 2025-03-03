from itertools import combinations, islice
from functools import partial
from typing import Callable, Tuple
import sys
import absl.app as app
from absl import flags
import jax.numpy as jnp
import chex
import jax
import xlron.train.parameter_flags
from xlron.environments.dataclasses import *
from xlron.environments import isrs_gn_model
from xlron.environments.env_funcs import generate_request_rsa, get_paths_se, required_slots, get_paths
from xlron.environments.wrappers import TimeIt
from xlron.train.train_utils import define_env
from xlron.heuristics.eval_heuristic import EvalState, get_warmup_fn, select_action_eval, Transition
from xlron.heuristics.heuristics import ksp_ff
from reconfigurable_routing_bounds import generate_request_list, sort_requests, get_active_requests

FLAGS = flags.FLAGS


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
            _env_step, runner_state, None, config.TOTAL_TIMESTEPS, unroll=min(config.TOTAL_TIMESTEPS, 5)
        )
        if config.PROFILE:
            jax.profiler.save_device_memory_profile("memory_scan.prof")

        metric = traj_episode.info

        return {"runner_state": runner_state, "metrics": metric}


    @jax.jit
    def run_reconfigurable_routing_bound_parallel(
            rng, requests, sort_indices, init_obs, env_states
    ):
        """This function runs the evaluate function for the given requests and returns the blocking probability.
        The requests array must be modified first to identify only active requests, which have been sorted in
        descending order of required resources. Then, all of the requests are allocated. If any are blocked, then
        the bitrate of the blocked request is set to 0 and the process is repeated.

        Args:
            rng: A PRNGKey.
            requests: Request array, with columns [source, bitrate, destination, arrival_time, holding_time].
            init_obs: Initial observation.
            env_state: The current environment state.
            env_params: The environment parameters.
            _sort_requests: Whether to sort the requests by required resources

        Returns:
            Tuple containing the (updated requests, state, parameters), and the blocking probability.
        """
        blocking_0 = jnp.ones(requests.shape[0])
        blocking_1 = jnp.ones(requests.shape[0])
        blocking_2 = jnp.zeros(requests.shape[0])

        def estimate_blocking(rng, state, init_obs, active_requests):
            state = state.replace(env_state=state.env_state.replace(list_of_requests=active_requests))
            runner_state = (state, init_obs, rng)
            out = _env_episode(runner_state)
            returns = out["metrics"]["returns"]
            _blocking = jnp.any(returns < 0).astype(jnp.float32)  # 1 if blocked 0 if not
            return _blocking

        def update_blocked_request(_blocking, _sort_indices, _requests):
            """We want to update the blocked request in all arrays so its bitrate is zero. """

            def return_blocked_array(i, s, arr):
                s_index = s[i]
                arr = arr.at[s_index, 1].set(0)
                return arr

            def update_blocking_index(i, val):
                arr = jax.lax.cond(val[0][i] == 1, lambda x: return_blocked_array(i, x[1], x[2]), lambda x: x[2], val)
                val = (val[0], val[1], arr)
                return val

            def get_blocked_array(b, s, arr):
                val = (b, s, arr)
                _, _, blocked_array = jax.lax.fori_loop(0, _blocking.shape[0], update_blocking_index, val)
                return blocked_array

            blocked_arrays = jax.vmap(get_blocked_array, in_axes=(None, None, 0))(_blocking, _sort_indices, _requests)
            return blocked_arrays

        def eval_fn(carry):
            _rng, _requests, _env_state, _init_obs, _sort_indices, blocking_t0, blocking_t1, blocking_t2 = carry
            new_blocking = jax.vmap(estimate_blocking)(_rng, _env_state, _init_obs, _requests)
            first_new_blocking = new_blocking > blocking_t2
            first_new_blocking_index = jnp.argmax(first_new_blocking)
            blocking_t3 = blocking_t2.at[first_new_blocking_index].set(new_blocking[first_new_blocking_index])

            updated_requests = update_blocked_request(blocking_t3, _sort_indices, _requests)
            # jax.debug.print("new_blocking {}", new_blocking, ordered=FLAGS.ORDERED)
            # jax.debug.print("first_new_blocking {} index {}", first_new_blocking.astype(jnp.float32), first_new_blocking_index, ordered=FLAGS.ORDERED)
            # jax.debug.print("blocking_t3 {}", blocking_t3, ordered=FLAGS.ORDERED)
            # jax.debug.print("requests {}", _requests, ordered=FLAGS.ORDERED)
            # jax.debug.print("updated_requests {}", updated_requests, ordered=FLAGS.ORDERED)
            return _rng, updated_requests, _env_state, _init_obs, _sort_indices, blocking_t1, blocking_t2, blocking_t3

        def cond_fn(carry):
            _, _, _, _, _, blocking_t0, blocking_t1, blocking_t2 = carry
            # jax.debug.print("condition", ordered=True)
            # jax.debug.print("blocking_t0 {}", blocking_t0, ordered=FLAGS.ORDERED)
            # jax.debug.print("blocking_t1 {}", blocking_t1, ordered=FLAGS.ORDERED)
            # jax.debug.print("blocking_t2 {}", blocking_t2, ordered=FLAGS.ORDERED)
            cond1 = jnp.any(blocking_t2 != blocking_t1)  # True if any mismatch
            cond2 = jnp.any(blocking_t2 != blocking_t0)  # True if any mismatch
            # If either condition is false, terminate the loop
            return jnp.logical_and(cond1, cond2)

        # While loop until blocking doesn't change
        blocking = jax.lax.while_loop(
            cond_fn, eval_fn, (rng, requests, env_states, init_obs, sort_indices, blocking_0, blocking_1, blocking_2)
        )[-1]
        return blocking

    return run_reconfigurable_routing_bound_parallel


def main(argv):

    # Define environment
    FLAGS.__setattr__("deterministic_requests", False)
    FLAGS.__setattr__("max_requests", FLAGS.TOTAL_TIMESTEPS)
    jax.config.update("jax_default_device", jax.devices()[int(FLAGS.VISIBLE_DEVICES)])
    print(f"Using device {jax.devices()[int(FLAGS.VISIBLE_DEVICES)]}")
    jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays
    jax.numpy.set_printoptions(linewidth=220)
    env, env_params = define_env(FLAGS)

    # Generate requests for parallel envs
    rng = jax.random.PRNGKey(FLAGS.SEED)
    setup_keys = jax.random.split(rng, FLAGS.NUM_ENVS)
    init_obs, env_state = env.reset(setup_keys[0], env_params)
    request_array = generate_request_list(setup_keys[0], FLAGS.TOTAL_TIMESTEPS, env_state, env_params)

    sorted_requests, sort_indices = sort_requests(request_array, env_params)
    active_requests = jax.vmap(get_active_requests, in_axes=(None, 0))(sorted_requests, sort_indices)
    # Set the requests arrays for each state
    inner_states = jax.vmap(lambda x, y: x.replace(list_of_requests=y), in_axes=(None, 0))(env_state.env_state, sorted_requests)
    env_states = jax.vmap(lambda x, y: x.replace(env_state=y), in_axes=(None, 0))(env_state, inner_states)

    # Add a new leading axis and repeat TOTAL_TIMESTEPS times
    env_keys = jax.random.split(rng, FLAGS.NUM_ENVS)
    env_keys = jnp.repeat(env_keys[0][None, :], int(FLAGS.TOTAL_TIMESTEPS), axis=0)
    init_obs = jnp.repeat(init_obs[None, :], int(FLAGS.TOTAL_TIMESTEPS), axis=0)

    # Define env again but this time with deterministic requests
    FLAGS.__setattr__("deterministic_requests", True)
    env, env_params = define_env(FLAGS)

    jax.debug.print("sorted_requests \n {}", sorted_requests, ordered=FLAGS.ORDERED)
    jax.debug.print("sort_indices \n {}", sort_indices, ordered=FLAGS.ORDERED)

    with TimeIt(tag='COMPILATION'):
        eval_fn = get_eval_fn(FLAGS, env, env_params)
        run_experiment = jax.jit(eval_fn).lower(
            env_keys, active_requests, sort_indices, init_obs, env_states
        ).compile()

    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS ** 2 * FLAGS.NUM_ENVS):
        blocking_events = run_experiment(
            env_keys, active_requests, sort_indices, init_obs, env_states
        )
        blocking_events = blocking_events.block_until_ready()

    jax.debug.print("blocking_events {}", blocking_events, ordered=FLAGS.ORDERED)
    jax.debug.print("1 - blocking_events {}", 1 - blocking_events, ordered=FLAGS.ORDERED)
    blocking_probs = jnp.sum(blocking_events) / FLAGS.TOTAL_TIMESTEPS
    blocking_prob_mean = jnp.mean(blocking_probs)
    blocking_prob_std = jnp.std(blocking_probs)
    blocking_prob_iqr_lower = jnp.percentile(blocking_probs, 25)
    blocking_prob_iqr_upper = jnp.percentile(blocking_probs, 75)
    print(f"Blocking Probability: {blocking_prob_mean:.5f} ± {blocking_prob_std:.5f}")
    print(f"Blocking Probability IQR: {blocking_prob_iqr_lower:.5f} - {blocking_prob_iqr_upper:.5f}")

if __name__ == "__main__":
    app.run(main)
