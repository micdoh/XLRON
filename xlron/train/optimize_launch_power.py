import wandb
import sys
import os
import time
import pathlib
import matplotlib.pyplot as plt
from absl import app, flags
import xlron.train.parameter_flags
import numpy as np
import jax
import jaxopt
from xlron.environments.env_funcs import *
from xlron.environments import isrs_gn_model
from xlron.environments.rsa import *
from xlron.heuristics.eval_heuristic import *

FLAGS = flags.FLAGS


def make_eval(config: flags.FlagValues) -> Callable:
    # INIT ENV
    env, env_params = define_env(config)

    def init_runner_state(rng: chex.PRNGKey, launch_power_array: Optional[jnp.ndarray] = None):
        # RESET ENV
        rng, warmup_rng, reset_key = jax.random.split(rng, 3)
        reset_key = jax.random.split(reset_key, config.NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, env_params)
        inner_state = env_state.env_state.replace(launch_power_array=launch_power_array) if launch_power_array is not None else env_state.env_state
        env_state = env_state.replace(env_state=inner_state)
        obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])

        # LOAD MODEL
        if config.EVAL_MODEL:
            network, last_obs = init_network(config, env, env_state, env_params)
            network_params = config.model["model"]["params"]
            apply = network.apply
            print('Evaluating model')
        else:
            network_params = apply = None

        eval_state = EvalState(apply_fn=apply, params=network_params)

        # Recreate DeepRMSA warmup period
        if config.ENV_WARMUP_STEPS:
            warmup_state = (warmup_rng, env_state, obsv)
            warmup_fn = get_warmup_fn(warmup_state, env, env_params, eval_state, config)
            env_state, obsv = warmup_fn(warmup_state)

        return (env_state, obsv, rng), eval_state

    def evaluate(runner_state, eval_state, num_steps: int, launch_power_array) -> Dict[str, Any]:
        def _env_step(runner_state, unused):
            env_state, last_obs, rng = runner_state
            rng, action_key, step_key = jax.random.split(rng, 3)

            # SELECT ACTION
            action = select_action_eval(config, env_state, env_params, env, eval_state, action_key, last_obs)

            # STEP ENV
            step_key = jax.random.split(step_key, config.NUM_ENVS)
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                step_key, env_state, action, env_params
            )
            obsv = (env_state.env_state, env_params) if config.USE_GNN else tuple([obsv])
            transition = Transition(done, action, reward, last_obs, info)
            runner_state = (env_state, obsv, rng)

            if config.DEBUG:
                jax.debug.print("link_slot_array {}", env_state.env_state.link_slot_array, ordered=config.ORDERED)
                jax.debug.print("link_slot_mask {}", env_state.env_state.link_slot_mask, ordered=config.ORDERED)
                jax.debug.print("action {}", action, ordered=config.ORDERED)
                jax.debug.print("reward {}", reward, ordered=config.ORDERED)

            # jax.debug.print("runner_state {}", runner_state, ordered=config.ORDERED)
            # jax.debug.print("transition {}", transition, ordered=config.ORDERED)
            return runner_state, transition

        runner_state = (runner_state[0].replace(env_state=runner_state[0].env_state.replace(launch_power_array=launch_power_array)),
                        runner_state[1], runner_state[2])
        jax.debug.print("runner_state.lp_array[-1] {}", runner_state[0].env_state.launch_power_array[-1][-1], ordered=config.ORDERED)
        runner_state, traj = jax.lax.scan(_env_step, runner_state, None, num_steps)
        metric = traj.info

        return {"runner_state": runner_state, "metrics": metric}

    return init_runner_state, evaluate


def generate_list_of_requests(num_nodes, directed, mean_datarate=600):
    sd_array = jnp.array(generate_source_dest_pairs(num_nodes, directed))
    # insert middle column of datarate
    datarate = jnp.full((sd_array.shape[0],), mean_datarate)
    sd_array = jnp.insert(sd_array, 1, datarate, axis=1)
    # convert to list of lists
    sd_array = sd_array.tolist()
    print("list of requests: ", sd_array)
    return sd_array


def main(argv):
    # We will compare 3 approaches in the paper:
    # 1. Optimise launch power with fixed power for all transceivers (i.e. single value)
    # 2. Optimise launch power with fixed power per path (i.e. table of values)
    # 3. Optimise launch power with RL (i.e. value determined by agent)

    # Set up config options
    jax.config.update("jax_debug_nans", True)
    #jax.config.update("jax_disable_jit", True)

    #_, env_params = define_env(FLAGS)

    rng = jax.random.PRNGKey(FLAGS.SEED)
    rng = jax.random.split(rng, FLAGS.NUM_LEARNERS)

    array_shape = (911),#(env_params.path_link_array.shape[0],)
    lower_bounds = jnp.full(array_shape, -1.).reshape((FLAGS.NUM_LEARNERS, -1))
    upper_bounds = jnp.full(array_shape, 3.).reshape((FLAGS.NUM_LEARNERS, -1))
    bounds = (lower_bounds, upper_bounds)
    # Fill launch power array with rnadom initial values between 1 and 3
    init_rng, guess_rng = jax.random.split(rng[0])
    launch_power_init = jax.random.uniform(init_rng, array_shape, minval=-1., maxval=3.).reshape((FLAGS.NUM_LEARNERS, -1))
    launch_power_guess = jax.random.uniform(guess_rng, array_shape, minval=-1., maxval=3.).reshape((FLAGS.NUM_LEARNERS, -1))

    # Add list of requests to FLAGS
    list_of_requests = generate_list_of_requests(14, True)
    # Add reverse to itself
    list_of_requests += [request.reverse() for request in list_of_requests]
    FLAGS.__setattr__("list_of_requests", list_of_requests)
    print("Length of list of requests: ", len(FLAGS.list_of_requests))
    # Set max_requests, max_timesteps, EVAL_STEPS
    FLAGS.__setattr__("max_requests", len(FLAGS.list_of_requests))
    FLAGS.__setattr__("max_timesteps", len(FLAGS.list_of_requests))

    rng = rng[0]
    with TimeIt("INITIALIZATION"):
        init_runner_state, evaluate = make_eval(FLAGS)
        runner_state_init, eval_state = init_runner_state(rng, launch_power_init)

    def objective(launch_power_array):
        #jax.debug.print("launch_power_array {}", launch_power_array, ordered=True)
        result = evaluate(runner_state_init, eval_state, FLAGS.TOTAL_TIMESTEPS, launch_power_array)
        metric = -jnp.sum(result["metrics"]["returns"])
        #metric = metric * jnp.sum(launch_power_array)  # Add some dependence on launch power
        jax.debug.print("metric {}", metric, ordered=True)
        return metric

    # Test gradient computation with value_and_grad
    value_grad_fn = jax.value_and_grad(objective)

    # Modified optimization setup
    def safe_objective(x):
        val, grad = value_grad_fn(x)
        # Add small L2 regularization for gradient stability
        val = val + 1e-4 * jnp.sum(x ** 2)
        return val

    # with TimeIt("EVAL OBJECTIVE TWICE"):
    #     jax.debug.print("val1 {}", objective(launch_power_init, runner_state_init), ordered=True)
    #     jax.debug.print("val2 {}", objective(launch_power_guess, runner_state_init), ordered=True)

    # Check gradient of objective with initial launch power
    with TimeIt("GRADIENT"):
        grad = jax.grad(objective)(launch_power_init)
        print(f"Gradient: {grad}")

    # Setup optimizer with modified parameters
    lbfgsb = jaxopt.LBFGSB(
        fun=objective,
        maxiter=100,
        maxls=50,  # More line search steps
        tol=1e-5,
        stepsize=0.01,
        verbose=True
    )

    # TODO - maybe use callback to replace the launch_power_array in runner_state

    sol, info = lbfgsb.run(
        launch_power_guess,
        bounds=bounds,
        #callback=callback
    )
    # Analyze results
    print("\nOptimization Results:")
    print(f"Initial value: {objective(launch_power_guess)}")
    print(f"Final value: {objective(sol)}")
    print(f"Solution range: [{jnp.min(sol)}, {jnp.max(sol)}]")
    print(f"Convergence info:", info)

    # Optimize with jaxopt
    #lbfgsb = jaxopt.LBFGSB(fun=lambda x: objective_jit(x), maxiter=100, stepsize=0.001)

    #print("Starting optimisation")
    #sol, info = lbfgsb.run(launch_power_guess, bounds=bounds)
    print(f"Solution: {sol}")
    print(f"Info: {info}")

    # def objective(launch_power_array, runner_state, eval_state):
    #     jax.debug.print("launch_power_array {}", launch_power_array, ordered=True)
    #     runner_state = runner_state[0].replace(
    #         env_state=runner_state[0].env_state.replace(launch_power_array=launch_power_array)), runner_state[1], \
    #     runner_state[2]
    #     result = evaluate(runner_state, eval_state, FLAGS.TOTAL_TIMESTEPS)
    #     return -jnp.sum(result["metrics"]["returns"])
    #
    # rng = rng[0]
    # with TimeIt("INITIALIZATION"):
    #     runner_state, eval_state = init_runner_state(rng, initial_launch_power)
    #
    # # Check gradient of objective with initial launch power
    # print("Checking gradient")
    # grad = jax.grad(objective)(initial_launch_power, runner_state, eval_state)
    # print(f"Gradient: {grad}")
    #
    # with TimeIt("JITTING OBJECTIVE"):
    #     objective_jit = jax.jit(objective)
    #
    # # Optimize with jaxopt
    # lbfgsb = jaxopt.LBFGSB(fun=lambda x: objective_jit(x, runner_state, eval_state), maxiter=10)
    #
    # print("Starting optimisation")
    # for i in range(FLAGS.OPTIMIZATION_ITERATIONS):
    #     sol, info = lbfgsb.run(initial_launch_power, bounds=bounds)
    #     print(f"Iteration {i + 1}:")
    #     print(f"Solution: {sol}")
    #     print(f"Info: {info}")
    #
    #     # Update runner_state with new launch_power_array
    #     runner_state = runner_state[0].replace(env_state=runner_state[0].env_state.replace(launch_power_array=sol)), \
    #     runner_state[1], runner_state[2]
    #
    #     # Run a few evaluation steps to update the runner_state
    #     result = evaluate(runner_state, eval_state, FLAGS.EVAL_STEPS)
    #     runner_state = result["runner_state"]
    #
    #     print(f"Evaluation metric: {-jnp.sum(result['metrics']['returns'])}")
    #
    #     initial_launch_power = sol  # Use the current solution as the starting point for the next iteration

    # # TODO - maybe vmap across seeds to compile different funcs for each seed
    # print("Starting compilation")
    # with TimeIt(tag='COMPILATION'):
    #     run_experiment = jax.jit(jax.vmap(make_eval(FLAGS), in_axes=(0, None)))#.lower(rng, initial_launch_power).compile()
    # print("Finished compilation")
    #
    # rng = rng[0]
    #
    # @jax.jit
    # def objective(launch_power_array, rng):
    #     rng, key = jax.random.split(rng)
    #     out = run_experiment(key.reshape((1, 2)), launch_power_array)
    #     regularization = 1e-6 * jnp.sum(launch_power_array ** 2)
    #     returns = jnp.sum(out["metrics"]["returns"])
    #     jax.debug.print("returns {}", returns, ordered=True)
    #     #out = jax.vmap(make_eval(FLAGS), in_axes=(0, None))(rng, launch_power_array)
    #     #out["metrics"]["returns"].block_until_ready()  # Wait for all devices to finish
    #     return jnp.sum(returns) - regularization
    #
    # # Optimise with jaxopt
    # lbfgsb = jaxopt.LBFGSB(fun=objective, stepsize=0.1, maxiter=2)
    # print("Starting optimisation")
    # with TimeIt("EXECUTION"):
    #     # state = lbfgsb.init_state(initial_launch_power, rng=rng, bounds=bounds)  # Initialize state
    #     # for i in range(lbfgsb.maxiter):
    #     #     rng, key = jax.random.split(rng)
    #     #     params, state = lbfgsb.update(state=state, params=state.params, rng=key, bounds=bounds)  # Update state
    #     # sol = params  # Get the solution from the final state
    #     # info = state.aux
    #
    #     #sol, info = lbfgsb.run(initial_launch_power, rng=rng, bounds=bounds)
    #
    #     optimize = jax.jit(partial(lbfgsb.run, bounds=bounds))
    #     sol, info = optimize(initial_launch_power, rng=rng)
    #
    # print(f"sol: {sol}")
    # print(f"info: {info}")


if __name__ == "__main__":
  app.run(main)
