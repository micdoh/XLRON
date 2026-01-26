import wandb
import sys
import os
import subprocess
import time
import pathlib
import math
import matplotlib.pyplot as plt
import absl
from absl import app, flags
import xlron.parameter_flags
import numpy as np
import pandas as pd
from box import Box
from typing import Optional, Union
from xlron import dtype_config

# Add differentiable optimization-specific flags
flags.DEFINE_boolean("ACTION_OPTIMIZATION", True, "Directly optimise rollout actions using first-order gradients from differentiable environment")
flags.DEFINE_float('OPTIMIZATION_LEARNING_RATE', 0.05,
                   'Learning rate for gradient-based action optimization')
flags.DEFINE_boolean('PATH_SLOT_ACTIONS', False,
                     'Use 2-part path-slot actions for optimization')

FLAGS = flags.FLAGS

def main(argv):

    config = process_config(FLAGS)
    dtype_config.initialize_dtypes(config)
    
    jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays
    # increase line length for numpy print options
    jax.numpy.set_printoptions(linewidth=220)

    # Check device count
    print(f"Available devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")
    num_devices = len(jax.devices())
    assert (num_devices == 1), "Please specify one device using VISIBLE_DEVICES flag or run train_multidevice.py"
    config.NUM_DEVICES = num_devices

    # Set flags for debugging
    jax.config.update("jax_debug_nans", config.DEBUG_NANS)
    jax.config.update("jax_disable_jit", config.DISABLE_JIT)
    jax.config.update("jax_enable_x64", config.ENABLE_X64)
    # The following flags can improve GPU performance for jaxlib>=0.4.18
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
        '--xla_gpu_enable_async_collectives=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
    )
    # Option to print memory usage for debugging OOM errors
    if config.PRINT_MEMORY_USE:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        os.environ["TF_CPP_VMODULE"] = "bfc_allocator=1"
    # Set the fraction of memory to pre-allocate
    if config.PREALLOCATE_MEM:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = config.PREALLOCATE_MEM_FRACTION
        print(f"XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    print(f"XLA_PYTHON_CLIENT_PREALLOCATE={os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")

    # Setup the project name, experiment name for wandb and plots
    run_name = create_run_name(config)
    project_name = config.PROJECT if config.PROJECT else run_name
    experiment_name = config.EXPERIMENT_NAME if config.EXPERIMENT_NAME else run_name

    # Setup wandb
    if config.WANDB:
        setup_wandb(config, project_name, experiment_name)

    # Print every flag and its name
    if config.DEBUG:
        print('non-flag arguments:', argv)
        jax.config.update("jax_debug_nans", True)
    if config.NO_TRUNCATE:
        jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays
        # increase line length for numpy print options
        jax.numpy.set_printoptions(linewidth=220)

    if not config.NO_PRINT_FLAGS:
        for name in config:
            print(name, config[name])

    rng = jax.random.PRNGKey(config.SEED)
    rng = jax.random.split(rng, config.NUM_LEARNERS)

    config.ROLLOUT_LENGTH = config.max_requests

    if (config.RETRAIN_MODEL or config.EVAL_MODEL) and not config.model:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        model = orbax_checkpointer.restore(pathlib.Path(config.MODEL_PATH))
        config.model = model

    NUM_UPDATES = (
            config.TOTAL_TIMESTEPS // config.ROLLOUT_LENGTH // config.NUM_ENVS
    )
    MINIBATCH_SIZE = (
            config.ROLLOUT_LENGTH * config.NUM_ENVS // config.NUM_MINIBATCHES
    )
    config.NUM_UPDATES = NUM_UPDATES
    config.MINIBATCH_SIZE = MINIBATCH_SIZE

    with TimeIt(tag="COMPILATION"):
        print(
            f"\n---BEGINNING COMPILATION---\n"
            f"Total timesteps per increment: {config.STEPS_PER_INCREMENT * config.NUM_LEARNERS}\n"
            f"Timesteps per environment per increment: {config.STEPS_PER_INCREMENT // config.NUM_ENVS}\n"
            f"Total updates per increment: {config.NUM_UPDATES * config.NUM_MINIBATCHES}\n"
        )

        rng = jax.random.PRNGKey(config.SEED)
        if config.NUM_LEARNERS > 1:
            rng = jax.random.split(rng, config.NUM_LEARNERS)
            experiment_fn = (
                get_learner_fn
                if not (config.EVAL_HEURISTIC or config.EVAL_MODEL)
                else get_eval_fn
            )
            experiment_input, env, env_params = jax.vmap(
                experiment_data_setup, axis_name="learner", in_axes=(None, 0)
            )(config, rng)
            experiment_fn = experiment_fn(env, env_params, experiment_input[0], config)
            run_experiment = (
                jax.jit(jax.vmap(experiment_fn, axis_name="learner"))
                .lower(experiment_input)
                .compile()
            )
        else:
            experiment_fn = (
                get_learner_fn
                if not (config.EVAL_HEURISTIC or config.EVAL_MODEL)
                else get_eval_fn
            )
            experiment_input, env, env_params = experiment_data_setup(config, rng)
            experiment_fn = experiment_fn(env, env_params, experiment_input, config)
            run_experiment = jax.jit(experiment_fn).lower(experiment_input).compile()
    
    # N.B. that increasing number of learner will increase the number of steps
    # (essentially training for total_timesteps separately per learner)

    start_time = time.time()
    with TimeIt(tag='EXECUTION', frames=config.TOTAL_TIMESTEPS * config.NUM_LEARNERS):
        out = run_experiment(experiment_input)
        out["final_actions"].block_until_ready()  # Wait for all devices to finish
    total_time = time.time() - start_time

    # Output leaf nodes have dimensions:
    # (learner, num_updates, rollout_length, num_envs)
    # For eval, output leaf nodes have dimensions:
    # (learner, num_updates, rollout_length, num_envs)

    print(f"Total time taken: {total_time:.2f} seconds")
    print("Final rewards:", out["final_reward"])
    print("Best rewards:", out["best_reward"])
    print("Heuristic rewards:", out["heuristic_reward"])
    print("Final actions:", out["final_actions"])
    print("Best actions:", out["best_actions"])
    print("Heuristic actions:", out["heuristic_actions"])
    actions = jnp.vstack((out["best_actions"], out["heuristic_actions"], out["final_actions"])).T
    print("All actions (best, heuristic, final):\n", actions)


if __name__ == "__main__":
    FLAGS(sys.argv)
    # JAM imports come after GPU selection (to avoid initializing a process on every GPU)
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint
    import optax
    from jax.lib import xla_bridge
    from xlron.environments.make_env import process_config
    from xlron.environments.env_funcs import create_run_name
    from xlron.environments.wrappers import TimeIt
    from optimize_actions_sgd import get_learner_fn
    from xlron.heuristics.eval_heuristic import get_eval_fn
    from xlron.train.train_utils import save_model, log_metrics, setup_wandb, experiment_data_setup
    app.run(main)
