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

FLAGS = flags.FLAGS


def restrict_visible_gpus(gpu_indices=None, auto_select=False):
    """
    Restricts JAX to only initialize and use specific GPUs.
    To avoid initializing a JAX process on every available device, must be called BEFORE importing JAX.
    Automated GPU selection is based on free memory.

    Args:
        gpu_indices: List of GPU indices to make visible (e.g. [0,3] for first and fourth GPUs)
        auto_select: If True, automatically select the GPU with most free memory
                     (overrides gpu_indices if both are provided)

    Returns:
        selected_gpu: The index of the selected GPU (for later reference)
    """
    if os.environ.get('COLAB_TPU_ADDR', False):
        print("Running on TPU")
        return

    def get_gpu_memory_info():
        """Get memory information for NVIDIA GPUs using nvidia-smi."""
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )

            gpu_info = []
            for line in result.strip().split('\n'):
                idx, free_mem = line.split(',')
                gpu_info.append((int(idx.strip()), int(free_mem.strip())))

            return gpu_info
        except:
            # If nvidia-smi fails, return an empty list and run on CPU
            return []

    # Auto-select GPU with most free memory if requested
    gpu_memory_info = get_gpu_memory_info()
    if not gpu_memory_info:
        print("Defaulting to CPU")
        return
    if auto_select:
        if gpu_memory_info:
            # Find GPU with most free memory
            selected_gpu, free_mem = max(gpu_memory_info, key=lambda x: x[1])
            gpu_indices = [selected_gpu]
            print(f"Auto-selected GPU {selected_gpu} with {free_mem} MB free memory")
        else:
            # Default to GPU 0 if we can't get memory info
            gpu_indices = [0]
            print("Could not get GPU memory info, defaulting to GPU 0")

    # Default to first GPU if nothing specified
    if gpu_indices is None:
        gpu_indices = [0]

    # Create comma-separated string of GPU indices
    visible_gpus = ','.join(str(idx) for idx in gpu_indices)

    # Set environment variables to restrict visible GPUs
    # This must happen BEFORE importing JAX
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

    # Prevent excess memory allocation
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    print(f"Restricted visible GPUs to: {visible_gpus}")
    return gpu_indices[0]  # Return the first (or only) selected GPU index


def main(argv):

    config = process_config(FLAGS)

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

    with TimeIt(tag='COMPILATION'):
        print(f"\n---BEGINNING COMPILATION---\n"
              f"Independent learners: {config.NUM_LEARNERS}\n"
              f"Environments per learner: {config.NUM_ENVS}\n"
              f"Number of devices: {num_devices}\n"
              f"Learners per device: {config.NUM_LEARNERS // num_devices}\n"
              f"Timesteps per learner: {config.TOTAL_TIMESTEPS}\n"
              f"Timesteps per environment: {config.TOTAL_TIMESTEPS // config.NUM_ENVS}\n"
              f"Total timesteps: {config.TOTAL_TIMESTEPS * config.NUM_LEARNERS}\n"
              f"Total updates: {config.NUM_UPDATES * config.NUM_MINIBATCHES}\n"
              f"Batch size: {config.NUM_ENVS * config.ROLLOUT_LENGTH}\n"
              f"Minibatch size: {config.MINIBATCH_SIZE}\n")

        experiment_fn = get_learner_fn if not (config.EVAL_HEURISTIC or config.EVAL_MODEL) else get_eval_fn
        experiment_input, env, env_params = jax.vmap(experiment_data_setup, axis_name='learner', in_axes=(None, 0))(config, rng)
        experiment_fn = experiment_fn(env, env_params, experiment_input[0], config)
        run_experiment = jax.jit(jax.vmap(experiment_fn, axis_name='learner')).lower(experiment_input).compile()

    # N.B. that increasing number of learner will increase the number of steps
    # (essentially training for total_timesteps separately per learner)

    start_time = time.time()
    with TimeIt(tag='EXECUTION', frames=config.TOTAL_TIMESTEPS * config.NUM_LEARNERS):
        out = run_experiment(experiment_input)
        out["metrics"]["returns"].block_until_ready()  # Wait for all devices to finish
    total_time = time.time() - start_time

    # Output leaf nodes have dimensions:
    # (learner, num_updates, rollout_length, num_envs)
    # For eval, output leaf nodes have dimensions:
    # (learner, num_updates, rollout_length, num_envs)

    # Save model params
    if config.SAVE_MODEL:
        # Merge seed_device and seed dimensions
        print(out["runner_state"])
        train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
        save_model(train_state, run_name, None)  # TODO - make flags compatible

    # END OF TRAINING

    def merge_func(x):
        # Original dims: (learner, num_updates, rollout_length, num_envs)
        # Compute the new shape
        learner, num_updates, rollout_length, num_envs = x.shape
        new_shape = (learner * num_envs, num_updates, rollout_length)
        # Perform a single transpose operation
        x = jnp.transpose(x, (0, 3, 1, 2))
        # Reshape to merge learner and num_envs dimensions
        return x.reshape(new_shape)

    log_metrics(config, out, experiment_name, total_time, merge_func)


if __name__ == "__main__":
    FLAGS(sys.argv)
    auto_select = False if FLAGS.VISIBLE_DEVICES else True
    selected_gpu = restrict_visible_gpus(gpu_indices=FLAGS.VISIBLE_DEVICES, auto_select=auto_select)
    print(f"Selected GPU: {selected_gpu}")
    # JAM imports come after GPU selection (to avoid initializing a process on every GPU)
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint
    from jax.lib import xla_bridge
    from xlron.environments.make_env import process_config
    from xlron.environments.env_funcs import create_run_name
    from xlron.environments.wrappers import TimeIt
    from xlron.train.ppo import get_learner_fn
    from xlron.heuristics.eval_heuristic import get_eval_fn
    from xlron.train.train_utils import save_model, log_metrics, setup_wandb, experiment_data_setup
    app.run(main)
