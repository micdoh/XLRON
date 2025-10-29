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
from typing import Optional, Union, Dict, Any

# JAX and related imports
from xlron import dtype_config
import jax
import jax.numpy as jnp
import orbax.checkpoint
from xlron.environments.make_env import process_config
from xlron.environments.env_funcs import create_run_name
from xlron.environments.wrappers import TimeIt
from xlron.train.ppo import get_learner_fn
from xlron.heuristics.eval_heuristic import get_eval_fn
from xlron.train.train_utils import save_model, print_metrics, plot_metrics, log_metrics, log_actions, setup_wandb, experiment_data_setup

FLAGS = flags.FLAGS

# Create a global mutable container to collect data
collected_states = []


def identify_default_device(gpu_index=None, auto_select=False):
    """
    Identifies and sets the default JAX device, preferring the GPU with most free memory.

    Args:
        gpu_index: Specific GPU index to use (0-indexed). If None and auto_select=False, uses GPU 0
        auto_select: If True, automatically select the GPU with most free memory
                     (overrides gpu_index if both are provided)

    Returns:
        jax_device: The selected JAX device object
    """
    # Check if running on TPU
    if os.environ.get('COLAB_TPU_ADDR', False):
        print("Running on TPU")
        device = jax.devices()[0]
        jax.config.update('jax_default_device', device)
        return device

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
        except Exception as e:
            print(f"Warning: Could not query GPU info: {e}")
            return []

    # Get available JAX devices
    jax_devices = jax.devices()

    # If no GPUs available, use default device (likely CPU)
    if not any(d.platform == 'gpu' for d in jax_devices):
        print("No GPUs detected, using default device (CPU)")
        device = jax_devices[0]
        jax.config.update('jax_default_device', device)
        return device

    # Get GPU memory information
    gpu_memory_info = get_gpu_memory_info()

    # Auto-select GPU with most free memory
    if auto_select and gpu_memory_info:
        selected_gpu_idx, free_mem = max(gpu_memory_info, key=lambda x: x[1])
        print(f"Auto-selected GPU {selected_gpu_idx} with {free_mem} MB free memory")

        # Display all GPU memory info for context
        print("All GPUs:")
        for idx, mem in gpu_memory_info:
            print(f"  GPU {idx}: {mem} MB free")

        gpu_index = selected_gpu_idx

    # Default to GPU 0 if nothing specified
    if gpu_index is None:
        gpu_index = 0
        print("No GPU specified, defaulting to GPU 0")

    # Find the corresponding JAX device
    try:
        device = [d for d in jax_devices if d.platform == 'gpu'][gpu_index]
        print(f"Setting default device to: {device}")
        jax.config.update('jax_default_device', device)
        return device
    except IndexError:
        print(f"Warning: GPU {gpu_index} not found, using first available GPU")
        device = [d for d in jax_devices if d.platform == 'gpu'][0]
        jax.config.update('jax_default_device', device)
        return device


def main(argv):

    config = process_config(FLAGS)
    
    def merge_func(x):
        # Original dims: (learner, num_updates, rollout_length, num_envs)
        # Compute the new shape
        # If X has only 2 or 3 dims then add more leading dims until it has 4
        if x.ndim == 2:
            x = x[None, :, :, None]
        elif x.ndim == 3 and config.NUM_LEARNERS == 1:
            x = x[None, :, :, :]
        elif x.ndim == 3 and config.NUM_ENVS > 1:
            x = x[:, :, :, None]
        learner, num_updates, rollout_length, num_envs = x.shape
        new_shape = (learner * num_envs, num_updates, rollout_length)
        # Perform a single transpose operation
        x = jnp.transpose(x, (0, 3, 1, 2))
        # Reshape to merge learner and num_envs dimensions
        return x.reshape(new_shape)

    # Check device count
    print(f"Available devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")
    num_devices = len(jax.devices())
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

    if (config.RETRAIN_MODEL or config.EVAL_MODEL) and not config.model:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        model = orbax_checkpointer.restore(pathlib.Path(config.MODEL_PATH))
        config.model = model
        
    print(
        f"Independent learners: {config.NUM_LEARNERS}\n"
        f"Environments per learner: {config.NUM_ENVS}\n"
        f"Number of devices: {num_devices}\n"
        f"Learners per device: {config.NUM_LEARNERS // num_devices}\n"
        f"Timesteps per learner: {config.TOTAL_TIMESTEPS}\n"
        f"Timesteps per environment: {config.TOTAL_TIMESTEPS // config.NUM_ENVS}\n"
        f"Total timesteps: {config.TOTAL_TIMESTEPS * config.NUM_LEARNERS}\n"
        f"Total updates: {config.NUM_INCREMENTS * config.NUM_UPDATES * config.NUM_MINIBATCHES}\n"
        f"Batch size: {config.NUM_ENVS * config.ROLLOUT_LENGTH}\n"
        f"Minibatch size: {config.MINIBATCH_SIZE}\n"
    )

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
            experiment_fn = get_learner_fn if not (config.EVAL_HEURISTIC or config.EVAL_MODEL) else get_eval_fn
            experiment_input, env, env_params = jax.vmap(experiment_data_setup, axis_name='learner', in_axes=(None, 0))(config, rng)
            experiment_fn = experiment_fn(env, env_params, experiment_input[0], config)
            run_experiment = jax.jit(jax.vmap(experiment_fn, axis_name='learner')).lower(experiment_input).compile()
        else:
            experiment_fn = get_learner_fn if not (config.EVAL_HEURISTIC or config.EVAL_MODEL) else get_eval_fn
            experiment_input, env, env_params = experiment_data_setup(config, rng)
            experiment_fn = experiment_fn(env, env_params, experiment_input, config)
            run_experiment = jax.jit(experiment_fn).lower(experiment_input).compile()
    
    
    # START TRAINING
    start_time = time.time()
    log_time = 0.0
    episode_count = update_count = step_count = 0
    processed_data_all: Dict[str, Dict[str, jax.Array]] = {}
    print(f"Running {config.NUM_INCREMENTS} increments of training")
    for i in range(config.NUM_INCREMENTS):
        print(f"\n---INCREMENT {i + 1}/{config.NUM_INCREMENTS}---")
        # Run the increment
        with TimeIt(tag="EXECUTION", frames=config.STEPS_PER_INCREMENT * config.NUM_LEARNERS):
            out = run_experiment(experiment_input)
            out["metrics"]["returns"].block_until_ready()  # Wait for all devices to finish
        run_time = time.time() - start_time - log_time
        # Save model params
        if config.SAVE_MODEL:
            # Merge seed_device and seed dimensions
            train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
            save_model(train_state, run_name, config)  # TODO - make flags compatible

        merged_out, processed_data = log_metrics(
            config,
            out,
            run_time,
            merge_func,
            episode_count=episode_count,
            update_count=update_count,
            step_count=step_count,
        )
        # Extend every item in processed data with new data
        episode_count += len(processed_data["returns"]["episode_end_mean"])
        step_count += config.STEPS_PER_INCREMENT // config.NUM_ENVS
        update_count += config.NUM_UPDATES * config.NUM_MINIBATCHES * config.UPDATE_EPOCHS
        # Concatenate arrays for each key
        processed_data_all = (
            processed_data
            if i == 0
            else jax.tree.map(lambda x, y: jnp.concatenate([x, y]), processed_data_all, processed_data)
        )
        log_time = log_time + time.time() - start_time - run_time
        # Update the experiment input for the next increment
        experiment_input = out["runner_state"]  # TrainState, EnvState, Obs, key, key

    # END OF TRAINING

    print_metrics(processed_data_all, config)
    if config.PLOTTING:
        plot_metrics(experiment_name, processed_data_all, config)
    if config.log_actions:
        log_actions(merged_out, processed_data, config)

if __name__ == "__main__":
    FLAGS(sys.argv)

    # Identify and set the default JAX device
    # If user specifies VISIBLE_DEVICES, use the first one; otherwise auto-select
    if FLAGS.VISIBLE_DEVICES:
        # Parse comma-separated GPU indices
        gpu_indices = [int(x) for x in FLAGS.VISIBLE_DEVICES.split(',')]
        default_device = identify_default_device(gpu_index=gpu_indices[0], auto_select=False)
    else:
        # Auto-select GPU with most free memory
        default_device = identify_default_device(auto_select=True)

    print(f"Default device set to: {default_device}")
    app.run(main)
