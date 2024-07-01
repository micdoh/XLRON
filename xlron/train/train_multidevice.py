import wandb
import sys
import os
import time
import pathlib
import math
import matplotlib.pyplot as plt
from absl import app, flags
import xlron.train.parameter_flags
import numpy as np
import pandas as pd

FLAGS = flags.FLAGS


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def main(argv):

    # Set visible devices
    if FLAGS.EMULATED_DEVICES:
        os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={FLAGS.EMULATED_DEVICES}"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.VISIBLE_DEVICES
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    # Allow time for environment variable to take effect
    time.sleep(2)

    # Jax imports must come after CUDA_VISIBLE_DEVICES is set
    import jax
    jax.config.update("jax_debug_nans", FLAGS.DEBUG_NANS)
    jax.config.update("jax_disable_jit", FLAGS.DEBUG)
    jax.config.update("jax_enable_x64", FLAGS.ENABLE_X64)
    print(f"Available devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")
    num_devices = len(jax.devices())  # or len(FLAGS.VISIBLE_DEVICES.split(","))
    FLAGS.__setattr__("NUM_DEVICES", num_devices)
    import jax.numpy as jnp
    import orbax.checkpoint
    from xlron.environments.env_funcs import create_run_name
    from xlron.environments.wrappers import TimeIt
    from xlron.train.ppo import make_train, reshape_keys
    from xlron.heuristics.eval_heuristic import make_eval
    from xlron.train.train_utils import merge_leading_dims, save_model, log_metrics
    # The following flags can improve GPU performance for jaxlib>=0.4.18
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
        '--xla_gpu_enable_async_collectives=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
    )

    # Option to print memory usage for debugging OOM errors
    if FLAGS.PRINT_MEMORY_USE:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        os.environ["TF_CPP_VMODULE"] = "bfc_allocator=1"

    # Set the fraction of memory to pre-allocate
    if FLAGS.PREALLOCATE_MEM:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = FLAGS.PREALLOCATE_MEM_FRACTION
        print(f"XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    print(f"XLA_PYTHON_CLIENT_PREALLOCATE={os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")

    run_name = create_run_name(FLAGS)
    project_name = FLAGS.PROJECT if FLAGS.PROJECT else run_name
    experiment_name = FLAGS.EXPERIMENT_NAME if FLAGS.EXPERIMENT_NAME else run_name

    if FLAGS.WANDB:
        wandb.setup(wandb.Settings(program="train.py", program_relpath="train.py"))
        run = wandb.init(
            project=project_name,
            save_code=True,  # optional
        )
        wandb.config.update(FLAGS)
        run.name = experiment_name
        wandb.define_metric('episode_count')
        wandb.define_metric("update_step")
        wandb.define_metric("lengths", step_metric="update_step")
        wandb.define_metric("returns", step_metric="update_step")
        wandb.define_metric("cum_returns", step_metric="update_step")
        wandb.define_metric("episode_lengths", step_metric="update_step")
        wandb.define_metric("episode_returns", step_metric="update_step")
        wandb.define_metric("episode_accepted_services", step_metric="episode_count")
        wandb.define_metric("episode_accepted_services_std", step_metric="episode_count")
        wandb.define_metric("episode_accepted_bitrate", step_metric="episode_count")
        wandb.define_metric("episode_accepted_bitrate_std", step_metric="episode_count")

    # Print every flag and its name
    if FLAGS.DEBUG:
        print('non-flag arguments:', argv)
        jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays
    for name in FLAGS:
        print(name, FLAGS[name].value)

    rng = jax.random.PRNGKey(FLAGS.SEED)

    make_func = make_train if not (FLAGS.EVAL_HEURISTIC or FLAGS.EVAL_MODEL) else make_eval

    if FLAGS.LOAD_MODEL or FLAGS.EVAL_MODEL:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        model = orbax_checkpointer.restore(pathlib.Path(FLAGS.MODEL_PATH))
        FLAGS.__setattr__("model", model)

    if FLAGS.NUM_LEARNERS > 1:
        rng = jax.random.split(rng, FLAGS.NUM_DEVICES*FLAGS.NUM_LEARNERS)
        rng = reshape_keys(rng, FLAGS.NUM_DEVICES, FLAGS.NUM_LEARNERS)
        # If running multiple independent learners, learners are distributed across devices.
        # Then, set the number of devices=1 to ensure environments aren't distributed across devices.
        FLAGS.__setattr__("NUM_DEVICES", 1)
    else:
        # Reshape keys to match the required (seed_device, seed) shape
        rng = jax.vmap(jax.random.split, in_axes=(0, None))(jax.random.split(rng, 1), 1)

    # Count total number of independent learners
    total_learners = math.prod(rng.shape[:-1])
    NUM_UPDATES = (
            FLAGS.TOTAL_TIMESTEPS // FLAGS.ROLLOUT_LENGTH // FLAGS.NUM_ENVS // FLAGS.NUM_DEVICES
    )
    MINIBATCH_SIZE = (
            FLAGS.NUM_DEVICES * FLAGS.NUM_ENVS * FLAGS.ROLLOUT_LENGTH // FLAGS.NUM_MINIBATCHES
    )
    FLAGS.__setattr__("NUM_UPDATES", NUM_UPDATES)
    FLAGS.__setattr__("MINIBATCH_SIZE", MINIBATCH_SIZE)

    with TimeIt(tag='COMPILATION'):
        print(f"\n---BEGINNING COMPILATION---\n"
              f"Independent learners: {total_learners}\n"
              f"Environments per learner: {FLAGS.NUM_ENVS}\n"
              f"Number of devices: {num_devices}\n"
              f"Seeds per device: {total_learners // num_devices}\n"
              f"Timesteps per learner: {FLAGS.TOTAL_TIMESTEPS}\n"
              f"Timesteps per environment: {FLAGS.TOTAL_TIMESTEPS // FLAGS.NUM_ENVS // FLAGS.NUM_DEVICES}\n"
              f"Total timesteps: {FLAGS.TOTAL_TIMESTEPS * total_learners}\n"
              f"Total updates: {FLAGS.NUM_UPDATES}\n"
              f"Batch size: {FLAGS.NUM_DEVICES * FLAGS.NUM_ENVS * FLAGS.ROLLOUT_LENGTH}\n"
              f"Minibatch size: {FLAGS.MINIBATCH_SIZE}\n")
        train_jit = jax.pmap(jax.vmap(make_func(FLAGS), axis_name='learner'), axis_name='device_learner').lower(rng).compile()

    # N.B. that increasing number of learners or devices will increase the number of steps
    # (essentially training separately for total_timesteps per learner)

    start_time = time.time()
    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS * total_learners):
        out = train_jit(rng)
        out["metrics"]["episode_returns"].block_until_ready()  # Wait for all devices to finish
    total_time = time.time() - start_time

    # Output leaf nodes have dimensions:
    # (device_learn, learner, num_updates, rollout_length, device_env, num_envs)
    # For eval, output leaf nodes have dimensions:
    # (device_learn, learner, num_updates, rollout_length, num_envs)

    # Save model params
    if FLAGS.SAVE_MODEL:
        # Merge seed_device and seed dimensions
        train_state = jax.tree.map(lambda x: merge_leading_dims(x, 2)[0], out["runner_state"])
        save_model(train_state, run_name, FLAGS)

    # END OF TRAINING

    def merge_func_train(x):
        # Original dims: (learner_device, learner, num_updates, rollout_length, env_device, num_envs)
        # Merge learner_device and learner dimensions
        x = merge_leading_dims(x, 2)
        # New dims: (num_learners, num_updates, rollout_length, env_device, num_envs)
        learner, num_updates, rollout_length, env_device, num_envs = x.shape
        # Merge env_device and num_envs dimensions
        x = jnp.transpose(x, (0, 3, 4, 1, 2))
        # Reshape to merge learner, env_device, num_envs dimensions
        new_shape = (learner * env_device * num_envs, num_updates, rollout_length)
        return x.reshape(new_shape)

    def merge_func_eval(x):
        # Original dims: (learner_device, learner, num_updates, rollout_length, num_envs)
        # Merge learner_device and learner dimensions
        x = merge_leading_dims(x, 2)
        # Compute the new shape
        learner, num_updates, rollout_length, num_envs = x.shape
        # Perform a single transpose operation
        x = jnp.transpose(x, (0, 3, 1, 2))
        # Reshape to merge learner and num_envs dimensions
        new_shape = (learner * num_envs, num_updates, rollout_length)
        return x.reshape(new_shape)

    merge_func = merge_func_train if not (FLAGS.EVAL_HEURISTIC or FLAGS.EVAL_MODEL) else merge_func_eval
    log_metrics(FLAGS, out, experiment_name, total_time, merge_func)


if __name__ == "__main__":
    app.run(main)
