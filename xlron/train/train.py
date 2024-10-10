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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.VISIBLE_DEVICES
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    # Allow time for environment variable to take effect
    time.sleep(2)

    # Jax imports must come after CUDA_VISIBLE_DEVICES is set
    import jax
    jax.config.update("jax_debug_nans", FLAGS.DEBUG_NANS)
    jax.config.update("jax_disable_jit", FLAGS.DISABLE_JIT)
    jax.config.update("jax_enable_x64", FLAGS.ENABLE_X64)
    print(f"Available devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")
    num_devices = len(jax.devices())
    assert (num_devices == 1), "Please specify one device using VISIBLE_DEVICES flag or run train_multidevice.py"
    FLAGS.__setattr__("NUM_DEVICES", num_devices)
    import jax.numpy as jnp
    import orbax.checkpoint
    from xlron.environments.env_funcs import create_run_name
    from xlron.environments.wrappers import TimeIt
    from xlron.train.ppo import learner_data_setup, get_learner_fn
    from xlron.heuristics.eval_heuristic import make_eval
    from xlron.train.train_utils import save_model, log_metrics, setup_wandb, define_env
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

    # Setup wandb
    if FLAGS.WANDB:
        setup_wandb(FLAGS, project_name, experiment_name)

    # Print every flag and its name
    if FLAGS.DEBUG:
        print('non-flag arguments:', argv)
        jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays

    if not FLAGS.NO_PRINT_FLAGS:
        for name in FLAGS:
            print(name, FLAGS[name].value)

    rng = jax.random.PRNGKey(FLAGS.SEED)
    rng = jax.random.split(rng, FLAGS.NUM_LEARNERS)

    if FLAGS.LOAD_MODEL or FLAGS.EVAL_MODEL:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        model = orbax_checkpointer.restore(pathlib.Path(FLAGS.MODEL_PATH))
        FLAGS.__setattr__("model", model)

    NUM_UPDATES = (
            FLAGS.TOTAL_TIMESTEPS // FLAGS.ROLLOUT_LENGTH // FLAGS.NUM_ENVS
    )
    MINIBATCH_SIZE = (
            FLAGS.ROLLOUT_LENGTH * FLAGS.NUM_ENVS // FLAGS.NUM_MINIBATCHES
    )
    FLAGS.__setattr__("NUM_UPDATES", NUM_UPDATES)
    FLAGS.__setattr__("MINIBATCH_SIZE", MINIBATCH_SIZE)

    with TimeIt(tag='COMPILATION'):
        print(f"\n---BEGINNING COMPILATION---\n"
              f"Independent learners: {FLAGS.NUM_LEARNERS}\n"
              f"Environments per learner: {FLAGS.NUM_ENVS}\n"
              f"Number of devices: {num_devices}\n"
              f"Learners per device: {FLAGS.NUM_LEARNERS // num_devices}\n"
              f"Timesteps per learner: {FLAGS.TOTAL_TIMESTEPS}\n"
              f"Timesteps per environment: {FLAGS.TOTAL_TIMESTEPS // FLAGS.NUM_ENVS}\n"
              f"Total timesteps: {FLAGS.TOTAL_TIMESTEPS * FLAGS.NUM_LEARNERS}\n"
              f"Total updates: {FLAGS.NUM_UPDATES * FLAGS.NUM_MINIBATCHES}\n"
              f"Batch size: {FLAGS.NUM_ENVS * FLAGS.ROLLOUT_LENGTH}\n"
              f"Minibatch size: {FLAGS.MINIBATCH_SIZE}\n")
        if not (FLAGS.EVAL_HEURISTIC or FLAGS.EVAL_MODEL):
            experiment_input, env, env_params = jax.vmap(learner_data_setup, axis_name='learner', in_axes=(None, 0))(FLAGS, rng)
            learn = get_learner_fn(env, env_params, experiment_input[0], FLAGS)
            run_experiment = jax.jit(jax.vmap(learn, axis_name='learner')).lower(experiment_input).compile()
        else:
            run_experiment = jax.jit(jax.vmap(make_eval(FLAGS))).lower(rng).compile()
            experiment_input = rng

    # N.B. that increasing number of learner will increase the number of steps
    # (essentially training for total_timesteps separately per learner)

    start_time = time.time()
    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS * FLAGS.NUM_LEARNERS):
        out = run_experiment(experiment_input)
        out["metrics"]["returns"].block_until_ready()  # Wait for all devices to finish
    total_time = time.time() - start_time

    # Output leaf nodes have dimensions:
    # (learner, num_updates, rollout_length, num_envs)
    # For eval, output leaf nodes have dimensions:
    # (learner, num_updates, rollout_length, num_envs)

    # Save model params
    if FLAGS.SAVE_MODEL:
        # Merge seed_device and seed dimensions
        train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
        save_model(train_state, run_name, FLAGS)

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

    log_metrics(FLAGS, out, experiment_name, total_time, merge_func)


if __name__ == "__main__":
    app.run(main)
