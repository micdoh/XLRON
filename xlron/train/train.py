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
    from xlron.train.train_utils import save_model
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
        wandb.define_metric("env_step")
        wandb.define_metric("lengths", step_metric="env_step")
        wandb.define_metric("returns", step_metric="env_step")
        wandb.define_metric("cum_returns", step_metric="env_step")
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
            experiment_input, env, env_params = jax.vmap(learner_data_setup, in_axes=(None, 0))(FLAGS, rng)
            learn = get_learner_fn(env, env_params, experiment_input[0], FLAGS)
            run_experiment = jax.jit(jax.vmap(learn)).lower(experiment_input).compile()
        else:
            run_experiment = jax.jit(jax.vmap(make_eval(FLAGS))).lower(rng).compile()
            experiment_input = rng

    # N.B. that increasing number of learner will increase the number of steps
    # (essentially training for total_timesteps separately per learner)

    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS * FLAGS.NUM_LEARNERS):
        out = run_experiment(experiment_input)
        out["metrics"]["episode_returns"].block_until_ready()  # Wait for all devices to finish

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

    merged_out = {k: jax.tree.map(merge_func, v) for k, v in out["metrics"].items()}
    get_mean = lambda x, y: x[y].mean(0).reshape(-1)
    get_std = lambda x, y: x[y].std(0).reshape(-1)

    if FLAGS.end_first_blocking:
        # Episode lengths are variable so  eturn the episode end values and the std of the episode end values
        episode_ends = np.where(merged_out["done"].reshape(-1) == 1)[0] - 1
        get_episode_end_mean = lambda x: x.reshape(-1)[episode_ends]
        get_episode_end_std = lambda x: jnp.full(x.reshape(-1)[episode_ends].shape, x.reshape(-1)[episode_ends].std())
    else:
        # Episode lengths are uniform so return the mean and std across envs at each episode end
        episode_ends = np.where(merged_out["done"].mean(0).reshape(-1) == 1)[0] - 1 \
            if not FLAGS.continuous_operation else np.arange(0, FLAGS.TOTAL_TIMESTEPS, FLAGS.max_timesteps)[1:].astype(int) - 1
        get_episode_end_mean = lambda x: x.mean(0).reshape(-1)[episode_ends]
        get_episode_end_std = lambda x: x.std(0).reshape(-1)[episode_ends]

    # Get episode end metrics
    returns_mean_episode_end = get_episode_end_mean(merged_out["returns"])
    returns_std_episode_end = get_episode_end_std(merged_out["returns"])
    lengths_mean_episode_end = get_episode_end_mean(merged_out["lengths"])
    lengths_std_episode_end = get_episode_end_std(merged_out["lengths"])
    cum_returns_mean_episode_end = get_episode_end_mean(merged_out["cum_returns"])
    cum_returns_std_episode_end = get_episode_end_std(merged_out["cum_returns"])
    accepted_services_mean_episode_end = get_episode_end_mean(merged_out["accepted_services"])
    accepted_services_std_episode_end = get_episode_end_std(merged_out["accepted_services"])
    accepted_bitrate_mean_episode_end = get_episode_end_mean(merged_out["accepted_bitrate"])
    accepted_bitrate_std_episode_end = get_episode_end_std(merged_out["accepted_bitrate"])
    total_bitrate_mean_episode_end = get_episode_end_mean(merged_out["total_bitrate"])
    total_bitrate_std_episode_end = get_episode_end_std(merged_out["total_bitrate"])
    utilisation_mean_episode_end = get_episode_end_mean(merged_out["utilisation"])
    utilisation_std_episode_end = get_episode_end_std(merged_out["utilisation"])
    service_blocking_probability_episode_end = 1 - (accepted_services_mean_episode_end / lengths_mean_episode_end)
    service_blocking_probability_std_episode_end = accepted_services_std_episode_end / lengths_mean_episode_end
    bitrate_blocking_probability_episode_end = 1 - (accepted_bitrate_mean_episode_end / total_bitrate_mean_episode_end)
    bitrate_blocking_probability_std_episode_end = accepted_bitrate_std_episode_end / total_bitrate_mean_episode_end

    returns_mean = get_mean(merged_out, "returns") if not FLAGS.end_first_blocking else returns_mean_episode_end
    returns_std = get_std(merged_out, "returns") if not FLAGS.end_first_blocking else returns_std_episode_end
    lengths_mean = get_mean(merged_out, "lengths") if not FLAGS.end_first_blocking else lengths_mean_episode_end
    lengths_std = get_std(merged_out, "lengths") if not FLAGS.end_first_blocking else lengths_std_episode_end
    cum_returns_mean = get_mean(merged_out, "cum_returns") if not FLAGS.end_first_blocking else cum_returns_mean_episode_end
    cum_returns_std = get_std(merged_out, "cum_returns") if not FLAGS.end_first_blocking else cum_returns_std_episode_end
    accepted_services_mean = get_mean(merged_out, "accepted_services") if not FLAGS.end_first_blocking else accepted_services_mean_episode_end
    accepted_services_std = get_std(merged_out, "accepted_services") if not FLAGS.end_first_blocking else accepted_services_std_episode_end
    accepted_bitrate_mean = get_mean(merged_out, "accepted_bitrate") if not FLAGS.end_first_blocking else accepted_bitrate_mean_episode_end
    accepted_bitrate_std = get_std(merged_out, "accepted_bitrate") if not FLAGS.end_first_blocking else accepted_bitrate_std_episode_end
    total_bitrate_mean = get_mean(merged_out, "total_bitrate") if not FLAGS.end_first_blocking else total_bitrate_mean_episode_end
    total_bitrate_std = get_std(merged_out, "total_bitrate") if not FLAGS.end_first_blocking else total_bitrate_std_episode_end
    utilisation_mean = get_mean(merged_out, "utilisation") if not FLAGS.end_first_blocking else utilisation_mean_episode_end
    utilisation_std = get_std(merged_out, "utilisation") if not FLAGS.end_first_blocking else utilisation_std_episode_end
    # get values of service and bitrate blocking probs
    service_blocking_probability = 1 - (accepted_services_mean / lengths_mean) if not FLAGS.end_first_blocking else service_blocking_probability_episode_end
    service_blocking_probability_std = accepted_services_std / lengths_mean if not FLAGS.end_first_blocking else service_blocking_probability_std_episode_end
    bitrate_blocking_probability = 1 - (accepted_bitrate_mean / total_bitrate_mean) if not FLAGS.end_first_blocking else bitrate_blocking_probability_episode_end
    bitrate_blocking_probability_std = accepted_bitrate_std / total_bitrate_mean if not FLAGS.end_first_blocking else bitrate_blocking_probability_std_episode_end


    if FLAGS.PLOTTING:
        if FLAGS.incremental_loading:
            plot_metric = accepted_services_mean
            plot_metric_std = accepted_services_std
            plot_metric_name = "Accepted Services"
        elif FLAGS.end_first_blocking:
            plot_metric = lengths_mean_episode_end
            plot_metric_std = lengths_std_episode_end
            plot_metric_name = "Episode Length"
        elif FLAGS.reward_type == "service":
            plot_metric = service_blocking_probability
            plot_metric_std = service_blocking_probability_std
            plot_metric_name = "Service Blocking Probability"
        else:
            plot_metric = bitrate_blocking_probability
            plot_metric_std = bitrate_blocking_probability_std
            plot_metric_name = "Bitrate Blocking Probability"

        # Do box and whisker plot of accepted services and bitrate at episode ends
        plt.boxplot(accepted_services_mean)
        plt.ylabel("Accepted Services")
        plt.title(experiment_name)
        plt.show()

        plt.boxplot(accepted_bitrate_mean)
        plt.ylabel("Accepted Bitrate")
        plt.title(experiment_name)
        plt.show()

        plot_metric = moving_average(plot_metric, min(100, int(len(plot_metric)/2)))
        plot_metric_std = moving_average(plot_metric_std, min(100, int(len(plot_metric_std)/2)))
        plt.plot(plot_metric)
        plt.fill_between(
            range(len(plot_metric)),
            plot_metric - plot_metric_std,
            plot_metric + plot_metric_std,
            alpha=0.2
        )
        plt.xlabel("Environment Step" if not FLAGS.end_first_blocking else "Episode Count")
        plt.ylabel(plot_metric_name)
        plt.title(experiment_name)
        plt.show()

    if FLAGS.DATA_OUTPUT_FILE:
        # Save episode end metrics to file
        df = pd.DataFrame({
            "accepted_services": accepted_services_mean_episode_end,
            "accepted_services_std": accepted_services_std_episode_end,
            "accepted_bitrate": accepted_bitrate_mean_episode_end,
            "accepted_bitrate_std": accepted_bitrate_std_episode_end,
            "service_blocking_probability": service_blocking_probability_episode_end,
            "service_blocking_probability_std": service_blocking_probability_std_episode_end,
            "bitrate_blocking_probability": bitrate_blocking_probability_episode_end,
            "bitrate_blocking_probability_std": bitrate_blocking_probability_std_episode_end,
            "total_bitrate": total_bitrate_mean_episode_end,
            "total_bitrate_std": total_bitrate_std_episode_end,
            "utilisation_mean": utilisation_mean_episode_end,
            "utilisation_std": utilisation_std_episode_end,
            "returns": returns_mean_episode_end,
            "returns_std": returns_std_episode_end,
            "cum_returns": cum_returns_mean_episode_end,
            "cum_returns_std": cum_returns_std_episode_end,
            "lengths": lengths_mean_episode_end,
            "lengths_std": lengths_std_episode_end,
        })
        df.to_csv(FLAGS.DATA_OUTPUT_FILE)

    if FLAGS.WANDB:
        # Log the data to wandb
        # Define the downsample factor to speed up upload to wandb
        # Then reshape the array and compute the mean
        chop = len(returns_mean) % FLAGS.DOWNSAMPLE_FACTOR
        cum_returns_mean = cum_returns_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        cum_returns_std = cum_returns_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        returns_mean = returns_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        returns_std = returns_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        lengths_mean = lengths_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        lengths_std = lengths_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        total_bitrate_mean = total_bitrate_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        total_bitrate_std = total_bitrate_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        service_blocking_probability = service_blocking_probability[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        service_blocking_probability_std = service_blocking_probability_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        bitrate_blocking_probability = bitrate_blocking_probability[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        bitrate_blocking_probability_std = bitrate_blocking_probability_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_services_mean = accepted_services_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_services_std = accepted_services_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_bitrate_mean = accepted_bitrate_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_bitrate_std = accepted_bitrate_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        utilisation_mean = utilisation_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        utilisation_std = utilisation_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)

        for i in range(len(episode_ends)):
            log_dict = {
                "episode_count": i,
                "episode_end_accepted_services": accepted_services_mean_episode_end[i],
                "episode_end_accepted_services_std": accepted_services_std_episode_end[i],
                "episode_end_accepted_bitrate": accepted_bitrate_mean_episode_end[i],
                "episode_end_accepted_bitrate_std": accepted_bitrate_std_episode_end[i],
            }
            wandb.log(log_dict)

        for i in range(len(returns_mean)):
            # Log the data
            log_dict = {
                "update_step": i*FLAGS.DOWNSAMPLE_FACTOR,
                "cum_returns_mean": cum_returns_mean[i],
                "cum_returns_std": cum_returns_std[i],
                "returns_mean": returns_mean[i],
                "returns_std": returns_std[i],
                "lengths_mean": lengths_mean[i],
                "lengths_std": lengths_std[i],
                "service_blocking_probability": service_blocking_probability[i],
                "service_blocking_probability_std": service_blocking_probability_std[i],
                "bitrate_blocking_probability": bitrate_blocking_probability[i],
                "bitrate_blocking_probability_std": bitrate_blocking_probability_std[i],
                "accepted_services_mean": accepted_services_mean[i],
                "accepted_services_std": accepted_services_std[i],
                "accepted_bitrate_mean": accepted_bitrate_mean[i],
                "accepted_bitrate_std": accepted_bitrate_std[i],
                "total_bitrate_mean": total_bitrate_mean[i],
                "total_bitrate_std": total_bitrate_std[i],
                "utilisation_mean": utilisation_mean[i],
                "utilisation_std": utilisation_std[i],
            }
            wandb.log(log_dict)


if __name__ == "__main__":
    app.run(main)
