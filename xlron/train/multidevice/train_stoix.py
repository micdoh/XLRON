import wandb
import sys
import os
from absl import app, flags
import xlron.train.parameter_flags
import time
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FLAGS = flags.FLAGS


def main(argv):

    # Set visible devices
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if FLAGS.EMULATED_DEVICES:
        xla_flags = xla_flags + f" --xla_force_host_platform_device_count={FLAGS.EMULATED_DEVICES}"
        os.environ['XLA_FLAGS'] = xla_flags
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
    num_devices = len(jax.devices())  # or len(FLAGS.VISIBLE_DEVICES.split(","))
    import jax.numpy as jnp
    import orbax.checkpoint
    from xlron.environments.env_funcs import create_run_name
    from xlron.environments.wrappers import TimeIt
    from xlron.train.ppo_stoix import setup_experiment
    from xlron.train.train_utils import unreplicate_n_dims, merge_leading_dims, save_model, moving_average
    from xlron.heuristics.eval_heuristic import make_eval
    # The following flags can improve GPU performance for jaxlib>=0.4.18
    os.environ['XLA_FLAGS'] = xla_flags + (
        ' --xla_gpu_enable_triton_softmax_fusion=true '
        ' --xla_gpu_triton_gemm_any=True '
        ' --xla_gpu_enable_async_collectives=true '
        ' --xla_gpu_enable_latency_hiding_scheduler=true '
        ' --xla_gpu_enable_highest_priority_async_stream=true '
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

    if FLAGS.LOAD_MODEL or FLAGS.EVAL_MODEL:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        model = orbax_checkpointer.restore(pathlib.Path(FLAGS.MODEL_PATH))
        FLAGS.__setattr__("model", model)

    with TimeIt(tag='COMPILATION'):
        if not (FLAGS.EVAL_HEURISTIC or FLAGS.EVAL_MODEL):
            run_experiment, experiment_input = setup_experiment(FLAGS)
            run_experiment = run_experiment.lower(experiment_input).compile()
        else:
            run_experiment = jax.jit(make_eval(FLAGS)).lower(rng).compile()
            experiment_input = rng

    print(f"Running {FLAGS.TOTAL_TIMESTEPS * num_devices * FLAGS.NUM_LEARNERS} timesteps over {num_devices} devices")
    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS * num_devices * FLAGS.NUM_LEARNERS):
        out = run_experiment(experiment_input)
        out["metrics"]["episode_returns"].block_until_ready()  # Wait for all devices to finish

    # Save model params
    if FLAGS.SAVE_MODEL:
        train_state = unreplicate_n_dims(out["runner_state"])
        save_model(train_state, run_name, FLAGS)

    # END OF TRAINING

    # Summarise the returns
    # Take mean on env dimension (2) and device dimension (0)
    # Dimensions are (num_devices, num_rollouts, num_envs, rollout_length)
    # After merging, out dimensions are (num_envs*num_devices, num_rollouts, rollout_length)
    merge_func = lambda x: merge_leading_dims(jnp.transpose(x, (0, 2, 1, 3)), 2)
    merged_out = {k: jax.tree.map(merge_func, v) for k, v in out["metrics"].items()}
    
    get_mean = lambda x, y: x[y].mean(0).reshape(-1)
    get_std = lambda x, y: x[y].std(0).reshape(-1)

    episode_returns_mean = get_mean(merged_out, "episode_returns")
    episode_returns_std = get_std(merged_out, "episode_returns")
    episode_lengths_mean = get_mean(merged_out, "episode_lengths")
    episode_lengths_std = get_std(merged_out, "episode_lengths")
    cum_returns_mean = get_mean(merged_out, "cum_returns")
    cum_returns_std = get_std(merged_out, "cum_returns")
    returns_mean = get_mean(merged_out, "returns")
    returns_std = get_std(merged_out, "returns")
    lengths_mean = get_mean(merged_out, "lengths")
    lengths_std = get_std(merged_out, "lengths")
    accepted_services_mean = get_mean(merged_out, "accepted_services")
    accepted_services_std = get_std(merged_out, "accepted_services")
    accepted_bitrate_mean = get_mean(merged_out, "accepted_bitrate")
    accepted_bitrate_std = get_std(merged_out, "accepted_bitrate")
    total_bitrate_mean = get_mean(merged_out, "total_bitrate")
    total_bitrate_std = get_std(merged_out, "total_bitrate")
    utilisation_mean = get_mean(merged_out, "utilisation")
    utilisation_std = get_std(merged_out, "utilisation")
    done = get_mean(merged_out, "done")

    episode_ends = np.where(done == 1)[0] if not FLAGS.continuous_operation else np.arange(0, len(done), FLAGS.max_requests)[1:].astype(int)
    # shift end indices by -1
    episode_ends = episode_ends - 1
    # get values of accepted services and bitrate at episode ends
    service_blocking_probability = 1 - (accepted_services_mean / lengths_mean)
    service_blocking_probability_std = accepted_services_std / lengths_mean
    bitrate_blocking_probability = 1 - (accepted_bitrate_mean / total_bitrate_mean)
    bitrate_blocking_probability_std = accepted_bitrate_std / total_bitrate_mean
    episode_end_accepted_services = accepted_services_mean[episode_ends]
    episode_end_accepted_services_std = accepted_services_std[episode_ends]
    episode_end_accepted_bitrate = accepted_bitrate_mean[episode_ends]
    episode_end_accepted_bitrate_std = accepted_bitrate_std[episode_ends]
    episode_end_service_blocking_probability = service_blocking_probability[episode_ends]
    episode_end_service_blocking_probability_std = service_blocking_probability_std[episode_ends]
    episode_end_bitrate_blocking_probability = bitrate_blocking_probability[episode_ends]
    episode_end_bitrate_blocking_probability_std = bitrate_blocking_probability_std[episode_ends]
    episode_end_total_bitrate = total_bitrate_mean[episode_ends]
    episode_end_total_bitrate_std = total_bitrate_std[episode_ends]

    if FLAGS.PLOTTING:
        if FLAGS.incremental_loading:
            plot_metric = accepted_services_mean
            plot_metric_std = accepted_services_std
            plot_metric_name = "Accepted Services"
        elif FLAGS.end_first_blocking:
            plot_metric = episode_lengths_mean
            plot_metric_std = episode_lengths_std
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
        plt.boxplot(episode_end_accepted_services)
        plt.ylabel("Accepted Services")
        plt.title(experiment_name)
        plt.show()

        plt.boxplot(episode_end_accepted_bitrate)
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
        plt.xlabel("Update Step")
        plt.ylabel(plot_metric_name)
        plt.title(experiment_name)
        plt.show()

    if FLAGS.DATA_OUTPUT_FILE:
        # Save episode end metrics to file
        df = pd.DataFrame({
            "accepted_services": episode_end_accepted_services,
            "accepted_services_std": episode_end_accepted_services_std,
            "accepted_bitrate": episode_end_accepted_bitrate,
            "accepted_bitrate_std": episode_end_accepted_bitrate_std,
            "service_blocking_probability": episode_end_service_blocking_probability,
            "service_blocking_probability_std": episode_end_service_blocking_probability_std,
            "bitrate_blocking_probability": episode_end_bitrate_blocking_probability,
            "bitrate_blocking_probability_std": episode_end_bitrate_blocking_probability_std,
            "total_bitrate": episode_end_total_bitrate,
            "total_bitrate_std": episode_end_total_bitrate_std,
            "utilisation_mean": utilisation_mean[episode_ends],
            "utilisation_std": utilisation_std[episode_ends],
            "returns": episode_returns_mean[episode_ends],
            "returns_std": episode_returns_std[episode_ends],
            "cum_returns": cum_returns_mean[episode_ends],
            "cum_returns_std": cum_returns_std[episode_ends],
            "lengths": lengths_mean[episode_ends],
            "lengths_std": lengths_std[episode_ends],
        })
        df.to_csv(FLAGS.DATA_OUTPUT_FILE)

    if FLAGS.WANDB:
        # Log the data to wandb
        # Define the downsample factor to speed up upload to wandb
        # Then reshape the array and compute the mean
        chop = len(episode_returns_mean) % FLAGS.DOWNSAMPLE_FACTOR
        episode_returns_mean = episode_returns_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        episode_returns_std = episode_returns_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        episode_lengths_mean = episode_lengths_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        episode_lengths_std = episode_lengths_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        cum_returns_mean = cum_returns_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        cum_returns_std = cum_returns_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        returns_mean = returns_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        returns_std = returns_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        lengths_mean = lengths_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        lengths_std = lengths_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        service_blocking_probability = service_blocking_probability[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        service_blocking_probability_std = service_blocking_probability_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        bitrate_blocking_probability = bitrate_blocking_probability[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        bitrate_blocking_probability_std = bitrate_blocking_probability_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_services_mean = accepted_services_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_services_std = accepted_services_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_bitrate_mean = accepted_bitrate_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_bitrate_std = accepted_bitrate_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)

        for i in range(len(episode_end_accepted_services)):
            log_dict = {
                "episode_count": i,
                "episode_end_accepted_services": episode_end_accepted_services[i],
                "episode_end_accepted_services_std": episode_end_accepted_services_std[i],
                "episode_end_accepted_bitrate": episode_end_accepted_bitrate[i],
                "episode_end_accepted_bitrate_std": episode_end_accepted_bitrate_std[i],
            }
            wandb.log(log_dict)

        for i in range(len(episode_returns_mean)):
            # Log the data
            log_dict = {
                "update_step": i*FLAGS.DOWNSAMPLE_FACTOR,
                "episode_returns_mean": episode_returns_mean[i],
                "episode_returns_std": episode_returns_std[i],
                "episode_lengths_mean": episode_lengths_mean[i],
                "episode_lengths_std": episode_lengths_std[i],
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
            }
            wandb.log(log_dict)


if __name__ == "__main__":
    app.run(main)
