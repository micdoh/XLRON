import wandb
import sys
import os
import time
import pathlib
import matplotlib.pyplot as plt
from absl import app, flags
import xlron.train.parameter_flags
import numpy as np


# TODO - Write function to profile execution time as a function of num_envs for a single device
# TODO - Write function to profile execution time as a function of num_envs and num_(emulated)_devices


FLAGS = flags.FLAGS


def main(argv):

    # Set visible devices
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
    num_devices = len(FLAGS.VISIBLE_DEVICES.split(",")) if FLAGS.USE_PMAP else 1
    import jax.numpy as jnp
    import orbax.checkpoint
    from flax.training import orbax_utils
    from xlron.environments.env_funcs import create_run_name
    from xlron.environments.wrappers import TimeIt
    from xlron.train.ppo import make_train
    from xlron.heuristics.eval_heuristic import make_eval
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
        wandb.define_metric("update_step")
        wandb.define_metric("lengths", step_metric="update_step")
        wandb.define_metric("returns", step_metric="update_step")
        wandb.define_metric("cum_returns", step_metric="update_step")
        wandb.define_metric("episode_lengths", step_metric="update_step")
        wandb.define_metric("episode_returns", step_metric="update_step")

    # Print every flag and its name
    if FLAGS.DEBUG:
        print('non-flag arguments:', argv)
        jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays
    for name in FLAGS:
        print(name, FLAGS[name].value)

    rng = jax.random.PRNGKey(FLAGS.SEED)

    make_func = make_train if not (FLAGS.EVAL_HEURISTIC or FLAGS.EVAL_MODEL) else make_eval

    if FLAGS.EVAL_MODEL:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        model = orbax_checkpointer.restore(pathlib.Path(FLAGS.MODEL_PATH))
        FLAGS.__setattr__("model", model)

    with TimeIt(tag='COMPILATION'):
        if FLAGS.USE_PMAP:
            # TODO - Fix this to be like Anakin architecture (share gradients across devices)
            # TODO - Retrieve the output using pmean/sum etc.
            FLAGS.ORDERED = False
            rng = jax.random.split(rng, num_devices*FLAGS.NUM_SEEDS)
            if FLAGS.NUM_SEEDS > 1:
                train_jit = jax.pmap(jax.vmap(make_func(FLAGS)), devices=jax.devices()).lower(rng).compile()
            else:
                train_jit = jax.pmap(make_func(FLAGS), devices=jax.devices()).lower(rng).compile()
        else:
            if FLAGS.NUM_SEEDS > 1:
                rng = jax.random.split(rng, FLAGS.NUM_SEEDS)
                train_jit = jax.jit(jax.vmap(make_func(FLAGS))).lower(rng).compile()
            else:
                train_jit = jax.jit(make_func(FLAGS)).lower(rng).compile()

    # N.B. that increasing number of seeds or devices will increase the number of steps
    # (essentially training separately per device/seed)
    print(f"Running {FLAGS.TOTAL_TIMESTEPS * num_devices * FLAGS.NUM_SEEDS} timesteps over {num_devices} devices")
    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS * num_devices * FLAGS.NUM_SEEDS):
        out = train_jit(rng)
        out["metrics"]["episode_returns"].block_until_ready()  # Wait for all devices to finish

    # Save model params
    if FLAGS.SAVE_MODEL:
        train_state = out["runner_state"][0]
        save_data = {"model": train_state, "config": FLAGS}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpoint
        er()
        save_args = orbax_utils.save_args_from_target(save_data)
        # Get path to current file
        model_path = FLAGS.MODEL_PATH if FLAGS.MODEL_PATH else pathlib.Path(__file__).resolve().parents[2] / "models" / run_name
        # If model_path dir already exists, append a number to the end
        i = 1
        while model_path.exists():
            model_path = FLAGS.MODEL_PATH if FLAGS.MODEL_PATH else pathlib.Path(__file__).resolve().parents[2] / "models" / f"{run_name}_{i}"
            i += 1
        print(f"Saving model to {model_path}")
        orbax_checkpointer.save(model_path, save_data, save_args=save_args)
        # TODO - Upload model to wandb
        if FLAGS.WANDB:
            print((model_path / "*").absolute())
            wandb.save(str((model_path / "*").absolute()), base_path=str(model_path.parent))

    # Summarise the returns
    if FLAGS.NUM_SEEDS > 1:
        # Take mean on env dimension (-1) then seed dimension (0)
        # For ref, dimension order is (num_seeds, num_updates, num_steps, num_envs)
        episode_returns_mean = out["metrics"]["episode_returns"].mean(-1).mean(0).reshape(-1)
        episode_returns_std = out["metrics"]["episode_returns"].mean(-1).std(0).reshape(-1)
        episode_lengths_mean = out["metrics"]["episode_lengths"].mean(-1).mean(0).reshape(-1)
        episode_lengths_std = out["metrics"]["episode_lengths"].mean(-1).std(0).reshape(-1)
        cum_returns_mean = out["metrics"]["cum_returns"].mean(-1).mean(0).reshape(-1)
        cum_returns_std = out["metrics"]["cum_returns"].mean(-1).std(0).reshape(-1)
        returns_mean = out["metrics"]["returns"].mean(-1).mean(0).reshape(-1)
        returns_std = out["metrics"]["returns"].mean(-1).std(0).reshape(-1)
        lengths_mean = out["metrics"]["lengths"].mean(-1).mean(0).reshape(-1)
        lengths_std = out["metrics"]["lengths"].mean(-1).std(0).reshape(-1)
        accepted_services_mean = out["metrics"]["accepted_services"].mean(-1).mean(0).reshape(-1)
        accepted_services_std = out["metrics"]["accepted_services"].mean(-1).std(0).reshape(-1)
        accepted_bitrate_mean = out["metrics"]["accepted_bitrate"].mean(-1).mean(0).reshape(-1)
        accepted_bitrate_std = out["metrics"]["accepted_bitrate"].mean(-1).std(0).reshape(-1)
        done = out["metrics"]["done"].mean(-1).mean(0).reshape(-1)
        # episode_accepted_services_mean = out["metrics"]["episode_accepted_services"].mean(-1).mean(0).reshape(-1)
        # episode_accepted_services_std = out["metrics"]["episode_accepted_services"].mean(-1).std(0).reshape(-1)
        # episode_accepted_bitrate_mean = out["metrics"]["episode_accepted_bitrate"].mean(-1).mean(0).reshape(-1)
        # episode_accepted_bitrate_std = out["metrics"]["episode_accepted_bitrate"].mean(-1).std(0).reshape(-1)
    else:
        # N.B. This is the same as the above code, but without the mean on the seed dimension
        # This means the results are still per update step
        episode_returns_mean = out["metrics"]["episode_returns"].mean(-1).reshape(-1)
        episode_returns_std = out["metrics"]["episode_returns"].std(-1).reshape(-1)
        episode_lengths_mean = out["metrics"]["episode_lengths"].mean(-1).reshape(-1)
        episode_lengths_std = out["metrics"]["episode_lengths"].std(-1).reshape(-1)
        cum_returns_mean = out["metrics"]["cum_returns"].mean(-1).reshape(-1)
        cum_returns_std = out["metrics"]["cum_returns"].std(-1).reshape(-1)
        returns_mean = out["metrics"]["returns"].mean(-1).reshape(-1)
        returns_std = out["metrics"]["returns"].std(-1).reshape(-1)
        lengths_mean = out["metrics"]["lengths"].mean(-1).reshape(-1)
        lengths_std = out["metrics"]["lengths"].std(-1).reshape(-1)
        accepted_services_mean = out["metrics"]["accepted_services"].mean(-1).reshape(-1)
        accepted_services_std = out["metrics"]["accepted_services"].std(-1).reshape(-1)
        accepted_bitrate_mean = out["metrics"]["accepted_bitrate"].mean(-1).reshape(-1)
        accepted_bitrate_std = out["metrics"]["accepted_bitrate"].std(-1).reshape(-1)
        done = out["metrics"]["done"].mean(-1).reshape(-1)
        # episode_accepted_services_mean = out["metrics"]["episode_accepted_services"].mean(-1).reshape(-1)
        # episode_accepted_services_std = out["metrics"]["episode_accepted_services"].std(-1).reshape(-1)
        # episode_accepted_bitrate_mean = out["metrics"]["episode_accepted_bitrate"].mean(-1).reshape(-1)
        # episode_accepted_bitrate_std = out["metrics"]["episode_accepted_bitrate"].std(-1).reshape(-1)

    episode_ends = np.where(done == 1)[0]
    # shift end indices by -1
    episode_ends = episode_ends - 1
    # get values of accepted services and bitrate at episode ends
    episode_end_accepted_services = accepted_services_mean[episode_ends]
    episode_end_accepted_bitrate = accepted_bitrate_mean[episode_ends]
    # Do box and whisker plot of accepted services and bitrate at episode ends
    plt.boxplot(episode_end_accepted_services)
    plt.title(experiment_name)
    plt.show()

    # This is valid for the case of +1 success -1 fail per request
    service_blocking_probability = 1 - ((episode_returns_mean + FLAGS.max_requests) / (2*FLAGS.max_requests))
    service_blocking_probability_std = episode_returns_std / (2*FLAGS.max_requests)

    service_blocking_probability = 1 - ((cum_returns_mean + lengths_mean) / (2*lengths_mean))
    service_blocking_probability_std = returns_std / (2*lengths_mean)

    # TODO - Define blocking probability metric
    # TODO - Get bit rate blocking and service blocking

    if FLAGS.incremental_loading:
        plot_metric = accepted_services_mean
        plot_metric_std = accepted_services_std
        plot_metric_name = "Accepted Services"
    elif FLAGS.end_first_blocking:
        plot_metric = episode_lengths_mean
        plot_metric_std = episode_lengths_std
        plot_metric_name = "Episode Length"
    else:
        plot_metric = service_blocking_probability
        plot_metric_std = service_blocking_probability_std
        plot_metric_name = "Service Blocking Probability"

    plt.boxplot(plot_metric[episode_ends])
    plt.title(experiment_name)
    plt.show()

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

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
    plt.savefig(f"{experiment_name}.png")

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
        accepted_services_mean = accepted_services_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_services_std = accepted_services_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_bitrate_mean = accepted_bitrate_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        accepted_bitrate_std = accepted_bitrate_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        done = done[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        # episode_accepted_services_mean = episode_accepted_services_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        # episode_accepted_services_std = episode_accepted_services_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        # episode_accepted_bitrate_mean = episode_accepted_bitrate_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        # episode_accepted_bitrate_std = episode_accepted_bitrate_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)

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
                "accepted_services_mean": accepted_services_mean[i],
                "accepted_services_std": accepted_services_std[i],
                "accepted_bitrate_mean": accepted_bitrate_mean[i],
                "accepted_bitrate_std": accepted_bitrate_std[i],
                # "episode_accepted_services_mean": episode_accepted_services_mean[i],
                # "episode_accepted_services_std": episode_accepted_services_std[i],
                # "episode_accepted_bitrate_mean": episode_accepted_bitrate_mean[i],
                # "episode_accepted_bitrate_std": episode_accepted_bitrate_std[i],
            }
            wandb.log(log_dict)

    plt.show()


if __name__ == "__main__":
    app.run(main)
