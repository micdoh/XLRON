import wandb
import sys
import os
import jax
import matplotlib.pyplot as plt
import orbax.checkpoint
from flax.training import orbax_utils
from absl import app, flags
from xlron.environments.env_funcs import *
from xlron.train.ppo import make_train
import xlron.train.parameter_flags

# TODO - Write function to profile execution time as a function of num_envs for a single device
# TODO - Write function to profile execution time as a function of num_envs and num_(emulated)_devices


FLAGS = flags.FLAGS


def main(argv):

    # Set visible devices
    if FLAGS.VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.VISIBLE_DEVICES
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

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

    if FLAGS.WANDB:
        wandb.setup(wandb.Settings(program="train.py", program_relpath="train.py"))
        run = wandb.init(
            project=FLAGS.PROJECT,
            save_code=True,  # optional
        )
        wandb.config.update(FLAGS)
        run.name = FLAGS.EXPERIMENT_NAME if FLAGS.EXPERIMENT_NAME else run.id
        wandb.define_metric("update_step")
        wandb.define_metric("returned_episode_returns_mean", step_metric="update_step")
        wandb.define_metric("returned_episode_returns_std", step_metric="update_step")
        wandb.define_metric("returned_episode_lengths_mean", step_metric="update_step")
        wandb.define_metric("returned_episode_lengths_std", step_metric="update_step")

    # Set the default device
    print(f"Available devices: {jax.devices()}")

    # Print every flag and its name
    if FLAGS.DEBUG:
        print('non-flag arguments:', argv)
        #jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays
    for name in FLAGS:
        print(name, FLAGS[name].value)

    rng = jax.random.PRNGKey(FLAGS.SEED)

    with TimeIt(tag='COMPILATION'):
        if FLAGS.USE_PMAP:
            # TODO - Fix this to be like Anakin architecture (share gradients across devices)
            # TODO - Retrieve the output using pmean/sum etc.
            FLAGS.ORDERED = False
            rng = jax.random.split(rng, len(jax.devices())*FLAGS.NUM_SEEDS)
            if FLAGS.NUM_SEEDS > 1:
                train_jit = jax.pmap(jax.vmap(make_train(FLAGS)), devices=jax.devices()).lower(rng).compile()
            else:
                train_jit = jax.pmap(make_train(FLAGS), devices=jax.devices()).lower(rng).compile()
        else:
            if FLAGS.NUM_SEEDS > 1:
                rng = jax.random.split(rng, FLAGS.NUM_SEEDS)
                train_jit = jax.jit(jax.vmap(make_train(FLAGS))).lower(rng).compile()
            else:
                train_jit = jax.jit(make_train(FLAGS)).lower(rng).compile()

    # N.B. that increasing number of seeds or devices will increase the number of steps
    # (essentially training separately per device/seed)
    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS * len(jax.devices()) * FLAGS.NUM_SEEDS):
        out = train_jit(rng)
        out["metrics"]["returned_episode_returns"].block_until_ready()  # Wait for all devices to finish

    # Save model params
    if FLAGS.SAVE_MODEL:
        train_state = out["runner_state"][0]
        save_data = {"model": train_state, "config": FLAGS}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(save_data)
        orbax_checkpointer.save(FLAGS.MODEL_PATH, save_data, save_args=save_args)

    # Summarise the returns
    if FLAGS.NUM_SEEDS > 1:
        # Take mean on env dimension (-1) then seed dimension (0)
        # For ref, dimension order is (num_seeds, num_updates, num_steps, num_envs)
        returned_episode_returns_mean = out["metrics"]["returned_episode_returns"].mean(-1).mean(0).reshape(-1)
        returned_episode_returns_std = out["metrics"]["returned_episode_returns"].mean(-1).std(0).reshape(-1)
        returned_episode_lengths_mean = out["metrics"]["returned_episode_lengths"].mean(-1).mean(0).reshape(-1)
        returned_episode_lengths_std = out["metrics"]["returned_episode_lengths"].mean(-1).std(0).reshape(-1)
    else:
        # N.B. This is the same as the above code, but without the mean on the seed dimension
        # This means the results are still per update step
        returned_episode_returns_mean = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
        returned_episode_returns_std = out["metrics"]["returned_episode_returns"].std(-1).reshape(-1)
        returned_episode_lengths_mean = out["metrics"]["returned_episode_lengths"].mean(-1).reshape(-1)
        returned_episode_lengths_std = out["metrics"]["returned_episode_lengths"].std(-1).reshape(-1)

    service_blocking_probability = 1 - ((returned_episode_returns_mean + FLAGS.max_requests) / (2*FLAGS.max_requests))

    plot_metric = returned_episode_lengths_mean if (FLAGS.env_type[:3] in ["rsa", "rms", "rwa"] and FLAGS.consecutive_loading) else returned_episode_returns_mean
    plot_metric_std = returned_episode_lengths_std if (FLAGS.env_type[:3] in ["rsa", "rms", "rwa"] and FLAGS.consecutive_loading) else returned_episode_returns_std
    plt.plot(plot_metric)
    plt.fill_between(
        range(len(plot_metric)),
        plot_metric - plot_metric_std,
        plot_metric + plot_metric_std,
        alpha=0.2
    )
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f"{FLAGS.EXPERIMENT_NAME}.png")
    plt.show()

    # TODO - Define blocking probability metric
    # TODO - Get bit rate blocking and service blocking

    if FLAGS.WANDB:
        # Log the data to wandb
        # Define the downsample factor to speed up upload to wandb
        # Then reshape the array and compute the mean
        chop = len(returned_episode_returns_mean) % FLAGS.DOWNSAMPLE_FACTOR
        returned_episode_returns_mean = returned_episode_returns_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        returned_episode_returns_std = returned_episode_returns_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        returned_episode_lengths_mean = returned_episode_lengths_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        returned_episode_lengths_std = returned_episode_lengths_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        service_blocking_probability = service_blocking_probability[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)

        for i in range(len(returned_episode_returns_mean)):
            # Log the data
            log_dict = {"update_step": i*FLAGS.DOWNSAMPLE_FACTOR,
                        "returned_episode_returns_mean": returned_episode_returns_mean[i],
                        "returned_episode_returns_std": returned_episode_returns_std[i],
                        "returned_episode_lengths_mean": returned_episode_lengths_mean[i],
                        "returned_episode_lengths_std": returned_episode_lengths_std[i],
                        "service_blocking_probability": service_blocking_probability[i]}
            wandb.log(log_dict)


if __name__ == "__main__":
    app.run(main)
