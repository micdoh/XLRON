import wandb
import os
import jax
import matplotlib.pyplot as plt
import orbax.checkpoint
from flax.training import orbax_utils
from absl import app, flags
from xlron.environments.env_funcs import *
from xlron.train.ppo import make_train
import xlron.train.parameter_flags


FLAGS = flags.FLAGS


def main(argv):

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


    # Set the number of host devices
    num_devices = FLAGS.NUM_DEVICES if FLAGS.NUM_DEVICES is not None else jax.local_device_count()

    # Set the default device
    print(f"Available devices: {jax.devices()}")
    if FLAGS.DEFAULT_DEVICE:
        jax.default_device = jax.devices()[FLAGS.DEFAULT_DEVICE]
        print(f"Using {jax.default_device}")

    # Print every flag and its name
    if FLAGS.DEBUG:
        print('non-flag arguments:', argv)
    for name in FLAGS:
        print(name, FLAGS[name].value)

    rng = jax.random.PRNGKey(FLAGS.SEED)

    with TimeIt(tag='COMPILATION'):
        if FLAGS.USE_PMAP:
            # TODO - Fix this to be like Anakin architecture (share gradients across devices)
            # TODO - Retrieve the output using pmean/sum etc.
            FLAGS.ORDERED = False
            rng = jax.random.split(rng, num_devices*FLAGS.NUM_SEEDS)
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
    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS * num_devices * FLAGS.NUM_SEEDS):
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
        returned_episode_lengths_std = out["metrics"]["returned_episode_lengths"].mean(-1).std(-1).reshape(-1)
    else:
        # N.B. This is the same as the above code, but without the mean on the seed dimension
        # This means the results are still per update step
        returned_episode_returns_mean = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
        returned_episode_returns_std = out["metrics"]["returned_episode_returns"].std(-1).reshape(-1)
        returned_episode_lengths_mean = out["metrics"]["returned_episode_lengths"].mean(-1).reshape(-1)
        returned_episode_lengths_std = out["metrics"]["returned_episode_lengths"].std(-1).reshape(-1)

    # Remove initial zeros before an episode is returned
    # if not FLAGS.consecutive_loading:
    #     mask = jnp.array(returned_episode_lengths_mean) > 0
    #     returned_episode_returns_mean = returned_episode_returns_mean[mask]
    #     returned_episode_returns_std = returned_episode_returns_std[mask]

    plt.plot(returned_episode_returns_mean)
    plt.fill_between(
        range(len(returned_episode_returns_mean)),
        returned_episode_returns_mean - returned_episode_returns_std,
        returned_episode_returns_mean + returned_episode_returns_std,
        alpha=0.2
    )
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f"{FLAGS.EXPERIMENT_NAME}.png")
    plt.show()

    # TODO - Define blocking probability metric

    if FLAGS.WANDB:
        # Log the data to wandb
        # Define the downsample factor to speed up upload to wandb
        # Then reshape the array and compute the mean
        chop = len(returned_episode_returns_mean) % FLAGS.DOWNSAMPLE_FACTOR
        returned_episode_returns_mean = returned_episode_returns_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        returned_episode_returns_std = returned_episode_returns_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        returned_episode_lengths_mean = returned_episode_lengths_mean[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)
        returned_episode_lengths_std = returned_episode_lengths_std[chop:].reshape(-1, FLAGS.DOWNSAMPLE_FACTOR).mean(axis=1)

        for i in range(len(returned_episode_returns_mean)):
            # Log the data
            log_dict = {"update_step": i*FLAGS.DOWNSAMPLE_FACTOR,
                        "returned_episode_returns_mean": returned_episode_returns_mean[i],
                        "returned_episode_returns_std": returned_episode_returns_std[i],
                        "returned_episode_lengths_mean": returned_episode_lengths_mean[i],
                        "returned_episode_lengths_std": returned_episode_lengths_std[i]}
            wandb.log(log_dict)


if __name__ == "__main__":
    app.run(main)
