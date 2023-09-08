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
    # TODO - Add wandb
    # TODO - For wandb, can run multiple seeds in parallel and log to same run (take mean/std of returns)
    # Set the number of (emulated) host devices
    num_devices = FLAGS.NUM_DEVICES if FLAGS.NUM_DEVICES is not None else jax.local_device_count()
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={num_devices}"

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

    # TODO - Remve the initial 0s from the returns
    if FLAGS.NUM_SEEDS > 1:
        for i in range(FLAGS.NUM_SEEDS):
            plt.plot(out["metrics"]["returned_episode_returns"][i].mean(-1).reshape(-1))
    else:
        plt.plot(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    #plt.savefig("ppo.png")
    plt.show()

    print(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))



if __name__ == "__main__":
    app.run(main)
