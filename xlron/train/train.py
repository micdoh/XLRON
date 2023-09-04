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
    # Set the number of (emulated) host devices
    num_devices = FLAGS.NUM_DEVICES if FLAGS.NUM_DEVICES is not None else jax.local_device_count()
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={num_devices}"

    # Set the default device
    jax.config.update("jax_default_device", FLAGS.DEFAULT_DEVICE)

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
            rng = jax.random.split(rng, num_devices)
            train_jit = jax.pmap(make_train(FLAGS), devices=jax.devices()).lower(rng).compile()
        else:
            train_jit = jax.jit(make_train(FLAGS)).lower(rng).compile()

    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS*num_devices):
        out = train_jit(rng)

    # Save model params
    if FLAGS.SAVE_MODEL:
        train_state = out["runner_state"][0]
        save_data = {"model": train_state, "config": FLAGS}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(save_data)
        orbax_checkpointer.save(FLAGS.MODEL_PATH, save_data, save_args=save_args)

    plt.plot(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.show()


if __name__ == "__main__":
    app.run(main)
