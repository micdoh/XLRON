import wandb
import os
import jax
import matplotlib.pyplot as plt
import orbax.checkpoint
from flax.training import orbax_utils
from absl import app, flags
from typing import NamedTuple
from xlron.environments.env_funcs import *
from xlron.train.ppo import make_train
from xlron.environments.vone import *
from xlron.environments.rsa import *
from xlron.environments.heuristics import *
import xlron.train.parameter_flags
from gymnax.wrappers.purerl import LogWrapper


FLAGS = flags.FLAGS

flags.DEFINE_string("path_heuristic", "ksp_ff", "Path heuristic to be evaluated")
flags.DEFINE_string("node_heuristic", "random", "Node heuristic to be evaluated")


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_eval(config):

    env_params = {
        "k": config.k,
        "load": config.load,
        "topology_name": config.topology_name,
        "mean_service_holding_time": config.mean_service_holding_time,
        "link_resources": config.link_resources,
        "max_requests": config.max_requests,
        "max_timesteps": config.max_timesteps,
        "min_slots": config.min_slots,
        "max_slots": config.max_slots,
    }
    if config.env_type.lower() == "vone":
        env_params["virtual_topologies"] = config.virtual_topologies
        env_params["min_node_resources"] = config.min_node_resources
        env_params["max_node_resources"] = config.max_node_resources
        env_params["node_resources"] = config.node_resources
        env, env_params = make_vone_env(**env_params)
    elif config.env_type.lower() == "rsa":
        env, env_params = make_rsa_env(**env_params)
    else:
        raise ValueError(f"Invalid environment type {config.env_type}")
    env = LogWrapper(env)


    def eval(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # COLLECT TRAJECTORIES
        def _env_episode(runner_state, unused):

            def _env_step(runner_state, unused):

                env_state, last_obs, rng = runner_state

                # SELECT ACTION
                if config.env_type.lower() == "vone":
                    action = jnp.array([0,0,0])
                    pass

                elif config.env_type.lower() == "rsa":
                    if config.path_heuristic.lower() == "ksp_ff":
                        action = jax.vmap(ksp_ff, in_axes=(0, None))(env_state.env_state, env_params)
                    elif config.path_heuristic.lower() == "ff_ksp":
                        action = jax.vmap(ff_ksp, in_axes=(0, None))(env_state.env_state, env_params)
                    else:
                        raise ValueError(f"Invalid path heuristic {config.path_heuristic}")

                else:
                    raise ValueError(f"Invalid environment type {config.env_type}")

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.NUM_ENVS)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    done, action, reward, last_obs, info
                )
                runner_state = (env_state, obsv, rng)

                if config.DEBUG:
                    jax.debug.print("link_slot_array {}", env_state.env_state.link_slot_array, ordered=config.ORDERED)
                    jax.debug.print("link_slot_mask {}", env_state.env_state.link_slot_mask, ordered=config.ORDERED)
                    jax.debug.print("action {}", action, ordered=config.ORDERED)
                    jax.debug.print("reward {}", reward, ordered=config.ORDERED)

                return runner_state, transition

            runner_state, traj_episode = jax.lax.scan(
                _env_step, runner_state, None, config.max_timesteps
            )

            metric = traj_episode.info

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (env_state, obsv, _rng)
        NUM_EPISODES = config.TOTAL_TIMESTEPS // config.max_timesteps
        runner_state, metric = jax.lax.scan(
            _env_episode, runner_state, None, NUM_EPISODES
        )
        return {"runner_state": runner_state, "metrics": metric}

    return eval


def main(argv):
    # TODO - Add wandb
    # Set the number of (emulated) host devices
    num_devices = FLAGS.NUM_DEVICES if FLAGS.NUM_DEVICES is not None else jax.local_device_count()
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={num_devices}"

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
            eval_jit = jax.pmap(make_eval(FLAGS), devices=jax.devices()).lower(rng).compile()
        else:
            eval_jit = jax.jit(make_eval(FLAGS)).lower(rng).compile()

    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS*num_devices):
        out = eval_jit(rng)

    plt.plot(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.show()


if __name__ == "__main__":
    app.run(main)
