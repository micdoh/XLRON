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
        "consecutive_loading": config.consecutive_loading,
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
                    raise NotImplementedError(f"VONE heuristics not yet implemented")

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
        NUM_EPISODES = config.TOTAL_TIMESTEPS // config.max_timesteps // config.NUM_ENVS
        runner_state, metric = jax.lax.scan(
            _env_episode, runner_state, None, NUM_EPISODES
        )
        return {"runner_state": runner_state, "metrics": metric}

    return eval


def main(argv):

    # TODO - can wrap this into train main() function (just have an EVAL flag and if statement for make_train/make_eval)
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

    # Print every flag and its name
    if FLAGS.DEBUG:
        print('non-flag arguments:', argv)
    for name in FLAGS:
        print(name, FLAGS[name].value)

    rng = jax.random.PRNGKey(FLAGS.SEED)

    with TimeIt(tag='COMPILATION'):
        if FLAGS.NUM_SEEDS > 1:
            rng = jax.random.split(rng, FLAGS.NUM_SEEDS)
            eval_jit = jax.jit(jax.vmap(make_eval(FLAGS))).lower(rng).compile()
        else:
            eval_jit = jax.jit(make_eval(FLAGS)).lower(rng).compile()

    with TimeIt(tag='EXECUTION', frames=FLAGS.TOTAL_TIMESTEPS * len(jax.devices()) * FLAGS.NUM_SEEDS):
        out = eval_jit(rng)
        out["metrics"]["returned_episode_returns"].block_until_ready()  # Wait for all devices to finish

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

    plot_metric = returned_episode_lengths_mean if (FLAGS.env_type == "rsa" and FLAGS.consecutive_loading) else returned_episode_returns_mean
    plot_metric_std = returned_episode_lengths_std if (FLAGS.env_type == "rsa" and FLAGS.consecutive_loading) else returned_episode_returns_std
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

    print(f"Final metrics: \n Mean: {plot_metric[-1]} \n Std: {plot_metric_std[-1]}")


if __name__ == "__main__":
    app.run(main)
