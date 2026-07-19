from typing import Callable

import jax
import jax.numpy as jnp
from box import Box
from flax.training.train_state import TrainState
from gymnax.environments import environment

from xlron import dtype_config
from xlron.environments.dataclasses import EnvParams, Transition
from xlron.train.train_utils import heuristic_eval_obs_placeholder, select_action_eval

# Core per-step scalars that LogWrapper puts in info and the metrics pipeline
# (process_metrics/print_metrics) consumes. In eval, each of these is stacked
# through the scan as its own (T,)-buffer, so every scalar costs one tiny
# dynamic-update-slice per step. We pack them into two vectors (integer-valued
# and float-valued) so the scan stacks 2 buffers instead of ~10, then unpack
# back to the same keys/dtypes after the scan. Split by value domain so integer
# counters are never squeezed through a float dtype (exactness for any run
# length). Keys absent from info (e.g. blocked_* on non-GN envs) are skipped.
_PACK_INT_KEYS = (
    "lengths",
    "accepted_services",
    "blocked_spectrum",
    "blocked_snr",
    "blocked_power",
    "terminal",
    "truncated",
)
_PACK_FLOAT_KEYS = (
    "returns",
    "cum_returns",
    "accepted_bitrate",
    "total_bitrate",
    "utilisation",
    "fragmentation",
)


def get_eval_fn(
    env: environment.Environment,
    env_params: EnvParams,
    eval_state: TrainState,
    config: Box,
) -> Callable:
    # Populated at trace time by _env_step (the scan body traces before the
    # post-scan unpack below runs); records which keys were packed and their
    # original dtypes so the unpack restores info exactly.
    packed_int_keys: list = []
    packed_float_keys: list = []
    packed_dtypes: dict = {}

    # Utilisation/fragmentation are constant zeros in the per-step info for all
    # RSA-family envs (log_metrics computes both per increment from the final
    # state and injects them into the reported stats); only VONE still reports
    # real per-step values. Dropping the dead keys avoids stacking them.
    drop_keys = (
        ("utilisation", "fragmentation")
        if "vone" not in getattr(config, "env_type", "").lower()
        else ()
    )

    def _pack_info(info: dict) -> dict:
        """Pack core per-step scalars into two vectors (see _PACK_*_KEYS)."""
        for key in drop_keys:
            info.pop(key, None)
        int_keys = [k for k in _PACK_INT_KEYS if k in info]
        float_keys = [k for k in _PACK_FLOAT_KEYS if k in info]
        packed_int_keys[:] = int_keys
        packed_float_keys[:] = float_keys
        packed_dtypes.update({k: jnp.asarray(info[k]).dtype for k in int_keys + float_keys})
        if int_keys:
            info["_packed_int"] = jnp.stack(
                [jnp.asarray(info.pop(k), dtype=dtype_config.LARGE_INT_DTYPE) for k in int_keys]
            )
        if float_keys:
            info["_packed_float"] = jnp.stack(
                [jnp.asarray(info.pop(k), dtype=dtype_config.LARGE_FLOAT_DTYPE) for k in float_keys]
            )
        return info

    def _unpack_info(info: dict) -> dict:
        """Invert _pack_info on the post-scan stacked info (packed axis is last)."""
        info = dict(info)
        packed_int = info.pop("_packed_int", None)
        packed_float = info.pop("_packed_float", None)
        for i, key in enumerate(packed_int_keys):
            info[key] = packed_int[..., i].astype(packed_dtypes[key])
        for i, key in enumerate(packed_float_keys):
            info[key] = packed_float[..., i].astype(packed_dtypes[key])
        return info

    # COLLECT TRAJECTORIES
    def _env_episode(runner_state, unused):
        def _env_step(runner_state, unused):
            eval_state, env_state, last_obs, step_key, rng_epoch = runner_state

            # Dedicated keys for action selection and env stepping; the parent step_key is
            # only ever split, never consumed directly.
            action_key, env_key, next_step_key = jax.random.split(step_key, 3)

            # SELECT ACTION
            select_action_state = (action_key, env_state, last_obs)
            env_state, action, _, _ = select_action_eval(
                select_action_state, env, env_params, eval_state, config
            )

            # STEP ENV
            obsv, env_state, reward, terminal, truncated, info = env.step(
                env_key, env_state, action, env_params
            )

            if config.USE_GNN or config.USE_TRANSFORMER:
                obsv = (env_state.env_state, env_params)
            elif config.EVAL_HEURISTIC:
                # Heuristic eval never reads the obs: carry the placeholder so the
                # ~4.4k-element get_obs concat inside env.step becomes dead code
                # (see heuristic_eval_obs_placeholder). EVAL_MODEL keeps real obs.
                obsv = heuristic_eval_obs_placeholder()
            else:
                obsv = tuple([obsv])
            transition = Transition(terminal, truncated, action, reward, last_obs, _pack_info(info))
            runner_state = (eval_state, env_state, obsv, next_step_key, rng_epoch)

            if getattr(config, "DEBUG", False):
                ordered = getattr(config, "ORDERED", False)
                jax.debug.print("request {}", env_state.env_state.request_array, ordered=ordered)
                jax.debug.print(
                    "link_slot_array {}",
                    env_state.env_state.link_slot_array,
                    ordered=ordered,
                )
                if env_params.__class__.__name__ == "RSAGNModelEnvParams":
                    jax.debug.print(
                        "link_snr_array {}",
                        env_state.env_state.link_snr_array,
                        ordered=ordered,
                    )
                    jax.debug.print(
                        "channel_power_array {}",
                        env_state.env_state.channel_power_array,
                        ordered=ordered,
                    )
                    jax.debug.print(
                        "modulation_format_index_array {}",
                        env_state.env_state.modulation_format_index_array,
                        ordered=ordered,
                    )
                    jax.debug.print(
                        "channel_centre_bw_array {}",
                        env_state.env_state.channel_centre_bw_array,
                        ordered=ordered,
                    )
                jax.debug.print(
                    "link_slot_mask {}", env_state.env_state.link_slot_mask, ordered=ordered
                )
                jax.debug.print("action {}", action, ordered=ordered)
                jax.debug.print("reward {}", reward, ordered=ordered)

            return runner_state, transition

        # VECTORISE ENV STEP
        _env_step_vmap = (
            jax.vmap(
                _env_step,
                in_axes=((None, 0, 0, 0, None), None),
                out_axes=((None, 0, 0, 0, None), 0),
            )
            if getattr(config, "NUM_ENVS", 1) > 1
            else _env_step
        )

        rng_step = runner_state[3]
        num_envs = getattr(config, "NUM_ENVS", 1)
        rng_step, *step_keys = jax.random.split(rng_step, num_envs + 1)
        step_keys = jnp.array(step_keys) if num_envs > 1 else step_keys[0]
        runner_state = runner_state[:3] + (step_keys,) + runner_state[4:]
        steps_per_env = getattr(config, "STEPS_PER_INCREMENT", 1000) // getattr(
            config, "NUM_ENVS", 1
        )
        runner_state, traj_episode = jax.lax.scan(
            _env_step_vmap,
            runner_state,
            None,
            steps_per_env,
        )
        runner_state = runner_state[:3] + (rng_step,) + runner_state[4:]

        metric = _unpack_info(traj_episode.info)

        return runner_state, metric

    def eval_fn(runner_state):
        runner_state, metric = jax.lax.scan(_env_episode, runner_state, None, 1)
        return {"runner_state": runner_state, "metrics": metric}

    return eval_fn
