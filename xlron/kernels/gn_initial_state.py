"""Validated physical-state bootstrap for isolated full-mask experiments."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def restore_physical_state(runner, params, config, path):
    """Restore admitted services, keeping fresh policy/optimizer and future RNGs.

    Saved replicas may seed a larger independent batch by cycling initial states.
    Future traffic keys remain the independent keys from experiment_data_setup.
    The snapshot carries a physical workload key; every recorded physical/traffic
    option must match. Observation and masking implementations may differ.
    """
    from xlron.environments.env_funcs import get_snr_link_array
    from xlron.train.train_utils import (
        heuristic_eval_obs_placeholder,
        reset_warmup_metric_counters,
    )

    if config.env_type != "rmsa_gn_model" or not config.continuous_operation:
        raise ValueError("Physical bootstrap requires continuous RMSA-GN")
    leaves, definition = jax.tree.flatten(runner[1].env_state)
    restored = []
    with np.load(Path(path), allow_pickle=False) as archive:
        recorded = json.loads(str(archive["workload_key"]))
        ignored = {"TOTAL_TIMESTEPS", "max_requests"}
        for name, expected in recorded.items():
            if name in ignored:
                continue
            actual = config.get(name)
            if isinstance(expected, list):
                actual = actual.split(",") if isinstance(actual, str) else list(actual)
                if expected and isinstance(expected[0], (int, float)):
                    actual = [float(x) for x in actual]
            if actual != expected:
                raise ValueError(f"Physical snapshot mismatch: {name}: {actual!r} != {expected!r}")
        for index, leaf in enumerate(leaves):
            value = archive[f"state_{index}"]
            if config.NUM_ENVS > 1:
                value = value[np.arange(config.NUM_ENVS) % value.shape[0]]
            else:
                value = value[0]
            if value.shape != leaf.shape or value.dtype != leaf.dtype:
                raise ValueError(f"Physical snapshot leaf {index} shape/dtype mismatch")
            restored.append(jnp.asarray(value))
    state = jax.tree.unflatten(definition, restored)
    refresh = lambda s: get_snr_link_array(s, params)
    state = state.replace(
        link_snr_array=(
            jax.jit(jax.vmap(refresh))(state) if config.NUM_ENVS > 1 else refresh(state)
        )
    )
    outer = reset_warmup_metric_counters(runner[1].replace(env_state=state))
    if config.USE_GNN or config.USE_TRANSFORMER:
        observation = (outer.env_state, params)
    elif config.EVAL_HEURISTIC:
        observation = heuristic_eval_obs_placeholder(config.NUM_ENVS)
    else:
        raise ValueError("Physical bootstrap supports transformers or heuristic evaluation")
    return (runner[0], outer, observation, runner[3], runner[4])
