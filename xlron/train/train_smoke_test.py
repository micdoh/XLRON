"""End-to-end CPU training smoke tests for the train.py entry point.

Regression cover for (a) the launch_power_type="rl" training path, which rotted
invisibly because nothing exercised the full train pipeline (init_network ->
warmup/select_action -> rollout -> _loss_fn) with a GN-model env, and (b) the
load-sweep path, which must re-run the warmup at each swept load. Each test runs
the real entry point in a subprocess with a tiny config; a clean exit is the main
assertion. Marked slow (~20s each, dominated by XLA compilation): deselect with
`pytest -m "not slow"`.
"""

import os
import subprocess
import sys

import pytest

BASE_ARGS = [
    "-m",
    "xlron.train.train",
    "--env_type=rsa_gn_model",
    "--topology_name=nsfnet_deeprmsa_directed",
    "--link_resources=10",
    "--k=4",
    "--values_bw=100",
    "--incremental_loading",
    "--slot_size=12.5",
    "--guardband=0",
    "--launch_power_type=rl",
    "--ROLLOUT_LENGTH=10",
    "--TOTAL_TIMESTEPS=20",
    "--NUM_ENVS=1",
    "--ENV_WARMUP_STEPS=0",
]


def _run_train(extra_args=()):
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, *BASE_ARGS, *extra_args],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, (
        f"training run failed (exit {result.returncode})\n"
        f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-3000:]}"
    )
    return result


@pytest.mark.slow
def test_launch_power_rl_training_smoke_continuous():
    """MLP power-only policy with the continuous (Beta) power head (the default)."""
    _run_train()


@pytest.mark.slow
def test_launch_power_rl_training_smoke_discrete():
    """MLP power-only policy with the discrete (Categorical) power head."""
    _run_train(["--discrete_launch_power"])


LOAD_SWEEP_ARGS = [
    "-m",
    "xlron.train.train",
    "--env_type=rsa",
    "--topology_name=5node_directed",
    "--link_resources=10",
    "--k=2",
    "--values_bw=1",
    "--slot_size=1",
    "--guardband=0",
    "--continuous_operation",
    "--EVAL_HEURISTIC",
    "--path_heuristic=ksp_ff",
    "--load=20",
    "--min_load=10",
    "--max_load=20",
    "--step_load=10",
    "--ENV_WARMUP_STEPS=10",
    "--ROLLOUT_LENGTH=10",
    "--TOTAL_TIMESTEPS=20",
    "--NUM_ENVS=1",
]


@pytest.mark.slow
def test_load_sweep_rewarms_each_load_smoke():
    """A load sweep must re-run the warmup at every swept load (regression: the
    sweep reused network state equilibrated at the original --load, biasing
    per-load steady-state metrics)."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, *LOAD_SWEEP_ARGS],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert result.returncode == 0, (
        f"load sweep run failed (exit {result.returncode})\n"
        f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-3000:]}"
    )
    # One re-warmup per swept load (2 loads: 10 and 20)
    assert result.stdout.count("Re-running warmup") == 2, (
        f"expected 2 re-warmups in stdout\n--- stdout tail ---\n{result.stdout[-3000:]}"
    )
