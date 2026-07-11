"""End-to-end CPU training smoke tests for the RL launch-power path.

Regression cover for the launch_power_type="rl" training path, which rotted invisibly
because nothing exercised the full train pipeline (init_network -> warmup/select_action
-> rollout -> _loss_fn) with a GN-model env. Covers both GN-model env variants:
rmsa_gn_model shares the flat path-slot action decode with rsa_gn_model but previously
crashed in _loss_fn because ppo.py gated the (path, power) tuple handling on the exact
env_type "rsa_gn_model". Each test runs the real entry point in a subprocess with a tiny
config; a clean exit is the assertion. Marked slow (~20s each, dominated by XLA
compilation): deselect with `pytest -m "not slow"`.
"""

import os
import subprocess
import sys

import pytest

BASE_ARGS = [
    "-m",
    "xlron.train.train",
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


def _run_train(env_type, extra_args=()):
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, *BASE_ARGS, f"--env_type={env_type}", *extra_args],
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
@pytest.mark.parametrize("env_type", ["rsa_gn_model", "rmsa_gn_model"])
def test_launch_power_rl_training_smoke_continuous(env_type):
    """MLP power-only policy with the continuous (Beta) power head (the default)."""
    _run_train(env_type)


@pytest.mark.slow
@pytest.mark.parametrize("env_type", ["rsa_gn_model", "rmsa_gn_model"])
def test_launch_power_rl_training_smoke_discrete(env_type):
    """MLP power-only policy with the discrete (Categorical) power head."""
    _run_train(env_type, ["--discrete_launch_power"])
