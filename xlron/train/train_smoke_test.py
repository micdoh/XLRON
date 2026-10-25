"""End-to-end CPU training smoke tests for the train.py entry point.

Regression cover for (a) the launch_power_type="rl" training path, which rotted
invisibly because nothing exercised the full train pipeline (init_network ->
warmup/select_action -> rollout -> _loss_fn) with a GN-model env — covering both
GN-model env variants: rmsa_gn_model shares the flat path-slot action decode with
rsa_gn_model but previously crashed in _loss_fn because ppo.py gated the
(path, power) tuple handling on the exact env_type "rsa_gn_model" — and (b) the
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


@pytest.mark.slow
@pytest.mark.parametrize("env_type", ["rsa_gn_model", "rmsa_gn_model"])
def test_fixed_power_routing_training_parallel_with_policy_warmup(env_type):
    """Routing MLPs must support policy warmup and a batched PPO update in GN envs."""
    _run_train(
        env_type,
        [
            "--launch_power_type=fixed",
            "--NUM_ENVS=2",
            "--ENV_WARMUP_STEPS=4",
            "--TOTAL_TIMESTEPS=40",
        ],
    )


@pytest.mark.slow
@pytest.mark.parametrize("env_type", ["rsa_gn_model", "rmsa_gn_model"])
def test_transformer_gn_training_with_policy_warmup(env_type):
    """GN transformer logits must match path-slot masks throughout real PPO."""
    _run_train(
        env_type,
        [
            "--launch_power_type=fixed",
            "--USE_TRANSFORMER",
            "--transformer_embedding_size=16",
            "--transformer_intermediate_size=32",
            "--NUM_ENVS=2",
            "--ENV_WARMUP_STEPS=2",
            "--TOTAL_TIMESTEPS=40",
        ],
    )


@pytest.mark.slow
@pytest.mark.parametrize("env_type", ["rsa_gn_model", "rmsa_gn_model"])
def test_transformer_snr_training_with_continuous_traffic(env_type):
    """The combined SNR features support the real batched PPO backward pass."""
    _run_train(
        env_type,
        [
            "--launch_power_type=fixed",
            "--USE_TRANSFORMER",
            "--continuous_operation",
            "--transformer_obs_type=occupancy_snr",
            "--transformer_embedding_size=16",
            "--transformer_intermediate_size=32",
            "--NUM_ENVS=2",
            "--ENV_WARMUP_STEPS=2",
            "--TOTAL_TIMESTEPS=40",
        ],
    )


@pytest.mark.slow
def test_transformer_snr_campaign_optimizer_and_checkpoint(tmp_path):
    """Exercise the campaign's optimizer, best-checkpoint save and model reload."""
    model_path = tmp_path / "snr.eqx"
    architecture = [
        "--launch_power_type=fixed",
        "--USE_TRANSFORMER",
        "--continuous_operation",
        "--incremental_loading=false",
        "--transformer_obs_type=occupancy_snr",
        "--transformer_embedding_size=16",
        "--transformer_intermediate_size=32",
        "--NUM_ENVS=2",
        "--ENV_WARMUP_STEPS=2",
        "--TOTAL_TIMESTEPS=40",
        f"--MODEL_PATH={model_path}",
    ]
    _run_train(
        "rmsa_gn_model",
        architecture
        + [
            "--NUM_MINIBATCHES=2",
            "--UPDATE_EPOCHS=2",
            "--LR=0.0005",
            "--SEPARATE_VF_OPTIMIZER",
            "--VF_LR=0.0001",
            "--OFF_POLICY_IAM",
            "--VALID_MASS_LOSS_COEF=0.0002",
            "--ENHANCED_LOGGING",
            "--SAVE_MODEL",
            "--EVAL_DURING_TRAINING",
            "--EVAL_FREQUENCY=1",
            "--EVAL_TIMESTEPS=20",
        ],
    )
    assert model_path.exists()
    _run_train("rmsa_gn_model", architecture + ["--EVAL_MODEL", "--deterministic"])


@pytest.mark.slow
@pytest.mark.parametrize("encoder,mask_mode", [("conv", "all_slots"), ("dense", "all_slots")])
def test_full_slot_lifetime_snr_ppo_and_checkpoint(tmp_path, encoder, mask_mode):
    path = tmp_path / "spectrum.eqx"
    args = [
        "--launch_power_type=fixed",
        "--USE_TRANSFORMER",
        "--continuous_operation",
        "--incremental_loading=false",
        "--transformer_obs_type=lifetime_snr",
        f"--transformer_spectrum_encoder={encoder}",
        "--transformer_embedding_size=16",
        "--transformer_intermediate_size=32",
        "--transformer_actor_pooling=sum",
        f"--gn_mask_mode={mask_mode}",
        "--transformer_modulation_features",
        "--gn_mask_env_batch_size=1",
        "--aggregate_slots=1",
        "--NUM_ENVS=2",
        "--ENV_WARMUP_STEPS=2",
        "--TOTAL_TIMESTEPS=40",
        "--SEPARATE_VF_OPTIMIZER",
        "--OFF_POLICY_IAM",
        "--VALID_MASS_LOSS_COEF=0.002",
        "--ENHANCED_LOGGING",
        f"--MODEL_PATH={path}",
    ]
    _run_train("rmsa_gn_model", args + ["--SAVE_MODEL"])
    assert path.is_file()
    _run_train("rmsa_gn_model", args + ["--EVAL_MODEL", "--deterministic"])


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
