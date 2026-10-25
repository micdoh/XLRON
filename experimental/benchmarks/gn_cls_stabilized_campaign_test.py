"""Paper-informed runs must preserve off-policy IAM, stabilization and full physics."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experimental.benchmarks.gn_cls_stabilized_campaign import build_jobs


def args_for(tmp_path, lane):
    return argparse.Namespace(
        gpu=3 if lane == "fine_tune" else 2,
        lane=lane,
        output=tmp_path,
        resume_model=tmp_path / "parent.eqx",
        project="XLRON-GN-CLS",
        load=5000,
        warmup=25000,
        pool_before_projection=False,
        train_timesteps=16777216,
        eval_timesteps=524288,
    )


@pytest.mark.parametrize("lane", ["fine_tune", "single_update"])
def test_stabilized_plan_preserves_iam_physics_and_separate_checkpoints(tmp_path, lane):
    args = args_for(tmp_path, lane)
    jobs = build_jobs(args)
    assert len(jobs) == (12 if lane == "fine_tune" else 8)
    trains = [j for j in jobs if j["kind"] == "train"]
    assert len(trains) == (3 if lane == "fine_tune" else 2)
    for job in trains:
        cmd = job["command"]
        for flag in (
            "--OFF_POLICY_IAM=true",
            "--IAM_RECENTER_CLIP=false",
            "--IAM_DAMPING=true",
            "--IAM_GATING=true",
            "--IAM_GATING_MIN_ACTIONS=2",
            "--CLIP_EPS=0.04",
            "--compute_dtype=float32",
            "--params_dtype=float32",
            "--mixed_precision=false",
            "--band_preference=C,L,S",
            "--link_resources=1668",
            "--load=5000",
            "--transformer_obs_type=occupancy_snr",
            "--GAMMA=0.996",
            "--GAE_LAMBDA=0.99",
        ):
            assert flag in cmd
        assert "--VALID_MASS_LOSS_COEF=0.0" not in cmd
        assert job["model_path"] != str(args.resume_model)
        if lane == "single_update":
            assert "--NUM_MINIBATCHES=1" in cmd and "--UPDATE_EPOCHS=1" in cmd
            assert "--ROLLOUT_LENGTH=64" in cmd
    if lane == "fine_tune":
        assert {
            next(x for x in t["command"] if x.startswith("--VALID_MASS_LOSS_COEF=")) for t in trains
        } == {"--VALID_MASS_LOSS_COEF=0.001", "--VALID_MASS_LOSS_COEF=0.002"}
    assert {
        next(x for x in j["command"] if x.startswith("--SEED="))
        for j in jobs
        if j["kind"] == "evaluation"
    } == {"--SEED=4242", "--SEED=4243", "--SEED=4244"}


def test_actual_cli_config_pins_new_loss_and_full_band_workload(tmp_path):
    jobs = build_jobs(args_for(tmp_path, "single_update"))
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(jobs))
    script = r"""
import argparse,json,sys
from absl import flags
import xlron.parameter_flags
from xlron.environments.make_env import make,process_config
jobs=json.load(open(sys.argv[1]))
for index in (0,1,4):
 command=jobs[index]['command']
 start=next(i for i,x in enumerate(command) if x.startswith('--env_type='))
 wrapper=argparse.ArgumentParser(add_help=False)
 wrapper.add_argument('--initial-model')
 _,remaining=wrapper.parse_known_args(command[start:])
 flags.FLAGS(['train',*remaining])
 config=process_config({k:flags.FLAGS[k].value for k in flags.FLAGS})
 _,params=make(config,log_wrapper=False)
 assert config.load==5000 and params.arrival_rate==200
 assert params.link_resources==1668 and params.coherent
 assert config.transformer_obs_type=='occupancy_snr'
 if jobs[index]['kind']=='train':
  assert config.OFF_POLICY_IAM and config.IAM_GATING and config.IAM_DAMPING
  assert config.VALID_MASS_LOSS_COEF==.002
  assert config.UPDATE_EPOCHS==1 and config.NUM_MINIBATCHES==1
  assert config.ROLLOUT_LENGTH==64
  assert str(config.compute_dtype)=='float32'
 flags.FLAGS.unparse_flags()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
