"""The campaign must construct the same intended physical workload in every mode."""

import os
import subprocess
import sys
from pathlib import Path


def test_campaign_effective_load_bands_and_physics_match_across_modes(tmp_path):
    # A subprocess isolates the real absl parser from pytest's global flags.
    script = r"""
import json,sys
from absl import flags
import xlron.parameter_flags
from xlron.environments.make_env import make,process_config

manifest=json.load(open(sys.argv[1]))
for index in (0,4,5):
    job=manifest["jobs"][index]
    command=job["command"]
    start=next(i for i,arg in enumerate(command) if arg.startswith("--env_type="))
    flags.FLAGS(["train",*command[start:]])
    config=process_config({key:flags.FLAGS[key].value for key in flags.FLAGS})
    _,params=make(config,log_wrapper=False)
    assert config.load == 5000, (job["kind"],config.load)
    assert params.arrival_rate == 200, (job["kind"],params.arrival_rate)
    assert params.mean_service_holding_time == 25
    assert params.link_resources == 1668
    assert params.coherent
    assert config.band_preference == "C,L,S"
    assert config.slot_size == 12.5
    assert config.power_per_channel == -10
    assert config.max_power_per_fibre == 21
    flags.FLAGS.unparse_flags()
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    root = Path(__file__).resolve().parents[2]
    plan = subprocess.run(
        [
            sys.executable,
            "-m",
            "experimental.benchmarks.gn_cls_training_campaign",
            "--gpu",
            "2",
            "--lane",
            "heuristics_then_train",
            "--output",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=root,
        check=False,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert plan.returncode == 0, plan.stderr
    preflight = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "manifest.json")],
        cwd=root,
        check=False,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert preflight.returncode == 0, preflight.stdout[-3000:] + preflight.stderr[-3000:]
