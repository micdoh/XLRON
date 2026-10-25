"""The representation comparison must preserve individual raw-slot actions."""

import argparse
from pathlib import Path

from experimental.benchmarks.gn_cls_observation_campaign import build_jobs


def test_plan_keeps_every_slot_and_changes_only_the_representation(tmp_path):
    args = argparse.Namespace(
        gpu=2,
        seed=42,
        output=tmp_path,
        mask_mode="all_slots",
        resume_model=Path("/tmp/provenance.eqx"),
        train_timesteps=16777216,
        eval_timesteps=524288,
        project="XLRON-GN-CLS",
        load=5000,
        warmup=25000,
        pool_before_projection=False,
    )
    jobs = build_jobs(args)
    trains = [j for j in jobs if j["kind"] == "train"]
    assert len(trains) == 4 and len(jobs) == 19
    for job in jobs:
        c = job["command"]
        assert "--aggregate_slots=1" in c and "--gn_mask_mode=all_slots" in c
        assert "--OFF_POLICY_IAM=true" in c and "--band_preference=C,L,S" in c
        assert "--NUM_MINIBATCHES=8" in c
        assert "--initial-model" not in c and "--RETRAIN_MODEL=false" in c
        assert len([a for a in c if a.startswith("--MODEL_PATH=")]) == 1
        if job["kind"] == "heuristic":
            assert "--EVAL_HEURISTIC=true" in c and "--transformer_modulation_features=false" in c
        else:
            assert "--transformer_modulation_features=true" in c
    assert len([j for j in jobs if j["kind"] == "heuristic"]) == 3
    assert {
        next(a for a in j["command"] if a.startswith("--transformer_spectrum_encoder="))
        for j in trains
    } == {"--transformer_spectrum_encoder=dense", "--transformer_spectrum_encoder=conv"}
