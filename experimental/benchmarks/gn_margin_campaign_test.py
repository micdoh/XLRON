"""Margin comparisons must isolate reserves/power without reusing fragile states."""

from argparse import Namespace

import pytest

from experimental.benchmarks.gn_margin_campaign import MARGINS, build_jobs


def args_for(tmp_path, lane, train=False, preflight=False):
    return Namespace(
        gpu=1,
        lane=lane,
        output=tmp_path,
        resume_model=tmp_path / "parent.eqx",
        project="test",
        load=5000,
        warmup=25000,
        pool_before_projection=False,
        train_timesteps=4194304,
        eval_timesteps=131072,
        num_envs=4,
        train_after=train,
        preflight=preflight,
    )


@pytest.mark.parametrize("lane", [*MARGINS, "power"])
def test_comparison_is_fresh_full_physical_and_paired(tmp_path, lane):
    args = args_for(tmp_path, lane)
    jobs = build_jobs(args)
    assert len(jobs) == (8 if lane == "power" else 4)
    admission, retained = MARGINS["reserve" if lane == "power" else lane]
    for job in jobs:
        cmd = job["command"]
        assert (
            "--initial-env-state" not in cmd and "--bootstrap-warmup-steps" not in cmd
        )
        assert "--ENV_WARMUP_STEPS=25000" in cmd
        assert "--gn_mask_mode=all_slots" in cmd and "--aggregate_slots=1" in cmd
        assert "--band_preference=C,L,S" in cmd and "--link_resources=1668" in cmd
        assert "--audit-existing-snr" in cmd
        assert (
            f"--snr_margin={admission}" in cmd
            and f"--existing_snr_margin={retained}" in cmd
        )
        assert "--launch_power_type=fixed" in cmd
    for heuristic in ("arbr_ff", "mscl_ksp"):
        matched = [j for j in jobs if f"--path_heuristic={heuristic}" in j["command"]]
        assert {
            x for j in matched for x in j["command"] if x.startswith("--SEED=")
        } == {"--SEED=4242", "--SEED=4243"}
    assert {
        x for j in jobs for x in j["command"] if x.startswith("--power_per_channel=")
    } == (
        {"--power_per_channel=-11", "--power_per_channel=-9"}
        if lane == "power"
        else {"--power_per_channel=-10"}
    )


def test_continuation_uses_same_reserve_and_keeps_unrestricted_slots(tmp_path):
    jobs = build_jobs(args_for(tmp_path, "reserve", train=True))
    train = jobs[-1]
    assert train["kind"] == "train"
    cmd = train["command"]
    for flag in (
        "--snr_margin=0.5",
        "--existing_snr_margin=0.5",
        "--USE_TRANSFORMER",
        "--transformer_modulation_features=true",
        "--upgrade-modulation-head",
        "--gn_mask_mode=all_slots",
        "--aggregate_slots=1",
        "--OFF_POLICY_IAM=true",
        "--VALID_MASS_LOSS_COEF=0.002",
        "--LR=0.0001",
        "--ENV_WARMUP_STEPS=25000",
    ):
        assert flag in cmd
    assert "--initial-env-state" not in cmd
    assert train["model_path"] != str(tmp_path / "parent.eqx")


def test_preflight_is_labeled_separately_and_short_warmup_cannot_be_mislabeled(
    tmp_path,
):
    args = args_for(tmp_path, "reserve", train=True, preflight=True)
    jobs = build_jobs(args)
    assert len(jobs) == 3 and all(j["name"].startswith("preflight-") for j in jobs)
    assert all("--ENV_WARMUP_STEPS=8" in j["command"] for j in jobs)
    assert all("--DOWNSAMPLE_FACTOR=1" in j["command"] for j in jobs)
    args.preflight = False
    args.warmup = 32
    with pytest.raises(ValueError, match="25,000"):
        build_jobs(args)
