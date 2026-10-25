"""The queue covers the full heuristic inventory and keeps evaluation independent."""

import json
from argparse import Namespace

from experimental.benchmarks.gn_cls_tuning_campaign import HEURISTICS, build_jobs, selection_score


def test_two_shards_cover_every_heuristic_and_continue_matching_models(tmp_path):
    jobs = []
    for shard in (0, 1):
        args = Namespace(
            gpu=shard + 2,
            shard=shard,
            output=tmp_path / str(shard),
            resume_model=tmp_path / f"parent-{shard}.eqx",
            project="test",
            load=5000,
            warmup=25000,
            pool_before_projection=False,
            train_timesteps=8388608,
            eval_timesteps=524288,
        )
        lane = build_jobs(args)
        jobs.extend(lane)
        for job in lane:
            command = job["command"]
            assert "--band_preference=C,L,S" in command
            assert "--link_resources=1668" in command
            assert "--load=5000" in command
            assert "--coherent" in command
            assert "--ENV_WARMUP_STEPS=25000" in command
        trains = [j for j in lane if j["kind"] == "train"]
        assert len(trains) == 3
        for train in trains:
            assert "--RETRAIN_MODEL" in train["command"]
            assert "--KEEP_VF=false" in train["command"]
            assert train["command"][train["command"].index("--initial-model") + 1] == str(
                args.resume_model
            )
            assert train["model_path"] != str(args.resume_model)
        assert trains[-1]["adaptive_parent"]
        assert all("--SEED=4244" in j["command"] for j in lane if j.get("selection"))
    heuristics = [j for j in jobs if j["kind"] == "heuristic"]
    assert len(heuristics) == 50
    for heuristic in HEURISTICS:
        matched = [j for j in heuristics if f"--path_heuristic={heuristic}" in j["command"]]
        assert {next(x for x in j["command"] if x.startswith("--SEED=")) for j in matched} == {
            "--SEED=4242",
            "--SEED=4243",
        }
    assert len({j["name"] for j in jobs}) == 76


def test_selection_rejects_missing_or_nonfinite_results(tmp_path):
    record = {"name": "trial"}
    assert selection_score(tmp_path, record) is None
    path = tmp_path / "trial.jsonl"
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "service_blocking_probability": {"mean": 0.3},
                    "bitrate_blocking_probability": {"mean": 0.44},
                }
            }
        )
        + "\n"
    )
    assert selection_score(tmp_path, record) == (0.3, 0.44)
    path.write_text(path.read_text().replace("0.3", "NaN"))
    assert selection_score(tmp_path, record) is None
