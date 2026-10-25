"""Publish reviewed full-mask timings and exact-output checks to the campaign W&B."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--name", default="full-physical-mask-exact-screening")
    args = p.parse_args()
    reports = []
    for path in sorted(args.directory.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "metrics" in data and path.name != "profile-timing.json":
            reports.append((path, data))
    if not reports:
        raise ValueError("No completed benchmark reports")
    baseline = next(data for path, data in reports if path.stem == "unpruned16")
    candidates = [
        (path, d)
        for path, d in reports
        if d.get("accuracy") and all(v["exact"] for v in d["accuracy"].values())
    ]
    best_path, best = min(candidates, key=lambda pair: pair[1]["metrics"]["rollout"]["median_ms"])
    with wandb.init(
        entity="micdoh",
        project="XLRON-GN-CLS",
        group="gn-cls-20261025-fullslot-observations",
        job_type="benchmark",
        name=args.name,
        config=dict(
            topology="nsfnet_deeprmsa_directed",
            bands="C,L,S",
            slots=1668,
            k=5,
            aggregate_slots=1,
            mask_mode="all_slots",
            load=5000,
            purpose="Matched outputs, environment only; excludes neural network/PPO and SNR-observation refresh",
        ),
    ) as run:
        table = wandb.Table(
            columns=[
                "variant",
                "envs",
                "candidate_batch",
                "env_chunk",
                "mask_env_chunk",
                "mask_ms",
                "rollout_ms",
                "env_steps_s",
                "temp_MiB",
                "exact_outputs",
            ]
        )
        for path, d in reports:
            m = d["metrics"]
            table.add_data(
                path.stem,
                d["args"]["envs"],
                d["args"]["gn_mask_batch_size"],
                d["args"].get("env_batch_size", 0),
                d["args"].get("gn_mask_env_batch_size", 0),
                m["mask"]["median_ms"],
                m["rollout"]["median_ms"],
                m["rollout"]["steps_per_s"],
                m["mask"]["temp_bytes"] / 2**20,
                all(v["exact"] for v in d.get("accuracy", {}).values()),
            )
        run.log({"benchmarks/full_mask": table})
        run.summary.update(
            {
                "mask/baseline_ms": baseline["metrics"]["mask"]["median_ms"],
                "mask/optimized_ms": best["metrics"]["mask"]["median_ms"],
                "mask/speedup": baseline["metrics"]["mask"]["median_ms"]
                / best["metrics"]["mask"]["median_ms"],
                "rollout/baseline_steps_s": baseline["metrics"]["rollout"]["steps_per_s"],
                "rollout/optimized_steps_s": best["metrics"]["rollout"]["steps_per_s"],
                "rollout/speedup": baseline["metrics"]["rollout"]["median_ms"]
                / best["metrics"]["rollout"]["median_ms"],
                "comparison/best_variant": best_path.stem,
                "comparison/exact_recorded_outputs": True,
                "comparison/independent_traffic": baseline["args"]["independent_streams"],
            }
        )
        artifact = wandb.Artifact("full-physical-mask-evidence", type="benchmark")
        for path in sorted(args.directory.iterdir()):
            if path.suffix in (".json", ".npz", ".csv"):
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        (args.directory / "wandb-report.json").write_text(
            json.dumps(dict(id=run.id, url=run.url), indent=2) + "\n"
        )
        print(run.url, flush=True)


if __name__ == "__main__":
    main()
