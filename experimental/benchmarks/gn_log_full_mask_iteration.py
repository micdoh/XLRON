"""Publish paired full-mask experiments, including unsuccessful tile controls."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    reports = {}
    for path in args.directory.glob("*.json"):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "metrics" in data and path.stem != "profile-timing":
            reports[path.stem] = data
    control = reports["control"]
    for name, data in reports.items():
        if name != "control" and not (
            data.get("accuracy") and all(result["exact"] for result in data["accuracy"].values())
        ):
            raise ValueError(name + " lacks exact-output validation")
    best = min(reports, key=lambda n: reports[n]["metrics"]["rollout"]["median_ms"])
    with wandb.init(
        entity="micdoh",
        project="XLRON-GN-CLS",
        job_type="benchmark",
        group="gn-cls-20261025-fullmask-iterations",
        name=args.name,
        config={
            "slots": 1668,
            "bands": "C,L,S",
            "k": 5,
            "envs": 16,
            "load": 5000,
            "aggregate_slots": 1,
            "gn_mask_mode": "all_slots",
            "resource_policy": json.loads((args.directory / "resource-policy.json").read_text()),
            "note": "Paired environment-only timings on shared idle capacity; compilation, warmup, inference and PPO excluded.",
        },
    ) as run:
        table = wandb.Table(
            columns=[
                "variant",
                "mask_ms",
                "rollout_ms",
                "environment_transitions_per_s",
                "mask_temporary_MiB",
                "exact_recorded_outputs",
            ]
        )
        for name, data in sorted(reports.items()):
            m = data["metrics"]
            table.add_data(
                name,
                m["mask"]["median_ms"],
                m["rollout"]["median_ms"],
                m["rollout"]["steps_per_s"],
                m["mask"]["temp_bytes"] / 2**20,
                True,
            )
        run.log({"benchmarks/full_mask": table})
        run.summary.update(
            {
                "comparison/best_variant": best,
                "comparison/exact_recorded_outputs": True,
                "comparison/rollout_speedup": control["metrics"]["rollout"]["median_ms"]
                / reports[best]["metrics"]["rollout"]["median_ms"],
                "benchmark/control_environment_transitions_per_s": control["metrics"]["rollout"][
                    "steps_per_s"
                ],
                "benchmark/best_environment_transitions_per_s": reports[best]["metrics"]["rollout"][
                    "steps_per_s"
                ],
            }
        )
        artifact = wandb.Artifact(args.name, type="benchmark")
        for path in sorted(args.directory.iterdir()):
            if path.suffix in (".json", ".csv", ".npz") or path.name == "native-tests.log":
                artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        (args.directory / "wandb-report.json").write_text(
            json.dumps({"id": run.id, "url": run.url}, indent=2) + "\n"
        )
        print(run.url, flush=True)


if __name__ == "__main__":
    main()
