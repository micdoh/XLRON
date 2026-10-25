"""Append eval/ aliases to completed campaign histories, preserving original data."""

import argparse
import ast
import json
from pathlib import Path

import wandb


def aliases(metrics, episodic):
    return {
        f"{metric}_{'episode_end_' if episodic else ''}{stat}": f"eval/{metric}_{stat}"
        for metric in metrics
        for stat in ("mean", "std", "iqr_lower", "iqr_upper")
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    module = ast.parse(Path("xlron/train/train_utils.py").read_text())
    metrics = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "metrics" for t in node.targets)
    )
    api = wandb.Api(timeout=60)
    runs = list(
        api.runs(
            "micdoh/XLRON-GN-CLS",
            filters={
                "state": "finished",
                "createdAt": {"$gte": "2026-10-24T00:00:00Z"},
                "$or": [{"config.EVAL_HEURISTIC": True}, {"config.EVAL_MODEL": True}],
            },
            order="+created_at",
        )
    )
    records = []
    for source in runs:
        if source.summary.get("eval/service_blocking_probability_mean") is not None:
            continue
        record = dict(id=source.id, name=source.name, url=source.url, status="planned")
        records.append(record)
        if args.limit and len(records) >= args.limit:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(dict(matching_finished_runs=len(runs), planned=len(records), apply=args.apply)),
        flush=True,
    )
    if not args.apply:
        return
    by_id = {r.id: r for r in runs}
    for record in records:
        source = by_id[record["id"]]
        mapping = aliases(metrics, not source.config.get("continuous_operation", False))
        rows = []
        for row in source.scan_history(page_size=1000):
            values = {
                new: row[old]
                for old, new in mapping.items()
                if isinstance(row.get(old), (float, int))
            }
            if values:
                for axis in ("env_step", "episode_count"):
                    if axis in row:
                        values[axis] = row[axis]
                if "training_time" in row:
                    values["eval/time_s"] = row["training_time"]
                rows.append(values)
        if not rows:
            record["status"] = "no_scalar_history"
            continue
        source_rows = len(rows)
        run = wandb.init(
            entity="micdoh",
            project="XLRON-GN-CLS",
            id=source.id,
            resume="must",
            settings=wandb.Settings(x_disable_stats=True, disable_git=True, save_code=False),
        )
        try:
            step = (
                "env_step" if source.config.get("continuous_operation", False) else "episode_count"
            )
            run.define_metric(step)
            run.define_metric("eval/*", step_metric=step)
            for row in rows:
                # Append new aliases; never overwrite or rewind original history.
                run.log(row)
            run.summary["logging/eval_namespace_version"] = 1
            run.summary["logging/eval_namespace_backfilled_rows"] = source_rows
            run.summary["logging/eval_namespace_note"] = (
                "Aliases of original completed evaluation history; no simulation rerun"
            )
            record.update(status="backfilled", rows=source_rows)
        except Exception as error:
            run.summary["logging/eval_namespace_backfill_error"] = str(error)
            record.update(status="error", error=str(error))
        finally:
            # A failed logging migration must not relabel completed simulation
            # results as a failed simulation.
            run.finish(exit_code=0)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
