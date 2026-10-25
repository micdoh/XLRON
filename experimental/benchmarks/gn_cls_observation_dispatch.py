"""Launch full-slot comparisons only after native and full-shape PPO preflight."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--preflight-pid", type=int, required=True)
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--provenance-checkpoint", type=Path, required=True)
    p.add_argument("--wandb-run")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "dispatch.json"

    def status(state, **kw):
        manifest.write_text(json.dumps(dict(state=state, **kw), indent=2) + "\n")
        print(json.dumps(dict(state=state, **kw)), flush=True)
        if args.wandb_run:
            import wandb

            run = wandb.Api().run("micdoh/XLRON-GN-CLS/" + args.wandb_run)
            run.summary["comparison/dispatch_state"] = state
            run.summary.update()

    status(
        "waiting_validation",
        mask_mode="all_slots",
        aggregate_slots=1,
        reason="Every spectrum-feasible start is checked with full admission physics before policy sampling",
    )
    while True:
        results = args.preflight / "results.json"
        rows = json.loads(results.read_text()) if results.exists() else []
        if any(row["exit_code"] for row in rows):
            status("validation_failed", results=rows)
            return
        models = args.benchmarks / "encoders.json"
        model_rows = json.loads(models.read_text())["results"] if models.exists() else []
        if len(rows) == 2 and len(model_rows) == 4:
            break
        if not psutil.pid_exists(args.preflight_pid):
            status(
                "validation_failed",
                reason="Preflight process ended without all required successful results",
            )
            return
        time.sleep(15)
    children = []
    for gpu, seed in ((2, 42), (3, 43)):
        lane = args.output / f"lane{gpu}"
        cmd = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "experimental.benchmarks.gn_cls_observation_campaign",
            "--gpu",
            str(gpu),
            "--seed",
            str(seed),
            "--output",
            str(lane),
            "--mask-mode",
            "all_slots",
            "--resume-model",
            str(args.provenance_checkpoint),
        ]
        with (args.output / f"controller{gpu}.log").open("w") as log:
            child = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        children.append(dict(gpu=gpu, seed=seed, pid=child.pid, command=cmd))
    status(
        "dispatched",
        controllers=children,
        initialization="fresh",
        aggregate_slots=1,
        mask_mode="all_slots",
    )


if __name__ == "__main__":
    main()
