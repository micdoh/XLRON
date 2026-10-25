"""Preserve an active campaign checkpoint before replacing its remaining queue."""

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import psutil
import wandb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane", type=Path, required=True)
    args = p.parse_args()
    root = args.lane.resolve()
    mp = root / "manifest.json"
    m = json.loads(mp.read_text())
    job = next(j for j in m["jobs"] if j["status"] == "running")
    controllers = []
    for proc in psutil.process_iter():
        try:
            command = proc.cmdline()
            if (
                proc.uids().real == os.getuid()
                and Path(proc.exe()).name.startswith("python")
                and "experimental.benchmarks.gn_cls_tuning_campaign" in command
                and str(root) in command
            ):
                controllers.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    assert len(controllers) == 1
    controller = controllers[0]
    children = [
        p
        for p in controller.children(recursive=True)
        if Path(p.exe()).name.startswith("python")
        and "xlron.kernels.gn_train_benchmark" in p.cmdline()
        and any(job["name"] in x for x in p.cmdline())
    ]
    assert len(children) == 1
    controller.send_signal(signal.SIGSTOP)
    checkpoint = Path(job["model_path"])
    print(
        json.dumps(
            dict(state="waiting_checkpoint", controller=controller.pid, child=children[0].pid)
        ),
        flush=True,
    )
    while not checkpoint.is_file():
        if not children[0].is_running():
            raise RuntimeError("Training ended without a checkpoint; controller remains paused")
        time.sleep(10)
    previous = None
    while True:
        current = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        if current == previous and current[0] > 1024:
            break
        previous = current
        time.sleep(2)
    child = children[0]
    try:
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=20)
    except psutil.NoSuchProcess:
        pass
    controller.send_signal(signal.SIGTERM)
    controller.send_signal(signal.SIGCONT)
    rows = [
        json.loads(s) for s in (root / (job["name"] + ".per-env.jsonl")).read_text().splitlines()
    ]
    steps = rows[-1]["measured_steps_per_env"] * 32
    note = f"Intentionally stopped after {steps:,} recorded steps to prioritize unrestricted-slot observation comparisons. Best checkpoint preserved. This is a partial run, not a completed 16M-step trial."
    job.update(
        status="interrupted_user_steering",
        reason=note,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    for j in m["jobs"]:
        if j["status"] == "planned":
            j.update(
                status="superseded_user_steering",
                reason="Replaced by unrestricted-slot observation comparison",
            )
    m["state"] = "superseded_user_steering"
    mp.write_text(json.dumps(m, indent=2) + "\n")
    run = wandb.Api().run("micdoh/XLRON-GN-CLS/" + job["wandb_run_id"])
    run.notes = ((run.notes or "") + "\n\n" + note).strip()
    run.update()
    run.summary["interrupted/recorded_training_steps"] = steps
    run.summary["interrupted/user_steering"] = True
    run.summary.update()
    print(json.dumps(dict(state="retired", checkpoint=str(checkpoint), steps=steps)), flush=True)


if __name__ == "__main__":
    main()
