"""Confirm combined mask changes, then benchmark matched transformer PPO."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-input", type=Path, required=True)
    parser.add_argument("--screen-size", type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = ["uv", "run", "--no-sync", "python", "-m"]
    mask_root = args.output / "mask"
    commands = [
        prefix
        + [
            "experimental.benchmarks.gn_full_mask_iteration",
            "--gpu",
            str(args.gpu),
            "--output",
            str(mask_root),
            "--state-input",
            str(args.state_input),
            "--confirm-screen-size",
            str(args.screen_size),
            "--profile",
        ]
    ]

    def run(name, command):
        with (args.output / (name + ".log")).open("w") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
        print(json.dumps({"job": name, "exit_code": result.returncode}), flush=True)
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(name + " failed")

    run("mask", commands[0])
    variants = {
        name: json.loads((mask_root / (name + ".json")).read_text())
        for name in ("optimized", "optimized-compact")
    }
    for name, data in variants.items():
        if not data.get("accuracy") or not all(item["exact"] for item in data["accuracy"].values()):
            raise AssertionError(name + " changed recorded physical outputs")
    best = min(variants, key=lambda n: variants[n]["metrics"]["rollout"]["median_ms"])
    flags = [
        "--reuse-screened-power",
        "--narrow-new-snr",
        "--reuse-unaffected-snr",
        "--screen-batch-size",
        str(args.screen_size),
    ]
    if best == "optimized-compact":
        flags.append("--compact-snr-updates")
    common = prefix + [
        "experimental.benchmarks.gn_full_mask_ppo_benchmark",
        "--gpu",
        str(args.gpu),
        "--gpu-memory-fraction",
        "0.25",
        "--state-input",
        str(args.state_input),
        "--iterations",
        "3",
        "--pack-old-predicates",
    ]
    for name, extra in (("ppo-control", []), ("ppo-optimized", flags)):
        command = (
            common
            + ["--output", str(args.output / name), "--name", "fullmask-h100-v14-" + name]
            + extra
        )
        if extra:
            command += ["--reference", str(args.output / "ppo-control" / "final-states.npz")]
        commands.append(command)
        run(name, command)
    command = prefix + [
        "experimental.benchmarks.gn_log_full_mask_iteration",
        "--directory",
        str(mask_root),
        "--name",
        "fullmask-combined-h100-v14",
    ]
    commands.append(command)
    run("wandb", command)
    (args.output / "status.json").write_text(
        json.dumps({"state": "complete", "best_variant": best}) + "\n"
    )


if __name__ == "__main__":
    main()
