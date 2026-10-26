"""Matched full-CLS margin and launch-power comparisons, always starting empty.

Each replica discards 25,000 requests under its trial's physical constraints.
The control retains the old 0.5 dB admission / 0 dB existing-path margins.
Power trials retain the new reserve while changing only fixed launch power.
"""

import argparse
import json
import subprocess
from pathlib import Path

from experimental.benchmarks.gn_cls_stabilized_campaign import (
    build_jobs as training_jobs,
)
from experimental.benchmarks.gn_cls_training_campaign import build_common
from experimental.benchmarks.gn_cls_tuning_campaign import replace_flags

MARGINS = {"control": (0.5, 0.0), "reserve": (0.5, 0.5), "admission": (1.0, 0.5)}


def optimized_flags():
    return [
        "--aggregate_slots=1",
        "--gn_mask_mode=all_slots",
        "--gn_mask_batch_size=64",
        "--gn_mask_env_batch_size=1",
        "--new-lightpath-prefilter",
        "--pack-old-predicates",
        "--reuse-screened-power",
        "--narrow-new-snr",
        "--reuse-unaffected-snr",
        "--screen-batch-size",
        "512",
        "--fused-endpoint-snr",
        "--generic-row-programs",
        "16",
        "--matmul-precision",
        "highest",
        "--audit-existing-snr",
        "--compute_dtype=float32",
        "--params_dtype=float32",
        "--mixed_precision=false",
    ]


def build_jobs(args):
    if args.warmup < 25000 and not args.preflight:
        raise ValueError(
            "Comparisons require at least 25,000 discarded requests per replica"
        )
    admission, retained = MARGINS["reserve" if args.lane == "power" else args.lane]
    margins = [f"--snr_margin={admission}", f"--existing_snr_margin={retained}"]
    common = replace_flags(build_common(args), *optimized_flags(), *margins)
    powers = (-11, -9) if args.lane == "power" else (-10,)
    jobs = []
    for power in powers:
        for heuristic in ("arbr_ff", "mscl_ksp"):
            for seed in (4242, 4243):
                name = f"{args.lane}-{heuristic}-seed{seed}-power{power}-admit{admission}-retain{retained}"
                command = replace_flags(
                    common,
                    "--EVAL_HEURISTIC=true",
                    "--USE_TRANSFORMER=false",
                    "--transformer_obs_type=departure",
                    "--mscl_interfering_k=1",
                    f"--path_heuristic={heuristic}",
                    f"--SEED={seed}",
                    f"--power_per_channel={power}",
                    f"--NUM_ENVS={args.num_envs}",
                    f"--TOTAL_TIMESTEPS={args.eval_timesteps}",
                    "--STEPS_PER_INCREMENT=8192",
                )
                jobs.append(dict(name=name, kind="heuristic", command=command))
    if args.train_after:
        if args.lane != "reserve":
            raise ValueError("The continuation lane uses the retained 0.5 dB reserve")
        template_args = argparse.Namespace(**(vars(args) | {"lane": "fine_tune"}))
        template = training_jobs(template_args)[0]
        name = "transformer-retained-margin0.5"
        model = args.output / (name + ".eqx")
        command = replace_flags(
            template["command"],
            *optimized_flags(),
            *margins,
            "--NUM_ENVS=16",
            "--STEPS_PER_INCREMENT=8192",
            "--transformer_spectrum_encoder=dense",
            "--transformer_actor_pooling=mean",
            "--transformer_modulation_features=true",
            "--upgrade-modulation-head",
            "--path_heuristic=arbr_ff",
            f"--MODEL_PATH={model}",
            "--EVAL_TIMESTEPS=1024",
            "--EVAL_FREQUENCY=32",
        )
        jobs.append(
            dict(name=name, kind="train", command=command, model_path=str(model))
        )
    if args.preflight:
        jobs = [jobs[0], jobs[2]] + [j for j in jobs if j["kind"] == "train"]
        for job in jobs:
            job["name"] = "preflight-" + job["name"]
            job["command"] = replace_flags(
                job["command"],
                "--ENV_WARMUP_STEPS=8",
                "--NUM_ENVS=4",
                "--ROLLOUT_LENGTH=32",
                "--TOTAL_TIMESTEPS=128",
                "--STEPS_PER_INCREMENT=128",
                "--DOWNSAMPLE_FACTOR=1",
                "--EVAL_DURING_TRAINING=false",
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--lane", choices=(*MARGINS, "power"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-model", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--eval-timesteps", type=int, default=131072)
    parser.add_argument("--train-timesteps", type=int, default=4194304)
    parser.add_argument("--warmup", type=int, default=25000)
    parser.add_argument("--load", type=float, default=5000)
    parser.add_argument("--project", default="XLRON-GN-CLS")
    parser.add_argument("--group", default="gn-cls-20261025-retained-margin")
    parser.add_argument("--train-after", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.pool_before_projection = False
    plan = args.output / "job_plan.json"
    if plan.exists():
        raise FileExistsError(plan)
    plan.write_text(json.dumps(build_jobs(args), indent=2) + "\n")
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "experimental.benchmarks.gn_cls_tuning_campaign",
        "--gpu",
        str(args.gpu),
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction),
        "--shard",
        "0",
        "--output",
        str(args.output),
        "--resume-model",
        str(args.resume_model),
        "--job-plan",
        str(plan),
        "--project",
        args.project,
        "--group",
        args.group,
    ]
    if args.dry_run:
        command.append("--dry-run")
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
