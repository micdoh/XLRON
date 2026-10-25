"""Matched full-resolution observation/encoder trials, from fresh initialization."""

import argparse
import json
import subprocess
from pathlib import Path

from experimental.benchmarks.gn_cls_stabilized_campaign import build_jobs as stabilized_jobs
from experimental.benchmarks.gn_cls_tuning_campaign import replace_flags


def build_jobs(args):
    base_args = argparse.Namespace(**vars(args))
    base_args.lane = "single_update"
    template = stabilized_jobs(base_args)[0]["command"]
    index = template.index("--initial-model")
    del template[index : index + 2]
    template = replace_flags(
        template,
        "--RETRAIN_MODEL=false",
        "--aggregate_slots=1",
        f"--gn_mask_mode={args.mask_mode}",
        "--LR=0.0005",
        "--NUM_MINIBATCHES=8",
        "--transformer_actor_pooling=sum",
        "--transformer_modulation_features=true",
        "--gn_mask_batch_size=64",
        "--gn_mask_env_batch_size=1",
        "--new-lightpath-prefilter",
        "--matmul-precision",
        "highest",
    )
    variants = [
        ("occupancy-snr-dense128", "occupancy_snr", "dense", 128),
        ("lifetime-snr-dense128", "lifetime_snr", "dense", 128),
        ("lifetime-snr-conv128", "lifetime_snr", "conv", 128),
        ("lifetime-snr-dense256", "lifetime_snr", "dense", 256),
    ]
    # Both seeds see every architecture, with opposite lane order to start the
    # structured and dense controls concurrently.
    if args.gpu == 3:
        variants = variants[2:] + variants[:2]

    def baseline(heuristic):
        # New full masks change the feasible set, so compare to heuristics using
        # the same full masks rather than the historical FF/LF results.
        seed = 4242 if args.gpu == 2 else 4243
        command = replace_flags(
            template,
            "--EVAL_HEURISTIC=true",
            "--USE_TRANSFORMER=false",
            "--transformer_modulation_features=false",
            "--transformer_obs_type=departure",
            "--SAVE_MODEL=false",
            "--EVAL_DURING_TRAINING=false",
            "--NUM_ENVS=16",
            f"--TOTAL_TIMESTEPS={args.eval_timesteps}",
            f"--SEED={seed}",
            f"--path_heuristic={heuristic}",
        )
        return dict(
            name=f"heuristic-{heuristic}-seed{seed}-all_slots", kind="heuristic", command=command
        )

    jobs = [baseline("arbr_ff")]
    for variant, obs, encoder, width in variants:
        name = f"{variant}-seed{args.seed}-{args.mask_mode}"
        model = args.output / (name + ".eqx")
        common = [
            f"--transformer_obs_type={obs}",
            f"--transformer_spectrum_encoder={encoder}",
            f"--transformer_embedding_size={width}",
            f"--transformer_intermediate_size={2 * width}",
        ]
        train = replace_flags(template, *common, f"--SEED={args.seed}", f"--MODEL_PATH={model}")
        jobs.append(dict(name=name, kind="train", command=train, model_path=str(model)))
        for seed in (4244, 4242, 4243):
            evaluation = replace_flags(
                train,
                "--EVAL_MODEL=true",
                "--SAVE_MODEL=false",
                "--EVAL_DURING_TRAINING=false",
                "--NUM_ENVS=16",
                f"--TOTAL_TIMESTEPS={args.eval_timesteps}",
                f"--SEED={seed}",
                "--deterministic",
            )
            jobs.append(
                dict(
                    name=f"eval-{name}-seed{seed}",
                    kind="evaluation",
                    parent=name,
                    selection=seed == 4244,
                    command=evaluation,
                )
            )
        if variant == variants[0][0]:
            jobs.extend(baseline(h) for h in ("ksp_ff", "mscl_ksp"))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mask-mode", choices=("all_slots",), default="all_slots")
    p.add_argument(
        "--resume-model",
        type=Path,
        required=True,
        help="Provenance checkpoint for the shared runner; trials initialize fresh",
    )
    p.add_argument("--train-timesteps", type=int, default=16777216)
    p.add_argument("--eval-timesteps", type=int, default=524288)
    p.add_argument("--project", default="XLRON-GN-CLS")
    p.add_argument("--group", default="gn-cls-20261025-fullslot-observations")
    p.add_argument("--load", type=float, default=5000)
    p.add_argument("--warmup", type=int, default=25000)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.pool_before_projection = False
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = args.output / "job_plan.json"
    if plan.exists():
        raise FileExistsError(plan)
    plan.write_text(json.dumps(build_jobs(args), indent=2) + "\n")
    (args.output / "comparison_plan.json").write_text(
        json.dumps(
            dict(
                initialization="fresh; optimizer and policy weights are not resumed",
                minibatch_size=256,
                mask_mode=args.mask_mode,
                aggregate_slots=1,
                seed=args.seed,
                purpose="Separate observation encoding, local feature retention, and hidden capacity",
            ),
            indent=2,
        )
        + "\n"
    )
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "experimental.benchmarks.gn_cls_tuning_campaign",
        "--gpu",
        str(args.gpu),
        "--shard",
        str(args.gpu - 2),
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
        cmd.append("--dry-run")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
