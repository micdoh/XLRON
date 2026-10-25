"""Paper-informed off-policy IAM continuations from the best occupancy+SNR model.

The fine_tune lane strengthens the valid-mass loss around the successful update
regime. The single_update lane compares learning rates with one optimizer update
per rollout, as in the reproduction commands. All jobs retain full C+L+S physics.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from experimental.benchmarks.gn_cls_training_campaign import build_common
from experimental.benchmarks.gn_cls_tuning_campaign import replace_flags

PAPER = "https://arxiv.org/html/2605.02075v2"


def build_jobs(args):
    common = build_common(args)
    architecture = [
        "--USE_TRANSFORMER",
        "--transformer_obs_type=occupancy_snr",
        "--transformer_num_layers=2",
        "--transformer_num_heads=4",
        "--transformer_embedding_size=128",
        "--transformer_intermediate_size=256",
    ]
    train_base = (
        replace_flags(common, "--ROLLOUT_LENGTH=128", "--STEPS_PER_INCREMENT=65536")
        + architecture
        + [
            "--initial-model",
            str(args.resume_model),
            "--RETRAIN_MODEL",
            "--KEEP_VF=false",
            "--NUM_ENVS=32",
            "--NUM_MINIBATCHES=8",
            "--UPDATE_EPOCHS=4",
            f"--TOTAL_TIMESTEPS={args.train_timesteps}",
            "--SEED=42",
            "--LR=0.0001",
            "--LR_SCHEDULE=cosine",
            "--LR_END_FRACTION=0.1",
            "--GAMMA=0.996",
            "--GAE_LAMBDA=0.99",
            "--CLIP_EPS=0.04",
            "--ENT_COEF=0.005",
            "--ENT_SCHEDULE=linear",
            "--VF_COEF=0.05",
            "--SEPARATE_VF_OPTIMIZER",
            "--VF_LR=0.00005",
            "--OFF_POLICY_IAM=true",
            "--IAM_RECENTER_CLIP=false",
            "--POSITIVE_ADV_ONLY=false",
            "--MU_WEIGHT_ACTOR=false",
            "--IAM_DAMPING=true",
            "--IAM_GATING=true",
            "--IAM_GATING_MIN_ACTIONS=2",
            "--VALID_MASS_TARGET=0.05",
            "--VALID_MASS_LOSS_COEF=0.002",
            "--VML_SCHEDULE=constant",
            "--VML_END_FRACTION=1.0",
            "--MAX_GRAD_NORM=0.5",
            "--ADAM_EPS=0.00001",
            "--compute_dtype=float32",
            "--params_dtype=float32",
            "--mixed_precision=false",
            "--REWARD_CENTERING=false",
            "--ENHANCED_LOGGING",
            "--SAVE_MODEL",
            "--OVERWRITE_MODEL",
            "--EVAL_DURING_TRAINING",
            "--EVAL_FREQUENCY=32",
            "--EVAL_TIMESTEPS=8192",
        ]
    )
    if args.lane == "fine_tune":
        variants = [
            ("finetune-vml0.002", ["--VALID_MASS_LOSS_COEF=0.002"]),
            ("finetune-vml0.001", ["--VALID_MASS_LOSS_COEF=0.001"]),
            (
                "finetune-vml0.002-target0.005",
                ["--VALID_MASS_LOSS_COEF=0.002", "--VALID_MASS_TARGET=0.005"],
            ),
        ]
    else:
        train_base = replace_flags(
            train_base,
            "--ROLLOUT_LENGTH=64",
            "--NUM_MINIBATCHES=1",
            "--UPDATE_EPOCHS=1",
            "--VF_COEF=0.1",
        )
        variants = [
            ("single-update-lr0.0005-vml0.002", ["--LR=0.0005"]),
            ("single-update-lr0.0001-vml0.002", ["--LR=0.0001"]),
        ]
    jobs = []
    for variant, overrides in variants:
        name = f"transformer-occupancy_snr-{variant}"
        model = args.output / (name + ".eqx")
        jobs.append(
            {
                "name": name,
                "kind": "train",
                "command": replace_flags(train_base, *overrides) + [f"--MODEL_PATH={model}"],
                "model_path": str(model),
                "paper_source": PAPER,
                "adaptations": "Same saved NSFNET 4-head architecture; GN workload, lower fine-tuning LR/entropy, no action aggregation; VML held constant to isolate coefficient",
            }
        )
        for seed in (4244, 4242, 4243):
            jobs.append(
                {
                    "name": f"eval-{name}-seed{seed}",
                    "kind": "evaluation",
                    "parent": name,
                    "selection": seed == 4244,
                    "command": common
                    + architecture
                    + [
                        "--NUM_ENVS=16",
                        f"--TOTAL_TIMESTEPS={args.eval_timesteps}",
                        f"--SEED={seed}",
                        "--EVAL_MODEL",
                        f"--MODEL_PATH={model}",
                        "--deterministic",
                    ],
                }
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--lane", choices=("fine_tune", "single_update"), required=True)
    parser.add_argument("--resume-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-timesteps", type=int, default=16_777_216)
    parser.add_argument("--eval-timesteps", type=int, default=524_288)
    parser.add_argument("--project", default="XLRON-GN-CLS")
    parser.add_argument("--entity", default="micdoh")
    parser.add_argument("--group", default="gn-cls-20261025-paper-stabilized")
    parser.add_argument("--load", type=float, default=5000)
    parser.add_argument("--warmup", type=int, default=25000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.pool_before_projection = False
    args.output = args.output.resolve()
    args.resume_model = args.resume_model.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = args.output / "job_plan.json"
    if plan.exists():
        raise FileExistsError(plan)
    if not args.dry_run and not args.resume_model.is_file():
        raise FileNotFoundError(args.resume_model)
    jobs = build_jobs(args)
    plan.write_text(json.dumps(jobs, indent=2) + "\n")
    (args.output / "paper_plan.json").write_text(
        json.dumps(
            {
                "paper": PAPER,
                "source_checkpoint": str(args.resume_model),
                "source_checkpoint_sha256": hashlib.sha256(
                    args.resume_model.read_bytes()
                ).hexdigest()
                if args.resume_model.is_file()
                else None,
                "lane": args.lane,
                "comparison": "Five off-policy IAM trials, all starting from the same best occupancy+SNR checkpoint",
                "numerical_loss": "Exact log-domain valid mass; empty masks contribute zero gradient and zero barrier",
                "paper_table_values": {
                    "actor_lr": 0.0015,
                    "critic_lr": 0.00005,
                    "clip": 0.04,
                    "gamma": 0.996,
                    "gae_lambda": 0.99,
                    "rollout": 64,
                    "vml_coefficients": [0.001, 0.002],
                },
                "adaptation": "Keep full C+L+S load 5000, float32, saved 4-head architecture, small entropy and lower continuation LR; constant VML for controlled coefficient comparison",
            },
            indent=2,
        )
        + "\n"
    )
    command = [
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
        "--entity",
        args.entity,
        "--group",
        args.group,
    ]
    if args.dry_run:
        command.append("--dry-run")
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
