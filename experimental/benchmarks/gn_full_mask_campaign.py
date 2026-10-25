"""Best observed PPO continuation and renewed full-physical-mask heuristics."""

import argparse
import json
import subprocess
from pathlib import Path

from experimental.benchmarks.gn_cls_stabilized_campaign import build_jobs as stabilized_jobs
from experimental.benchmarks.gn_cls_tuning_campaign import HEURISTICS, replace_flags


def build_jobs(args):
    template_args = argparse.Namespace(
        **vars(args), lane="fine_tune", warmup=0, pool_before_projection=False
    )
    template = stabilized_jobs(template_args)[0]["command"]
    common = replace_flags(
        template,
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
        "--initial-env-state",
        str(args.initial_state),
        "--ENV_WARMUP_STEPS=0",
        "--NUM_ENVS=32",
        "--STEPS_PER_INCREMENT=8192",
        "--EVAL_FREQUENCY=32",
        "--EVAL_TIMESTEPS=1024",
        "--transformer_spectrum_encoder=dense",
        "--transformer_actor_pooling=mean",
        "--transformer_modulation_features=true",
        "--upgrade-modulation-head",
    )
    if args.kind == "training":
        pilot = "preflight-best-hp-full-physical"
        name = "transformer-occupancy-snr-best-hp-full-physical"
        pilot_command = replace_flags(
            common,
            "--TOTAL_TIMESTEPS=8192",
            "--EVAL_DURING_TRAINING=false",
            f"--MODEL_PATH={args.output / (pilot + '.eqx')}",
        )
        primary = replace_flags(common, f"--MODEL_PATH={args.output / (name + '.eqx')}")
        jobs = [
            dict(name=pilot, kind="train", command=pilot_command),
            dict(
                name=name,
                kind="train",
                parent=pilot,
                command=primary,
                model_path=str(args.output / (name + ".eqx")),
            ),
        ]
        for seed in (4244, 4242, 4243):
            evaluation = [x for x in primary if x != "--upgrade-modulation-head"]
            index = evaluation.index("--initial-model")
            del evaluation[index : index + 2]
            evaluation = replace_flags(
                evaluation,
                "--RETRAIN_MODEL=false",
                "--EVAL_MODEL=true",
                "--SAVE_MODEL=false",
                "--EVAL_DURING_TRAINING=false",
                "--NUM_ENVS=16",
                "--bootstrap-warmup-steps",
                "25000",
                f"--SEED={seed}",
                f"--TOTAL_TIMESTEPS={args.eval_timesteps}",
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
        return jobs

    common = [x for x in common if x != "--upgrade-modulation-head"]
    index = common.index("--initial-model")
    del common[index : index + 2]
    common = replace_flags(
        common,
        "--RETRAIN_MODEL=false",
        "--EVAL_HEURISTIC=true",
        "--USE_TRANSFORMER=false",
        "--transformer_modulation_features=false",
        "--transformer_obs_type=departure",
        "--SAVE_MODEL=false",
        "--EVAL_DURING_TRAINING=false",
        "--NUM_ENVS=16",
        f"--TOTAL_TIMESTEPS={args.eval_timesteps}",
        "--bootstrap-warmup-steps",
        "25000",
    )
    ordered = ["arbr_ff", "mscl_ksp", "ksp_ff"] + [
        h for h in HEURISTICS if h not in ("arbr_ff", "mscl_ksp", "ksp_ff")
    ]
    variants = [(h, 1) for h in ordered] + [(h, 0) for h in ("ksp_mscl", "mscl_ksp")]
    return [
        dict(
            name=f"heuristic-{h}-n{n}-seed{seed}-full-physical",
            kind="heuristic",
            command=replace_flags(
                common, f"--path_heuristic={h}", f"--mscl_interfering_k={n}", f"--SEED={seed}"
            ),
        )
        for h, n in variants
        for seed in (4242, 4243)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--kind", choices=("training", "heuristics"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-model", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--train-timesteps", type=int, default=8388608)
    parser.add_argument("--eval-timesteps", type=int, default=524288)
    parser.add_argument("--load", type=float, default=5000)
    parser.add_argument("--project", default="XLRON-GN-CLS")
    parser.add_argument("--group", default="gn-cls-20261025-full-physical-best-hp")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
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
        "--shard",
        "0" if args.kind == "training" else "1",
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
    if args.gpu_memory_fraction is not None:
        command += ["--gpu-memory-fraction", str(args.gpu_memory_fraction)]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
