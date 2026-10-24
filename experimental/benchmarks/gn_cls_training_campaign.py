"""Reproducible full C+L+S heuristic/PPO campaign on an explicitly free GPU.

Run two lanes on Turin after tests and profiling:
  uv run --no-sync python -m experimental.benchmarks.gn_cls_training_campaign \
      --gpu 2 --lane heuristics_then_train --output /tmp/gn-cls/lane2
  uv run --no-sync python -m experimental.benchmarks.gn_cls_training_campaign \
      --gpu 3 --lane train --output /tmp/gn-cls/lane3

Every child logs online to W&B. Manifests retain exact commands, safe environment
overrides, run IDs, exit status and timing. Failures never count as evaluations.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--lane", choices=("train", "heuristics_then_train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", default="XLRON-GN-CLS")
    parser.add_argument("--entity", default="micdoh")
    parser.add_argument("--group", default="gn-cls-20261024")
    parser.add_argument("--wandb-app-url", help="Override the configured W&B application URL")
    parser.add_argument("--train-timesteps", type=int, default=10_485_760)
    parser.add_argument("--eval-timesteps", type=int, default=524_288)
    parser.add_argument("--warmup", type=int, default=25_000)
    parser.add_argument("--load", type=float, default=5000.0)
    parser.add_argument("--pool-before-projection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    app_url = args.wandb_app_url
    if app_url is None and not args.dry_run:
        import wandb

        app_url = wandb.Settings(base_url=wandb.Api(timeout=30).settings["base_url"]).app_url
    app_url = (app_url or "https://wandb.ai").rstrip("/")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        used = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={args.gpu}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if used > 512:
            raise RuntimeError(f"GPU {args.gpu} has {used} MiB allocated; refusing to start")

    common = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "xlron.kernels.gn_train_benchmark",
        "--xpm-backend",
        "pallas_grouped",
        "--step-xpm-backend",
        "pallas_grouped",
        "--candidate-snr-backend",
        "pallas",
        "--path-snr-backend",
        "pallas_precomputed",
        "--xpm-tile",
        "16",
        "32",
        "4",
        "--env_type=rmsa_gn_model",
        "--topology_name=nsfnet_deeprmsa_directed",
        "--band_preference=C,L,S",
        "--inter_band_gap_ghz=25",
        "--link_resources=1668",
        "--slot_size=12.5",
        "--k=5",
        "--values_bw=100,200,400",
        f"--load={args.load}",
        "--guardband=0",
        "--coherent",
        "--power_per_channel=-10",
        "--max_power_per_fibre=21",
        "--launch_power_type=fixed",
        "--continuous_operation",
        f"--ENV_WARMUP_STEPS={args.warmup}",
        "--warmup_action_type=heuristic",
        "--path_heuristic=ksp_ff",
        "--ROLLOUT_LENGTH=64",
        "--STEPS_PER_INCREMENT=32768",
        f"--VISIBLE_DEVICES={args.gpu}",
        "--PREALLOCATE_MEM=false",
        "--WANDB",
        f"--PROJECT={args.project}",
        "--DOWNSAMPLE_FACTOR=64",
    ]
    if args.pool_before_projection:
        common.append("--pool-before-projection")
    jobs = []
    if args.lane == "heuristics_then_train":
        for heuristic in ("ksp_ff", "ff_ksp", "ksp_mscl", "mscl_ksp"):
            # Each heuristic also warms up using its own policy.
            command = [x for x in common if not x.startswith("--path_heuristic=")]
            jobs.append(
                (
                    f"heuristic-{heuristic}",
                    "heuristic",
                    command
                    + [
                        "--EVAL_HEURISTIC",
                        f"--path_heuristic={heuristic}",
                        "--SEED=4242",
                        "--NUM_ENVS=16",
                        f"--TOTAL_TIMESTEPS={args.eval_timesteps}",
                    ],
                )
            )

    obs = "snr" if args.lane == "heuristics_then_train" else "occupancy_snr"
    lr = "0.001" if obs == "snr" else "0.0005"
    label = f"transformer-{obs}-lr{lr}"
    model_path = output / f"{label}.eqx"
    if model_path.exists() and not args.dry_run:
        raise FileExistsError(f"Use a fresh output directory; {model_path} already exists")
    architecture = [
        "--USE_TRANSFORMER",
        f"--transformer_obs_type={obs}",
        "--transformer_num_layers=2",
        "--transformer_num_heads=4",
        "--transformer_embedding_size=128",
        "--transformer_intermediate_size=256",
    ]
    train = (
        common
        + architecture
        + [
            "--NUM_ENVS=32",
            "--NUM_MINIBATCHES=8",
            "--UPDATE_EPOCHS=4",
            f"--TOTAL_TIMESTEPS={args.train_timesteps}",
            "--SEED=42",
            f"--LR={lr}",
            "--LR_SCHEDULE=cosine",
            "--LR_END_FRACTION=0.05",
            "--GAMMA=0.996",
            "--GAE_LAMBDA=0.99",
            "--CLIP_EPS=0.04",
            "--ENT_COEF=0.0175",
            "--ENT_SCHEDULE=linear",
            "--VF_COEF=0.05",
            "--SEPARATE_VF_OPTIMIZER",
            "--VF_LR=0.0001",
            "--OFF_POLICY_IAM",
            "--VALID_MASS_LOSS_COEF=0.0002",
            "--VML_SCHEDULE=constant",
            "--VML_END_FRACTION=0.01",
            "--ENHANCED_LOGGING",
            "--SAVE_MODEL",
            f"--MODEL_PATH={model_path}",
            "--EVAL_DURING_TRAINING",
            "--EVAL_FREQUENCY=16",
            "--EVAL_TIMESTEPS=32768",
        ]
    )
    jobs.append((label, "train", train))
    for seed in (4242, 4243):
        jobs.append(
            (
                f"eval-{label}-seed{seed}",
                "evaluation",
                common
                + architecture
                + [
                    "--NUM_ENVS=16",
                    f"--TOTAL_TIMESTEPS={args.eval_timesteps}",
                    f"--SEED={seed}",
                    "--EVAL_MODEL",
                    f"--MODEL_PATH={model_path}",
                    "--deterministic",
                ],
            )
        )

    sources = [
        "xlron/environments/env_funcs.py",
        "xlron/environments/make_env.py",
        "xlron/environments/gn_model/isrs_gn_model.py",
        "xlron/models/transformer.py",
        "xlron/train/ppo.py",
        "xlron/train/train_utils.py",
        "xlron/parameter_flags.py",
        "xlron/kernels/gn_xpm.py",
        "xlron/kernels/gn_xpm_legacy.py",
        "xlron/kernels/gn_path_snr.py",
        "xlron/kernels/gn_train_benchmark.py",
    ]
    source_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_hashes,
        "source_fingerprint": fingerprint,
        "jobs": [],
    }
    manifest_path = output / "manifest.json"

    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    failed_training = False
    for name, kind, command in jobs:
        run_id = uuid.uuid4().hex[:8]
        command += [f"--EXPERIMENT_NAME={name}", f"--DATA_OUTPUT_FILE={output / (name + '.jsonl')}"]
        overrides = {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0",
            "WANDB_MODE": "online",
            "WANDB_ENTITY": args.entity,
            "WANDB_RUN_GROUP": args.group,
            "WANDB_JOB_TYPE": kind,
            "WANDB_TAGS": f"full-cls,gn-kernels,{kind}",
            "WANDB_RUN_ID": run_id,
            "WANDB_RESUME": "never",
            "WANDB_DIR": str(output),
            "XLRON_SOURCE_FINGERPRINT": fingerprint,
        }
        record = {
            "name": name,
            "kind": kind,
            "command": command,
            "environment": overrides,
            "wandb_url": f"{app_url}/{args.entity}/{args.project}/runs/{run_id}",
            "status": "planned",
        }
        manifest["jobs"].append(record)
        if kind == "evaluation" and failed_training:
            record["status"] = "skipped_training_failed"
            save()
            continue
        if args.dry_run:
            save()
            continue
        started = time.monotonic()
        with (output / (name + ".log")).open("w") as log:
            child = subprocess.Popen(
                command, env=dict(os.environ, **overrides), stdout=log, stderr=subprocess.STDOUT
            )
            record.update(status="running", pid=child.pid)
            save()
            print(
                json.dumps({"name": name, "pid": child.pid, "wandb_url": record["wandb_url"]}),
                flush=True,
            )
            code = child.wait()
        record.update(
            status="completed" if code == 0 else "failed",
            exit_code=code,
            wall_s=time.monotonic() - started,
        )
        save()
        print(json.dumps({"name": name, "exit_code": code, "wall_s": record["wall_s"]}), flush=True)
        failed_training |= kind == "train" and code != 0
    if any(job["status"] == "failed" for job in manifest["jobs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
