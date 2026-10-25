"""All implemented GN heuristics, followed by checkpoint-based PPO tuning.

Two workers shard the heuristic inventory, then tune one observation mode each.
Workers reserve only GPUs 2/3 and wait while another process owns GPU memory.
Every child runs online in W&B; exact plans and outcomes are saved before launch.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from experimental.benchmarks.gn_cls_training_campaign import build_common

HEURISTICS = (
    "ksp_ff",
    "ff_ksp",
    "kmc_ff",
    "kmf_ff",
    "ksp_mu",
    "ksp_mu_nonrel",
    "ksp_mu_unique",
    "mu_ksp",
    "mu_ksp_nonrel",
    "mu_ksp_unique",
    "kca_ff",
    "arbr_ff",
    "kme_ff",
    "ksp_bf",
    "bf_ksp",
    "ksp_lf",
    "lf_ksp",
    "ksp_ef",
    "ksp_flf",
    "flf_ksp",
    "ksp_flef",
    "ksp_mscl",
    "mscl_ksp",
)
SEEDS = (4242, 4243)


def replace_flags(command, *flags):
    """Replace whole absl flags without retaining an earlier conflicting value."""
    names = {flag.split("=", 1)[0] for flag in flags}
    return [arg for arg in command if arg.split("=", 1)[0] not in names] + list(flags)


def build_jobs(args):
    common = build_common(args)
    jobs = []
    variants = [(name, 1) for name in HEURISTICS] + [(name, 0) for name in ("ksp_mscl", "mscl_ksp")]
    for index, (heuristic, interfering_k) in enumerate(variants):
        if index % 2 != args.shard:
            continue
        for seed in SEEDS:
            name = f"heuristic-{heuristic}-n{interfering_k}-seed{seed}"
            command = replace_flags(common, f"--path_heuristic={heuristic}") + [
                "--EVAL_HEURISTIC",
                f"--mscl_interfering_k={interfering_k}",
                f"--SEED={seed}",
                "--NUM_ENVS=16",
                f"--TOTAL_TIMESTEPS={args.eval_timesteps}",
            ]
            jobs.append({"name": name, "kind": "heuristic", "command": command})

    obs = "snr" if args.shard == 0 else "occupancy_snr"
    architecture = [
        "--USE_TRANSFORMER",
        f"--transformer_obs_type={obs}",
        "--transformer_num_layers=2",
        "--transformer_num_heads=4",
        "--transformer_embedding_size=128",
        "--transformer_intermediate_size=256",
    ]
    jobs.append(
        {
            "name": f"eval-transformer-{obs}-initial-seed4244",
            "kind": "evaluation",
            "selection": True,
            "selection_model": str(args.resume_model),
            "command": common
            + architecture
            + [
                "--NUM_ENVS=16",
                f"--TOTAL_TIMESTEPS={args.eval_timesteps}",
                "--SEED=4244",
                "--EVAL_MODEL",
                f"--MODEL_PATH={args.resume_model}",
                "--deterministic",
            ],
        }
    )
    # Model weights continue; the optimizer and its schedules start afresh.
    # The last variant selects among the initial checkpoint and the first two.
    for variant in ("lower_lr_entropy", "on_policy", "long_horizon"):
        name = f"transformer-{obs}-{variant}"
        model = args.output / f"{name}.eqx"
        train = (
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
                "--VALID_MASS_LOSS_COEF=0.0002",
                "--VML_SCHEDULE=constant",
                "--VML_END_FRACTION=0.01",
                "--ENHANCED_LOGGING",
                "--SAVE_MODEL",
                "--OVERWRITE_MODEL",
                f"--MODEL_PATH={model}",
                "--EVAL_DURING_TRAINING",
                "--EVAL_FREQUENCY=16",
                "--EVAL_TIMESTEPS=8192",
            ]
        )
        if variant != "lower_lr_entropy":
            train = replace_flags(
                train, "--OFF_POLICY_IAM=false", "--CLIP_EPS=0.1", "--VALID_MASS_LOSS_COEF=0.0"
            )
        if variant == "long_horizon":
            train = replace_flags(
                train,
                "--ROLLOUT_LENGTH=256",
                "--LR=0.00005",
                "--GAMMA=0.9995",
                "--GAE_LAMBDA=0.995",
                "--ENT_COEF=0.002",
                "--REWARD_CENTERING=true",
            )
        jobs.append(
            {
                "name": name,
                "kind": "train",
                "command": train,
                "model_path": str(model),
                "adaptive_parent": variant == "long_horizon",
                "variant": variant,
            }
        )
        # 4244 selects the parent for the adaptive continuation. The reported
        # 4242/4243 comparison remains separate from this selection seed.
        for seed in (4244, *SEEDS):
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


def save_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def wait_for_gpu(gpu, status):
    """Hold our advisory lease, then wait for other GPU users without touching them."""
    lease = Path(f"/tmp/xlron-gn-gpu-{gpu}-{os.getuid()}.lock").open("a+")  # noqa: SIM115 -- caller closes the campaign lease
    last_report = 0.0
    while True:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() - last_report > 600:
                status("waiting_lease", gpu=gpu)
                last_report = time.monotonic()
            time.sleep(30)
    while True:
        used = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if used <= 512:
            status("gpu_ready", gpu=gpu, memory_mib=used)
            return lease
        if time.monotonic() - last_report > 600:
            status("waiting_gpu", gpu=gpu, memory_mib=used)
            last_report = time.monotonic()
        time.sleep(30)


def selection_score(output, record):
    """Finite service BP, with bitrate BP retained as the secondary criterion."""
    import math

    try:
        result = json.loads((output / (record["name"] + ".jsonl")).read_text().splitlines()[-1])
        metrics = result["metrics"]
        score = tuple(
            float(metrics[key]["mean"])
            for key in ("service_blocking_probability", "bitrate_blocking_probability")
        )
        return score if all(math.isfinite(x) for x in score) else None
    except (KeyError, ValueError, OSError, IndexError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-model", type=Path, required=True)
    parser.add_argument("--project", default="XLRON-GN-CLS")
    parser.add_argument("--entity", default="micdoh")
    parser.add_argument("--group", default="gn-cls-20261025-all-heuristics-tuning")
    parser.add_argument("--train-timesteps", type=int, default=8_388_608)
    parser.add_argument("--eval-timesteps", type=int, default=524_288)
    parser.add_argument("--warmup", type=int, default=25_000)
    parser.add_argument("--load", type=float, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.pool_before_projection = False
    args.output = args.output.resolve()
    args.resume_model = args.resume_model.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Use a fresh output directory: {manifest_path}")
    if not args.dry_run and not args.resume_model.is_file():
        raise FileNotFoundError(args.resume_model)
    paths = sorted(Path("xlron").rglob("*.py")) + sorted(
        Path("experimental/benchmarks").glob("gn_cls*.py")
    )
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    jobs = build_jobs(args)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "gpu": args.gpu,
        "source_sha256": hashes,
        "source_fingerprint": fingerprint,
        "initial_checkpoint": str(args.resume_model),
        "initial_checkpoint_sha256": hashlib.sha256(args.resume_model.read_bytes()).hexdigest()
        if args.resume_model.is_file()
        else None,
        "optimizer_restored": False,
        "state": "planned",
        "jobs": jobs,
    }
    for record in jobs:
        run_id = uuid.uuid4().hex[:8]
        record.update(status="planned", wandb_run_id=run_id)
        record["command"] += [
            f"--EXPERIMENT_NAME={record['name']}",
            f"--DATA_OUTPUT_FILE={args.output / (record['name'] + '.jsonl')}",
            "--per-env-output",
            str(args.output / (record["name"] + ".per-env.jsonl")),
        ]
    save_manifest(manifest_path, manifest)
    if args.dry_run:
        return

    def status(state, **details):
        manifest.update(state=state, updated_utc=datetime.now(UTC).isoformat(), **details)
        save_manifest(manifest_path, manifest)
        print(json.dumps(dict(state=state, **details)), flush=True)

    import wandb

    app_url = wandb.Settings(base_url=wandb.Api(timeout=30).settings["base_url"]).app_url.rstrip(
        "/"
    )
    for record in jobs:
        record["wandb_url"] = (
            f"{app_url}/{args.entity}/{args.project}/runs/{record['wandb_run_id']}"
        )
    save_manifest(manifest_path, manifest)
    lease = wait_for_gpu(args.gpu, status)
    failed_parents = set()
    candidates = []
    try:
        for record in jobs:
            changed = [
                path
                for path, digest in hashes.items()
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
            ]
            if changed:
                status("source_changed", changed_files=changed)
                raise RuntimeError(
                    "Campaign sources changed after planning; refusing to mislabel runs"
                )
            if record.get("parent") in failed_parents:
                record["status"] = "skipped_training_failed"
                save_manifest(manifest_path, manifest)
                continue
            if record.get("adaptive_parent"):
                if candidates:
                    best = min(candidates, key=lambda x: x[0])
                    command = record["command"]
                    command[command.index("--initial-model") + 1] = best[1]
                    record.update(
                        selected_checkpoint=best[1], selection_score=best[0], selection_seed=4244
                    )
                else:
                    record["selected_checkpoint"] = str(args.resume_model)
            # Recheck external users before each child, retaining our lease.
            while True:
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
                if used <= 512:
                    break
                status("waiting_gpu", gpu=args.gpu, memory_mib=used)
                time.sleep(30)
            overrides = {
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0",
                "WANDB_MODE": "online",
                "WANDB_ENTITY": args.entity,
                "WANDB_RUN_GROUP": args.group,
                "WANDB_JOB_TYPE": record["kind"],
                "WANDB_TAGS": f"full-cls,all-heuristics,hp-tuning,{record['kind']}",
                "WANDB_RUN_ID": record["wandb_run_id"],
                "WANDB_RESUME": "never",
                "WANDB_DIR": str(args.output),
                "XLRON_SOURCE_FINGERPRINT": fingerprint,
            }
            record["environment"] = overrides
            started = time.monotonic()
            with (args.output / (record["name"] + ".log")).open("w") as log:
                child = subprocess.Popen(
                    record["command"],
                    env=dict(os.environ, **overrides),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                record.update(
                    status="running", pid=child.pid, started_utc=datetime.now(UTC).isoformat()
                )
                status("running", active_job=record["name"])
                code = child.wait()
            record.update(
                status="completed" if code == 0 else "failed",
                exit_code=code,
                wall_s=time.monotonic() - started,
            )
            if record["kind"] == "train" and code:
                failed_parents.add(record["name"])
            if record.get("selection") and code == 0:
                score = selection_score(args.output, record)
                model_path = record.get("selection_model")
                if model_path is None:
                    parent = next(j for j in jobs if j["name"] == record["parent"])
                    model_path = parent["model_path"]
                if score is not None and Path(model_path).is_file():
                    candidates.append((score, model_path))
            status("between_jobs", last_job=record["name"], exit_code=code)
        status(
            "completed_with_failures" if any(j["status"] == "failed" for j in jobs) else "completed"
        )
    finally:
        lease.close()


if __name__ == "__main__":
    main()
