"""Run bounded, reproducible kernel comparisons on an explicitly selected GPU.

Invoke with uv run --no-sync. Each child has a log, exact command, exit status,
wall time and separate synchronized environment or real-PPO measurements.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["env", "train"], required=True)
    parser.add_argument("--backends", nargs="+", required=True)
    parser.add_argument("--gpu", choices=["2", "3"], required=True)
    parser.add_argument("--slots", type=int, default=1376)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--compiler-profile", choices=["default", "library"], default="default")
    parser.add_argument(
        "--path-snr-backend", choices=["jax", "pallas", "pallas_fused", "pallas_precomputed"], default="jax"
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    child_env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=args.gpu, XLA_PYTHON_CLIENT_PREALLOCATE="false"
    )
    if args.compiler_profile == "library":
        child_env["XLA_FLAGS"] = (
            child_env.get("XLA_FLAGS", "")
            + " --xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
        )
    for backend in args.backends:
        stem = Path(f"{args.prefix}-{backend}")
        stem.parent.mkdir(parents=True, exist_ok=True)
        if stem.with_suffix(".meta.json").exists():
            raise FileExistsError(f"Refusing to overwrite recorded run {stem}")
        command = [
            sys.executable,
            "-m",
            f"xlron.kernels.gn_{'environment' if args.mode == 'env' else 'train'}_benchmark",
            "--xpm-backend",
            backend,
            "--path-snr-backend",
            args.path_snr_backend,
        ]
        if args.mode == "env":
            command += [
                "--compiler-profile",
                args.compiler_profile,
                "--slots",
                str(args.slots),
                "--envs",
                str(args.envs),
                "--warmup-steps",
                str(args.warmup),
                "--steps",
                str(args.steps),
                "--independent-streams",
                "--kernels",
                "link",
                "mask",
                "rollout",
                "--iterations",
                str(args.iterations),
                "--output",
                f"{stem}.json",
                "--snapshot",
                f"{stem}.npz",
            ]
            if args.reference:
                command += ["--reference", str(args.reference)]
        else:
            command += [
                "--env_type=rmsa_gn_model",
                "--topology_name=nsfnet_deeprmsa_directed",
                f"--link_resources={args.slots}",
                "--k=5",
                "--values_bw=100,200,400",
                "--slot_size=12.5",
                "--guardband=0",
                "--power_per_channel=-10",
                "--max_power_per_fibre=21",
                "--launch_power_type=fixed",
                "--continuous_operation",
                f"--NUM_ENVS={args.envs}",
                f"--ROLLOUT_LENGTH={args.steps}",
                f"--TOTAL_TIMESTEPS={args.timesteps}",
                f"--STEPS_PER_INCREMENT={args.timesteps // 2}",
                f"--ENV_WARMUP_STEPS={args.warmup}",
                "--NUM_MINIBATCHES=4",
                "--UPDATE_EPOCHS=4",
                "--PREALLOCATE_MEM=false",
                f"--VISIBLE_DEVICES={args.gpu}",
                f"--DATA_OUTPUT_FILE={stem}.jsonl",
            ]
            if args.transformer:
                command.append("--USE_TRANSFORMER")
        print(f"Starting {stem}", flush=True)
        start = time.perf_counter()
        with Path(f"{stem}.log").open("w") as log:
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                status = {"exit_code": result.returncode}
            except subprocess.TimeoutExpired:
                status = {"exit_code": None, "timeout_s": args.timeout}
        metadata = {
            "command": command,
            "gpu": args.gpu,
            "backend": backend,
            "compiler_profile": args.compiler_profile,
            "xla_flags": child_env.get("XLA_FLAGS", ""),
            "wall_s": time.perf_counter() - start,
            **status,
        }
        Path(f"{stem}.meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
        if status["exit_code"] != 0:
            print(Path(f"{stem}.log").read_text()[-5000:], flush=True)


if __name__ == "__main__":
    main()
