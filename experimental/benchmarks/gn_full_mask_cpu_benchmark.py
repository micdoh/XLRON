"""Bounded CPU feasibility check: exact full C+L+S mask, no GPU kernels."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--state-input", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))[-8:]
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        JAX_DEFAULT_MATMUL_PRECISION="highest",
        PYTHONPATH=str(Path.cwd()),
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false",
    )
    command = [
        sys.executable,
        "-m",
        "xlron.kernels.gn_environment_benchmark",
        "--xpm-backend",
        "jax",
        "--new-lightpath-prefilter",
        "--slots",
        "1668",
        "--envs",
        "1",
        "--k",
        "5",
        "--load",
        "5000",
        "--coherent",
        "--band-preference",
        "C,L,S",
        "--gn-mask-mode",
        "all_slots",
        "--gn-mask-batch-size",
        "1",
        "--gn-mask-env-batch-size",
        "1",
        "--state-input",
        str(args.state_input),
        "--kernels",
        "mask",
        "--iterations",
        "1",
        "--warm-executions",
        "0",
        "--output",
        str(args.output / "cpu.json"),
    ]
    with (args.output / "benchmark.log").open("w") as log:
        child = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            code = 124
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    (args.output / "status.json").write_text(
        json.dumps(
            dict(
                exit_code=code,
                cpu=model,
                affinity=cpus,
                timeout_s=args.timeout,
                command=command,
                note="Single unwarmed CPU execution; bounded total wall time includes compilation",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(exit_code=code, cpu=model)), flush=True)


if __name__ == "__main__":
    main()
