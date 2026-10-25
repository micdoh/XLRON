"""Native tests and short full-shape PPO updates before long observation trials."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from experimental.benchmarks.gn_cls_observation_campaign import build_jobs
from experimental.benchmarks.gn_cls_tuning_campaign import replace_flags, wait_for_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.seed = 42
    args.mask_mode = "all_slots"
    args.resume_model = Path("/tmp/unused-provenance")
    args.train_timesteps = 8192
    args.eval_timesteps = 8192
    args.project = "XLRON-GN-CLS"
    args.load = 5000
    args.warmup = 32
    args.pool_before_projection = False
    lease = wait_for_gpu(
        args.gpu, lambda state, **kw: print(json.dumps(dict(state=state, **kw)), flush=True)
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(Path.cwd()),
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_DEFAULT_MATMUL_PRECISION="highest",
        XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0",
        WANDB_MODE="online",
        WANDB_ENTITY="micdoh",
        WANDB_RUN_GROUP="gn-cls-20261025-fullslot-preflight",
        WANDB_JOB_TYPE="preflight",
        WANDB_TAGS="full-cls,full-slots,short-preflight",
    )
    records = []

    def execute(command, path, timeout):
        with path.open("w") as log:
            child = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                return child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Include uv's Python child in timeout cleanup.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                return 124

    try:
        tests = [
            sys.executable,
            "-m",
            "pytest",
            "xlron/models/spectrum_transformer_test.py",
            "xlron/models/modulation_actor_test.py",
            "xlron/environments/env_funcs_test.py::TestMaskSlotsRmsaGnModel",
                        "xlron/environments/env_funcs_test.py::TestGetSnrLinkArray",
            "-q",
            "-x",
            "--tb=short",
        ]
        code = execute(tests, args.output / "native-tests.log", 420)
        print(json.dumps(dict(native_test_exit=code)), flush=True)
        if code:
            raise RuntimeError("Native tests failed")
        for job in build_jobs(args):
            if job["kind"] != "train" or not any(s in job["name"] for s in ("conv128", "dense256")):
                continue
            name = "preflight-" + job["name"]
            command = replace_flags(
                job["command"],
                "--STEPS_PER_INCREMENT=4096",
                "--EVAL_DURING_TRAINING=false",
                f"--EXPERIMENT_NAME={name}",
                f"--DATA_OUTPUT_FILE={args.output / (name + '.jsonl')}",
            )
            code = execute(command, args.output / (name + ".log"), 900)
            records.append(dict(name=name, exit_code=code, command=command))
            (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(dict(name=name, exit_code=code)), flush=True)
            if code:
                raise RuntimeError("Full-shape PPO preflight failed")
    finally:
        lease.close()


if __name__ == "__main__":
    main()
