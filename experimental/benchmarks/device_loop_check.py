"""Check whether XLA device-side while loops are safe for RMSA-GN on this JAX/GPU stack.

`--xla_gpu_enable_command_buffer=+WHILE` runs XLA while loops as CUDA-graph
conditional nodes (CONDITIONAL is already a JAX 0.12 default). On JAX 0.12.0.1 it
segfaults inside libcuda during RMSA-GN warmup when processes share an A100; see
docs/gn_model_performance.md. Re-run this after upgrading JAX, CUDA or the driver,
on an A100 that nothing else is using:

  CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \\
    uv run --no-sync python -m experimental.benchmarks.device_loop_check

Each round starts --concurrent processes of the departure-load sweep configuration
that crashed, together on the visible GPU (a lone process did not crash); a round
takes about ten minutes with the flag. Each process's exit code is reported.
"""

import argparse
import contextlib
import os
import subprocess
import sys
import time

WHILE_FLAG = "--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL"
# RMSA-GN, C+L+S NSFNET, full physical masks, Pallas kernels, immediate departures.
SWEEP = [
    "--env_type=rmsa_gn_model",
    "--topology_name=nsfnet_deeprmsa_directed",
    "--band_preference=C,L,S",
    "--inter_band_gap_ghz=25",
    "--link_resources=1668",
    "--slot_size=12.5",
    "--k=5",
    "--values_bw=100,200,400",
    "--load=4500",
    "--guardband=0",
    "--coherent",
    "--power_per_channel=-10",
    "--max_power_per_fibre=21",
    "--launch_power_type=fixed",
    "--continuous_operation",
    "--aggregate_slots=1",
    "--gn_mask_mode=all_slots",
    "--gn_mask_batch_size=64",
    "--gn_mask_env_batch_size=1",
    "--gn_kernel_backend=pallas",
    "--departure_policy=immediate",
    "--snr_margin=0.5",
    "--EVAL_HEURISTIC",
    "--path_heuristic=arbr_ff",
    "--SEED=4243",
    "--NUM_ENVS=4",
    "--TOTAL_TIMESTEPS=4096",
    "--STEPS_PER_INCREMENT=4096",
]


def run_round(device_loops, args, round_index):
    """Start args.concurrent processes together; return (name, exit code, seconds, log)."""
    flags = os.environ.get("XLA_FLAGS", "").split()
    flags = [f for f in flags if not f.startswith("--xla_gpu_enable_command_buffer=")]
    if device_loops:
        flags.append(WHILE_FLAG)
    env = dict(os.environ, XLA_FLAGS=" ".join(flags))
    command = [sys.executable, "-X", "faulthandler", "-m", "xlron.train.train", *SWEEP,
               f"--ENV_WARMUP_STEPS={args.warmup_steps}"]  # fmt: skip
    start = time.monotonic()
    running = []
    with contextlib.ExitStack() as logs:
        for process in range(args.concurrent):
            name = f"{'while' if device_loops else 'default'}_{round_index}_{process}"
            log = os.path.join(args.log_dir, f"{name}.log")
            out = logs.enter_context(open(log, "w"))
            popen = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT)
            running.append((name, popen, log))
        return [(name, popen.wait(), time.monotonic() - start, log) for name, popen, log in running]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--concurrent", type=int, default=3, help="Processes per round")
    parser.add_argument("--warmup-steps", type=int, default=15000)
    parser.add_argument("--log-dir", default="runs/device_loop_check")
    parser.add_argument(
        "--with-default", action="store_true", help="Also run rounds without WHILE (slow)"
    )
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    arms = [False, True] if args.with_default else [True]
    failures = 0
    for round_index in range(args.rounds):
        for device_loops in arms:
            for name, code, seconds, log in run_round(device_loops, args, round_index):
                # A negative code is the signal number (-11 is SIGSEGV).
                failures += device_loops and code != 0
                print(f"{name:12s} exit={code:4d} {seconds:6.0f}s  {log}", flush=True)
    print(f"device-side loops: {failures} of {args.rounds * args.concurrent} processes failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
