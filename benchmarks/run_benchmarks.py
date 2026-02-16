#!/usr/bin/env python3
"""XLRON Benchmarking Suite.

Runs structured sweeps across environment types, topologies, and parallelization
levels to measure simulation throughput (FPS) via heuristic evaluation.

Usage:
    # Run all sweep groups
    python benchmarks/run_benchmarks.py --output_dir=benchmarks/results

    # Run specific sweep groups
    python benchmarks/run_benchmarks.py --groups=num_envs,topology

    # Dry run (print commands without executing)
    python benchmarks/run_benchmarks.py --dry_run

    # Resume (skip runs whose output file already exists)
    python benchmarks/run_benchmarks.py --resume

    # CPU-only runs
    python benchmarks/run_benchmarks.py --groups=device --device=cpu
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# -- Constants ----------------------------------------------------------------

DIRECTED_TOPOLOGIES = [
    "5node_directed",
    "conus_directed",
    "cost239_deeprmsa_directed",
    "cost239_ptrnet_published_directed",
    "cost239_ptrnet_real_directed",
    "german17_directed",
    "jpn48_directed",
    "nsfnet_deeprmsa_directed",
    "usnet_gcnrnn_directed",
    "usnet_ptrnet_directed",
]

NUM_ENVS_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
LINK_RESOURCES_VALUES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
K_VALUES = [5, 10, 25, 50]

# Shared flags for all heuristic evaluation runs
HEURISTIC_BASE = [
    "--EVAL_HEURISTIC",
    "--path_heuristic=ksp_ff",
    "--ROLLOUT_LENGTH=128",
]

# Per-env-type base configurations
ENV_BASES = {
    "rwa": [
        "--env_type=rwa",
        "--continuous_operation",
        "--ENV_WARMUP_STEPS=0",
        "--load=250",
    ],
    "rmsa": [
        "--env_type=rmsa",
        "--values_bw=100",
        "--slot_size=12.5",
        "--continuous_operation",
        "--ENV_WARMUP_STEPS=0",
        "--load=250",
    ],
    "rsa_gn_model": [
        "--env_type=rsa_gn_model",
        "--slot_size=100",
        "--NUM_ENVS=1",
        "--continuous_operation",
        "--ENV_WARMUP_STEPS=0",
        "--load=250",
    ],
    "rmsa_gn_model": [
        "--env_type=rmsa_gn_model",
        "--slot_size=100",
        "--NUM_ENVS=1",
        "--continuous_operation",
        "--ENV_WARMUP_STEPS=0",
        "--load=250",
    ],
    "rwa_lightpath_reuse": [
        "--env_type=rwa_lightpath_reuse",
        "--incremental_loading",
        "--max_requests=10000",
        "--scale_factor=1.0",
    ],
}


def _timesteps_for_num_envs(num_envs: int) -> int:
    """Scale TOTAL_TIMESTEPS with NUM_ENVS to keep wall-clock reasonable."""
    if num_envs <= 16:
        return 100_000
    elif num_envs <= 256:
        return 500_000
    else:
        return 1_000_000


def _build_command(
    env_flags: list[str],
    extra_flags: list[str],
    output_file: str,
) -> list[str]:
    """Build the subprocess command list."""
    return [
        sys.executable,
        "-m",
        "xlron.train.train",
        *HEURISTIC_BASE,
        *env_flags,
        *extra_flags,
        f"--DATA_OUTPUT_FILE={output_file}",
    ]


# -- Sweep Group Generators --------------------------------------------------


def _group_num_envs() -> list[dict]:
    """Group 1: NUM_ENVS scaling (rwa/rmsa on NSFNET)."""
    runs = []
    for env_type in ["rwa", "rmsa"]:
        for num_envs in NUM_ENVS_VALUES:
            ts = _timesteps_for_num_envs(num_envs)
            runs.append(
                {
                    "env_flags": ENV_BASES[env_type],
                    "extra_flags": [
                        "--topology_name=nsfnet_deeprmsa_directed",
                        "--link_resources=100",
                        "--k=5",
                        f"--NUM_ENVS={num_envs}",
                        f"--TOTAL_TIMESTEPS={ts}",
                    ],
                    "label": f"num_envs_{env_type}_ne{num_envs}",
                }
            )
    return runs


def _group_topology() -> list[dict]:
    """Group 2: Topology scaling (rwa/rmsa, all directed topos)."""
    runs = []
    for env_type in ["rwa", "rmsa"]:
        for topo in DIRECTED_TOPOLOGIES:
            runs.append(
                {
                    "env_flags": ENV_BASES[env_type],
                    "extra_flags": [
                        f"--topology_name={topo}",
                        "--link_resources=100",
                        "--k=5",
                        "--NUM_ENVS=64",
                        "--TOTAL_TIMESTEPS=500000",
                    ],
                    "label": f"topology_{env_type}_{topo}",
                }
            )
    return runs


def _group_link_resources() -> list[dict]:
    """Group 3: Link resources scaling (rwa/rmsa on NSFNET)."""
    runs = []
    for env_type in ["rwa", "rmsa"]:
        for lr in LINK_RESOURCES_VALUES:
            runs.append(
                {
                    "env_flags": ENV_BASES[env_type],
                    "extra_flags": [
                        "--topology_name=nsfnet_deeprmsa_directed",
                        f"--link_resources={lr}",
                        "--k=5",
                        "--NUM_ENVS=64",
                        "--TOTAL_TIMESTEPS=500000",
                    ],
                    "label": f"link_resources_{env_type}_lr{lr}",
                }
            )
    return runs


def _group_k_paths() -> list[dict]:
    """Group 4: K-paths scaling (rwa/rmsa on NSFNET)."""
    runs = []
    for env_type in ["rwa", "rmsa"]:
        for k in K_VALUES:
            runs.append(
                {
                    "env_flags": ENV_BASES[env_type],
                    "extra_flags": [
                        "--topology_name=nsfnet_deeprmsa_directed",
                        "--link_resources=100",
                        f"--k={k}",
                        "--NUM_ENVS=64",
                        "--TOTAL_TIMESTEPS=500000",
                    ],
                    "label": f"k_paths_{env_type}_k{k}",
                }
            )
    return runs


def _group_gn_bands() -> list[dict]:
    """Group 5: GN model band scaling (C / C+L / C+L+S)."""
    runs = []
    for env_type in ["rsa_gn_model", "rmsa_gn_model"]:
        for bands in ["C", "C,L", "C,L,S"]:
            band_label = bands.replace(",", "")
            runs.append(
                {
                    "env_flags": ENV_BASES[env_type],
                    "extra_flags": [
                        "--topology_name=nsfnet_deeprmsa_directed",
                        "--k=5",
                        f"--band_preference={bands}",
                        "--TOTAL_TIMESTEPS=10000",
                    ],
                    "label": f"gn_bands_{env_type}_{band_label}",
                }
            )
    return runs


def _group_device() -> list[dict]:
    """Group 6: CPU vs GPU comparison."""
    runs = []
    for device in ["cpu", "gpu"]:
        for env_type in ["rwa", "rmsa"]:
            for num_envs in [1, 16, 64, 256, 1024]:
                runs.append(
                    {
                        "env_flags": ENV_BASES[env_type],
                        "extra_flags": [
                            "--topology_name=nsfnet_deeprmsa_directed",
                            "--link_resources=100",
                            "--k=5",
                            f"--NUM_ENVS={num_envs}",
                            "--TOTAL_TIMESTEPS=100000",
                        ],
                        "label": f"device_{device}_{env_type}_ne{num_envs}",
                        "device": device,
                    }
                )
    return runs


def _group_rwa_lr() -> list[dict]:
    """Group 7: rwa_lightpath_reuse across topologies and k values."""
    runs = []
    for topo in DIRECTED_TOPOLOGIES:
        for k in K_VALUES:
            runs.append(
                {
                    "env_flags": ENV_BASES["rwa_lightpath_reuse"],
                    "extra_flags": [
                        f"--topology_name={topo}",
                        "--link_resources=100",
                        f"--k={k}",
                        "--NUM_ENVS=1",
                        "--TOTAL_TIMESTEPS=10000",
                    ],
                    "label": f"rwa_lr_{topo}_k{k}",
                }
            )
    return runs


def _group_cross_env() -> list[dict]:
    """Group 8: Cross-env-type comparison on same config."""
    runs = []
    for env_type in ["rwa", "rmsa", "rsa_gn_model", "rmsa_gn_model"]:
        ts = 10000 if "gn_model" in env_type else 100000
        runs.append(
            {
                "env_flags": ENV_BASES[env_type],
                "extra_flags": [
                    "--topology_name=nsfnet_deeprmsa_directed",
                    "--link_resources=100",
                    "--k=5",
                    "--NUM_ENVS=1",
                    f"--TOTAL_TIMESTEPS={ts}",
                ],
                "label": f"cross_env_{env_type}_ne1",
            }
        )
    # Also rwa/rmsa at NUM_ENVS=64
    for env_type in ["rwa", "rmsa"]:
        runs.append(
            {
                "env_flags": ENV_BASES[env_type],
                "extra_flags": [
                    "--topology_name=nsfnet_deeprmsa_directed",
                    "--link_resources=100",
                    "--k=5",
                    "--NUM_ENVS=64",
                    "--TOTAL_TIMESTEPS=500000",
                ],
                "label": f"cross_env_{env_type}_ne64",
            }
        )
    return runs


SWEEP_GROUPS = {
    "num_envs": _group_num_envs,
    "topology": _group_topology,
    "link_resources": _group_link_resources,
    "k_paths": _group_k_paths,
    "gn_bands": _group_gn_bands,
    "device": _group_device,
    "rwa_lr": _group_rwa_lr,
    "cross_env": _group_cross_env,
}


# -- Execution ----------------------------------------------------------------


def run_benchmark(
    run_config: dict,
    output_dir: Path,
    dry_run: bool = False,
    timeout: int = 1800,
) -> bool:
    """Execute a single benchmark run.

    Returns True on success, False on failure/timeout.
    """
    label = run_config["label"]
    device = run_config.get("device", "gpu")
    output_file = str(output_dir / f"{label}.jsonl")

    cmd = _build_command(run_config["env_flags"], run_config["extra_flags"], output_file)

    env = os.environ.copy()
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"

    cmd_str = " ".join(cmd)
    if device == "cpu":
        cmd_str = f"JAX_PLATFORMS=cpu {cmd_str}"

    print(f"\n{'=' * 70}")
    print(f"RUN: {label}")
    print(f"CMD: {cmd_str}")
    print(f"{'=' * 70}")

    if dry_run:
        return True

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"FAILED: {label}")
            print(f"STDERR (last 500 chars): {result.stderr[-500:]}")
            return False
        print(f"DONE: {label}")
        return True
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({timeout}s): {label}")
        return False


def main():
    parser = argparse.ArgumentParser(description="XLRON Benchmark Suite")
    parser.add_argument(
        "--output_dir", default="benchmarks/results", help="Directory for JSONL output files"
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated sweep groups to run (default: all). "
        f"Available: {', '.join(SWEEP_GROUPS)}",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "gpu"],
        help="Force all non-'device' group runs onto this device and prefix "
        "output filenames with the device name. If not set, defaults to "
        "GPU. The 'device' group always runs both CPU and GPU regardless.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--resume", action="store_true", help="Skip runs whose output file already exists"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Per-run timeout in seconds (default: 1800)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select groups
    if args.groups:
        selected = [g.strip() for g in args.groups.split(",")]
        unknown = [g for g in selected if g not in SWEEP_GROUPS]
        if unknown:
            print(f"Unknown groups: {unknown}. Available: {list(SWEEP_GROUPS)}")
            sys.exit(1)
    else:
        selected = list(SWEEP_GROUPS)

    # Build all runs, applying --device override
    all_runs = []
    for group_name in selected:
        runs = SWEEP_GROUPS[group_name]()
        for r in runs:
            r["group"] = group_name
            # The 'device' group already has per-run device settings; skip override
            if group_name != "device" and args.device is not None:
                r["device"] = args.device
                # Prefix label so CPU/GPU results don't collide
                if not r["label"].startswith(f"{args.device}_"):
                    r["label"] = f"{args.device}_{r['label']}"
        all_runs.extend(runs)

    print(f"Total benchmark runs: {len(all_runs)}")
    print(f"Groups: {', '.join(selected)}")

    completed = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    current_group = None
    for run_config in all_runs:
        group = run_config["group"]
        if group != current_group:
            current_group = group
            group_runs = [r for r in all_runs if r["group"] == group]
            print(f"\n{'#' * 70}")
            print(f"SWEEP GROUP: {group} ({len(group_runs)} runs)")
            print(f"{'#' * 70}")

        label = run_config["label"]
        output_file = output_dir / f"{label}.jsonl"

        if args.resume and output_file.exists():
            print(f"SKIP (exists): {label}")
            skipped += 1
            completed += 1
            continue

        success = run_benchmark(run_config, output_dir, args.dry_run, args.timeout)
        if success:
            completed += 1
        else:
            failed += 1

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUITE COMPLETE")
    print(f"Completed: {completed}/{len(all_runs)} (skipped: {skipped}), Failed: {failed}")
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
