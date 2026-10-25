"""Native correctness, matched 16-state rollouts and timed-region CUDA profiling."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from experimental.benchmarks.gn_cls_tuning_campaign import wait_for_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--state-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunking", action="store_true")
    parser.add_argument("--predicate-packing", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=str(Path.cwd()),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0",
    )
    if args.gpu_memory_fraction is not None:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.gpu_memory_fraction)
    lease = wait_for_gpu(
        args.gpu,
        lambda state, **kw: print(json.dumps(dict(state=state, **kw)), flush=True),
        args.gpu_memory_fraction,
    )
    (args.output / "resource-policy.json").write_text(
        json.dumps(
            dict(
                gpu=args.gpu,
                jax_memory_fraction=args.gpu_memory_fraction,
                shared_allocations=args.gpu_memory_fraction is not None,
                note="Shared VRAM permitted by user; require idle compute and at least twice our memory budget free before launch. Compare paired runs on this GPU; do not treat these as exclusive-GPU timings.",
            ),
            indent=2,
        )
        + "\n"
    )
    records = []

    def run(name, command, timeout=600):
        print(json.dumps(dict(start=name)), flush=True)
        with (args.output / (name + ".log")).open("w") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            )
        records.append(dict(name=name, exit_code=result.returncode, command=command))
        (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(dict(name=name, exit_code=result.returncode)), flush=True)
        if result.returncode:
            raise RuntimeError(name + " failed")

    try:
        hashes = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*Path("xlron").rglob("*.py"), *Path("tests").rglob("*.py")])
        }
        (args.output / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        run(
            "native-tests",
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "pytest",
                "-q",
                "xlron/environments/env_funcs_test.py::TestMaskSlotsRmsaGnModel",
                        "xlron/environments/env_funcs_test.py::TestGetSnrLinkArray",
                "xlron/models/modulation_actor_test.py",
                "xlron/kernels/gn_predicate_pack_test.py",
                "xlron/kernels/gn_path_snr_test.py::test_packed_old_path_still_rejects_degradation_on_another_shared_link",
            ],
        )
        common = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "xlron.kernels.gn_environment_benchmark",
            "--xpm-backend",
            "pallas_grouped",
            "--step-xpm-backend",
            "pallas_grouped",
            "--candidate-snr-backend",
            "pallas",
            "--path-snr-backend",
            "pallas_precomputed",
            "--compiler-profile",
            "library",
            "--xpm-tile",
            "16",
            "32",
            "4",
            "--slots",
            "1668",
            "--envs",
            "16",
            "--k",
            "5",
            "--load",
            "5000",
            "--coherent",
            "--band-preference",
            "C,L,S",
            "--gn-mask-mode",
            "all_slots",
            "--state-input",
            str(args.state_input),
            "--independent-streams",
        ]
        reference = args.output / "unpruned16.npz"
        variants = [
            ("unpruned16", ["--no-full-format-pruning"]),
            (
                "screened16",
                [
                    "--new-lightpath-prefilter",
                    "--gn-mask-batch-size",
                    "64",
                    "--reference",
                    str(reference),
                ],
            ),
        ]
        variants.append(
            (
                "screened-integrated1",
                [
                    "--new-lightpath-prefilter",
                    "--gn-mask-batch-size",
                    "64",
                    "--gn-mask-env-batch-size",
                    "1",
                    "--reference",
                    str(reference),
                ],
            )
        )
        if args.chunking:
            variants += [
                (
                    f"screened-chunk{size}",
                    [
                        "--new-lightpath-prefilter",
                        "--gn-mask-batch-size",
                        "64",
                        "--env-batch-size",
                        str(size),
                        "--reference",
                        str(reference),
                    ],
                )
                for size in (8, 4, 2, 1)
            ]
        if args.predicate_packing:
            variants.append(
                (
                    "screened-packed",
                    [
                        "--new-lightpath-prefilter",
                        "--gn-mask-batch-size",
                        "64",
                        "--gn-mask-env-batch-size",
                        "1",
                        "--pack-old-predicates",
                        "--reference",
                        str(reference),
                    ],
                )
            )
        for name, flags in variants:
            run(
                name,
                common
                + flags
                + [
                    "--kernels",
                    "mask",
                    "rollout",
                    "--steps",
                    "8",
                    "--iterations",
                    "10",
                    "--output",
                    str(args.output / (name + ".json")),
                    "--snapshot",
                    str(args.output / (name + ".npz")),
                ],
            )
        nsys = "/apps/cuda/cuda-12.9/bin/nsys"
        trace = args.output / "screened16-profile"
        best_name, best_flags = min(
            variants[1:],
            key=lambda item: json.loads((args.output / (item[0] + ".json")).read_text())["metrics"][
                "mask"
            ]["median_ms"],
        )
        profile_flags = best_flags[: best_flags.index("--reference")]
        run(
            "nsys",
            [
                nsys,
                "profile",
                "--trace=cuda,nvtx",
                "--cuda-graph-trace=node",
                "--sample=none",
                "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--force-overwrite=true",
                "--output",
                str(trace),
            ]
            + common
            + profile_flags
            + [
                "--kernels",
                "mask",
                "--iterations",
                "10",
                "--profile",
                "--output",
                str(args.output / "profile-timing.json"),
            ],
        )
        run(
            "nsys-stats",
            [
                nsys,
                "stats",
                "--report",
                "cuda_gpu_kern_sum,cuda_api_sum",
                "--format",
                "csv",
                "--output",
                str(args.output / "cuda"),
                str(trace) + ".nsys-rep",
            ],
        )
    finally:
        lease.close()


if __name__ == "__main__":
    main()
