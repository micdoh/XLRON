"""Paired full-mask tuning on frozen physical states and a bounded GPU pool."""

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
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-input", type=Path, required=True)
    parser.add_argument("--reuse-power", action="store_true")
    parser.add_argument("--native-tests", action="store_true")
    parser.add_argument("--unaffected", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compact-updates", action="store_true")
    parser.add_argument("--screen-tuning", action="store_true")
    parser.add_argument("--group-widths", action="store_true")
    parser.add_argument("--endpoint-kernel", action="store_true")
    parser.add_argument("--screen-sizes", type=int, nargs="+", default=[32, 128, 256, 512])
    parser.add_argument("--confirm-screen-size", type=int)
    parser.add_argument(
        "--test-selection",
        default="not test_reduced_candidate_evidence_keeps_all_original_predicates",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(args.gpu_memory_fraction),
        PYTHONPATH=str(Path.cwd()),
    )
    lease = wait_for_gpu(
        args.gpu,
        lambda state, **kw: print(json.dumps(dict(state=state, **kw)), flush=True),
        args.gpu_memory_fraction,
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
        "--gn-mask-batch-size",
        "64",
        "--gn-mask-env-batch-size",
        "1",
        "--state-input",
        str(args.state_input),
        "--independent-streams",
        "--new-lightpath-prefilter",
        "--pack-old-predicates",
        "--kernels",
        "mask",
        "rollout",
        "--steps",
        "8",
        "--iterations",
        "10",
    ]
    reference = args.output / "control.npz"
    variants = [("control", [])]
    if args.group_widths or args.endpoint_kernel:
        common += [
            "--reuse-screened-power",
            "--narrow-new-snr",
            "--reuse-unaffected-snr",
            "--screen-batch-size",
            "512",
        ]
        variants += (
            [
                ("selected-xpm", ["--endpoint-xpm-backend", "pallas"]),
                ("fused-endpoint", ["--fused-endpoint-snr"]),
                ("persistent-8", ["--generic-row-programs", "8"]),
                ("persistent-16", ["--generic-row-programs", "16"]),
                ("fused-persistent", ["--fused-endpoint-snr", "--generic-row-programs", "16"]),
            ]
            if args.endpoint_kernel
            else [("grouped-widths", ["--group-widths"])]
        ) + [("control-recheck", [])]
    elif args.confirm_screen_size:
        optimized = [
            "--reuse-screened-power",
            "--narrow-new-snr",
            "--reuse-unaffected-snr",
            "--screen-batch-size",
            str(args.confirm_screen_size),
        ]
        variants += [
            ("optimized", optimized),
            ("optimized-compact", optimized + ["--compact-snr-updates"]),
            ("control-recheck", []),
        ]
    elif args.screen_tuning:
        variants += [
            (f"screen-{size}", ["--screen-batch-size", str(size)]) for size in args.screen_sizes
        ]
    elif args.reuse_power:
        variants.append(("reuse-power", ["--reuse-screened-power"]))
        variants.append(("narrow-new", ["--narrow-new-snr"]))
        variants.append(("combined", ["--reuse-screened-power", "--narrow-new-snr"]))
        if args.unaffected:
            variants.append(("unaffected", ["--reuse-unaffected-snr"]))
            variants.append(
                (
                    "combined-unaffected",
                    ["--reuse-screened-power", "--narrow-new-snr", "--reuse-unaffected-snr"],
                )
            )
        if args.compact_updates:
            variants.append(
                ("compact-updates", ["--reuse-screened-power", "--compact-snr-updates"])
            )
            variants.append(
                (
                    "compact-unaffected",
                    ["--reuse-screened-power", "--compact-snr-updates", "--reuse-unaffected-snr"],
                )
            )
    else:
        variants += [
            (f"path-{slots}-{warps}", ["--path-tile", str(slots), str(warps)])
            for slots, warps in ((64, 2), (64, 4), (128, 2), (256, 4), (256, 8))
        ]
    records = []
    try:
        (args.output / "source_sha256.json").write_text(
            json.dumps(
                {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted([*Path("xlron").rglob("*.py"), *Path("tests").rglob("*.py")])
                },
                indent=2,
            )
            + "\n"
        )
        (args.output / "resource-policy.json").write_text(
            json.dumps(
                {
                    "gpu": args.gpu,
                    "memory_fraction": args.gpu_memory_fraction,
                    "shared_capacity": True,
                    "preallocation": False,
                },
                indent=2,
            )
            + "\n"
        )
        if args.native_tests:
            with (args.output / "native-tests.log").open("w") as output:
                result = subprocess.run(
                    [
                        "uv",
                        "run",
                        "--no-sync",
                        "pytest",
                        "-q",
                        "xlron/environments/env_funcs_test.py::TestMaskSlotsRmsaGnModel",
                        "xlron/environments/env_funcs_test.py::TestGetSnrLinkArray",
                        "xlron/kernels/gn_path_snr_test.py",
                        *(
                            [
                                "xlron/kernels/gn_xpm_legacy_test.py",
                                "xlron/kernels/gn_endpoint_snr_test.py",
                            ]
                            if args.endpoint_kernel
                            else []
                        ),
                        "-k",
                        args.test_selection,
                    ],
                    env=env,
                    check=False,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
            print(json.dumps({"name": "native-tests", "exit_code": result.returncode}), flush=True)
            if result.returncode:
                raise RuntimeError("native tests failed")
        for name, flags in variants:
            command = (
                common
                + flags
                + [
                    "--output",
                    str(args.output / (name + ".json")),
                    "--snapshot",
                    str(args.output / (name + ".npz")),
                ]
            )
            if name != "control":
                command += ["--reference", str(reference)]
            print(json.dumps({"start": name}), flush=True)
            with (args.output / (name + ".log")).open("w") as output:
                result = subprocess.run(
                    command,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
            records.append({"name": name, "exit_code": result.returncode, "command": command})
            (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps({"name": name, "exit_code": result.returncode}), flush=True)
            if result.returncode:
                raise RuntimeError(name + " failed")
            if name != "control":
                accuracy = json.loads((args.output / (name + ".json")).read_text()).get("accuracy")
                if not accuracy or not all(item["exact"] for item in accuracy.values()):
                    raise AssertionError(name + " changed recorded physical outputs")
        if args.profile:
            best, flags = min(
                variants,
                key=lambda item: json.loads((args.output / (item[0] + ".json")).read_text())[
                    "metrics"
                ]["mask"]["median_ms"],
            )
            nsys = "/apps/cuda/cuda-12.9/bin/nsys"
            trace = args.output / "profile"
            command = [
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
            command += (
                common
                + flags
                + [
                    "--kernels",
                    "mask",
                    "--profile",
                    "--output",
                    str(args.output / "profile-timing.json"),
                ]
            )
            with (args.output / "profile.log").open("w") as output:
                result = subprocess.run(
                    command,
                    env=dict(env, XLRON_DUMP_HLO=str(args.output / "profile.hlo")),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError("profile failed")
            subprocess.run(
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
                env=env,
                check=True,
            )
            (args.output / "profile-variant.json").write_text(json.dumps({"variant": best}) + "\n")
    finally:
        lease.close()


if __name__ == "__main__":
    main()
