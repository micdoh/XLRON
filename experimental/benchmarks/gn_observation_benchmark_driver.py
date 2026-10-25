"""Reserve a free experiment GPU and compare masks and spectrum encoders."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from experimental.benchmarks.gn_cls_tuning_campaign import wait_for_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--state-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoders-only", action="store_true")
    parser.add_argument("--full-masks-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lease = wait_for_gpu(
        args.gpu, lambda state, **kw: print(json.dumps(dict(state=state, **kw)), flush=True)
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0",
    )
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted([*Path("xlron").rglob("*.py"), *Path("tests").rglob("*.py")])
    }
    (args.output / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
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
        "4",
        "--k",
        "5",
        "--load",
        "5000",
        "--coherent",
        "--band-preference",
        "C,L,S",
        "--state-input",
        str(args.state_input),
        "--kernels",
        "mask",
        "rollout",
        "--steps",
        "1",
        "--iterations",
        "2",
    ]
    records = []
    try:
        variants = [(mode, []) for mode in ("selected_slot", "all_slots", "ff_lf")]
        if args.full_masks_only:
            variants = [
                ("full-unpruned8", ["--gn-mask-mode", "all_slots", "--no-full-format-pruning"])
            ]
            variants += [
                (
                    f"full-pruned{batch}",
                    ["--gn-mask-mode", "all_slots", "--gn-mask-batch-size", str(batch)],
                )
                for batch in (8, 16, 32, 64)
            ]
            variants += [
                (
                    f"full-screened{batch}",
                    [
                        "--gn-mask-mode",
                        "all_slots",
                        "--gn-mask-batch-size",
                        str(batch),
                        "--new-lightpath-prefilter",
                    ],
                )
                for batch in (32, 64)
            ]
        for mode, overrides in [] if args.encoders_only else variants:
            reference = args.output / "full-unpruned8.npz"
            command = common + [
                "--output",
                str(args.output / (mode + ".json")),
            ]
            if args.full_masks_only:
                command += overrides + ["--snapshot", str(args.output / (mode + ".npz"))]
                if mode != "full-unpruned8":
                    command += ["--reference", str(reference)]
            else:
                command += ["--gn-mask-mode", mode]
            print(json.dumps(dict(start=mode, command=command)), flush=True)
            started = time.monotonic()
            with (args.output / (mode + ".log")).open("w") as log:
                try:
                    result = subprocess.run(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=450
                    )
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            records.append(dict(mode=mode, exit_code=code, wall_s=time.monotonic() - started))
            (args.output / "driver_results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(records[-1]), flush=True)
            if args.full_masks_only and code:
                raise RuntimeError("Full physical mask benchmark or exact comparison failed")
        if args.full_masks_only:
            return
        command = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-m",
            "experimental.benchmarks.gn_spectrum_encoder_benchmark",
            "--state-input",
            str(args.state_input),
            "--output",
            str(args.output / "encoders.json"),
            "--envs",
            "16",
            "--iterations",
            "10",
        ]
        with (args.output / "encoders.log").open("w") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600
            )
        print(json.dumps(dict(encoder_exit=result.returncode)), flush=True)
    finally:
        lease.close()


if __name__ == "__main__":
    main()
