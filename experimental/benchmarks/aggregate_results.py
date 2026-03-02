#!/usr/bin/env python3
"""Aggregate individual JSONL benchmark files into a single CSV.

Reads all .jsonl files from the results directory, flattens the nested
JSON structure (config, timing, metrics), and writes one row per run.

Usage:
    python benchmarks/aggregate_results.py
    python benchmarks/aggregate_results.py --results_dir=benchmarks/results --output=benchmarks/results/benchmark_results.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def aggregate(
    results_dir: str = "benchmarks/results",
    output_csv: str = "benchmarks/results/benchmark_results.csv",
) -> pd.DataFrame:
    """Aggregate JSONL benchmark files into a single CSV.

    Args:
        results_dir: Directory containing per-run .jsonl files.
        output_csv: Path for the output CSV.

    Returns:
        DataFrame with one row per run.
    """
    rows = []
    for jsonl_file in sorted(Path(results_dir).glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                row = {
                    "file": jsonl_file.stem,
                    "run_type": data.get("run_type"),
                    "timestamp": data.get("timestamp"),
                }

                # Flatten config
                for k, v in data.get("config", {}).items():
                    row[f"config_{k}"] = v

                # Flatten timing
                for k, v in data.get("timing", {}).items():
                    row[f"timing_{k}"] = v

                # Flatten metrics (metric_name -> {mean, std, ...})
                for metric_name, stats in data.get("metrics", {}).items():
                    for stat_name, stat_val in stats.items():
                        row[f"metric_{metric_name}_{stat_name}"] = stat_val

                rows.append(row)

    if not rows:
        print(f"No JSONL files found in {results_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Derive device and sweep group from filename.
    # Filenames follow the pattern: [{device}_]{group}_{details}
    # e.g. "cpu_num_envs_rwa_ne64", "device_gpu_rmsa_ne16", "num_envs_rwa_ne1"
    import re

    _KNOWN_GROUPS = [
        "num_envs", "topology", "link_resources", "k_paths",
        "gn_bands", "device", "rwa_lr", "cross_env", "config_grid",
        "k_grid",
    ]
    _GROUP_PATTERN = "|".join(re.escape(g) for g in _KNOWN_GROUPS)

    def _parse_filename(name: str) -> tuple[str, str, bool]:
        """Return (device, group, valid) from a benchmark filename.

        Filenames follow: {server_device}_{group}_{details}
        The server_device prefix indicates which machine ran the benchmark.
        For the 'device' group, the JAX platform is encoded in the details.
        For the 'config_grid' group, the JAX platform is also in the details.

        The third return value ``valid`` is False when the server prefix
        doesn't match the claimed device.  For ``device`` and ``config_grid``
        groups, the target JAX platform is encoded in the filename details.
        Cross-server runs (e.g. ``cpu_config_grid_gpu_*`` or
        ``gpu_config_grid_cpu_*``) are unreliable because
        ``JAX_PLATFORMS=cpu`` on a GPU server doesn't produce comparable
        results to a dedicated CPU server, and a CPU server requesting GPU
        silently falls back to CPU.  We require server == device for these
        groups.
        """
        # Match device prefix then group name
        m = re.match(rf"^(?:(cpu|gpu)_)?({_GROUP_PATTERN})_", name)
        if m:
            server = m.group(1) or "gpu"
            group = m.group(2)
            device = server  # default: device = server
            valid = True
            # 'device' group: the run's JAX platform is in the next segment
            # e.g. "gpu_device_cpu_rmsa_ne1" or legacy "device_cpu_rmsa_ne1"
            if group == "device":
                rest = name[m.end():]
                m2 = re.match(r"^(cpu|gpu)_", rest)
                if m2:
                    device = m2.group(1)
                    # Only keep runs where server matches claimed device
                    if device != server:
                        valid = False
            # 'config_grid' / 'k_grid' groups: the run's JAX platform is in the next segment
            # e.g. "gpu_config_grid_cpu_lr128_ne64", "gpu_k_grid_cpu_k5_ne64"
            elif group in ("config_grid", "k_grid"):
                rest = name[m.end():]
                m2 = re.match(r"^(cpu|gpu)_", rest)
                if m2:
                    device = m2.group(1)
                    # Only keep runs where server matches claimed device
                    if device != server:
                        valid = False
            return device, group, valid
        return "unknown", "unknown", True

    parsed = df["file"].apply(_parse_filename)
    df["device"] = parsed.apply(lambda x: x[0])
    df["group"] = parsed.apply(lambda x: x[1])
    valid = parsed.apply(lambda x: x[2])
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        print(f"Dropping {n_invalid} rows where server != claimed device")
        df = df[valid].reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(
        f"Aggregated {len(rows)} runs from {len(list(Path(results_dir).glob('*.jsonl')))} files to {output_csv}"
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate XLRON benchmark results")
    parser.add_argument(
        "--results_dir", default="benchmarks/results", help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--output", default="benchmarks/results/benchmark_results.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    aggregate(results_dir=args.results_dir, output_csv=args.output)
