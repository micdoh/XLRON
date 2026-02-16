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
        "gn_bands", "device", "rwa_lr", "cross_env",
    ]
    _GROUP_PATTERN = "|".join(re.escape(g) for g in _KNOWN_GROUPS)

    def _parse_filename(name: str) -> tuple[str, str]:
        """Return (device, group) from a benchmark filename."""
        # Match optional device prefix, then the group name
        m = re.match(rf"^(?:(cpu|gpu)_)?({_GROUP_PATTERN})_", name)
        if m:
            device = m.group(1) or "gpu"
            group = m.group(2)
            # The 'device' group encodes device in the next segment
            if group == "device":
                m2 = re.match(r"^device_(cpu|gpu)_", name)
                if m2:
                    device = m2.group(1)
            return device, group
        return "unknown", "unknown"

    parsed = df["file"].apply(_parse_filename)
    df["device"] = parsed.apply(lambda x: x[0])
    df["group"] = parsed.apply(lambda x: x[1])
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
