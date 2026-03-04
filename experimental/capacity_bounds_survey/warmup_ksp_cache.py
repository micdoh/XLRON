"""Generate KSP cache files for all K-sensitivity runs.

Runs the same topology+K combinations as 05_run_k_sensitivity.py but with
minimal resources (link_resources=2, 10 timesteps) so each run compiles
and exits quickly, leaving behind the cached KSP arrays.
"""

import sys
import subprocess
from pathlib import Path

from config import (
    K_SENSITIVITY_MIN_NODES,
    K_SENSITIVITY_VALUES,
    SHARED_FLAGS,
    get_topology_list,
    load_heuristic_selection,
    load_load_ranges,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    topologies = get_topology_list()
    ranges = load_load_ranges()
    heur_selection = load_heuristic_selection()

    large_topos = [t for t in topologies if t["num_nodes"] > K_SENSITIVITY_MIN_NODES]
    total = len(large_topos) * len(K_SENSITIVITY_VALUES)
    print(f"Warming KSP cache: {len(large_topos)} topologies x {len(K_SENSITIVITY_VALUES)} K values = {total} runs")

    done = 0
    failed = 0
    for topo in large_topos:
        name = topo["topology_name"]
        best_heuristic = heur_selection.get(name, "ksp_ff")

        for k_val in K_SENSITIVITY_VALUES:
            done += 1
            print(f"[{done}/{total}] {name}  K={k_val}", end=" ", flush=True)

            # Minimal flags: same topology/env_type/modulations but tiny resources
            flags = {
                "env_type": SHARED_FLAGS["env_type"],
                "topology_name": name,
                "link_resources": 2,
                "slot_size": SHARED_FLAGS["slot_size"],
                "guardband": SHARED_FLAGS["guardband"],
                "values_bw": SHARED_FLAGS["values_bw"],
                "mean_service_holding_time": SHARED_FLAGS["mean_service_holding_time"],
                "modulations_csv_filepath": SHARED_FLAGS["modulations_csv_filepath"],
                "k": k_val,
                "load": 10,
                "path_heuristic": best_heuristic,
                "EVAL_HEURISTIC": True,
                "continuous_operation": True,
                "ENV_WARMUP_STEPS": 0,
                "NUM_ENVS": 1,
                "TOTAL_TIMESTEPS": 10,
                "STEPS_PER_INCREMENT": 10,
                "ROLLOUT_LENGTH": 10,
            }

            cmd = ["uv", "run", "python", "-m", "xlron.train.train"]
            for key, val in flags.items():
                if isinstance(val, bool):
                    if val:
                        cmd.append(f"--{key}")
                else:
                    cmd.append(f"--{key}={val}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(PROJECT_ROOT),
            )

            if result.returncode == 0:
                print("OK")
            else:
                print("FAILED")
                failed += 1
                # Print last few lines of stderr for debugging
                if result.stderr:
                    for line in result.stderr.strip().split("\n")[-5:]:
                        print(f"    {line}")

    print(f"\nDone: {done - failed}/{done} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
