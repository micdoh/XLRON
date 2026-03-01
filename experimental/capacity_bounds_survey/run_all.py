#!/usr/bin/env python
"""Master entrypoint: run all phases of the capacity bounds survey.

Usage:
    uv run python experimental/capacity_bounds_survey/run_all.py

Runs all 5 phases sequentially. Resume-safe: each phase skips
topologies that already have output files. Can be killed and
restarted at any time.

Individual phases can also be run standalone:
    uv run python experimental/capacity_bounds_survey/01_discover_load_ranges.py
    uv run python experimental/capacity_bounds_survey/03_run_cutset_bounds.py
    ...etc
"""

import importlib
import os
import sys
import time
from pathlib import Path

# Ensure we can import from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_phase(module_name: str, phase_num: int, description: str):
    """Import and run a phase module's main() function."""
    print()
    print("#" * 60)
    print(f"# Phase {phase_num}: {description}")
    print("#" * 60)
    print()

    start = time.time()
    module = importlib.import_module(module_name)
    module.main()
    elapsed = time.time() - start

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    print(f"\nPhase {phase_num} completed in {hours}h {minutes}m")


def main():
    # Change to project root so paths work correctly
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)

    print("=" * 60)
    print("All-Topology Capacity Bounds Survey")
    print("=" * 60)
    print(f"Working directory: {project_root}")
    print()

    # Show config summary
    from config import SHARED_FLAGS, get_topology_list
    topos = get_topology_list()
    print(f"Topologies: {len(topos)}")
    print(f"Traffic model: {SHARED_FLAGS['env_type']}, "
          f"{SHARED_FLAGS['link_resources']} slots x {SHARED_FLAGS['slot_size']} GHz, "
          f"guardband={SHARED_FLAGS['guardband']}, "
          f"bw={SHARED_FLAGS['values_bw']} Gbps, "
          f"K={SHARED_FLAGS['k']}")
    print()

    total_start = time.time()

    phases = [
        ("01_discover_load_ranges", 1, "Discover load ranges + heuristic eval"),
        ("03_run_cutset_bounds",    2, "Cut-set bounds"),
        ("04_run_rr_bounds",        3, "Reconfigurable routing bounds"),
        ("05_run_k_sensitivity",    4, "K-sensitivity experiment"),
        ("06_analyze_results",      5, "Analysis and plotting"),
    ]

    for module_name, phase_num, description in phases:
        run_phase(module_name, phase_num, description)

    total_elapsed = time.time() - total_start
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    print()
    print("=" * 60)
    print(f"All phases complete! Total time: {hours}h {minutes}m")
    print(f"Results: experimental/capacity_bounds_survey/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
