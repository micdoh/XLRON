"""Phase 2: Run full heuristic evaluation at the two bracket loads.

For each topology, runs at exactly the two loads from Phase 1 that
bracket 0.1% blocking (one below, one above). Uses higher-quality
statistical parameters than the quick discovery probes.
"""

from config import (
    FULL_HEURISTIC_PARAMS,
    RESULTS_DIR,
    build_command,
    load_load_ranges,
    get_topology_list,
    run_command,
)


def main():
    print("=" * 60)
    print("Phase 2: Running heuristic evaluation at bracket loads")
    print("=" * 60)

    topologies = get_topology_list()
    ranges = load_load_ranges()
    output_dir = RESULTS_DIR / "heuristic_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0

    for topo in topologies:
        name = topo["topology_name"]

        output_file = output_dir / f"{name}.jsonl"
        if output_file.exists() and output_file.stat().st_size > 0:
            skipped += 1
            continue

        if name not in ranges or ranges[name].get("status") == "failed":
            print(f"  Skipping {name}: no load range discovered")
            failed += 1
            continue

        entry = ranges[name]
        load_low = entry["load_low"]
        load_high = entry["load_high"]
        step = load_high - load_low

        if step <= 0:
            print(f"  Skipping {name}: invalid load range [{load_low}, {load_high}]")
            failed += 1
            continue

        print(f"[{completed + skipped + failed + 1}/{len(topologies)}] {name}: "
              f"loads [{load_low}, {load_high}]")

        cmd = build_command(
            script="xlron.train.train",
            topology=name,
            extra_flags={
                **FULL_HEURISTIC_PARAMS,
                "EVAL_HEURISTIC": True,
                "load": load_high,
                "min_load": load_low,
                "max_load": load_high,
                "step_load": step,
            },
            output_file=str(output_file),
        )

        result = run_command(cmd, timeout=7200)
        if result.returncode == 0:
            completed += 1
        else:
            failed += 1

    print(f"\nDone: {completed} completed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
