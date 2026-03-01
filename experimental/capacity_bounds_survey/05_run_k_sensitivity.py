"""Phase 5: K-sensitivity experiment for larger topologies.

For topologies with >30 nodes, run heuristic eval at a fixed load
(near the 0.1% blocking point) while varying K from 5 to 100.
Each K value requires separate compilation since K affects path computation.
"""

import json

from config import (
    FULL_HEURISTIC_PARAMS,
    K_SENSITIVITY_MIN_NODES,
    K_SENSITIVITY_VALUES,
    RESULTS_DIR,
    SHARED_FLAGS,
    build_command,
    get_topology_list,
    load_load_ranges,
    parse_jsonl_blocking,
    run_command,
)


def get_target_load(name: str, ranges: dict) -> float | None:
    """Get the load to use for K-sensitivity (midpoint of discovered range)."""
    if name not in ranges or ranges[name].get("status") == "failed":
        return None
    entry = ranges[name]
    load_low = entry["load_low"]
    load_high = entry["load_high"]
    # Use geometric mean of low and high (closer to 0.1% blocking)
    return round((load_low * load_high) ** 0.5)


def main():
    print("=" * 60)
    print("Phase 5: K-sensitivity experiment")
    print("=" * 60)

    topologies = get_topology_list()
    ranges = load_load_ranges()
    output_dir = RESULTS_DIR / "k_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to large topologies only
    large_topos = [t for t in topologies if t["num_nodes"] > K_SENSITIVITY_MIN_NODES]
    print(f"Running K-sensitivity for {len(large_topos)} topologies with >{K_SENSITIVITY_MIN_NODES} nodes")
    print(f"K values: {K_SENSITIVITY_VALUES}")

    completed = 0
    skipped = 0
    failed = 0

    for topo in large_topos:
        name = topo["topology_name"]

        # Check if already done
        output_file = output_dir / f"{name}.jsonl"
        if output_file.exists() and output_file.stat().st_size > 0:
            # Check if all K values are present
            existing = parse_jsonl_blocking(output_file)
            if len(existing) >= len(K_SENSITIVITY_VALUES):
                skipped += 1
                continue

        target_load = get_target_load(name, ranges)
        if target_load is None:
            print(f"  Skipping {name}: no load range discovered")
            failed += 1
            continue

        print(f"\n[{completed + skipped + failed + 1}/{len(large_topos)}] {name} "
              f"(nodes={topo['num_nodes']}), load={target_load}")

        # Clear the output file for a fresh run
        if output_file.exists():
            output_file.unlink()

        topo_failed = False
        for k_val in K_SENSITIVITY_VALUES:
            # Each K value is a separate run (different compilation)
            temp_file = output_dir / f"{name}_k{k_val}.jsonl"

            print(f"  K={k_val}", end=" ", flush=True)

            cmd = build_command(
                script="xlron.train.train",
                topology=name,
                extra_flags={
                    **FULL_HEURISTIC_PARAMS,
                    "EVAL_HEURISTIC": True,
                    "load": target_load,
                    "k": k_val,
                },
                output_file=str(temp_file),
            )

            result = run_command(cmd, timeout=7200)

            if result.returncode == 0:
                # Read the result and append to the main output file
                results = parse_jsonl_blocking(temp_file)
                if results:
                    bp = results[0]["blocking_mean"]
                    print(f"-> BP={bp*100:.4f}%")

                    # Append to consolidated output file
                    with open(temp_file) as f_in, open(output_file, "a") as f_out:
                        for line in f_in:
                            f_out.write(line)
                else:
                    print("-> no data")
                # Clean up temp file
                temp_file.unlink(missing_ok=True)
            else:
                print("-> FAILED")
                topo_failed = True
                temp_file.unlink(missing_ok=True)

        if topo_failed:
            failed += 1
        else:
            completed += 1

    print(f"\nDone: {completed} completed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
