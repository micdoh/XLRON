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
    ProgressTracker,
    build_command,
    format_duration,
    format_timing_breakdown,
    get_topology_list,
    load_heuristic_selection,
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
    heur_selection = load_heuristic_selection()
    output_dir = RESULTS_DIR / "k_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to large topologies only
    large_topos = [t for t in topologies if t["num_nodes"] > K_SENSITIVITY_MIN_NODES]
    print(f"Running K-sensitivity for {len(large_topos)} topologies with >{K_SENSITIVITY_MIN_NODES} nodes")
    print(f"K values: {K_SENSITIVITY_VALUES}")

    progress = ProgressTracker(len(large_topos), "Phase 5")

    for topo in large_topos:
        name = topo["topology_name"]

        # Check which K values are already done
        output_file = output_dir / f"{name}.jsonl"
        completed_k = set()
        if output_file.exists() and output_file.stat().st_size > 0:
            with open(output_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    k_done = obj.get("config", {}).get("k")
                    if k_done is not None:
                        completed_k.add(int(k_done))

            if completed_k >= set(K_SENSITIVITY_VALUES):
                progress.item_done(progress.item_start(), "skipped")
                continue

        target_load = get_target_load(name, ranges)
        if target_load is None:
            print(f"  Skipping {name}: no load range discovered")
            progress.item_done(progress.item_start(), "failed")
            continue

        best_heuristic = heur_selection.get(name, "ksp_ff")
        t_start = progress.item_start()
        remaining_k = [k for k in K_SENSITIVITY_VALUES if k not in completed_k]
        print(progress.header(
            progress.processed + 1, name,
            f"(nodes={topo['num_nodes']}), load={target_load}, heuristic={best_heuristic}"
            + (f", resuming ({len(completed_k)}/{len(K_SENSITIVITY_VALUES)} done)" if completed_k else "")
        ))

        topo_failed = False
        for k_val in remaining_k:
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
                    "path_heuristic": best_heuristic,
                },
                output_file=str(temp_file),
            )

            result = run_command(cmd, timeout=7200)

            if result.returncode == 0:
                # Read the result and append to the main output file
                results = parse_jsonl_blocking(temp_file)
                if results:
                    bp = results[0]["blocking_mean"]
                    timing_str = format_timing_breakdown(temp_file)
                    timing_part = f"  [{timing_str}]" if timing_str else ""
                    print(f"-> BP={bp*100:.4f}%{timing_part}")

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
            progress.item_done(t_start, "failed")
        else:
            progress.item_done(t_start, "completed")
        print(f"  [wall={format_duration(progress._durations[-1])}]")

    print(f"\n{progress.summary_line()}")


if __name__ == "__main__":
    main()
