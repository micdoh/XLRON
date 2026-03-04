"""Phase 3: Run cut-set bounds with adaptive load selection.

For each topology, probes at the heuristic's load_high to find the cut-set
blocking at that load, then uses the heuristic gradient (dBP/dLoad) to
estimate where the cut-set method crosses 0.1% blocking.

Unlike the heuristic, cut-set bounds may have either higher or lower capacity,
so the search adapts to whichever direction is needed.

Uses up to MAX_PROBES adaptive probes per topology. Resumes from existing
probe data if the output file already contains partial results.

Selects CUTSET_EXHAUSTIVE for topologies with <= CUTSET_EXHAUSTIVE_MAX_NODES nodes,
shortest-paths method for larger topologies.
"""

from config import (
    CUTSET_EXHAUSTIVE_MAX_NODES,
    FULL_CUTSET_PARAMS,
    RESULTS_DIR,
    TARGET_BP,
    ProgressTracker,
    build_command,
    compute_heuristic_gradient,
    estimate_next_load,
    format_duration,
    format_timing_breakdown,
    get_topology_list,
    load_load_ranges,
    parse_jsonl_blocking,
    run_command,
)
MAX_PROBES = 5


def run_cutset_probe(name, load, topo, probe_file, timeout=14400):
    """Run cut-set bounds at a single load and return blocking probability."""
    extra_flags = dict(FULL_CUTSET_PARAMS)
    extra_flags["load"] = load

    if topo["num_nodes"] <= CUTSET_EXHAUSTIVE_MAX_NODES:
        extra_flags["CUTSET_EXHAUSTIVE"] = True
        extra_flags["CUTSET_BATCH_SIZE"] = 512
        extra_flags["CUTSET_ITERATIONS"] = 32

    extra_flags["CUTSET_TOP_K"] = 256
    extra_flags["cutset_link_selection_mode"] = "least_congested"

    cmd = build_command(
        script="xlron.bounds.cutsets_bounds",
        topology=name,
        extra_flags=extra_flags,
        output_file=str(probe_file),
    )

    result = run_command(cmd, timeout=timeout)
    if result.returncode != 0:
        return None

    results = parse_jsonl_blocking(probe_file)
    if not results:
        return None

    return results[0]["blocking_mean"]


def has_bracket(all_bps):
    """Check if we have probes on both sides of TARGET_BP."""
    has_lower = any(0 < bp < TARGET_BP for _, bp in all_bps)
    has_upper = any(bp >= TARGET_BP for _, bp in all_bps)
    return has_lower and has_upper


def choose_next_load(all_bps, gradient):
    """Choose the next load to probe based on existing results.

    Strategy:
    - If we have both zero-BP and high-BP probes: bisect between them.
    - If all probes are zero or below target: increase load.
    - If all probes are at/above target: decrease load.
    - For the first adaptive probe, use gradient-based estimate.

    Returns None if all_bps is empty (caller should fall back to load_high).
    """
    if not all_bps:
        return None

    zero_loads = [l for l, bp in all_bps if bp <= 0]
    high_loads = [l for l, bp in all_bps if bp >= TARGET_BP]
    below_loads = [l for l, bp in all_bps if 0 < bp < TARGET_BP]

    # Bisect between zero/below and high
    if high_loads and (zero_loads or below_loads):
        # Use the highest load that's below target as the lower bound
        below_all = zero_loads + below_loads
        lo = max(below_all)
        hi = min(high_loads)
        load = round((lo + hi) / 2)
        # Ensure we're not re-probing the same load
        if load <= lo:
            load = lo + 1
        elif load >= hi:
            load = hi - 1
        if load <= 0:
            load = 1
        return load

    if not high_loads:
        # All below target or zero — need higher load
        highest_load = max(l for l, _ in all_bps)
        highest_bp = max(bp for l, bp in all_bps if l == highest_load)
        if highest_bp <= 0:
            # All zero — double the highest
            return round(highest_load * 2)
        else:
            # Have some non-zero BP — use gradient if available
            return estimate_next_load(highest_load, highest_bp, gradient)

    # All at/above target — need lower load
    lowest_load = min(l for l, _ in all_bps)
    lowest_bp = min(bp for l, bp in all_bps if l == lowest_load)
    return estimate_next_load(lowest_load, lowest_bp, gradient)


def load_existing_probes(output_file):
    """Load (load, bp) pairs from an existing output file."""
    all_bps = []
    results = parse_jsonl_blocking(output_file)
    for r in results:
        all_bps.append((r["load"], r["blocking_mean"]))
    return all_bps


def main():
    print("=" * 60)
    print("Phase 3: Running cut-set bounds (adaptive load selection)")
    print(f"  Max probes per topology: {MAX_PROBES}")
    print("=" * 60)

    topologies = get_topology_list()
    ranges = load_load_ranges()
    output_dir = RESULTS_DIR / "cutset_bounds"
    probe_dir = output_dir / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressTracker(len(topologies), "Phase 3")

    for topo in topologies:
        name = topo["topology_name"]

        if name not in ranges or ranges[name].get("status") == "failed":
            print(f"  Skipping {name}: no load range discovered")
            progress.item_done(progress.item_start(), "failed")
            continue

        entry = ranges[name]
        load_high = entry.get("load_high", 0)

        if load_high <= 0:
            print(f"  Skipping {name}: invalid load_high={load_high}")
            progress.item_done(progress.item_start(), "failed")
            continue

        t_start = progress.item_start()
        method = "exhaustive" if topo["num_nodes"] <= CUTSET_EXHAUSTIVE_MAX_NODES else "shortest-paths"
        print(progress.header(progress.processed + 1, name, f"({method})"))

        gradient = compute_heuristic_gradient(entry)
        output_file = output_dir / f"{name}.jsonl"
        probe_files = []

        # Resume: load existing probes from output file
        all_bps = load_existing_probes(output_file)
        existing_count = len(all_bps)
        if existing_count > 0:
            print(f"  Resuming with {existing_count} existing probes")
            if has_bracket(all_bps):
                print(f"  -> Already brackets 0.1%")
                progress.item_done(t_start, "completed")
                continue

        # Adaptive probe loop
        consecutive_failures = 0
        for probe_num in range(existing_count + 1, MAX_PROBES + 1):
            if has_bracket(all_bps):
                break

            # Choose load
            next_load = choose_next_load(all_bps, gradient)
            if next_load is None:
                next_load = load_high  # Fallback for first probe or empty results

            # Skip if we already have this load
            existing_loads = {l for l, _ in all_bps}
            if next_load in existing_loads:
                next_load = next_load + 1

            probe_file = probe_dir / f"{name}_p{probe_num}.jsonl"
            print(f"  Probe {probe_num}: load={next_load}", end=" ", flush=True)

            bp = run_cutset_probe(name, next_load, topo, probe_file)
            if bp is not None:
                print(f"-> BP={bp*100:.4f}%")
                timing_str = format_timing_breakdown(probe_file)
                if timing_str:
                    print(f"    [{timing_str}]")
                probe_files.append(probe_file)
                all_bps.append((next_load, bp))
                consecutive_failures = 0
            else:
                print("-> FAILED")
                probe_file.unlink(missing_ok=True)
                consecutive_failures += 1
                if consecutive_failures >= 2 and not all_bps:
                    print(f"  -> Bailing out after {consecutive_failures} consecutive failures with no data")
                    break

        # Append new probe data to output file
        if probe_files:
            with open(output_file, "a") as f_out:
                for pf in probe_files:
                    with open(pf) as f_in:
                        for line in f_in:
                            f_out.write(line)

            # Clean up probe files
            for pf in probe_files:
                pf.unlink(missing_ok=True)

        if has_bracket(all_bps):
            print(f"  -> Brackets 0.1%")
        elif all_bps:
            print(f"  -> WARNING: Does not bracket 0.1% ({len(all_bps)} probes)")
        else:
            print(f"  -> No successful probes")

        progress.item_done(t_start, "completed" if all_bps else "failed")
        print(f"  [wall={format_duration(progress._durations[-1])}]")

    print(f"\n{progress.summary_line()}")


if __name__ == "__main__":
    main()
