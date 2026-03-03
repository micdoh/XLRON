"""Phase 3: Run cut-set bounds with adaptive load selection.

For each topology, probes at the heuristic's load_high to find the cut-set
blocking at that load, then uses the heuristic gradient (dBP/dLoad) to
estimate where the cut-set method crosses 0.1% blocking.

Unlike the heuristic, cut-set bounds may have either higher or lower capacity,
so the search adapts to whichever direction is needed.

Uses up to MAX_PROBES adaptive probes per topology. Resumes from existing
probe data if the output file already contains partial results.

Selects CUTSET_EXHAUSTIVE for topologies with <= 30 nodes,
shortest-paths method for larger topologies.
"""

from config import (
    FULL_CUTSET_PARAMS,
    RESULTS_DIR,
    TARGET_BP,
    build_command,
    compute_heuristic_gradient,
    estimate_next_load,
    get_topology_list,
    load_load_ranges,
    parse_jsonl_blocking,
    run_command,
)


CUTSET_EXHAUSTIVE_MAX_NODES = 30
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

    completed = 0
    skipped = 0
    failed = 0

    for topo in topologies:
        name = topo["topology_name"]

        if name not in ranges or ranges[name].get("status") == "failed":
            print(f"  Skipping {name}: no load range discovered")
            failed += 1
            continue

        entry = ranges[name]
        load_high = entry["load_high"]

        if load_high <= 0:
            print(f"  Skipping {name}: invalid load_high={load_high}")
            failed += 1
            continue

        # --- Load existing probes if output file exists ---
        output_file = output_dir / f"{name}.jsonl"
        all_bps = []
        probe_files = []

        if output_file.exists() and output_file.stat().st_size > 0:
            all_bps = load_existing_probes(output_file)
            if has_bracket(all_bps):
                skipped += 1
                continue
            print(f"\n[{completed + skipped + failed + 1}/{len(topologies)}] {name} "
                  f"(resuming from {len(all_bps)} existing probes)")
            for load, bp in all_bps:
                print(f"  Existing: load={load} -> BP={bp*100:.4f}%")
        else:
            print(f"\n[{completed + skipped + failed + 1}/{len(topologies)}] {name}")

        gradient = compute_heuristic_gradient(entry)
        probed_loads = {l for l, _ in all_bps}
        start_probe = len(all_bps) + 1

        # --- Probe loop ---
        # Run up to MAX_PROBES, then up to 2 extra if still no bracket
        max_this_run = MAX_PROBES + 2
        consecutive_failures = 0
        for probe_num in range(start_probe, max_this_run + 1):
            # Stop at MAX_PROBES if we have a bracket; otherwise continue
            if probe_num > MAX_PROBES and has_bracket(all_bps):
                break

            # Bail out if all probes so far have failed (topology is broken)
            if consecutive_failures >= 2 and not all_bps:
                print(f"  Giving up on {name}: {consecutive_failures} consecutive failures with no successful probes")
                break

            # Choose load for this probe
            if not all_bps:
                # No successful probes yet — fall back to load_high
                load = load_high
            elif probe_num == 1:
                load = load_high
            elif len(all_bps) == 1:
                # First adaptive probe — use gradient-based estimate
                prev_load, prev_bp = all_bps[0]
                load = estimate_next_load(prev_load, prev_bp, gradient)
            else:
                load = choose_next_load(all_bps, gradient)

            # Skip if we've already probed this load
            if load in probed_loads:
                # Nudge slightly
                load = load + 1
            probed_loads.add(load)

            p_file = probe_dir / f"{name}_p{probe_num}.jsonl"
            print(f"  Probe {probe_num}: load={load}", end=" ", flush=True)
            bp = run_cutset_probe(name, load, topo, p_file)

            if bp is not None:
                print(f"-> BP={bp*100:.4f}%")
                probe_files.append(p_file)
                all_bps.append((load, bp))
                consecutive_failures = 0
            else:
                print("-> FAILED")
                p_file.unlink(missing_ok=True)
                consecutive_failures += 1

            # Check if we have a bracket
            if has_bracket(all_bps):
                break

        # --- Append new probes to output file ---
        with open(output_file, "a") as f_out:
            for pf in probe_files:
                if pf.exists():
                    with open(pf) as f_in:
                        for line in f_in:
                            f_out.write(line)

        # Clean up probe files
        for pf in probe_files:
            pf.unlink(missing_ok=True)

        if has_bracket(all_bps):
            print(f"  -> Brackets 0.1%")
        else:
            print(f"  -> WARNING: Does not bracket 0.1% ({len(all_bps)} probes)")

        completed += 1

    print(f"\nDone: {completed} completed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
