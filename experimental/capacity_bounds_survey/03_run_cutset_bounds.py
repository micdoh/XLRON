"""Phase 3: Run cut-set bounds with sweep-based load selection.

For each topology, runs a sweep of ~10 loads in a single subprocess call,
so JAX compiles only once per sweep. Uses load ranges from Phase 1 to
determine the initial sweep range, then runs targeted follow-up sweeps
if the first doesn't bracket 0.1% blocking.

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
    format_duration,
    format_timing_breakdown,
    get_topology_list,
    load_load_ranges,
    parse_jsonl_blocking,
    run_command,
)

MAX_SWEEPS = 3
N_POINTS = 10
MIN_LOAD = 2
MAX_LOAD = 50000


def run_cutset_sweep(name, sweep_min, sweep_max, step, topo, sweep_file, timeout=14400):
    """Run cut-set bounds across a load sweep. Returns list of {load, blocking_mean, ...}."""
    extra_flags = dict(FULL_CUTSET_PARAMS)
    extra_flags["min_load"] = sweep_min
    extra_flags["max_load"] = sweep_max
    extra_flags["step_load"] = step

    if topo["num_nodes"] <= CUTSET_EXHAUSTIVE_MAX_NODES:
        extra_flags["CUTSET_EXHAUSTIVE"] = True
        extra_flags["CUTSET_BATCH_SIZE"] = 512
        extra_flags["CUTSET_ITERATIONS"] = 32

    extra_flags["CUTSET_TOP_K"] = 256

    cmd = build_command(
        script="xlron.bounds.cutsets_bounds",
        topology=name,
        extra_flags=extra_flags,
        output_file=str(sweep_file),
    )

    result = run_command(cmd, timeout=timeout)
    if result.returncode != 0:
        return None

    return parse_jsonl_blocking(sweep_file)


def has_bracket(all_probes):
    """Check if we have probes on both sides of TARGET_BP."""
    has_lower = any(0 < bp < TARGET_BP for bp in all_probes.values())
    has_upper = any(bp >= TARGET_BP for bp in all_probes.values())
    return has_lower and has_upper


def compute_sweep_range(sweep_num, entry, all_probes):
    """Determine the min/max for the next sweep based on results so far."""
    load_low = entry.get("load_low", 0)
    load_high = entry.get("load_high", 0)

    if sweep_num == 0:
        # Broad initial sweep: 0.5x lower heuristic load to 2x upper.
        # Cutset capacity can be significantly higher or lower than heuristic.
        sweep_min = max(MIN_LOAD, round(load_low * 0.5))
        sweep_max = min(MAX_LOAD, round(load_high * 2))
        return sweep_min, sweep_max

    # Targeted sweep based on previous results
    sorted_probes = sorted(all_probes.items())
    bps = [b for _, b in sorted_probes]
    loads = [l for l, _ in sorted_probes]

    if all(b == 0 for b in bps):
        # All zero blocking — need much higher loads
        return max(loads), min(MAX_LOAD, max(loads) * 3)

    if all(b >= TARGET_BP for b in bps):
        # All at/above target — need lower loads
        return max(MIN_LOAD, round(min(loads) / 3)), min(loads)

    if all(b < TARGET_BP for b in bps):
        # All below target — need higher loads
        return max(loads), min(MAX_LOAD, round(max(loads) * 2))

    # Mixed results — zoom into the transition region
    below_target = [l for l, b in all_probes.items() if b < TARGET_BP]
    above_target = [l for l, b in all_probes.items() if b >= TARGET_BP]

    if below_target and above_target:
        lo = max(below_target)
        hi = min(above_target)
        margin = max(1, round((hi - lo) * 0.2))
        return max(MIN_LOAD, lo - margin), min(MAX_LOAD, hi + margin)

    # Fallback
    return max(MIN_LOAD, round(min(loads) * 0.5)), min(MAX_LOAD, round(max(loads) * 2))


def main():
    print("=" * 60)
    print("Phase 3: Running cut-set bounds (sweep-based)")
    print(f"  Max sweeps per topology: {MAX_SWEEPS}, ~{N_POINTS} loads per sweep")
    print("=" * 60)

    topologies = get_topology_list()
    ranges = load_load_ranges()
    output_dir = RESULTS_DIR / "cutset_bounds"
    sweep_dir = output_dir / "sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

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

        # Check for existing complete results
        output_file = output_dir / f"{name}.jsonl"
        existing_probes = {}
        if output_file.exists() and output_file.stat().st_size > 0:
            results = parse_jsonl_blocking(output_file)
            existing_probes = {r["load"]: r["blocking_mean"] for r in results}
            if has_bracket(existing_probes):
                progress.item_done(progress.item_start(), "skipped")
                continue

        t_start = progress.item_start()
        method = "exhaustive" if topo["num_nodes"] <= CUTSET_EXHAUSTIVE_MAX_NODES else "shortest-paths"
        print(progress.header(progress.processed + 1, name, f"({method})"))

        all_probes = dict(existing_probes)
        sweep_files = []

        if existing_probes:
            print(f"  Resuming with {len(existing_probes)} existing probes")

        for sweep_num in range(MAX_SWEEPS):
            if has_bracket(all_probes):
                break

            sweep_min, sweep_max = compute_sweep_range(sweep_num, entry, all_probes)

            if sweep_max <= sweep_min:
                sweep_max = sweep_min + N_POINTS

            step = max(round((sweep_max - sweep_min) / (N_POINTS - 1)), 1)
            sweep_max = sweep_min + step * (N_POINTS - 1)  # exact N_POINTS

            print(f"  Sweep {sweep_num + 1}: [{sweep_min}, {sweep_max}] step={step}")

            sweep_file = sweep_dir / f"{name}_sweep{sweep_num}.jsonl"
            entries = run_cutset_sweep(name, sweep_min, sweep_max, step, topo, sweep_file)

            if entries is None:
                print(f"  -> Sweep FAILED")
                break

            sweep_files.append(sweep_file)
            for e in entries:
                all_probes[e["load"]] = e["blocking_mean"]
                print(f"    load={e['load']:>6} -> BP={e['blocking_mean']*100:.4f}%")

            timing_str = format_timing_breakdown(sweep_file)
            if timing_str:
                print(f"    [{timing_str}]")

        # Append new sweep data to output file
        if sweep_files:
            with open(output_file, "a") as f_out:
                for sf in sweep_files:
                    if sf.exists():
                        with open(sf) as f_in:
                            f_out.write(f_in.read())

        if has_bracket(all_probes):
            print(f"  -> Brackets 0.1%")
        elif all_probes:
            print(f"  -> WARNING: Does not bracket 0.1% ({len(all_probes)} probes)")
        else:
            print(f"  -> No successful probes")

        progress.item_done(t_start, "completed" if all_probes else "failed")
        print(f"  [wall={format_duration(progress._durations[-1])}]")

    print(f"\n{progress.summary_line()}")


if __name__ == "__main__":
    main()
