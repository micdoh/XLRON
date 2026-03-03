"""Phase 4: Run reconfigurable routing bounds with adaptive load selection.

Strategy:
1. Single-trial probes starting at heuristic's load_high, incrementing ~10%.
2. Any probe with blocking > 0% but <= 1% is selected for a full 10-trial run.
3. If a probe shows blocking > 1%, scale back (too expensive due to defrag).
4. Stop probing once blocking exceeds 0.1% (TARGET_BP).
5. Full 10-trial runs at selected loads produce the actual data.
"""

from config import (
    FULL_RR_PARAMS,
    RESULTS_DIR,
    TARGET_BP,
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

MAX_PROBES = 30
LOAD_INCREMENT = 1.10  # 10% increment between probes
SCALE_BACK_FACTOR = 0.95  # 5% reduction when blocking > 1%
MAX_BLOCKING_FOR_FULL_RUN = 0.01  # 1% - don't do full runs above this


def run_rr(name, load, output_file, heuristic="ksp_ff", num_trials=1, timeout=14400):
    """Run RR bounds at a single load and return blocking probability."""
    extra_flags = dict(FULL_RR_PARAMS)
    extra_flags["load"] = load
    extra_flags["path_heuristic"] = heuristic
    extra_flags["num_trials"] = num_trials

    cmd = build_command(
        script="xlron/bounds/reconfigurable_routing_bounds.py",
        topology=name,
        extra_flags=extra_flags,
        output_file=str(output_file),
    )

    result = run_command(cmd, timeout=timeout)
    if result.returncode != 0:
        return None

    results = parse_jsonl_blocking(output_file)
    if not results:
        return None

    return results[0]["blocking_mean"]


def main():
    print("=" * 60)
    print("Phase 4: Running RR bounds (adaptive load selection)")
    print("=" * 60)

    topologies = get_topology_list()
    ranges = load_load_ranges()
    heur_selection = load_heuristic_selection()
    output_dir = RESULTS_DIR / "rr_bounds"
    probe_dir = output_dir / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressTracker(len(topologies), "Phase 4")

    for topo in topologies:
        name = topo["topology_name"]

        output_file = output_dir / f"{name}.jsonl"
        if output_file.exists() and output_file.stat().st_size > 0:
            progress.item_done(progress.item_start(), "skipped")
            continue

        if name not in ranges or ranges[name].get("status") == "failed":
            print(f"  Skipping {name}: no load range discovered")
            progress.item_done(progress.item_start(), "failed")
            continue

        entry = ranges[name]
        load_high = entry["load_high"]

        if load_high <= 0:
            print(f"  Skipping {name}: invalid load_high={load_high}")
            progress.item_done(progress.item_start(), "failed")
            continue

        best_heuristic = heur_selection.get(name, "ksp_ff")
        t_start = progress.item_start()
        print(progress.header(progress.processed + 1, name, f"(heuristic={best_heuristic})"))

        # =============================================================
        # Phase 1: Cheap single-trial probes to find loads with blocking
        # =============================================================
        print("  --- Phase 1: Single-trial probes ---")
        current_load = load_high
        probe_num = 0
        selected_loads = []  # loads to do full 10-trial runs at
        probed_loads = set()  # prevent re-probing the same rounded load
        phase1_failed = False

        while probe_num < MAX_PROBES:
            # Avoid re-probing the same load (rounding can cause duplicates)
            if current_load in probed_loads:
                current_load = round(current_load * 1.05)
                if current_load in probed_loads:
                    break
            probed_loads.add(current_load)

            probe_num += 1
            pfile = probe_dir / f"{name}_probe{probe_num}.jsonl"
            print(f"  Probe {probe_num}: load={current_load}", end=" ", flush=True)
            bp = run_rr(name, current_load, pfile, heuristic=best_heuristic, num_trials=1)

            if bp is None:
                print("-> FAILED")
                pfile.unlink(missing_ok=True)
                phase1_failed = True
                break

            print(f"-> BP={bp*100:.4f}%")
            pfile.unlink(missing_ok=True)

            if bp > MAX_BLOCKING_FOR_FULL_RUN:
                # > 1% blocking - too expensive for defrag, scale back
                print(f"    Blocking > 1%, scaling back")
                current_load = max(round(current_load * SCALE_BACK_FACTOR), 1)
                continue

            if bp > 0:
                # Blocking detected but <= 1% - select for full run
                selected_loads.append(current_load)

            if bp >= TARGET_BP:
                # Found our target - done probing
                print(f"    Found load above {TARGET_BP*100:.1f}% target")
                break

            # Not at target yet - increment load by ~10%
            current_load = round(current_load * LOAD_INCREMENT)

        if phase1_failed and not selected_loads:
            progress.item_done(t_start, "failed")
            print(f"  [wall={format_duration(progress._durations[-1])}]")
            continue

        if not selected_loads:
            print(f"  WARNING: No loads with blocking > 0% found in {probe_num} probes")
            progress.item_done(t_start, "failed")
            print(f"  [wall={format_duration(progress._durations[-1])}]")
            continue

        # =============================================================
        # Phase 2: Full 10-trial runs at selected loads
        # =============================================================
        selected_loads = sorted(set(selected_loads))
        print(f"  --- Phase 2: Full 10-trial runs at {len(selected_loads)} load(s): {selected_loads} ---")
        full_files = []

        for load in selected_loads:
            full_file = probe_dir / f"{name}_full_{load}.jsonl"
            print(f"  Full run: load={load} (10 trials)", end=" ", flush=True)
            bp = run_rr(name, load, full_file, heuristic=best_heuristic, num_trials=10)

            if bp is not None:
                print(f"-> BP={bp*100:.4f}%")
                timing_str = format_timing_breakdown(full_file)
                if timing_str:
                    print(f"    [{timing_str}]")
                full_files.append(full_file)
            else:
                print("-> FAILED")
                full_file.unlink(missing_ok=True)

        # Combine full run results into output
        if full_files:
            with open(output_file, "w") as f_out:
                for ff in full_files:
                    with open(ff) as f_in:
                        for line in f_in:
                            f_out.write(line)

            # Clean up
            for ff in full_files:
                ff.unlink(missing_ok=True)

            print(f"  -> Wrote {len(full_files)} load(s) to {output_file.name}")
            progress.item_done(t_start, "completed")
        else:
            print(f"  -> All full runs failed")
            progress.item_done(t_start, "failed")

        print(f"  [wall={format_duration(progress._durations[-1])}]")

    print(f"\n{progress.summary_line()}")


if __name__ == "__main__":
    main()
