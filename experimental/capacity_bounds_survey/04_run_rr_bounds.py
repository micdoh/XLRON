"""Phase 4: Run reconfigurable routing bounds with adaptive load selection.

RR bounds always have higher capacity than the heuristic (lower blocking
at the same load). The first probe at heuristic's load_high will typically
show blocking below 0.1%, and we use the heuristic gradient to estimate
where RR bounds cross 0.1%.
"""

from config import (
    FULL_RR_PARAMS,
    RESULTS_DIR,
    TARGET_BP,
    ProgressTracker,
    build_command,
    compute_heuristic_gradient,
    estimate_next_load,
    format_duration,
    format_timing_breakdown,
    get_topology_list,
    load_heuristic_selection,
    load_load_ranges,
    parse_jsonl_blocking,
    run_command,
)


def run_rr_probe(name, load, probe_file, heuristic="ksp_ff", timeout=14400):
    """Run RR bounds at a single load and return blocking probability."""
    extra_flags = dict(FULL_RR_PARAMS)
    extra_flags["load"] = load
    extra_flags["path_heuristic"] = heuristic

    cmd = build_command(
        script="xlron/bounds/reconfigurable_routing_bounds.py",
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

        gradient = compute_heuristic_gradient(entry)
        probe_files = []
        all_bps = []  # (load, bp) pairs

        # --- Probe 1: run at heuristic's load_high ---
        p1_file = probe_dir / f"{name}_p1.jsonl"
        print(f"  Probe 1: load={load_high}", end=" ", flush=True)
        bp1 = run_rr_probe(name, load_high, p1_file, heuristic=best_heuristic)

        if bp1 is None:
            print("-> FAILED")
            p1_file.unlink(missing_ok=True)
            progress.item_done(t_start, "failed")
            print(f"  [wall={format_duration(progress._durations[-1])}]")
            continue
        print(f"-> BP={bp1*100:.4f}%")
        timing_str = format_timing_breakdown(p1_file)
        if timing_str:
            print(f"    [{timing_str}]")
        probe_files.append(p1_file)
        all_bps.append((load_high, bp1))

        # --- Probe 2: estimate load on other side of 0.1% ---
        load2 = estimate_next_load(load_high, bp1, gradient)
        p2_file = probe_dir / f"{name}_p2.jsonl"
        print(f"  Probe 2: load={load2}", end=" ", flush=True)
        bp2 = run_rr_probe(name, load2, p2_file, heuristic=best_heuristic)

        if bp2 is not None:
            print(f"-> BP={bp2*100:.4f}%")
            timing_str = format_timing_breakdown(p2_file)
            if timing_str:
                print(f"    [{timing_str}]")
            probe_files.append(p2_file)
            all_bps.append((load2, bp2))
        else:
            print("-> FAILED")
            p2_file.unlink(missing_ok=True)

        # --- Check bracket ---
        has_lower = any(0 < bp < TARGET_BP for _, bp in all_bps)
        has_upper = any(bp >= TARGET_BP for _, bp in all_bps)

        # --- Probe 3 (fallback if no bracket) ---
        if not (has_lower and has_upper) and len(all_bps) >= 2:
            if not has_upper:
                highest_load = max(l for l, _ in all_bps)
                load3 = round(highest_load * 2)
            elif not has_lower:
                lowest_load = min(l for l, _ in all_bps)
                load3 = max(round(lowest_load * 0.5), 1)
            else:
                load3 = None

            if load3 is not None:
                p3_file = probe_dir / f"{name}_p3.jsonl"
                print(f"  Probe 3: load={load3}", end=" ", flush=True)
                bp3 = run_rr_probe(name, load3, p3_file, heuristic=best_heuristic)
                if bp3 is not None:
                    print(f"-> BP={bp3*100:.4f}%")
                    timing_str = format_timing_breakdown(p3_file)
                    if timing_str:
                        print(f"    [{timing_str}]")
                    probe_files.append(p3_file)
                    all_bps.append((load3, bp3))
                else:
                    print("-> FAILED")
                    p3_file.unlink(missing_ok=True)

        # --- Combine probes into output ---
        with open(output_file, "w") as f_out:
            for pf in probe_files:
                with open(pf) as f_in:
                    for line in f_in:
                        f_out.write(line)

        # Clean up probe files
        for pf in probe_files:
            pf.unlink(missing_ok=True)

        has_lower = any(0 < bp < TARGET_BP for _, bp in all_bps)
        has_upper = any(bp >= TARGET_BP for _, bp in all_bps)

        if has_lower and has_upper:
            print(f"  -> Brackets 0.1%")
        else:
            print(f"  -> WARNING: Does not bracket 0.1% ({len(all_bps)} probes)")

        progress.item_done(t_start, "completed")
        print(f"  [wall={format_duration(progress._durations[-1])}]")

    print(f"\n{progress.summary_line()}")


if __name__ == "__main__":
    main()
