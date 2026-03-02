"""Phase 3: Run cut-set bounds with adaptive load selection.

For each topology, probes at the heuristic's load_high to find the cut-set
blocking at that load, then uses the heuristic gradient (dBP/dLoad) to
estimate where the cut-set method crosses 0.1% blocking.

Unlike the heuristic, cut-set bounds may have either higher or lower capacity,
so the search adapts to whichever direction is needed.

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


def main():
    print("=" * 60)
    print("Phase 3: Running cut-set bounds (adaptive load selection)")
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

        output_file = output_dir / f"{name}.jsonl"
        if output_file.exists() and output_file.stat().st_size > 0:
            skipped += 1
            continue

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

        print(f"\n[{completed + skipped + failed + 1}/{len(topologies)}] {name}")

        gradient = compute_heuristic_gradient(entry)
        probe_files = []
        all_bps = []  # (load, bp) pairs

        # --- Probe 1: run at heuristic's load_high ---
        p1_file = probe_dir / f"{name}_p1.jsonl"
        print(f"  Probe 1: load={load_high}", end=" ", flush=True)
        bp1 = run_cutset_probe(name, load_high, topo, p1_file)

        if bp1 is None:
            print("-> FAILED")
            p1_file.unlink(missing_ok=True)
            failed += 1
            continue
        print(f"-> BP={bp1*100:.4f}%")
        probe_files.append(p1_file)
        all_bps.append((load_high, bp1))

        # --- Probe 2: estimate load on other side of 0.1% ---
        load2 = estimate_next_load(load_high, bp1, gradient)
        p2_file = probe_dir / f"{name}_p2.jsonl"
        print(f"  Probe 2: load={load2}", end=" ", flush=True)
        bp2 = run_cutset_probe(name, load2, topo, p2_file)

        if bp2 is not None:
            print(f"-> BP={bp2*100:.4f}%")
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
            # Check if we have both zero-BP and high-BP probes (common case:
            # 0% at one load, >>0.1% at another). Bisect between them.
            zero_loads = [l for l, bp in all_bps if bp <= 0]
            high_loads = [l for l, bp in all_bps if bp >= TARGET_BP]

            if zero_loads and high_loads:
                # Bisect between the highest zero-BP load and lowest high-BP load
                lo = max(zero_loads)
                hi = min(high_loads)
                load3 = round((lo + hi) / 2)
                # Ensure we're not re-probing the same load
                if load3 == lo:
                    load3 = lo + 1
                elif load3 == hi:
                    load3 = hi - 1
            elif not has_upper:
                # All below 0.1% - need higher load
                highest_load = max(l for l, _ in all_bps)
                load3 = round(highest_load * 2)
            elif not has_lower:
                # All above 0.1% - need lower load
                lowest_load = min(l for l, _ in all_bps)
                load3 = max(round(lowest_load * 0.5), 1)
            else:
                load3 = None

            if load3 is not None:
                p3_file = probe_dir / f"{name}_p3.jsonl"
                print(f"  Probe 3: load={load3}", end=" ", flush=True)
                bp3 = run_cutset_probe(name, load3, topo, p3_file)
                if bp3 is not None:
                    print(f"-> BP={bp3*100:.4f}%")
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

        completed += 1

    print(f"\nDone: {completed} completed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
