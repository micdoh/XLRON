"""Phase 2: Compare FF-KSP vs KSP-FF heuristics per topology.

For each topology, runs FF-KSP at load_high (from Phase 1) and compares
blocking with KSP-FF. If FF-KSP gives strictly lower blocking, re-runs
Phase 1 discovery with FF-KSP and updates load_ranges.json.

Saves heuristic_selection.json mapping each topology to its best heuristic.
"""

import importlib
import json

from config import (
    QUICK_SCAN_PARAMS,
    RESULTS_DIR,
    build_command,
    get_topology_list,
    load_heuristic_selection,
    load_load_ranges,
    parse_jsonl_blocking,
    run_command,
    save_load_ranges,
)


def save_heuristic_selection(selection: dict):
    """Save heuristic selection to JSON."""
    path = RESULTS_DIR / "heuristic_selection.json"
    with open(path, "w") as f:
        json.dump(selection, f, indent=2)


def run_single_probe(topology, load, heuristic, probe_file, timeout=3600):
    """Run heuristic eval at a single load point. Returns blocking_mean or None."""
    cmd = build_command(
        script="xlron.train.train",
        topology=topology,
        extra_flags={
            **QUICK_SCAN_PARAMS,
            "EVAL_HEURISTIC": True,
            "load": load,
            "path_heuristic": heuristic,
        },
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
    print("Phase 2: Comparing KSP-FF vs FF-KSP heuristics")
    print("=" * 60)

    topologies = get_topology_list()
    ranges = load_load_ranges()
    selection = load_heuristic_selection()
    probe_dir = RESULTS_DIR / "probes"
    heuristic_dir = RESULTS_DIR / "heuristic_eval"
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Build topology stats lookup (needed for Phase 1 re-run)
    topo_stats = {t["topology_name"]: t for t in topologies}

    completed = 0
    skipped = 0
    failed = 0
    switched = 0

    for topo in topologies:
        name = topo["topology_name"]

        # Resume: skip if already selected
        if name in selection:
            skipped += 1
            continue

        # Need load_ranges entry from Phase 1
        if name not in ranges or ranges[name].get("status") == "failed":
            print(f"  Skipping {name}: no load range from Phase 1")
            failed += 1
            continue

        entry = ranges[name]
        load_high = entry.get("load_high", 0)
        bp_high = entry.get("bp_high")

        if load_high <= 0 or bp_high is None:
            # Default to ksp_ff for edge cases
            selection[name] = "ksp_ff"
            save_heuristic_selection(selection)
            completed += 1
            continue

        print(f"\n[{completed + skipped + failed + 1}/{len(topologies)}] {name}")
        print(f"  KSP-FF at load={load_high}: BP={bp_high*100:.4f}%")

        # Run FF-KSP probe at load_high
        probe_file = probe_dir / f"{name}_ffksp_compare.jsonl"
        ff_ksp_bp = run_single_probe(name, load_high, "ff_ksp", probe_file)

        if ff_ksp_bp is None:
            print(f"  FF-KSP probe FAILED, defaulting to ksp_ff")
            selection[name] = "ksp_ff"
            save_heuristic_selection(selection)
            probe_file.unlink(missing_ok=True)
            failed += 1
            continue

        print(f"  FF-KSP at load={load_high}: BP={ff_ksp_bp*100:.4f}%")
        probe_file.unlink(missing_ok=True)

        if ff_ksp_bp < bp_high:
            print(f"  -> FF-KSP is BETTER ({(bp_high - ff_ksp_bp)*100:.4f}pp lower)")
            print(f"  -> Re-running Phase 1 with FF-KSP...")
            selection[name] = "ff_ksp"
            switched += 1

            # Delete old heuristic_eval data
            old_heuristic_file = heuristic_dir / f"{name}.jsonl"
            old_heuristic_file.unlink(missing_ok=True)

            # Remove old load_ranges entry so discover_bracket runs fresh
            del ranges[name]
            save_load_ranges(ranges)

            # Re-run Phase 1 discovery with ff_ksp
            phase1 = importlib.import_module("01_discover_load_ranges")
            result = phase1.discover_bracket(
                name, topo_stats[name], probe_dir, heuristic_dir,
                extra_flags={"path_heuristic": "ff_ksp"},
            )
            ranges[name] = result
            save_load_ranges(ranges)

            if result["status"] in ("ok", "ok_relaxed"):
                print(f"  -> New bracket: load_low={result['load_low']} "
                      f"(BP={result['bp_low']*100:.4f}%), "
                      f"load_high={result['load_high']} "
                      f"(BP={result['bp_high']*100:.4f}%)")
            else:
                print(f"  -> Re-run status: {result['status']}")
        else:
            print(f"  -> KSP-FF is same or better, keeping ksp_ff")
            selection[name] = "ksp_ff"

        save_heuristic_selection(selection)
        completed += 1

    print(f"\nDone: {completed} compared, {skipped} skipped, {failed} failed")
    print(f"Switched to FF-KSP: {switched} topologies")
    print(f"Selection saved to {RESULTS_DIR / 'heuristic_selection.json'}")


if __name__ == "__main__":
    main()
