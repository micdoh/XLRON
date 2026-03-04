"""Phase 2: Compare FF-KSP vs KSP-FF heuristics per topology.

For each topology, runs FF-KSP at load_high (from Phase 1) and compares
blocking with KSP-FF. If FF-KSP gives strictly lower blocking, re-runs
Phase 1 discovery with FF-KSP and updates load_ranges.json.

Saves heuristic_selection.json mapping each topology to its best heuristic.
"""

import importlib
import json
from pathlib import Path

from config import (
    QUICK_SCAN_PARAMS,
    RESULTS_DIR,
    ProgressTracker,
    build_command,
    format_duration,
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


def extract_bracket_from_jsonl(jsonl_path: Path, heuristic: str) -> dict | None:
    """Extract load bracket from existing heuristic_eval JSONL for a specific heuristic.

    Filters entries by config.path_heuristic, then applies the same bracket
    logic as Phase 1's discover_bracket. Returns a load_ranges-style dict
    or None if insufficient data.
    """
    from importlib import import_module
    phase1 = import_module("01_discover_load_ranges")

    if not jsonl_path.exists():
        return None

    # Parse entries filtered by heuristic
    probes = {}  # load -> bp_mean
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            config = obj.get("config", {})
            if config.get("path_heuristic") != heuristic:
                continue
            metrics = obj.get("metrics", {})
            bp = metrics.get("service_blocking_probability", {})
            load = config.get("load", 0)
            if load and isinstance(bp, dict) and "mean" in bp:
                probes[int(load)] = bp["mean"]

    if not probes:
        return None

    # Reuse Phase 1 bracket logic
    below = phase1.find_best_below(probes)
    above = phase1.find_best_above(probes)

    if below and above:
        return {
            "status": "ok",
            "load_low": below[0],
            "load_high": above[0],
            "bp_low": below[1],
            "bp_high": above[1],
            "probes": probes,
        }

    # Fallback: relax constraints
    below_any = [(l, b) for l, b in probes.items() if b < phase1.BP_TARGET]
    above_any = [(l, b) for l, b in probes.items() if b >= phase1.BP_TARGET]

    if below_any and above_any:
        best_below = max(below_any, key=lambda x: x[1])
        best_above = min(above_any, key=lambda x: x[1])
        return {
            "status": "ok_relaxed",
            "load_low": best_below[0],
            "load_high": best_above[0],
            "bp_low": best_below[1],
            "bp_high": best_above[1],
            "probes": probes,
        }

    return None


def run_single_probe(topology, load, heuristic, probe_file, timeout=12000):
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

    progress = ProgressTracker(len(topologies), "Phase 2")
    switched = 0

    for topo in topologies:
        name = topo["topology_name"]

        # Resume: skip if already selected AND load_ranges data matches
        if name in selection:
            selected_heur = selection[name]
            needs_rerun = (
                selected_heur != "ksp_ff"
                and name in ranges
                and ranges[name].get("heuristic") != selected_heur
            )
            if not needs_rerun:
                # Data is correct (ksp_ff default or already re-run), just backfill field
                if name in ranges and "heuristic" not in ranges[name]:
                    ranges[name]["heuristic"] = selected_heur
                    save_load_ranges(ranges)
                progress.item_done(progress.item_start(), "skipped")
                continue

            # Heuristic was switched but load_ranges not updated yet
            t_start = progress.item_start()
            print(progress.header(progress.processed + 1, name))

            # Try to extract bracket from existing heuristic_eval data
            heuristic_file = heuristic_dir / f"{name}.jsonl"
            existing = extract_bracket_from_jsonl(heuristic_file, selected_heur)

            if existing is not None:
                print(f"  Recomputed bracket from existing {selected_heur} data")
                existing["heuristic"] = selected_heur
                ranges[name] = existing
                save_load_ranges(ranges)

                if existing["status"] in ("ok", "ok_relaxed"):
                    print(f"  -> load_low={existing['load_low']} "
                          f"(BP={existing['bp_low']*100:.4f}%), "
                          f"load_high={existing['load_high']} "
                          f"(BP={existing['bp_high']*100:.4f}%)")
                else:
                    print(f"  -> Status: {existing['status']}")
                progress.item_done(t_start, "completed")
                print(f"  [wall={format_duration(progress._durations[-1])}]")
                continue

            # No existing data — re-run Phase 1
            print(f"  Re-running Phase 1 with {selected_heur} (no existing data)")

            # Rename old heuristic_eval data (keep for reference)
            archive = heuristic_dir / f"{name}_ksp_ff.jsonl"
            if heuristic_file.exists() and not archive.exists():
                heuristic_file.rename(archive)

            # Remove old load_ranges entry so discover_bracket runs fresh
            del ranges[name]
            save_load_ranges(ranges)

            phase1 = importlib.import_module("01_discover_load_ranges")
            result = phase1.discover_bracket(
                name, topo_stats[name], probe_dir, heuristic_dir,
                extra_flags={"path_heuristic": selected_heur},
            )
            result["heuristic"] = selected_heur
            ranges[name] = result
            save_load_ranges(ranges)

            if result["status"] in ("ok", "ok_relaxed"):
                print(f"  -> New bracket: load_low={result['load_low']} "
                      f"(BP={result['bp_low']*100:.4f}%), "
                      f"load_high={result['load_high']} "
                      f"(BP={result['bp_high']*100:.4f}%)")
            else:
                print(f"  -> Re-run status: {result['status']}")
            progress.item_done(t_start, "completed")
            print(f"  [wall={format_duration(progress._durations[-1])}]")
            continue

        # Need load_ranges entry from Phase 1
        if name not in ranges or ranges[name].get("status") == "failed":
            print(f"  Skipping {name}: no load range from Phase 1")
            progress.item_done(progress.item_start(), "failed")
            continue

        entry = ranges[name]
        load_high = entry.get("load_high", 0)
        bp_high = entry.get("bp_high")

        if load_high <= 0 or bp_high is None:
            # Default to ksp_ff for edge cases
            selection[name] = "ksp_ff"
            if name in ranges:
                ranges[name]["heuristic"] = "ksp_ff"
                save_load_ranges(ranges)
            save_heuristic_selection(selection)
            progress.item_done(progress.item_start(), "completed")
            continue

        t_start = progress.item_start()
        print(progress.header(progress.processed + 1, name))
        print(f"  KSP-FF at load={load_high}: BP={bp_high*100:.4f}%")

        # Run FF-KSP probe at load_high
        probe_file = probe_dir / f"{name}_ffksp_compare.jsonl"
        ff_ksp_bp = run_single_probe(name, load_high, "ff_ksp", probe_file)

        if ff_ksp_bp is None:
            print(f"  FF-KSP probe FAILED, defaulting to ksp_ff")
            selection[name] = "ksp_ff"
            ranges[name]["heuristic"] = "ksp_ff"
            save_load_ranges(ranges)
            save_heuristic_selection(selection)
            probe_file.unlink(missing_ok=True)
            progress.item_done(t_start, "failed")
            print(f"  [wall={format_duration(progress._durations[-1])}]")
            continue

        print(f"  FF-KSP at load={load_high}: BP={ff_ksp_bp*100:.4f}%")
        probe_file.unlink(missing_ok=True)

        if ff_ksp_bp < bp_high:
            print(f"  -> FF-KSP is BETTER ({(bp_high - ff_ksp_bp)*100:.4f}pp lower)")
            print(f"  -> Re-running Phase 1 with FF-KSP...")
            selection[name] = "ff_ksp"
            switched += 1

            # Rename old KSP-FF heuristic_eval data (keep for reference)
            old_heuristic_file = heuristic_dir / f"{name}.jsonl"
            ksp_ff_archive = heuristic_dir / f"{name}_ksp_ff.jsonl"
            if old_heuristic_file.exists():
                old_heuristic_file.rename(ksp_ff_archive)

            # Remove old load_ranges entry so discover_bracket runs fresh
            del ranges[name]
            save_load_ranges(ranges)

            # Re-run Phase 1 discovery with ff_ksp
            phase1 = importlib.import_module("01_discover_load_ranges")
            result = phase1.discover_bracket(
                name, topo_stats[name], probe_dir, heuristic_dir,
                extra_flags={"path_heuristic": "ff_ksp"},
            )
            result["heuristic"] = "ff_ksp"
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
            ranges[name]["heuristic"] = "ksp_ff"
            save_load_ranges(ranges)

        save_heuristic_selection(selection)
        progress.item_done(t_start, "completed")
        print(f"  [wall={format_duration(progress._durations[-1])}]")

    print(f"\n{progress.summary_line()}")
    print(f"Switched to FF-KSP: {switched} topologies")
    print(f"Selection saved to {RESULTS_DIR / 'heuristic_selection.json'}")


if __name__ == "__main__":
    main()
