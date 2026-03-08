"""Phase 4: Run reconfigurable routing bounds with adaptive load selection.

Strategy:
1. Single-trial probes starting at heuristic's load_high, incrementing ~10%.
2. Any probe with blocking > 0% but <= 1% is selected for a full 10-trial run.
3. If a probe shows blocking > 1%, scale back (too expensive due to defrag).
4. Stop probing once blocking exceeds 0.1% (TARGET_BP).
5. Refinement phase: binary search to ensure bracketing of TARGET_BP.
6. Full 10-trial runs at selected loads produce the actual data.

Retry mode (--retry): re-processes topologies whose existing output
doesn't have valid interpolation at TARGET_BP.

Resume-safe: probe state is persisted to a sidecar JSON file after each
probe. If interrupted and restarted, probing resumes from where it left off.
Full 10-trial results are appended to the output file incrementally.
"""

import json

import numpy as np
from scipy import interpolate as scipy_interpolate

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
MAX_REFINE_PROBES = 8
MAX_RETRY_ROUNDS = 10  # Max probe→refine→fullrun cycles per topology in retry mode
LOAD_INCREMENT = 1.10  # 10% increment between probes
SCALE_BACK_FACTOR = 0.95  # 5% reduction when blocking > 1%
MAX_BLOCKING_FOR_FULL_RUN = 0.01  # 1% - don't do full runs above this
MIN_REFINE_STEP_FRAC = 0.05  # Minimum 5% step between refinement probes


def _probe_state_path(probe_dir, name):
    """Path to the sidecar JSON that persists probe results across interruptions."""
    return probe_dir / f"{name}_probe_state.json"


def save_probe_state(probe_dir, name, probe_results, completed_full_loads=None):
    """Persist probe results (and which full runs are done) to a sidecar JSON.

    Called after each probe/full-run so progress survives interruption.
    """
    state = {
        "probe_results": {str(k): v for k, v in probe_results.items()},
    }
    if completed_full_loads is not None:
        state["completed_full_loads"] = sorted(completed_full_loads)
    path = _probe_state_path(probe_dir, name)
    # Write atomically: write to temp then rename
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(path)


def load_probe_state(probe_dir, name):
    """Load persisted probe state. Returns (probe_results, completed_full_loads) or (None, None)."""
    path = _probe_state_path(probe_dir, name)
    if not path.exists():
        return None, None
    try:
        with open(path) as f:
            state = json.load(f)
        probe_results = {int(k): v for k, v in state["probe_results"].items()}
        completed_full_loads = set(state.get("completed_full_loads", []))
        return probe_results, completed_full_loads
    except (json.JSONDecodeError, KeyError, ValueError):
        return None, None


def clear_probe_state(probe_dir, name):
    """Remove the sidecar JSON after successful completion."""
    path = _probe_state_path(probe_dir, name)
    path.unlink(missing_ok=True)


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


def has_valid_interpolation(entries, target_bp=TARGET_BP):
    """Check if a list of {load, blocking_mean} entries can interpolate to target_bp."""
    if len(entries) < 2:
        return False
    bps = np.array([e["blocking_mean"] for e in entries])
    valid = bps > 0
    if valid.sum() < 2:
        return False
    bps_valid = bps[valid]
    return bps_valid.min() < target_bp < bps_valid.max()


def categorize_probes(probe_results):
    """Categorize probe results into below/above target and zero/too-high groups.

    Args:
        probe_results: dict mapping load -> bp

    Returns:
        (zero_loads, below_target, above_target, too_high) as sorted lists of loads.
    """
    zero_loads = sorted(l for l, bp in probe_results.items() if bp == 0)
    below_target = sorted(l for l, bp in probe_results.items() if 0 < bp < TARGET_BP)
    above_target = sorted(l for l, bp in probe_results.items()
                          if TARGET_BP <= bp <= MAX_BLOCKING_FOR_FULL_RUN)
    too_high = sorted(l for l, bp in probe_results.items() if bp > MAX_BLOCKING_FOR_FULL_RUN)
    return zero_loads, below_target, above_target, too_high


def refine_probes(name, probe_results, heuristic, probe_dir, save_fn=None):
    """Do targeted probes to bracket TARGET_BP.

    Uses a two-phase strategy:
    1. **Gallop**: from zero-blocking loads, take big multiplicative jumps
       (LOAD_INCREMENT) upward to quickly find any non-zero blocking.
    2. **Narrow**: binary search between the last zero and the first
       non-zero to bracket TARGET_BP precisely.

    Modifies probe_results in place. Returns True if any probes were run.
    save_fn: optional callback called after each successful probe to persist state.
    """
    refine_num = 0

    def do_probe(load):
        nonlocal refine_num
        load = round(load)
        if load in probe_results or load <= 0:
            return probe_results.get(load)
        refine_num += 1
        pfile = probe_dir / f"{name}_refine{refine_num}.jsonl"
        print(f"  Refine {refine_num}: load={load}", end=" ", flush=True)
        bp = run_rr(name, load, pfile, heuristic=heuristic, num_trials=1)
        pfile.unlink(missing_ok=True)
        if bp is None:
            print("-> FAILED")
            return None
        print(f"-> BP={bp*100:.4f}%")
        probe_results[load] = bp
        if save_fn:
            save_fn()
        return bp

    def gallop_up(start):
        """Gallop upward from start by LOAD_INCREMENT until we find non-zero blocking.

        Takes big multiplicative jumps (10% each). No ceiling — if previous
        1-trial probes showed above-target blocking at some load, gallop may
        jump past it. If it lands on a cached load, do_probe returns the
        cached result and gallop stops naturally. If it jumps over it,
        it finds the real transition at a wider spacing.

        Returns (last_zero, first_nonzero_load) or (last_zero, None) if
        probes exhausted.
        """
        current = start
        last_zero = start
        for _ in range(MAX_REFINE_PROBES):
            current = round(current * LOAD_INCREMENT)
            bp = do_probe(current)
            if bp is None:
                return last_zero, None
            if bp > 0:
                return last_zero, current
            last_zero = current
        return last_zero, None

    def narrow_down(lo, hi):
        """Binary search between lo (zero/below-target) and hi (above-target/non-zero).

        Stops when a below-target point is found or the gap is exhausted.
        Uses MIN_REFINE_STEP_FRAC to avoid tiny steps.
        """
        for _ in range(MAX_REFINE_PROBES):
            min_step = max(round(lo * MIN_REFINE_STEP_FRAC), 1)
            if hi - lo <= min_step:
                break  # Gap too small to subdivide further
            mid = round((lo + hi) / 2)
            # Enforce minimum step from lo
            mid = max(mid, lo + min_step)
            mid = min(mid, hi - 1)
            if mid <= lo or mid >= hi:
                break
            bp = do_probe(mid)
            if bp is None:
                break
            if bp == 0:
                lo = mid
            elif bp < TARGET_BP:
                break  # Found a below-target point — bracket achieved
            elif bp <= MAX_BLOCKING_FOR_FULL_RUN:
                hi = mid  # Above target but usable — tighten upper bound
            else:
                hi = mid  # Too high (>1%) — tighten upper bound

    zero_loads, below_target, above_target, too_high = categorize_probes(probe_results)
    has_below = len(below_target) > 0
    has_above = len(above_target) > 0

    # Case A: Have above-target loads but no below-target (need lower bracket)
    if has_above and not has_below:
        lowest_above = min(above_target)
        if not zero_loads:
            # No zero loads — search downward from above-target to find zero/below
            base = lowest_above
            for frac in [0.90, 0.80, 0.70, 0.60, 0.50]:
                load = round(base * frac)
                bp = do_probe(load)
                if bp is None:
                    break
                if bp == 0:
                    break  # Found zero — gallop will handle it
                if 0 < bp < TARGET_BP:
                    break  # Found below-target — bracket done
            zero_loads, below_target, above_target, too_high = categorize_probes(probe_results)

        # Gallop up from highest zero, then narrow to find below-target
        if zero_loads and not below_target:
            last_zero, first_nonzero = gallop_up(max(zero_loads))
            zero_loads, below_target, above_target, too_high = categorize_probes(probe_results)
            # Narrow between highest zero and lowest non-zero (of any kind)
            nonzero = above_target + too_high
            if not below_target and zero_loads and nonzero:
                narrow_down(max(zero_loads), min(nonzero))

    # Re-categorize after Case A
    zero_loads, below_target, above_target, too_high = categorize_probes(probe_results)
    has_below = len(below_target) > 0
    has_above = len(above_target) > 0

    # Case B: Have below-target loads but no above-target (need upper bracket)
    if has_below and not has_above:
        highest_below = max(below_target)
        current = highest_below
        for _ in range(MAX_REFINE_PROBES):
            current = round(current * LOAD_INCREMENT)
            bp = do_probe(current)
            if bp is None:
                break
            if bp >= TARGET_BP:
                break

    # Re-categorize after Case B
    zero_loads, below_target, above_target, too_high = categorize_probes(probe_results)
    has_below = len(below_target) > 0
    has_above = len(above_target) > 0

    # Case C: Only zeros and too-high (>1%) — gallop up then narrow
    if not has_below and not has_above and zero_loads and too_high:
        last_zero, first_nonzero = gallop_up(max(zero_loads))
        zero_loads, below_target, above_target, too_high = categorize_probes(probe_results)
        # Narrow between highest zero and lowest usable non-zero
        upper_bound = above_target + too_high
        if zero_loads and upper_bound and not (below_target and above_target):
            narrow_down(max(zero_loads), min(upper_bound))

    return refine_num > 0


def main(retry=False):
    label = "Phase 4b: Retrying RR bounds" if retry else "Phase 4: Running RR bounds (adaptive load selection)"
    print("=" * 60)
    print(label)
    print("=" * 60)

    topologies = get_topology_list()
    ranges = load_load_ranges()
    heur_selection = load_heuristic_selection()
    output_dir = RESULTS_DIR / "rr_bounds"
    probe_dir = output_dir / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressTracker(len(topologies), "Phase 4b" if retry else "Phase 4")

    for topo in topologies:
        name = topo["topology_name"]

        output_file = output_dir / f"{name}.jsonl"

        # ---------------------------------------------------------
        # Check for persisted probe state from a previous interrupted run
        # ---------------------------------------------------------
        saved_probes, saved_full_loads = load_probe_state(probe_dir, name)
        has_saved_state = saved_probes is not None

        # ---------------------------------------------------------
        # Skip / retry logic
        # ---------------------------------------------------------
        if has_saved_state:
            # Interrupted previous run — resume from saved probe state.
            # Also merge actual full-run results from the output file, since
            # 10-trial values are more accurate than the original 1-trial probes.
            if output_file.exists() and output_file.stat().st_size > 0:
                for e in parse_jsonl_blocking(output_file):
                    saved_probes[e["load"]] = e["blocking_mean"]
                    saved_full_loads.add(e["load"])
            print(f"\n  Resuming {name}: found saved probe state "
                  f"({len(saved_probes)} probes, {len(saved_full_loads)} full runs done)")
        elif output_file.exists() and output_file.stat().st_size > 0:
            if retry:
                # Check if existing output has valid interpolation
                existing = parse_jsonl_blocking(output_file)
                if has_valid_interpolation(existing):
                    progress.item_done(progress.item_start(), "skipped")
                    continue
                # Invalid interpolation — load existing results as seed data
                print(f"\n  Retrying {name}: existing output lacks valid interpolation")
                saved_probes = {e["load"]: e["blocking_mean"] for e in existing}
                # Mark existing loads as already having full-run data so we
                # don't repeat the same 10-trial runs on every retry.
                saved_full_loads = set(e["load"] for e in existing)
                # Keep existing output file — new full runs will be appended.
                save_probe_state(probe_dir, name, saved_probes, saved_full_loads)
            else:
                progress.item_done(progress.item_start(), "skipped")
                continue
        else:
            saved_probes = None
            saved_full_loads = set()

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

        # Track all probe results: load -> bp
        probe_results = dict(saved_probes) if saved_probes else {}
        completed_full_loads = set(saved_full_loads) if saved_full_loads else set()

        # Helper to persist state after each probe
        def _save():
            save_probe_state(probe_dir, name, probe_results, completed_full_loads)

        # =============================================================
        # Phase 1: Cheap single-trial probes to find loads with blocking
        # =============================================================
        # Skip Phase 1 if we already have probe data with non-zero blocking
        # (from saved state or retry seed). If all probes are zero, Phase 1
        # must run to find loads where blocking actually occurs.
        has_nonzero_probes = (saved_probes is not None and len(saved_probes) > 0
                              and any(bp > 0 for bp in saved_probes.values()))
        skip_phase1 = has_nonzero_probes

        if not skip_phase1:
            print("  --- Phase 1: Single-trial probes ---")
            # Start above the highest known zero-blocking load (if any),
            # since those loads are already known to have no blocking.
            if probe_results:
                max_zero = max((l for l, bp in probe_results.items() if bp == 0), default=0)
                current_load = max(load_high, round(max_zero * LOAD_INCREMENT))
            else:
                current_load = load_high
            probe_num = 0
            phase1_failed = False

            while probe_num < MAX_PROBES:
                if current_load in probe_results:
                    current_load = round(current_load * 1.05)
                    if current_load in probe_results:
                        break
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
                probe_results[current_load] = bp
                _save()

                if bp > MAX_BLOCKING_FOR_FULL_RUN:
                    print(f"    Blocking > 1%, scaling back")
                    current_load = max(round(current_load * SCALE_BACK_FACTOR), 1)
                    continue

                if bp >= TARGET_BP:
                    print(f"    Found load above {TARGET_BP*100:.1f}% target")
                    break

                current_load = round(current_load * LOAD_INCREMENT)

            if phase1_failed and not any(bp > 0 for bp in probe_results.values()):
                progress.item_done(t_start, "failed")
                print(f"  [wall={format_duration(progress._durations[-1])}]")
                continue

        # =============================================================
        # Probe → Refine → Full-run loop
        # In retry mode, repeat until output has valid interpolation.
        # In normal mode, run once.
        # =============================================================
        max_rounds = MAX_RETRY_ROUNDS if retry else 1
        for round_num in range(max_rounds):
            if round_num > 0:
                print(f"  --- Round {round_num + 1}: re-probing after full-run results ---")
                # Update probe_results with actual full-run values from the
                # output file, since they supersede stale 1-trial probes.
                if output_file.exists() and output_file.stat().st_size > 0:
                    for e in parse_jsonl_blocking(output_file):
                        probe_results[e["load"]] = e["blocking_mean"]

            # ---------------------------------------------------------
            # Refinement probes to ensure bracketing of TARGET_BP
            # ---------------------------------------------------------
            _, below_target, above_target, _ = categorize_probes(probe_results)
            needs_refine = not (below_target and above_target)

            if needs_refine and any(bp > 0 for bp in probe_results.values()):
                print("  --- Refinement probes (bracketing) ---")
                refine_probes(name, probe_results, best_heuristic, probe_dir, save_fn=_save)

            # Build selected_loads from all probes with 0 < bp <= MAX_BLOCKING
            selected_loads = sorted(
                l for l, bp in probe_results.items()
                if 0 < bp <= MAX_BLOCKING_FOR_FULL_RUN
            )

            if not selected_loads:
                any_blocking = any(bp > 0 for bp in probe_results.values())
                if not any_blocking:
                    print(f"  WARNING: No loads with blocking > 0% found")
                else:
                    print(f"  WARNING: All non-zero blocking > {MAX_BLOCKING_FOR_FULL_RUN*100:.0f}%")
                break  # Can't make progress — exit the round loop

            # ---------------------------------------------------------
            # Full 10-trial runs at selected loads
            # ---------------------------------------------------------
            remaining_loads = [l for l in selected_loads if l not in completed_full_loads]
            if remaining_loads:
                print(f"  --- Full 10-trial runs at {len(remaining_loads)} load(s): {remaining_loads} ---")
                if completed_full_loads:
                    print(f"    (skipping {len(completed_full_loads)} already-completed load(s))")
            else:
                if round_num == 0:
                    print(f"  --- All {len(selected_loads)} full runs already completed ---")

            for load in remaining_loads:
                full_file = probe_dir / f"{name}_full_{load}.jsonl"
                print(f"  Full run: load={load} (10 trials)", end=" ", flush=True)
                bp = run_rr(name, load, full_file, heuristic=best_heuristic, num_trials=10)

                if bp is not None:
                    print(f"-> BP={bp*100:.4f}%")
                    timing_str = format_timing_breakdown(full_file)
                    if timing_str:
                        print(f"    [{timing_str}]")
                    with open(output_file, "a") as f_out:
                        with open(full_file) as f_in:
                            for line in f_in:
                                f_out.write(line)
                    full_file.unlink(missing_ok=True)
                    # Update probe_results with the actual full-run value
                    probe_results[load] = bp
                    completed_full_loads.add(load)
                    _save()
                else:
                    print("-> FAILED")
                    full_file.unlink(missing_ok=True)

            # Check if we now have valid interpolation
            if output_file.exists() and output_file.stat().st_size > 0:
                final_entries = parse_jsonl_blocking(output_file)
                if has_valid_interpolation(final_entries):
                    break  # Success — exit the round loop

            if not remaining_loads:
                # No new full runs were done and still no valid interpolation.
                # Next round will re-probe with updated data.
                if round_num + 1 >= max_rounds:
                    break
                # Continue to next round — refinement will find new loads
                continue

        # Check final result
        if completed_full_loads:
            print(f"  -> Wrote {len(completed_full_loads)} load(s) to {output_file.name}")
            final_entries = parse_jsonl_blocking(output_file)
            if has_valid_interpolation(final_entries):
                clear_probe_state(probe_dir, name)
                progress.item_done(t_start, "completed")
            else:
                print(f"  -> Still lacks valid interpolation — probe state preserved for next retry")
                progress.item_done(t_start, "completed")
        else:
            print(f"  -> All full runs failed")
            # Keep probe state so next retry can build on accumulated probes
            progress.item_done(t_start, "failed")

        print(f"  [wall={format_duration(progress._durations[-1])}]")

    print(f"\n{progress.summary_line()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry", action="store_true",
                        help="Re-process topologies with invalid interpolation")
    args = parser.parse_args()
    main(retry=args.retry)
