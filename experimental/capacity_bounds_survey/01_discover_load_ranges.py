"""Phase 1: Discover two load points bracketing 0.1% blocking for each topology.

Uses load sweeps (~10 loads per compilation) instead of individual probes
to minimize JAX compilation overhead. Each sweep compiles once, then runs
all load points sequentially. Typically finds brackets in 1 sweep.

The sweep results double as the final heuristic evaluation data — all load
points are saved to heuristic_eval/ for interpolation in Phase 5.

Results are saved incrementally to load_ranges.json for resume support.
"""

import sys
from pathlib import Path

from config import (
    QUICK_SCAN_PARAMS,
    RESULTS_DIR,
    build_command,
    estimate_initial_load,
    get_topology_list,
    load_load_ranges,
    parse_jsonl_blocking,
    run_command,
    save_load_ranges,
)


# 0.1% blocking = 0.001 as a fraction
BP_TARGET = 0.001

# Acceptable ranges for the two bracket points (as fractions)
BP_BELOW_MIN = 0.0001   # 0.01%
BP_BELOW_MAX = 0.001    # 0.1%  (exclusive — must be strictly below target)
BP_ABOVE_MIN = 0.001    # 0.1%  (inclusive — at or above target)
BP_ABOVE_MAX = 0.01     # 1%

MAX_SWEEPS = 3
N_POINTS = 10
MIN_LOAD = 2
MAX_LOAD = 50000


def run_sweep(topology, sweep_min, sweep_max, step, sweep_file, timeout=3600, extra_flags=None):
    """Run a heuristic eval load sweep. Returns list of {load, blocking_mean, ...}."""
    flags = {
        **QUICK_SCAN_PARAMS,
        "EVAL_HEURISTIC": True,
        "load": sweep_min,
        "min_load": sweep_min,
        "max_load": sweep_max,
        "step_load": step,
    }
    if extra_flags:
        flags.update(extra_flags)
    cmd = build_command(
        script="xlron.train.train",
        topology=topology,
        extra_flags=flags,
        output_file=str(sweep_file),
    )

    result = run_command(cmd, timeout=timeout)
    if result.returncode != 0:
        return None

    return parse_jsonl_blocking(sweep_file)


def find_best_below(probes: dict) -> tuple[int, float] | None:
    """Find the probe closest to (but below) 0.1% blocking."""
    candidates = [(l, b) for l, b in probes.items()
                   if BP_BELOW_MIN <= b < BP_BELOW_MAX]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])


def find_best_above(probes: dict) -> tuple[int, float] | None:
    """Find the probe closest to (but at/above) 0.1% blocking."""
    candidates = [(l, b) for l, b in probes.items()
                   if BP_ABOVE_MIN <= b <= BP_ABOVE_MAX]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])


def compute_sweep_range(sweep_num, initial_load, all_probes):
    """Determine the min/max for the next sweep based on results so far."""
    if sweep_num == 0:
        # Wide initial sweep: 0.3x to 3x estimated load
        sweep_min = max(MIN_LOAD, round(initial_load * 0.3))
        sweep_max = min(MAX_LOAD, round(initial_load * 3.0))
        return sweep_min, sweep_max

    # Targeted sweep based on previous results
    sorted_probes = sorted(all_probes.items())
    bps = [b for _, b in sorted_probes]
    loads = [l for l, _ in sorted_probes]

    if all(b < BP_BELOW_MIN for b in bps):
        # All too low — need much higher loads
        return max(loads), min(MAX_LOAD, max(loads) * 5)

    if all(b > BP_ABOVE_MAX for b in bps):
        # All too high — need much lower loads
        return max(MIN_LOAD, round(min(loads) / 5)), min(loads)

    # Mixed results — zoom into the transition region
    below_target = [l for l, b in all_probes.items() if b < BP_TARGET]
    above_target = [l for l, b in all_probes.items() if b >= BP_TARGET]

    if below_target and above_target:
        lo = max(below_target)
        hi = min(above_target)
        margin = max(1, round((hi - lo) * 0.3))
        return max(MIN_LOAD, lo - margin), min(MAX_LOAD, hi + margin)

    # Only one side — extend in the missing direction
    if not above_target:
        return max(loads), min(MAX_LOAD, max(loads) * 3)
    return max(MIN_LOAD, round(min(loads) / 3)), min(loads)


def discover_bracket(topology, stats, probe_dir, heuristic_dir, extra_flags=None):
    """Find two loads that bracket 0.1% blocking using load sweeps.

    Saves ALL sweep results to heuristic_eval/ for use as final data.
    Returns dict with status, load_low, load_high, bp_low, bp_high, probes.

    Args:
        extra_flags: Optional dict of flags to override defaults (e.g. path_heuristic).
    """
    initial_load = estimate_initial_load(stats)
    print(f"  Initial load estimate: {initial_load} Erlang")

    heuristic_file = heuristic_dir / f"{topology}.jsonl"
    all_probes = {}  # load -> bp
    sweep_files = []

    for sweep_num in range(MAX_SWEEPS):
        sweep_min, sweep_max = compute_sweep_range(
            sweep_num, initial_load, all_probes
        )

        if sweep_max <= sweep_min:
            sweep_max = sweep_min + N_POINTS

        step = max(round((sweep_max - sweep_min) / (N_POINTS - 1)), 1)
        sweep_max = sweep_min + step * (N_POINTS - 1)  # exact N_POINTS

        print(f"  Sweep {sweep_num + 1}: [{sweep_min}, {sweep_max}] step={step}")

        sweep_file = probe_dir / f"{topology}_sweep{sweep_num}.jsonl"
        entries = run_sweep(topology, sweep_min, sweep_max, step, sweep_file, extra_flags=extra_flags)

        if entries is None:
            break

        sweep_files.append(sweep_file)
        for e in entries:
            all_probes[e["load"]] = e["blocking_mean"]
            print(f"    load={e['load']:>6} -> BP={e['blocking_mean']*100:.4f}%")

        # Check for brackets
        below = find_best_below(all_probes)
        above = find_best_above(all_probes)

        if below and above:
            _save_heuristic_results(heuristic_file, sweep_files)
            return {
                "status": "ok",
                "load_low": below[0],
                "load_high": above[0],
                "bp_low": below[1],
                "bp_high": above[1],
                "probes": {int(k): v for k, v in all_probes.items()},
            }

    # Save whatever data we have (useful for interpolation even without bracket)
    _save_heuristic_results(heuristic_file, sweep_files)

    if not all_probes:
        return {"status": "failed", "probes": {}}

    # Fallback: relax constraints — use any two probes that bracket BP_TARGET
    below_any = [(l, b) for l, b in all_probes.items() if b < BP_TARGET]
    above_any = [(l, b) for l, b in all_probes.items() if b >= BP_TARGET]

    if below_any and above_any:
        best_below = max(below_any, key=lambda x: x[1])
        best_above = min(above_any, key=lambda x: x[1])
        return {
            "status": "ok_relaxed",
            "load_low": best_below[0],
            "load_high": best_above[0],
            "bp_low": best_below[1],
            "bp_high": best_above[1],
            "probes": {int(k): v for k, v in all_probes.items()},
        }

    # Can't bracket at all
    sorted_probes = sorted(all_probes.items())
    if all(b < BP_BELOW_MIN for _, b in sorted_probes):
        return {"status": "ultra-high-capacity",
                "load_low": sorted_probes[-1][0],
                "load_high": round(sorted_probes[-1][0] * 1.5),
                "probes": {int(k): v for k, v in all_probes.items()}}
    if all(b > BP_ABOVE_MAX for _, b in sorted_probes):
        return {"status": "ultra-low-capacity",
                "load_low": max(MIN_LOAD, round(sorted_probes[0][0] * 0.5)),
                "load_high": sorted_probes[0][0],
                "probes": {int(k): v for k, v in all_probes.items()}}

    return {"status": "failed", "probes": {int(k): v for k, v in all_probes.items()}}


def _save_heuristic_results(heuristic_file, sweep_files):
    """Concatenate all sweep JSONL files into the heuristic eval output."""
    with open(heuristic_file, "w") as f_out:
        for sf in sweep_files:
            if sf.exists():
                with open(sf) as f_in:
                    f_out.write(f_in.read())


def main():
    print("=" * 60)
    print("Phase 1: Discovering load brackets for 0.1% blocking")
    print("=" * 60)

    topologies = get_topology_list()
    ranges = load_load_ranges()
    probe_dir = RESULTS_DIR / "probes"
    heuristic_dir = RESULTS_DIR / "heuristic_eval"
    probe_dir.mkdir(parents=True, exist_ok=True)
    heuristic_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0

    for topo in topologies:
        name = topo["topology_name"]

        heuristic_file = heuristic_dir / f"{name}.jsonl"
        if name in ranges and heuristic_file.exists() and heuristic_file.stat().st_size > 0:
            skipped += 1
            continue

        print(f"\n[{completed + skipped + failed + 1}/{len(topologies)}] {name} "
              f"(nodes={topo['num_nodes']}, edges={topo['num_edges']})")

        result = discover_bracket(name, topo, probe_dir, heuristic_dir)
        ranges[name] = result
        save_load_ranges(ranges)

        if result["status"] in ("ok", "ok_relaxed"):
            print(f"  -> load_low={result['load_low']} (BP={result['bp_low']*100:.4f}%), "
                  f"load_high={result['load_high']} (BP={result['bp_high']*100:.4f}%)")
            completed += 1
        elif result["status"] in ("ultra-low-capacity", "ultra-high-capacity"):
            print(f"  -> {result['status']}: [{result.get('load_low', '?')}, {result.get('load_high', '?')}]")
            completed += 1
        else:
            print(f"  -> FAILED")
            failed += 1

    print(f"\nDone: {completed} discovered, {skipped} skipped (already done), {failed} failed")
    print(f"Results saved to {RESULTS_DIR / 'load_ranges.json'}")
    print(f"Heuristic eval results saved to {heuristic_dir}/")


if __name__ == "__main__":
    main()
