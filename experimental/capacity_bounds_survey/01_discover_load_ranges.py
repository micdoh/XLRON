"""Phase 1: Discover two load points bracketing 0.1% blocking for each topology.

Uses binary search with quick heuristic evaluations to find:
  - load_low:  a load where blocking is in [0.01%, 0.1%)
  - load_high: a load where blocking is in [0.1%, 1%]

These two points are sufficient to interpolate the load at exactly 0.1% blocking.
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

MAX_ITERATIONS = 10
MIN_LOAD = 2
MAX_LOAD = 50000


def probe_blocking(topology: str, load: float, output_dir: Path) -> float | None:
    """Run a quick heuristic evaluation and return the mean blocking probability."""
    probe_file = output_dir / f"{topology}_probe_{int(load)}.jsonl"

    cmd = build_command(
        script="xlron.train.train",
        topology=topology,
        extra_flags={
            **QUICK_SCAN_PARAMS,
            "EVAL_HEURISTIC": True,
            "load": load,
        },
        output_file=str(probe_file),
    )

    result = run_command(cmd, timeout=1800)
    if result.returncode != 0:
        return None

    results = parse_jsonl_blocking(probe_file)
    if not results:
        return None

    return results[0]["blocking_mean"]


def discover_bracket(topology: str, stats: dict, output_dir: Path) -> dict:
    """Find two loads that bracket 0.1% blocking probability.

    Returns dict with:
        load_low:  load where BP is in [0.01%, 0.1%)
        load_high: load where BP is in [0.1%, 1%]
        status: "ok", "failed", "ultra-low-capacity", "ultra-high-capacity"
        probes: {load: bp} dict of all probes taken
    """
    initial_load = estimate_initial_load(stats)
    print(f"  Initial load estimate: {initial_load} Erlang")

    probes = {}  # load -> blocking_probability

    def do_probe(load: float) -> float | None:
        load = max(MIN_LOAD, min(MAX_LOAD, round(load)))
        if load in probes:
            return probes[load]
        print(f"  Probe: load={load}", end=" ", flush=True)
        bp = probe_blocking(topology, load, output_dir)
        if bp is None:
            print("-> FAILED")
            return None
        probes[load] = bp
        print(f"-> BP={bp*100:.4f}%")
        return bp

    def find_best_below() -> tuple[int, float] | None:
        """Find the probe closest to (but below) 0.1% blocking."""
        candidates = [(l, b) for l, b in probes.items()
                       if BP_BELOW_MIN <= b < BP_BELOW_MAX]
        if not candidates:
            return None
        # Pick the one with BP closest to target from below
        return max(candidates, key=lambda x: x[1])

    def find_best_above() -> tuple[int, float] | None:
        """Find the probe closest to (but at/above) 0.1% blocking."""
        candidates = [(l, b) for l, b in probes.items()
                       if BP_ABOVE_MIN <= b <= BP_ABOVE_MAX]
        if not candidates:
            return None
        # Pick the one with BP closest to target from above
        return min(candidates, key=lambda x: x[1])

    # Step 1: Initial probe to orient ourselves
    bp = do_probe(initial_load)
    if bp is None:
        return {"status": "failed", "probes": probes}

    # Step 2: Coarse search — get into the right ballpark
    for _ in range(MAX_ITERATIONS):
        if len(probes) >= MAX_ITERATIONS:
            break

        # Check if we already have both brackets
        below = find_best_below()
        above = find_best_above()
        if below and above:
            return {
                "status": "ok",
                "load_low": below[0],
                "load_high": above[0],
                "bp_low": below[1],
                "bp_high": above[1],
                "probes": {int(k): v for k, v in probes.items()},
            }

        # Find the probe closest to target to guide our next probe
        sorted_probes = sorted(probes.items())
        loads = [l for l, _ in sorted_probes]
        bps = [b for _, b in sorted_probes]

        if all(b < BP_BELOW_MIN for b in bps):
            # All probes too low — need much higher load
            next_load = max(loads) * 2
        elif all(b > BP_ABOVE_MAX for b in bps):
            # All probes too high — need much lower load
            next_load = min(loads) / 2
        elif not below:
            # Need a probe below 0.1% — find the lowest-BP probe and go lower
            lowest_bp_load = min(probes.items(), key=lambda x: x[1])[0]
            lowest_bp = probes[lowest_bp_load]
            if lowest_bp >= BP_TARGET:
                # All probes are above target, need to go lower
                next_load = lowest_bp_load * 0.7
            else:
                # Have something below target but not in [0.01%, 0.1%) range
                # It's below 0.01% — need to increase slightly
                # Find the closest probe above target for interpolation
                above_probes = [(l, b) for l, b in probes.items() if b >= BP_TARGET]
                if above_probes:
                    above_load = min(above_probes, key=lambda x: x[1])[0]
                    next_load = (lowest_bp_load + above_load) / 2
                else:
                    next_load = lowest_bp_load * 1.5
        elif not above:
            # Need a probe above 0.1% — find the highest-BP probe and go higher
            highest_bp_load = max(probes.items(), key=lambda x: x[1])[0]
            highest_bp = probes[highest_bp_load]
            if highest_bp < BP_TARGET:
                # All probes below target, need to go higher
                next_load = highest_bp_load * 1.5
            else:
                # Have something above target but not in [0.1%, 1%] range
                # It's above 1% — need to decrease slightly
                below_probes = [(l, b) for l, b in probes.items() if b < BP_TARGET]
                if below_probes:
                    below_load = max(below_probes, key=lambda x: x[1])[0]
                    next_load = (highest_bp_load + below_load) / 2
                else:
                    next_load = highest_bp_load * 0.7

        next_load = max(MIN_LOAD, min(MAX_LOAD, round(next_load)))
        if next_load in probes:
            # Nudge to avoid re-probing
            next_load = round(next_load * 1.05)
            if next_load in probes:
                next_load = round(next_load * 0.9)

        bp = do_probe(next_load)
        if bp is None:
            return {"status": "failed", "probes": {int(k): v for k, v in probes.items()}}

    # Final check — use best available even if not perfect brackets
    below = find_best_below()
    above = find_best_above()

    if below and above:
        return {
            "status": "ok",
            "load_low": below[0],
            "load_high": above[0],
            "bp_low": below[1],
            "bp_high": above[1],
            "probes": {int(k): v for k, v in probes.items()},
        }

    # Fallback: relax constraints — use any two probes that bracket BP_TARGET
    below_any = [(l, b) for l, b in probes.items() if b < BP_TARGET]
    above_any = [(l, b) for l, b in probes.items() if b >= BP_TARGET]

    if below_any and above_any:
        best_below = max(below_any, key=lambda x: x[1])
        best_above = min(above_any, key=lambda x: x[1])
        return {
            "status": "ok_relaxed",
            "load_low": best_below[0],
            "load_high": best_above[0],
            "bp_low": best_below[1],
            "bp_high": best_above[1],
            "probes": {int(k): v for k, v in probes.items()},
        }

    # Can't bracket at all
    sorted_probes = sorted(probes.items())
    if all(b < BP_BELOW_MIN for _, b in sorted_probes):
        return {"status": "ultra-high-capacity",
                "load_low": sorted_probes[-1][0],
                "load_high": round(sorted_probes[-1][0] * 1.5),
                "probes": {int(k): v for k, v in probes.items()}}
    if all(b > BP_ABOVE_MAX for _, b in sorted_probes):
        return {"status": "ultra-low-capacity",
                "load_low": max(MIN_LOAD, round(sorted_probes[0][0] * 0.5)),
                "load_high": sorted_probes[0][0],
                "probes": {int(k): v for k, v in probes.items()}}

    return {"status": "failed", "probes": {int(k): v for k, v in probes.items()}}


def main():
    print("=" * 60)
    print("Phase 1: Discovering load brackets for 0.1% blocking")
    print("=" * 60)

    topologies = get_topology_list()
    ranges = load_load_ranges()
    probe_dir = RESULTS_DIR / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0

    for topo in topologies:
        name = topo["topology_name"]

        if name in ranges:
            skipped += 1
            continue

        print(f"\n[{completed + skipped + failed + 1}/{len(topologies)}] {name} "
              f"(nodes={topo['num_nodes']}, edges={topo['num_edges']})")

        result = discover_bracket(name, topo, probe_dir)
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


if __name__ == "__main__":
    main()
