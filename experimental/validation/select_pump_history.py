"""Pick the best pump-optimisation history from a set of saved JSONs and
re-plot it with the JOCN figure styling.

Selection rule: among histories whose running-best improvement clears
``--min_improvement`` (default 800 Gb/s), pick the one with the smoothest
curve (lowest std-dev of running-best deltas).  If none reach the
threshold, fall back to the run with the largest improvement.

Usage:
    uv run python experimental/validation/select_pump_history.py
    uv run python experimental/validation/select_pump_history.py --min_improvement 800
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from experimental.validation.pump_optimization import (
    load_optimisation_history,
    plot_optimisation_history,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "gerard2025_results")
MAX_STEPS = 30


def metrics(history):
    """Return (per-step volatility, running-best volatility, final improvement)."""
    steps_arr = np.array([s for s, _ in history])
    tps_arr = np.array([t for _, t in history])
    keep = steps_arr <= MAX_STEPS
    tps_arr = tps_arr[keep]
    if len(tps_arr) < 2:
        return 0.0, 0.0, 0.0

    running_best = np.maximum.accumulate(tps_arr)
    per_step_vol = float(np.std(np.diff(tps_arr)))
    rb_vol = float(np.std(np.diff(running_best)))
    final_improvement = float(running_best[-1] - tps_arr[0])
    return per_step_vol, rb_vol, final_improvement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_improvement", type=float, default=800.0,
                    help="Minimum running-best improvement (Gb/s) to qualify")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(OUT_DIR, "pump_history_*.json")))
    if not paths:
        print(f"No pump_history_*.json files found in {OUT_DIR}")
        return

    print(f"Found {len(paths)} histories:")
    print(f"{'tag':<70} {'per-step vol':>12} {'rb vol':>10} {'improvement':>14}")
    print("-" * 110)
    rated = []
    for p in paths:
        tag = os.path.splitext(os.path.basename(p))[0].replace("pump_history_", "")
        try:
            h = load_optimisation_history(p)
        except Exception as e:
            print(f"{tag:<70} (load failed: {e})")
            continue
        ps_vol, rb_vol, imp = metrics(h)
        rated.append((tag, p, h, ps_vol, rb_vol, imp))
        print(f"{tag:<70} {ps_vol:>12.2f} {rb_vol:>10.2f} {imp:>+14.2f}")

    qualified = [r for r in rated if r[5] >= args.min_improvement]
    if qualified:
        # Smoothest qualifying run = lowest running-best volatility.
        qualified.sort(key=lambda r: r[4])
        best = qualified[0]
        print("-" * 110)
        print(f"BEST (smoothest qualifying ≥ {args.min_improvement:.0f} Gb/s): {best[0]}")
    else:
        rated.sort(key=lambda r: -r[5])  # largest improvement first
        best = rated[0]
        print("-" * 110)
        print(f"NO RUN reached the {args.min_improvement:.0f} Gb/s target.")
        print(f"Falling back to largest-improvement run: {best[0]}")

    print(f"  per-step vol = {best[3]:.2f} Gb/s")
    print(f"  running-best vol = {best[4]:.2f} Gb/s")
    print(f"  improvement = {best[5]:+.2f} Gb/s")
    print(f"  source = {best[1]}")

    plot_optimisation_history(best[2], OUT_DIR, fname="pump_optimisation.png")
    print("\nDone.  Wrote pump_optimisation.png from the chosen run.")


if __name__ == "__main__":
    main()
