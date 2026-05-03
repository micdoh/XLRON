# Reproducing the ECOC 2026 capacity bounds paper

This page documents how to reproduce the figures and tables from:

> **Doherty, M.\*, Deng, B.\*, Beghelli, A., Toni, L., Bayvel, P.** *Comparison of Dynamic Elastic Optical Network Capacity Bound Estimation Methods*.
> Submitted to ECOC 2026 (pending publication). \*Equal contribution.

The paper compares three methods of estimating the capacity of dynamically-operated elastic optical networks across the **118 real-world topologies from [TopologyBench](https://github.com/TopologyBench/Real-Topologies)**:

- **KSP** — K-shortest-path heuristics (KSP-FF and FF-KSP, picking the better of the two per topology) as a realistic on-line baseline.
- **CS** — Cut-set bound: each bottleneck cut is treated as one aggregated resource of `n_i` fibres per slot, removing the per-link routing constraint within a cut while still enforcing spectrum continuity across cuts. Top-10 and top-100 most congested cuts are evaluated.
- **RPD** — Resource-prioritized defragmentation: an omniscient controller that reassigns all active connections on would-be-blocking events, prioritising by slot-hops.

The capacity metric is the offered load (Erlangs) at 0.1% blocking probability.

All scripts live under [`experimental/capacity_bounds_survey/`](https://github.com/micdoh/XLRON/tree/main/experimental/capacity_bounds_survey).

---

## Run the full sweep

The sweep runs in six phases. The master entrypoint is resume-safe (each phase skips topologies that already have output files), so it can be killed and restarted at any time.

```bash
uv run python experimental/capacity_bounds_survey/run_all.py
```

Or run individual phases as standalone scripts:

| Phase | Script | What it does |
| --- | --- | --- |
| 1 | `01_discover_load_ranges.py` | Probe each topology to find the load range that brackets 0.1 % blocking. |
| 2 | `02_compare_heuristics.py` | KSP-FF vs FF-KSP across the load range; pick the winner per topology. Output: `results/heuristic_selection.json`. |
| 3 | `03_run_cutset_bounds.py` | CS bound with the top-10 and top-100 most congested cuts. Calls `xlron.bounds.cutsets_bounds`. |
| 3b | `03b_run_cutset_bounds_top1pct.py` | CS bound with the top-1 % of cuts (sensitivity check). |
| 4 | `04_run_rr_bounds.py` | RPD bound. Calls `xlron/bounds/reconfigurable_routing_bounds.py`. |
| 5 | `05_run_k_sensitivity.py` | KSP K-sensitivity sweep on the larger topologies (K = 25, 50, 75, 100). |
| 6 | `06_analyze_results.py` | Interpolate every method's bp-vs-load curve to find the load at 0.1 % blocking, write `results/summary_table.csv`, and generate the figures. |

`config.py` defines the topology list and per-method defaults; `warmup_ksp_cache.py` precomputes the K-shortest-path cache so the bounds runs don't pay that cost.

> **Compute requirement.** This is a heavy sweep — 118 topologies × 4 methods (KSP-FF, KSP-FF/FF-KSP winner, CS-top10, CS-top100, RPD) × multiple loads each, on top of the K-sensitivity sub-sweep. Expect tens of GPU-hours on an A100 even with the resume-safe restart logic. Single topologies are individually reproducible by editing `config.get_topology_list()`.

---

## Plot the figures

`06_analyze_results.py` writes plots to `experimental/capacity_bounds_survey/results/figures/`. The three figures used in the paper are:

| Paper figure | Plot file |
| --- | --- |
| Heuristic-selection summary (which of KSP-FF / FF-KSP wins per topology) | `heuristic_selection.png` |
| Bound gap scatter (Δ % over KSP for CS-top10, CS-top100, RPD) | `bounds_gap_scatter.png` |
| All bounds normalised by edge count | `all_bounds_normalized_by_edges.png` |

Copies of the rendered PNGs that ship with the manuscript live alongside the LaTeX source in the [ECOC-2026-Capacity-Bounds](https://github.com/micdoh/ECOC-2026-Capacity-Bounds) sister repository.

---

## Headline numbers

From `summary_table.csv` aggregated over 118 topologies:

| Method | Median Δ over KSP at 0.1 % blocking |
| --- | ---: |
| Cut-set, top-10 cuts | **+9.4 %** |
| Cut-set, top-100 cuts | **+5.8 %** |
| Resource-prioritised defragmentation | **+17.0 %** |

The larger RPD gap implies that **spectrum fragmentation is a bigger source of lost capacity than routing inefficiency** — spectrum-allocation is the higher-leverage optimisation target. CS can underestimate (negative Δ) on a small number of large topologies for the mathematical reason set out in [`cutsets_explanation.tex`](https://github.com/micdoh/ECOC-2026-Capacity-Bounds/blob/main/cutsets_explanation.tex) (feasibility scales as $\bar p^T$ across $T$ traversed cuts vs $\bar q^H$ across $H$ physical hops; on large topologies $T \gg H$).

See [Capacity Bound Estimation](capacity_bounds.md) for the underlying XLRON estimator implementations (`xlron.bounds.cutsets_bounds`, `xlron.bounds.reconfigurable_routing_bounds`) and example single-topology commands.

---

## Citation

Pending publication. Until the proceedings appear, please use:

```bibtex
@unpublished{doherty_deng_capacity_bounds_2026,
  author  = {Doherty, Michael and Deng, Bohua and Beghelli, Alejandra and Toni, Laura and Bayvel, Polina},
  title   = {Comparison of Dynamic Elastic Optical Network Capacity Bound Estimation Methods},
  note    = {Submitted to ECOC 2026},
  year    = {2026}
}
```
