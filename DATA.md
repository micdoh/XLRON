# Experimental data

The `experimental/` directory contains scripts that generate, evaluate, or plot
results for individual experiments and papers. Each experiment owns its own
`results/` and `figures/` subdirectories, which are **gitignored** so the
repository stays small.

## Layout

```
experimental/
├── JOCN2024/
│   ├── generate_data/         # scripts that produce the CSVs
│   ├── generate_plots/        # scripts that read results/ and write figures/
│   ├── results/               # gitignored — CSVs, JSONLs, raw run outputs
│   │   └── bounds/            # cut-set / reconfigurable-routing bounds data
│   └── figures/               # gitignored — generated PNGs
├── ISRSGN2025/
│   └── results/
├── OFC2025/
│   └── results/
├── JOCN_SI/
│   └── results/
├── capacity_bounds_survey/
│   └── results/
├── large_topologies/
│   └── results/
├── benchmarks/
│   └── results/
├── differentiable/
│   └── figures/
└── ...
```

Plotting scripts resolve data relative to their own location, e.g.
```python
csv_path = Path(__file__).resolve().parents[1] / 'results' / 'experiment_results_eval.csv'
```
so they work as long as the data sits in the experiment's `results/` directory.

## Reproducing the data

Each experiment is reproducible from the code in this repo. A typical run looks
like one of the commands documented in `CLAUDE.md` or the experiment-specific
README, e.g.

```bash
# JOCN2024 cut-set bounds
uv run python -m xlron.bounds.cutsets_bounds \
    --topology_name=nsfnet_deeprmsa_directed --env_type=rmsa \
    --link_resources=100 --k=50 --load=250 --continuous_operation \
    --max_requests=100000 --num_trials=10 \
    --CUTSET_EXHAUSTIVE --CUTSET_TOP_K=256 \
    --DATA_OUTPUT_FILE=experimental/JOCN2024/results/bounds/experiment_results_cutsets_bounds.jsonl
```

## Snapshot of published results

The exact data used in the publications is archived externally so figures can
be regenerated without re-running multi-hour sweeps.

* **Zenodo DOI**: _TBD — to be minted at publication time_
* **Contents**: per-paper tarballs (`JOCN2024_results.tar.gz`,
  `ISRSGN2025_results.tar.gz`, …) that unpack into the corresponding
  `experimental/<paper>/results/` directory.

To regenerate a paper's figures from the snapshot:

```bash
# replace <PAPER> with JOCN2024, ISRSGN2025, OFC2025, JOCN_SI
curl -L -o /tmp/<PAPER>_results.tar.gz <ZENODO_URL>
tar -xzf /tmp/<PAPER>_results.tar.gz -C experimental/<PAPER>/
uv run python experimental/<PAPER>/generate_plots/<script>.py
```
