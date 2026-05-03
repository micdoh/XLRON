# Reproducing the *Reinforcement Learning: Hype or Hope?* JOCN paper

This page documents how to reproduce the figures and tables from:

> **Doherty, M., Beghelli, A.** *Reinforcement Learning for Dynamic Resource Allocation in Optical Networks: Hype or Hope?*
> Journal of Optical Communications and Networking **17**(9), D1 (2025).
> DOI: [10.1364/JOCN.559990](https://doi.org/10.1364/JOCN.559990) · [arXiv:2406.01919](https://arxiv.org/abs/2406.01919)

The paper is a systematic literature survey and benchmarking study covering ~100 RL-for-RMSA papers. Its main empirical finding is that the published RL solutions in five highly-cited benchmark settings are matched or beaten by simple KSP-FF / FF-KSP heuristics with a sufficient number of candidate paths and an appropriate path-sort criterion. It also introduces a defragmentation-based capacity bound estimator (reconfigurable routing) and uses cut-set bounds as an upper-bound capacity reference.

All scripts referenced here live under [`experimental/JOCN2024/`](https://github.com/micdoh/XLRON/tree/main/experimental/JOCN2024).

---

## Generate the data

The `generate_data/` directory contains seven shell scripts that wrap `xlron.train.train`, `xlron.bounds.cutsets_bounds`, and `xlron.bounds.reconfigurable_routing_bounds` for every topology / load / K combination used in the paper. Run them from the repository root with the project virtual environment active.

```bash
# Heuristic sweeps used for the bp-vs-load comparison plots
bash experimental/JOCN2024/generate_data/heuristic_evaluation.sh         # benchmark settings (KSP-FF k∈{5,50})
bash experimental/JOCN2024/generate_data/heuristic_evaluation_bounds.sh  # finer load sweeps used as benchmark baselines

# Heuristic comparison studies (Section 4 of the paper)
bash experimental/JOCN2024/generate_data/heuristic_comparison.sh         # 6 heuristics × 4 topologies × K∈{2..26}
bash experimental/JOCN2024/generate_data/heuristic_comparison_k.sh       # K-sweep used in path-stat analysis
bash experimental/JOCN2024/generate_data/heuristic_comparison_traffic.sh # SBP vs traffic load for each heuristic / topology

# Capacity bound estimates
bash experimental/JOCN2024/generate_data/run_cutsets_bounds.sh           # Cruzado-style cut-set bound (top-K = 256)
bash experimental/JOCN2024/generate_data/run_reconfigurable_routing_bounds.sh  # Defragmentation bound

# Path statistics (used by paths_lengths_hops_plots.py)
bash experimental/JOCN2024/generate_data/unique_paths.sh
```

Each script writes a JSONL file in the working directory; the plotting scripts expect those JSONL files to be aggregated to CSV and placed under `experimental/JOCN2024/results/` with the filenames hard-coded in `plot_heuristic_comparison.py` (e.g.\ `heuristic_comparison_high_traffic_new.csv`, `k_traffic_comparison_new_new.csv`, `experiment_results_traffic.csv`, `experiment_results_eval.csv`, `experiment_results_eval_bounds.csv`, `experiment_results_bounds.csv`, `experiment_results_unique_paths.csv`). A small JSONL→CSV concatenation step (e.g. `pandas.read_json(..., lines=True).to_csv(...)`) is required between the data-generation and plotting steps.

> **Compute requirement.** The full sweep is heavy: each `--EVAL_HEURISTIC` run uses `NUM_ENVS=200` (for bounds runs) or `NUM_ENVS=10` (for benchmark settings) on an A100 with 100k–2.6M total timesteps per point. The cut-set bound runs use `--CUTSET_EXHAUSTIVE --CUTSET_BATCH_SIZE=512 --CUTSET_TOP_K=256`. End-to-end the sweep takes a few hours on a single A100.

---

## Plot the figures

```bash
# Main bp-vs-load comparison (RL benchmarks vs heuristics vs bounds)
uv run python experimental/JOCN2024/generate_plots/plot_heuristic_comparison.py

# JPN48-specific plot (used in the paper's MaskRSA section)
uv run python experimental/JOCN2024/generate_plots/plot_jpn48.py

# Path-length / hop distributions for KSP-FF and FF-KSP at varying K
uv run python experimental/JOCN2024/generate_plots/paths_lengths_hops_plots.py
uv run python experimental/JOCN2024/generate_plots/plot_path_length_comparisons.py

# Literature-survey bar chart (review of ~100 RL papers)
uv run python experimental/JOCN2024/generate_plots/plot_literature_review_bar_chart.py

# Sensitivity studies on simulation choices
uv run python experimental/JOCN2024/generate_plots/plot_truncation.py   # truncate_holding_time effect
uv run python experimental/JOCN2024/generate_plots/plot_warmup.py       # ENV_WARMUP_STEPS effect

# Numerical summary tables in the paper
uv run python experimental/JOCN2024/generate_plots/summarise_bounds_table.py
uv run python experimental/JOCN2024/generate_plots/summarise_review_table.py
```

All output figures are written to `experimental/JOCN2024/generate_plots/plots/`.

The headline figure used in the *Graph Transformer* paper (`bounds_comparison_new_with_rl.png`) is also produced by `plot_heuristic_comparison.py` — the Transformer-RL curves in that figure come from `experimental/JOCN2024/generate_data/eval_transformers.sh` (see [Reproducing the Graph Transformer paper](reproduce_jocn_transformer.md)).

---

## Benchmark settings reproduced

| Setting | Topology | Env | K | Notes |
| --- | --- | --- | --- | --- |
| DeepRMSA / RewardRMSA / GCN-RMSA | NSFNET, COST239, USNET | `rmsa`, 100 FSU | 5 / 50 | distance-adaptive modulation |
| MaskRSA | NSFNET, JPN48 | `rmsa`, 80 FSU, 12.5 GHz slots, no guardband | 5 / 50 | `mean_service_holding_time=12` |
| PtrNet-RSA-40 | NSFNET, COST239, USNET | `rsa`, 40 FSU, slot_size=1, no guardband | 5 / 50 | `values_bw=1` |
| PtrNet-RSA-80 | NSFNET, COST239, USNET | `rsa`, 80 FSU, slot_size=1, no guardband | 5 / 50 | `values_bw=1,…,4` (mixed bw) |

Both `--weight=weight` (km-sorted paths) and the default (hop-sorted paths) are run for every benchmark; the paper compares these against the published RL results.

---

## Citation

```bibtex
@article{doherty_reinforcement_2025,
  author  = {Doherty, Michael and Beghelli, Alejandra},
  title   = {Reinforcement Learning for Dynamic Resource Allocation in Optical Networks: Hype or Hope?},
  journal = {Journal of Optical Communications and Networking},
  volume  = {17},
  number  = {9},
  pages   = {D1},
  year    = {2025},
  doi     = {10.1364/JOCN.559990},
  url     = {https://opg.optica.org/abstract.cfm?URI=jocn-17-9-D1}
}
```
