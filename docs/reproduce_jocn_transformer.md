# Reproducing the Graph Transformer paper

This page documents how to reproduce the figures and tables from:

> **Doherty, M., Beghelli, A., Toni, L.** *Graph Transformers and Stabilized Reinforcement Learning for Large-Scale Dynamic Routing, Modulation and Spectrum Allocation in Elastic Optical Networks*. (In preparation; targeted at the Journal of Optical Communications and Networking.)

The paper makes two main experimental claims:

1. **Benchmark comparison** — On four standard topologies (NSFNET, COST239, USNET, JPN48) and five RL benchmark settings (DeepRMSA, RewardRMSA, GCN-RMSA, MaskRSA, PtrNet-RSA), the Graph Transformer is the first RL method to consistently match or exceed the strongest heuristic baseline (Section 3).
2. **Scalability** — On TopologyBench's USA100 (100 nodes) and TataInd (143 nodes) topologies — the largest dynamic RMSA instances ever attempted with RL — the Transformer supports 3–4% higher load than FF-KSP (`K`=70 / `K`=90) at 0.1% blocking (Section 4).

---

## Section 3 — Benchmark comparison

The Section 3 figure compares the Graph Transformer against five published RL methods, the strongest heuristic in each setting, and two capacity bound estimates (cut-set and reconfigurable-routing).

The **comparison plot itself (Fig. 7, `bounds_comparison_new_with_rl.png`)** is built using results from scripts in [`experimental/JOCN2024/`](https://github.com/micdoh/XLRON/tree/main/experimental/JOCN2024) — the same directory used for the *Reinforcement Learning: Hype or Hope?* paper, since the heuristic and bounds curves are reused. See [Reproducing the JOCN 2024 (Hype or Hope) paper](reproduce_jocn2024.md) for the heuristic and bounds runs in detail. The Transformer evaluation curves overlaid on top of those baselines come from [`experimental/JOCN2024/generate_data/eval_transformers.sh`](https://github.com/micdoh/XLRON/tree/main/experimental/JOCN2024/generate_data/eval_transformers.sh).

### 1. Train the Transformer policies

The trained Equinox checkpoints expected by the evaluation script are listed below. Train them with `--SAVE_MODEL --MODEL_PATH=<filename>.eqx` (one A100, ~30 min – 1 h 40 min each):

| Benchmark setting | Topology | Model file |
| --- | --- | --- |
| DeepRMSA / RewardRMSA / GCN-RMSA | NSFNET | `nsfnet_maskrsa_43_1.eqx` (RMSA) and `nsfnet_rsa80.eqx` (RSA) |
| DeepRMSA / RewardRMSA / GCN-RMSA | COST239 | `cost239_deeprmsa_13.eqx` |
| DeepRMSA / RewardRMSA / GCN-RMSA | USNET | `usnet_2.eqx` |
| MaskRSA | JPN48 | `jpn48_maskrsa.eqx` |
| MaskRSA | NSFNET | `nsfnet_maskrsa_43_1.eqx` |
| PtrNet-RSA-40 | NSFNET, USNET, COST239 | `*_rsa40.eqx` |
| PtrNet-RSA-80 | NSFNET, USNET, COST239 | `*_rsa80.eqx` |

A representative training command (NSFNET / RMSA / DeepRMSA setting):

```bash
uv run xlron/train/train.py \
  --topology_name=nsfnet_deeprmsa_directed \
  --env_type=rmsa --link_resources=100 --k=50 \
  --load=145 --mean_service_holding_time=20 --truncate_holding_time \
  --modulations_csv_filepath=./xlron/data/modulations/modulations_deeprmsa.csv \
  --max_requests=13000 --ENV_WARMUP_STEPS=0 --relative_arrival_times \
  --USE_TRANSFORMER --transformer_num_layers=2 --transformer_num_heads=4 \
  --aggregate_slots=20 \
  --OFF_POLICY_IAM --VALID_MASS_LOSS_COEF=0.0002 --VML_SCHEDULE=constant \
  --LR=5e-3 --LR_SCHEDULE=cosine \
  --ENT_COEF=0.0175 --ENT_SCHEDULE=linear \
  --VF_COEF=0.05 --SEPARATE_VF_OPTIMIZER --VF_LR=1e-4 \
  --GAMMA=0.996 --GAE_LAMBDA=0.99 --CLIP_EPS=0.04 \
  --ROLLOUT_LENGTH=64 --NUM_ENVS=200 \
  --TOTAL_TIMESTEPS=100000000 --STEPS_PER_INCREMENT=5150000 \
  --WANDB --PROJECT=TRANSFORMER_TRAIN --DOWNSAMPLE_FACTOR=100 \
  --SAVE_MODEL --MODEL_PATH=./episodic_20_8_10.eqx
```

The full set of training commands for every benchmark setting is in `experimental/JOCN2024/generate_data/eval_transformers.sh` — that file contains both the `--EVAL_MODEL` evaluation commands shown below and the corresponding training settings (drop `--EVAL_MODEL --MODEL_PATH=...` and add `--SAVE_MODEL` to train from scratch).

### 2. Evaluate the trained Transformers and the heuristics

```bash
# Heuristic and bounds baselines (also used by the JOCN 2024 paper)
bash experimental/JOCN2024/generate_data/heuristic_evaluation.sh
bash experimental/JOCN2024/generate_data/run_cutsets_bounds.sh
bash experimental/JOCN2024/generate_data/run_reconfigurable_routing_bounds.sh

# Transformer evaluation across all four topologies and five RL settings
bash experimental/JOCN2024/generate_data/eval_transformers.sh
bash experimental/JOCN2024/generate_data/eval_transformers_bounds.sh
```

These produce the JSONL/CSV results consumed by the comparison plotting script. The Transformer evaluation reuses each saved `.eqx` checkpoint via `--EVAL_MODEL --MODEL_PATH=...`.

### 3. Plot Fig. 7

```bash
uv run python experimental/JOCN2024/generate_plots/plot_heuristic_comparison.py
```

Output: `experimental/JOCN2024/generate_plots/plots/bounds_comparison_new_with_rl.png` (paper Fig. 7 — *bp vs load* across NSFNET / COST239 / USNET / JPN48 with all five benchmark settings).

The summary tables comparing supported load at 0.1% blocking come from `summarise_bounds_table.py` and `summarise_review_table.py` in the same directory.

---

## Section 4 — Large-Scale Experiments (USA100 and TataInd)

All Section 4 figures live under [`experimental/large_topologies/`](https://github.com/micdoh/XLRON/tree/main/experimental/large_topologies). The trained Transformer checkpoints, FF-KSP trajectories, ablation runs, and slot-occupancy heatmaps are all included in `experimental/large_topologies/results/{usa100,tataind}/`.

### 1. Train the large-topology Transformers

Both runs use a single H100 (74 GB), 12 parallel envs, 40M steps, ~4–5 hours wall-clock.

```bash
# USA100 — 100 nodes, 342 directed links, k=70
uv run xlron/train/train.py \
  --topology_name=usa100_directed \
  --env_type=rmsa --link_resources=320 \
  --k=70 --aggregate_slots=80 \
  --load=620 --mean_service_holding_time=25 \
  --max_requests=25000 --ENV_WARMUP_STEPS=0 --relative_arrival_times \
  --USE_TRANSFORMER --transformer_num_layers=2 --transformer_num_heads=8 \
  --transformer_embedding_size=128 \
  --OFF_POLICY_IAM --VALID_MASS_LOSS_COEF=0.002 --VML_SCHEDULE=linear --VML_END_FRACTION=0.5 \
  --LR=1.5e-3 --LR_SCHEDULE=cosine \
  --ENT_COEF=0.01 --ENT_SCHEDULE=cosine \
  --VF_COEF=0.1 --SEPARATE_VF_OPTIMIZER --VF_LR=5e-5 \
  --GAMMA=0.996 --GAE_LAMBDA=0.99 --CLIP_EPS=0.04 \
  --ROLLOUT_LENGTH=64 --NUM_ENVS=12 \
  --TOTAL_TIMESTEPS=40000000 \
  --SAVE_MODEL --MODEL_PATH=./usa100_transformer.eqx \
  --WANDB --PROJECT=LARGE_TRANSFORMER

# TataInd — 143 nodes, 362 directed links, k=90
uv run xlron/train/train.py \
  --topology_name=tataind_directed \
  --env_type=rmsa --link_resources=320 \
  --k=90 --aggregate_slots=80 \
  --load=450 --mean_service_holding_time=25 \
  --max_requests=25000 --ENV_WARMUP_STEPS=0 --relative_arrival_times \
  --USE_TRANSFORMER --transformer_num_layers=2 --transformer_num_heads=8 \
  --transformer_embedding_size=128 \
  --OFF_POLICY_IAM --VALID_MASS_LOSS_COEF=0.001 --VML_SCHEDULE=linear --VML_END_FRACTION=0.5 \
  --LR=1.5e-3 --LR_SCHEDULE=cosine \
  --ENT_COEF=0.015 --ENT_SCHEDULE=cosine \
  --VF_COEF=0.1 --SEPARATE_VF_OPTIMIZER --VF_LR=5e-5 \
  --GAMMA=0.996 --GAE_LAMBDA=0.99 --CLIP_EPS=0.04 \
  --ROLLOUT_LENGTH=64 --NUM_ENVS=12 \
  --TOTAL_TIMESTEPS=40000000 \
  --SAVE_MODEL --MODEL_PATH=./tataind_transformer.eqx \
  --WANDB --PROJECT=LARGE_TRANSFORMER
```

Hyperparameters that differ between USA100 and TataInd are summarized in Table 2 of the paper.

### 2. Heuristic benchmarks (FF-KSP K-sweep)

The choice of `K`=70 (USA100) and `K`=90 (TataInd) for FF-KSP is justified in Section 4.1 by sweeping `K` from 10 to 100 and recording blocking probability at 620 / 450 Erlang. Reproduce with `--EVAL_HEURISTIC --path_heuristic=ff_ksp` over a `K` sweep:

```bash
for K in 10 20 30 40 50 60 70 80 90 100; do
  uv run python -m xlron.train.train \
    --env_type=rmsa --topology_name=usa100_directed --link_resources=320 \
    --k=$K --load=620 --mean_service_holding_time=25 \
    --max_requests=100000 --continuous_operation --ENV_WARMUP_STEPS=3000 \
    --EVAL_HEURISTIC --path_heuristic=ff_ksp \
    --NUM_ENVS=10 --TOTAL_TIMESTEPS=1000000 \
    --DATA_OUTPUT_FILE=experimental/large_topologies/results/usa100/usa100_ff_ksp_K${K}.jsonl
done
# Repeat with topology_name=tataind_directed and load=450 for TataInd
```

### 3. Evaluate Transformer vs FF-KSP across loads

For each topology, evaluate at multiple loads and dump full per-request trajectories:

```bash
# USA100 — load sweep with Transformer
for LOAD in 550 600 620 650 680 700 750; do
  uv run python -m xlron.train.train \
    --env_type=rmsa --topology_name=usa100_directed --link_resources=320 \
    --k=70 --aggregate_slots=80 --load=$LOAD --mean_service_holding_time=25 \
    --USE_TRANSFORMER --transformer_num_layers=2 --transformer_num_heads=8 --transformer_embedding_size=128 \
    --OFF_POLICY_IAM --max_requests=100000 --continuous_operation --ENV_WARMUP_STEPS=3000 \
    --EVAL_MODEL --MODEL_PATH=./usa100_transformer.eqx \
    --NUM_ENVS=10 --TOTAL_TIMESTEPS=1000000 \
    --DATA_OUTPUT_FILE=experimental/large_topologies/results/usa100/usa100_transformer_eval_results.jsonl
done

# Single full episode at training load — produces the trajectory CSVs used for path/spectral analysis
uv run python -m xlron.train.train \
  --env_type=rmsa --topology_name=usa100_directed --link_resources=320 \
  --k=70 --aggregate_slots=80 --load=620 \
  --USE_TRANSFORMER --transformer_num_layers=2 --transformer_num_heads=8 \
  --max_requests=100000 --ENV_WARMUP_STEPS=0 --relative_arrival_times \
  --EVAL_MODEL --MODEL_PATH=./usa100_transformer.eqx \
  --EPISODE_DATA_OUTPUT_FILE=experimental/large_topologies/results/usa100/usa100_transformer_traj.csv
```

Repeat with `--EVAL_HEURISTIC --path_heuristic=ff_ksp` to produce `usa100_ff_ksp_*` files. The same patterns apply to TataInd (`load=450`, `k=90`).

### 4. Slot-occupancy and link-usage processing

```bash
uv run python experimental/large_topologies/process_slot_occupancy.py
uv run python experimental/large_topologies/analyze_path_lengths.py --topology_name=usa100_directed --k=70
uv run python experimental/large_topologies/analyze_path_lengths.py --topology_name=tataind_directed --k=90
uv run python experimental/large_topologies/analyze_action_path_ratios.py
```

These read the trajectory CSVs and produce the `*_slot_occupancy.npz` and `*_link_usage.npy` artifacts consumed by the plotting script.

### 5. Ablation runs (Fig. 9)

The five ablation variants live in `experimental/large_topologies/results/{usa100,tataind}/ablations/`:

| Variant | Drop |
| --- | --- |
| `original` (full model) | — (paper "All Features") |
| `onpolicy` | swap `--OFF_POLICY_IAM` for on-policy IAM (`--ON_POLICY_IAM`) |
| `nodamping` | drop `--LOSS_DAMPING_VALID_MASS_TARGET` |
| `nogating` | drop hard gating (`--LOSS_DAMPING_K_MIN=0`) |
| `nogating_nodamping` | drop both |
| `novml` | set `--VALID_MASS_LOSS_COEF=0` |
| `gating1` | use `--LOSS_DAMPING_K_MIN=1` (keep transitions with ≥1 valid action) |
| `ffksp` | `--EVAL_HEURISTIC --path_heuristic=ff_ksp` baseline |

Each variant is a separate training run with the corresponding flag change.

### 6. Plot all Section 4 figures

```bash
uv run python experimental/large_topologies/plot_large_topologies.py
```

| Paper figure | Plot file |
| --- | --- |
| Fig. 8 — TataInd and USA100 topologies | `figures/topologies.png` (handled separately by `topology_visualization` scripts) |
| Fig. 9 — Ablation: blocking over training | `figures/ablation_blocking.png` |
| Fig. 10 — Loss components over training | `figures/loss_components.png` |
| Fig. 11 — Blocking probability vs load | `figures/blocking_vs_load.png` |
| Fig. 12 — Bitrate blocking over a single episode | `figures/bitrate_blocking_over_steps.png` |
| Fig. 13 — Mean path length (km, hops) over an episode | `figures/path_comparison.png` |
| Fig. 14 — Per-request path length delta | `figures/path_delta.png` |
| Fig. 15 — Path-length distributions | `figures/path_boxplots.png` |
| Fig. 16 — Per-link FSU occupancy difference | `figures/slot_occupancy_diff.png` |
| Fig. 17 — Per-link usage difference | `figures/link_usage_delta.png` |

The TopologyBench source for USA100 and TataInd is at <https://github.com/TopologyBench/Real-Topologies>; both topologies are bundled with XLRON via [`xlron/data/topologies/topology_bench_to_xlron_conversion.py`](https://github.com/micdoh/XLRON/blob/main/xlron/data/topologies/topology_bench_to_xlron_conversion.py).

---

## Hardware

- **Section 3 training:** NVIDIA A100 80 GB, 200 parallel envs, 100M steps, 30 min – 1 h 40 min per run.
- **Section 4 training:** NVIDIA H100 80 GB, 12 parallel envs, 40M steps, ~4–5 h per topology, 74 GB GPU memory.
- **Section 4 evaluation:** A100 or H100 sufficient; 100k requests per evaluation point.

---

## Citation

This paper is **in preparation** — citation details will be updated once the manuscript is published.

```bibtex
@unpublished{doherty_graph_transformer,
  author  = {Doherty, Michael and Beghelli, Alejandra and Toni, Laura},
  title   = {Graph Transformers and Stabilized Reinforcement Learning for Large-Scale Dynamic Routing, Modulation and Spectrum Allocation in Elastic Optical Networks},
  note    = {In preparation},
  year    = {2026}
}
```
