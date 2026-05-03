# Reproducing the RWA-LR (Lightpath Reuse) paper

This page documents how to reproduce the figures from:

> **Doherty, M., Beghelli, A.** *Reinforcement Learning with Graph Attention for Routing and Wavelength Assignment with Lightpath Reuse*.
> Optical Network Design and Modelling (ONDM) 2025.
> Preprint: [arXiv:2502.14741](https://arxiv.org/abs/2502.14741).

The paper trains a Graph-Attention-based RL agent for **Routing and Wavelength Assignment with Lightpath Reuse (RWA-LR)** — the variant of RWA in which an established lightpath can carry several requests up to its capacity, so the agent's job is to balance routing decisions against capacity packing on existing lightpaths. The XLRON environment used is `--env_type=rwa_lightpath_reuse`.

All scripts referenced live under [`experimental/ONDM2025/`](https://github.com/micdoh/XLRON/tree/main/experimental/ONDM2025).

---

## 1. Train the GNN policy

Single A100, 100 parallel envs, 200M total env steps, NSFNET (100 FSU/link).

```bash
uv run python -m xlron.train.train \
  --env_type=rwa_lightpath_reuse \
  --topology_name=nsfnet_deeprmsa_undirected \
  --link_resources=100 --k=5 \
  --values_bw=100 --scale_factor=0.2 --symbol_rate=100 \
  --incremental_loading \
  --max_requests=10000 --ROLLOUT_LENGTH=150 \
  --TOTAL_TIMESTEPS=200000000 --NUM_ENVS=100 --NUM_SEEDS=1 \
  --UPDATE_EPOCHS=10 \
  --USE_GNN \
    --gnn_mlp_layers=2 --message_passing_steps=3 \
    --gnn_latent=128 \
    --output_nodes_size=1 --output_globals_size=1 \
  --LR=1.9432e-05 --LR_SCHEDULE=warmup_cosine \
  --WARMUP_PEAK_MULTIPLIER=2 --WARMUP_END_FRACTION=0.1 --WARMUP_STEPS_FRACTION=0.1 \
  --GAMMA=0.9186 --GAE_LAMBDA=0.9842 \
  --DOWNSAMPLE_FACTOR=100 \
  --SAVE_MODEL --MODEL_PATH=./RWA_LR_NSFNET_GNN.eqx \
  --WANDB --PROJECT=RWA_LR_SWEEP
```

These hyperparameters are the best-found configuration from the W&B sweep documented in the paper. Drop the `--SAVE_MODEL`/`--MODEL_PATH` for a dry run.

---

## 2. Evaluate the trained policy

End-on-first-blocking evaluation that records the number of accepted requests per episode:

```bash
uv run python -m xlron.train.train \
  --env_type=rwa_lightpath_reuse \
  --topology_name=nsfnet_deeprmsa_undirected \
  --link_resources=100 --k=5 \
  --values_bw=100 --scale_factor=0.2 --symbol_rate=100 \
  --incremental_loading --end_first_blocking \
  --max_requests=10000 --ROLLOUT_LENGTH=150 \
  --TOTAL_TIMESTEPS=1000000 --NUM_ENVS=1 \
  --USE_GNN --gnn_mlp_layers=2 --message_passing_steps=3 \
    --gnn_latent=128 \
    --output_nodes_size=1 --output_globals_size=1 \
  --EVAL_MODEL --MODEL_PATH=./RWA_LR_NSFNET_GNN.eqx \
  --DATA_OUTPUT_FILE=./rwalr_model_firstblock_eval.jsonl \
  --EPISODE_DATA_OUTPUT_FILE=./rwalr_model_firstblock_eval.csv
```

Run the same command without `--EVAL_MODEL --MODEL_PATH` and with `--EVAL_HEURISTIC --path_heuristic=ksp_ff` (or other heuristics from `xlron/heuristics/heuristics.py`) to produce the heuristic baselines.

---

## 3. Plot the figures

```bash
uv run jupyter nbconvert --to notebook --execute experimental/ONDM2025/rwa_lr_training.ipynb
uv run jupyter nbconvert --to notebook --execute experimental/ONDM2025/rwa_lr_evaluation.ipynb
```

The notebooks read the per-step training CSV (`rwa_lr_training.csv` + std variant) and the per-episode evaluation CSVs to produce:

- The training-curve figure `rwalr_training.png` (episode returns vs env steps for KSP-FF and the trained GNN).
- The evaluation comparison: accepted requests at first-block vs traffic load, GNN vs KSP-FF / FF-KSP, with the `kpath_plots.ipynb` from `experimental/JOCN2024/` providing the K-sweep context.

---

## 4. Notes

- The paper uses `--scale_factor=0.2` to set the per-lightpath capacity scaling specific to the RWA-LR formulation; `symbol_rate` sets the symbol rate used in capacity calculation. See [`docs/training.md`](training.md) and [`xlron/parameter_flags.py`](https://github.com/micdoh/XLRON/blob/main/xlron/parameter_flags.py) for the meaning of each flag.
- `--incremental_loading` (no service expiry) is what defines the "RWA" steady-state-loading regime evaluated in the paper, vs the dynamic RMSA setting used in the [Graph Transformer paper](reproduce_jocn_transformer.md).
- The W&B sweep configuration that produced the hyperparameters above lives in [`docs/example_sweep_config.yml`](https://github.com/micdoh/XLRON/blob/main/docs/example_sweep_config.yml) as a template.

---

## Citation

```bibtex
@inproceedings{doherty_rwalr_2025,
  author    = {Doherty, Michael and Beghelli, Alejandra},
  title     = {Reinforcement Learning with Graph Attention for Routing and Wavelength Assignment with Lightpath Reuse},
  booktitle = {Optical Network Design and Modelling (ONDM)},
  year      = {2025},
  note      = {arXiv:2502.14741},
  url       = {https://arxiv.org/abs/2502.14741}
}
```
