# Reproducing the XLRON framework paper

This page documents how to reproduce the figures and tables from:

> **Doherty, M., Beghelli, A., Jarmolovičius, M., Deng, B., Killey, R., Bayvel, P., Toni, L.** *XLRON: A Framework for Hardware-Accelerated and Differentiable Simulation of Optical Networks*. (In preparation; targeted at the Journal of Optical Communications and Networking.)

The paper makes four main claims that are demonstrated experimentally:

1. **Throughput** — XLRON is faster than competing optical-network simulators, with 222–1,494× speedup for end-to-end RL training (Section 2).
2. **Physical layer accuracy** — XLRON's ISRS GN model with DRA agrees with the Gerard et al. 2025 C+L-band experiment to within 0.5 dB (Section 3).
3. **Differentiable simulation** — XLRON is the first fully differentiable optical-network simulator and can perform gradient-based RSA (Section 4).
4. **Interfaces** — both a CLI and a browser GUI expose the same configuration surface (Section 5).

All scripts referenced below live under [`experimental/`](https://github.com/micdoh/XLRON/tree/main/experimental). Run everything from the repository root with the project virtual environment activated (`uv sync`, then `uv run …`).

---

## Section 2 — Comparisons and Performance Benchmarks

The benchmarking results live under [`experimental/benchmarks/`](https://github.com/micdoh/XLRON/tree/main/experimental/benchmarks).

### Run the throughput sweeps

```bash
# All sweep groups (env-type cross-comparison, num_envs, FSU/k, GN-band scaling, topology heatmap)
uv run python experimental/benchmarks/run_benchmarks.py \
  --output_dir=experimental/benchmarks/results

# Or run individual groups
uv run python experimental/benchmarks/run_benchmarks.py --groups=num_envs,topology
uv run python experimental/benchmarks/run_benchmarks.py --groups=device --device=cpu

# Resume after interruption
uv run python experimental/benchmarks/run_benchmarks.py --resume
```

The sweep launches `xlron.train.train --EVAL_HEURISTIC --path_heuristic=ksp_ff` runs across environment types (`rwa`, `rmsa`, `rwa_lightpath_reuse`, `rsa_gn_model`, `rmsa_gn_model`), parallel-environment counts (1 → 4096), FSU per link (50 → 500), candidate-path counts (k = 5, 10, 25, 50), GN-model band configurations (C, C+L, C+L+S), and topologies (5node, NSFNET, COST239, German17, USNET, JPN48, CONUS, COST239-PtrNet variants).

Each run writes one JSON line to `experimental/benchmarks/results/`. After the sweeps finish, aggregate and plot:

```bash
uv run python experimental/benchmarks/plot_benchmarks.py
```

| Paper figure | Plot file |
| --- | --- |
| Fig. 1 — Throughput across environment types (1 env) | `figures/cross_env_comparison.png` |
| Fig. 2 — SPS vs. number of parallel envs | `figures/fps_vs_num_envs.png` |
| Fig. 3 — GPU speedup vs. CPU | `figures/gpu_speedup.png` |
| Fig. 4 — JIT compilation time vs. parallel envs | `figures/compilation_time_vs_num_envs.png` |
| Fig. 5 — SPS vs. FSU/link and k | `figures/fps_vs_fsu_and_k_by_num_envs.png` |
| Fig. 6 — GN-model SPS across bands (C, C+L, C+L+S) | `figures/gn_band_scaling.png` |
| Fig. 7 — Throughput heatmap (topology × env type) | `figures/heatmap.png` |
| Table 4 — Topology properties | `figures/topology_table.png` (also `topology_stats.py`) |

### DeepRMSA RL training comparison (Fig. 8)

The end-to-end RL training comparison reproduces the benchmark from Chen *et al.* 2019 and compares XLRON, the original DeepRMSA codebase, and Optical-RL-Gym.

```bash
# 1. Train DeepRMSA with XLRON (single A100, 512 envs, ~75 s compile + 12 s train)
uv run python -m xlron.train.train \
  --env_type=deeprmsa \
  --topology_name=nsfnet_deeprmsa_directed \
  --link_resources=100 --k=5 \
  --load=250 --continuous_operation --truncate_holding_time \
  --ENV_WARMUP_STEPS=3000 \
  --ROLLOUT_LENGTH=128 --NUM_ENVS=512 \
  --TOTAL_TIMESTEPS=20000000 --STEPS_PER_INCREMENT=128000 \
  --LR=5e-4 --LR_SCHEDULE=warmup_cosine \
  --DATA_OUTPUT_FILE=experimental/benchmarks/results/deeprmsa_benchmark.csv

# 2. Train original DeepRMSA: clone https://github.com/micdoh/DeepRMSA
#    and run its training script (TF2 fork of Chen et al. 2019).
#    Output expected at experimental/benchmarks/results/deeprmsa_original_training_results.csv

# 3. Train Optical-RL-Gym: clone https://github.com/carlosnatalino/optical-rl-gym
#    and run examples/stable_baselines3/DeepRMSA.ipynb.
#    Output expected at experimental/benchmarks/results/training_iqr_data.csv

# 4. Plot the combined figure
uv run python experimental/benchmarks/plot_deeprmsa_comparison.py
```

This produces `experimental/benchmarks/figures/deeprmsa_bp_combined.png` (paper Fig. 8) — the dual-panel SBP-vs-steps and SBP-vs-time plot showing XLRON's 222–1,494× wall-clock speedup and lower blocking through invalid action masking.

### Cross-library throughput table (Table 1)

The numbers for FUSION, ON-Gym, and Flex Net Sim in Table 1 are reproduced from Bórquez-Paredes *et al.* 2026. The XLRON and GNPy entries come from the sweeps above; the GNPy entry is a per-lightpath benchmark on the same 14-node NSFNET, 96 channels at 50 GHz. There is no automated runner for the GNPy comparison — it uses the standard GNPy `path_request_run.py` workflow on a 100-request set.

---

## Section 3 — Physical Layer Model (Gerard et al. 2025 validation)

All physical-layer figures live under [`experimental/validation/`](https://github.com/micdoh/XLRON/tree/main/experimental/validation).

```bash
uv run python -m experimental.validation.gerard2025_validation
```

This single script reproduces the 90-channel C+L-band system from Gerard *et al.* 2025 (15 × 100 km TXF, hybrid backward-Raman + EDFA, 100 GHz spacing) using `rsa_gn_model` configured via `gui.presets.PRESETS["gerard2025"]`. Outputs are written to `experimental/validation/gerard2025_results/`.

| Paper figure | Plot file |
| --- | --- |
| Fig. 9 — Gain budget (Raman + EDFA vs span loss) | `plot_gain_budget.png` |
| Fig. 10 — Per-channel SNR metrics (GOSNR, OSNR_ASE, OSNR_NL, received) | `plot1_2_combined_snr_metrics.png` |
| Fig. 11 — Per-band averaged comparison (XLRON vs Gerard Table I) | `plot3_band_comparison.png` |
| Fig. 12 — Ablation: GOSNR vs launch power, removing DRA / Nyquist subchannels / coherent ASE | `plot5_ablation_sweep.png` |

The script also prints the per-band agreement (within 0.3–0.5 dB across GOSNR, OSNR_ASE, OSNR_NL) used in Section 3.4.

### Raman pump power optimization (Fig. 13)

```bash
uv run python -m experimental.validation.pump_optimization
```

This runs gradient-based optimization of the five backward-Raman pump powers using Adam through the differentiable DRA pipeline (see [Differentiable DRA Pipeline](differentiable_dra.md) for the IFT/JVP details). It starts from the equal-pump configuration described in the paper and drives the total Shannon–Hartley throughput from ~70.2 Tb/s up to ~71.1 Tb/s within 30 iterations. Output is `pump_optimisation.png`.

`select_pump_history.py` extracts and replots the convergence trace from the saved optimization log.

---

## Section 4 — Differentiable Simulation

All differentiable-simulation figures live under [`experimental/differentiable/`](https://github.com/micdoh/XLRON/tree/main/experimental/differentiable).

### Reward landscape visualization (Figs. 14–15)

```bash
# Topology test case schematic (Fig. 14)
uv run python experimental/differentiable/plot_topology_diagram.py

# 2-request RSA reward landscape, gradient field, optimization trajectories (Fig. 15)
uv run python experimental/differentiable/gradient_sense_check.py
```

Outputs:

| Paper figure | Plot file |
| --- | --- |
| Fig. 14 — Small topology used for test case | `figures/topology_test_case.png` |
| Fig. 15 — Combined reward landscape, gradient surface, trajectories, gradient arrows | `figures/combined_landscape_view_full_gaussian.png` |

### Direct RSA optimization on NSFNET (Fig. 16)

```bash
# Case 0: from-scratch optimization, 2,000 requests, 20,000 iterations
uv run python experimental/differentiable/optimize_actions.py \
  --topology_name=nsfnet_nevin_undirected \
  --link_resources=100 --k=5 \
  --max_requests=2000 --incremental_loading \
  --differentiable --temperature=2.7e-4 \
  --LR=1.0 --num_iterations=20000

# Case 1: from-heuristic init, 1,000 iterations
uv run python experimental/differentiable/optimize_actions.py \
  --topology_name=nsfnet_nevin_undirected \
  --link_resources=100 --k=5 \
  --max_requests=2000 --incremental_loading \
  --differentiable --temperature=5.0e-4 \
  --LR=1.0 --num_iterations=1000 \
  --init_from_heuristic=ksp_ff

# Plot the convergence figure
uv run python experimental/differentiable/plot_optimization_convergence.py
```

This produces `figures/optimization_convergence.png` (Fig. 16): Case 0 reaches 994/2000 accepted services from scratch; Case 1 briefly improves the KSP-FF baseline from 1,037 → 1,039 before degrading.

`case_1.py` and `optimize_actions_sgd.py` provide the SGD variants used in the paper's discussion section. The `differentiable.ipynb` notebook contains the original interactive version of the same experiments.

---

## Section 5 — Interfaces

The GUI screenshot (Fig. 17) and the `render` visualization (Fig. 18) are not regenerated by a script — they are screen captures of the live application:

```bash
# Launch the GUI
xlron

# Or, open the render visualization for any environment by passing --PLOTTING
uv run python -m xlron.train.train \
  --env_type=rmsa \
  --topology_name=nsfnet_deeprmsa_directed \
  --link_resources=100 --k=5 --load=250 \
  --continuous_operation --ENV_WARMUP_STEPS=3000 \
  --EVAL_HEURISTIC --path_heuristic=ksp_ff \
  --PLOTTING --TOTAL_TIMESTEPS=10000 --NUM_ENVS=1
```

---

## Hardware and software

All XLRON results in the paper used:

- **GPU:** NVIDIA A100 80 GB
- **CPU:** Apple M1 Pro 10-core (also used for all GNPy and physical-layer benchmarks)
- **JAX / jaxlib:** 0.6.2 / 0.9.1
- Single device per run (no multi-GPU sharding)

For the cross-library Table 1 numbers, FUSION / ON-Gym / Flex Net Sim values are reproduced verbatim from Bórquez-Paredes *et al.* 2026.

---

## Citation

This paper is **in preparation** — citation details will be updated once the manuscript is published. In the meantime, please cite the codebase via the OFC 2024 entry on the [Papers](papers.md) page.

```bibtex
@unpublished{doherty_xlron_framework,
  author  = {Doherty, Michael and Beghelli, Alejandra and Jarmolovi\v{c}ius, Mindaugas and Deng, Bohua and Killey, Robert and Bayvel, Polina and Toni, Laura},
  title   = {{XLRON}: A Framework for Hardware-Accelerated and Differentiable Simulation of Optical Networks},
  note    = {In preparation},
  year    = {2026}
}
```
