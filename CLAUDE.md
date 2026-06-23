# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working with the XLRON codebase.

## Project Overview

XLRON ("ex-el-er-on") is a JAX-based library for simulating optical network resource allocation problems and training reinforcement learning agents. It provides gym-style environments that run entirely on GPU/TPU for massively parallel training.

**Documentation site**: https://micdoh.github.io/XLRON/

## Build and Run Commands

Run Python entrypoints with `uv run` in this repo to use the project interpreter with all dependencies.

```bash
# Install dependencies
uv sync

# Run training
uv run python -m xlron.train.train --env_type=rmsa --topology_name=nsfnet_deeprmsa_directed --link_resources=100 --continuous_operation --ENV_WARMUP_STEPS=3000 --truncate_holding_time --k=50 --ROLLOUT_LENGTH=128 --TOTAL_TIMESTEPS=1280 --STEPS_PER_INCREMENT=128 --NUM_ENVS=1

# Run with heuristic evaluation
uv run python -m xlron.train.train --env_type=rmsa --topology_name=nsfnet_deeprmsa_directed --link_resources=100 --k=50 --load=250 --continuous_operation --ENV_WARMUP_STEPS=3000 --TOTAL_TIMESTEPS=20000000 --NUM_ENVS=2000 --EVAL_HEURISTIC --path_heuristic=ksp_ff

# Run capacity bound estimation
uv run python -m xlron.bounds.cutsets_bounds --topology_name=nsfnet_deeprmsa_directed --env_type=rmsa --link_resources=100 --k=50 --load=250 --continuous_operation --max_requests=100000 --num_trials=10 --CUTSET_EXHAUSTIVE --CUTSET_TOP_K=256
uv run python xlron/bounds/reconfigurable_routing_bounds.py --topology_name=nsfnet_deeprmsa_directed --env_type=rmsa --link_resources=100 --k=50 --load=250 --continuous_operation --path_heuristic=ksp_ff --TOTAL_TIMESTEPS=13000 --NUM_ENVS=1 --COMPILE_RR_BOUNDS

# Run tests
uv run pytest .

# Run single test file
uv run pytest xlron/environments/env_funcs_test.py -v
```

## Architecture

### Core Structure

```
xlron/
├── environments/          # JAX-based optical network environments
│   ├── dataclasses.py     # Flax struct dataclasses for state/params
│   ├── env_funcs.py       # Core environment functions used in step, reset, etc.
│   ├── diff_utils.py      # Differentiable approximations for discrete ops
│   ├── make_env.py        # Environment factory (config → env + params)
│   └── wrappers.py        # Gym-style wrappers (LogWrapper, etc.)
├── train/
│   ├── train.py           # Main training entry point (handles both RL and eval)
│   ├── train_utils.py     # Training utilities (schedules, optimizers, logging, metrics)
│   └── ppo.py             # PPO implementation (rollout, advantages, loss, updates)
├── heuristics/            # Classical algorithms (KSP-FF, FF-KSP, etc.)
│   ├── heuristics.py      # All heuristic implementations
│   └── eval_heuristic.py  # Evaluation wrapper (get_eval_fn)
├── bounds/                # Capacity bound estimation
│   ├── cutsets_bounds.py   # Cut-set based bounds (standalone script)
│   └── reconfigurable_routing_bounds.py  # Defragmentation-based bounds (standalone script)
├── models/                # Neural network architectures
│   ├── mlp.py             # ActorCriticMLP
│   ├── gnn.py             # ActorCriticGNN (Jraph-based, with optional GAT attention)
│   └── transformer.py     # ActorCriticTransformer (with WIRE positional encodings)
├── dtype_config.py        # Device-aware dtype configuration
└── parameter_flags.py     # Command-line flags (absl)
```

### GN Model with DRA (`isrs_gn_model_dra.py`)

The DRA (Distributed Raman Amplification) module extends the ISRS GN model with Raman-pump-aware NLI and ASE calculations. Key files: `environments/gn_model/isrs_gn_model_dra.py` (DRA model), `environments/gn_model/isrs_gn_model.py` (base ISRS model).

- **Fit parameters**: Shape `(6, num_channels, max_spans)`. Indices 0-4: `[C_f, a_f, C_b, a_b, a]` for the semi-analytical power profile. Index 5: per-channel ODE Raman gain (linear).
- **Fitting**: `fit_dra_params_triangular()` runs at env creation time: solves Raman ODE via `jax.experimental.ode.odeint`, fits profiles via `jaxopt.LevenbergMarquardt`, stores ODE Raman gain.
- **NLI**: `gn_model_dra()` computes NLI using 9 Raman mode combinations (forward/backward pump interactions) via the eta helper functions.
- **ASE noise**: `get_snr_dra()` uses a Friis cascade model for the hybrid Raman+EDFA amplifier. `NF_hybrid = NF_DRA + (NF_EDFA - 1) / G_raman` where `NF_DRA = 1/G_raman + 2*n_sp*(1 - 1/G_raman)` with `n_sp ≈ 1.13` (phonon factor). Total ASE per span: `P_ASE = NF_hybrid * G_total * h * f * B`, accumulated as `num_spans * P_ASE`.
- **Signal-signal ISRS**: Excluded from the Raman ODE (`g_R[:num_channels, :num_channels] = 0`) to avoid double-counting with the GN model's perturbative ISRS tilt.

### Key Design Patterns

1. **Flax Struct Dataclasses**: State and parameters use `@struct.dataclass` with `pytree_node=False` for static fields that shouldn't be traced by JAX.

2. **Functional Style**: All environment functions are pure functions that take state/params and return new state. No side effects.

3. **JIT Compilation**: Functions use `@jax.jit` or `@partial(jax.jit, static_argnums=...)` for compilation. Static arguments must be hashable.

4. **Differentiable Operations**: `diff_utils.py` provides differentiable approximations using straight-through gradient estimators and temperature-controlled soft functions.

5. **Equinox Models**: Neural networks use Equinox (`eqx.Module`). Models are partitioned into params/static via `eqx.partition` for JAX tracing. `TrainState` in `train_utils.py` holds the model, optimizer state, and schedules.

## Three Execution Modes

All three modes share the same environment/traffic flags. The mode is selected by flags:

### 1. RL Training (default)
No special flag needed. Uses `get_learner_fn` from `ppo.py`. The training loop: rollout → advantage estimation → PPO updates (scan over epochs/minibatches). Key flags: `--LR`, `--GAMMA`, `--GAE_LAMBDA`, `--CLIP_EPS`, `--VF_COEF`, `--ENT_COEF`, `--ROLLOUT_LENGTH`, `--UPDATE_EPOCHS`, `--NUM_MINIBATCHES`. See `docs/training.md`.

### 2. Heuristic Evaluation (`--EVAL_HEURISTIC`)
Uses `get_eval_fn` from `eval_heuristic.py`. No neural network is created. Action selection dispatches to heuristic functions based on `--path_heuristic`. See `docs/heuristic_evaluation.md`.

### 3. Model Evaluation (`--EVAL_MODEL`)
Uses `get_eval_fn` with a loaded model (`--MODEL_PATH`). Runs the trained policy without gradient updates.

## Environment Types

- `rsa` - Routing and Spectrum Assignment
- `rmsa` - Routing, Modulation and Spectrum Assignment
- `rwa` - Convenience wrapper: RSA with datarate=1, slot_size=1, spectral_efficiency=1
- `deeprmsa` - DeepRMSA with path statistics observation (specific obs/action space)
- `rwa_lightpath_reuse` - RWA with lightpath reuse (has `--scale_factor`, `--symbol_rate`)
- `vone` - Virtual Optical Network Embedding
- `rsa_gn_model` / `rmsa_gn_model` - With ISRS GN model physical layer impairments

## Key Configuration Flags

### Environment
- `--env_type` - Environment type
- `--topology_name` - Network topology (must match a file in `xlron/data/topologies/` without `.json`)
- `--link_resources` - Number of frequency slots per link
- `--k` - Number of K-shortest paths (flag name is `k`, not `k_paths`)
- `--load` - Traffic load in Erlangs (primary traffic parameter; arrival_rate = load / mean_service_holding_time)
- `--mean_service_holding_time` - Default: 25. Load is what matters; this is derived.
- `--continuous_operation` - Don't reset env between episodes (standard for steady-state measurements)
- `--ENV_WARMUP_STEPS` - Steps before collecting stats (e.g. 3000 for steady-state)
- `--slot_size` - Spectral width per slot in GHz (default: 12.5)
- `--guardband` - Guard band slots (default: 1)
- `--modulations_csv_filepath` - Modulation format definitions CSV
- `--values_bw` - Comma-separated bandwidth request values
- `--incremental_loading` - Non-expiring requests (for capacity measurement)
- `--end_first_blocking` - End episode on first block (used with incremental_loading)
- `--truncate_holding_time` - Truncate to < 2*mean (for DeepRMSA paper compatibility)
- `--max_requests` - Episode length for episodic mode only; ignored with continuous_operation (set to TOTAL_TIMESTEPS automatically)
- `--path_sort_criteria` - How K paths are sorted: `spectral_resources` (default), `hops`, `distance`, `hops_distance`, `capacity` (RWA-LR only)

### Training (PPO)
- `--LR` - Learning rate (most important hyperparameter)
- `--GAMMA` - Discount factor (default: 0.999)
- `--GAE_LAMBDA` - GAE lambda (None = auto-anneal from INITIAL_LAMBDA to FINAL_LAMBDA)
- `--CLIP_EPS` - PPO clipping (default: 0.2)
- `--VF_COEF` - Value loss coefficient (default: 0.5)
- `--ENT_COEF` - Entropy bonus coefficient (default: 0.0)
- `--ROLLOUT_LENGTH` - Steps per rollout before PPO update (default: 150)
- `--UPDATE_EPOCHS` - Passes over rollout buffer per update (default: 1)
- `--NUM_MINIBATCHES` - Minibatches per epoch (default: 1)
- `--TOTAL_TIMESTEPS` - Total env steps across all envs
- `--NUM_ENVS` - Parallel environments (batch_size = NUM_ENVS * ROLLOUT_LENGTH)
- `--STEPS_PER_INCREMENT` - Steps between logging intervals
- `--LR_SCHEDULE` - `cosine` (default), `warmup_cosine`, `linear`, `constant`
- `--REWARD_CENTERING` - Subtract running average reward from TD errors
- `--OFF_POLICY_IAM` - Off-policy invalid action masking (ratio = unmasked/masked policy)
- `--VALID_MASS_LOSS_COEF` - Auxiliary loss to keep unmasked policy on valid actions
- `--PRIO_ALPHA` / `--PRIO_BETA0` - Prioritized experience replay (0.0 = uniform)
- `--RHO_CLIP` / `--C_CLIP` - VTrace-style importance ratio clipping (<=0 = standard GAE)
- `--SEPARATE_VF_OPTIMIZER` - Separate optimizer for critic with own LR/schedule
- `--REWARD_SCALE` - Multiply all rewards by this factor

### Model Architecture
- Default: MLP (`--NUM_LAYERS`, `--NUM_UNITS`, `--ACTIVATION`)
- GNN: `--USE_GNN` (`--message_passing_steps`, `--edge_embedding_size`, `--node_embedding_size`, `--attn_mlp_layers`)
- Transformer: `--USE_TRANSFORMER` (`--transformer_num_layers`, `--transformer_num_heads`, `--transformer_embedding_size`, `--transformer_obs_type`)
- `--aggregate_slots` - Reduce action space by grouping slots into blocks of N

### Heuristics
- `--EVAL_HEURISTIC` - Run heuristic evaluation instead of RL
- `--path_heuristic` - Algorithm: `ksp_ff`, `ksp_lf`, `ksp_bf`, `ksp_mu`, `ff_ksp`, `lf_ksp`, `bf_ksp`, `mu_ksp`, `kmc_ff`, `kmf_ff`, `kme_ff`, `kca_ff`

### Capacity Bounds (standalone scripts, not through train.py)
- Cut-sets (`python -m xlron.bounds.cutsets_bounds`): `--max_requests` (requests per trial), `--num_trials`, `--CUTSET_EXHAUSTIVE`, `--CUTSET_TOP_K`, `--cutset_link_selection_mode`
- Reconfigurable routing (`python xlron/bounds/reconfigurable_routing_bounds.py`): `--COMPILE_RR_BOUNDS`, `--path_heuristic`. Forces `relative_arrival_times=False` and `max_requests=TOTAL_TIMESTEPS` internally.

### Differentiable Mode
- `--differentiable` - Enable differentiable approximations (default: False)
- `--temperature` - Temperature for soft approximations (default: 1.0)

### Logging
- `--WANDB` - Log to Weights & Biases
- `--SAVE_MODEL` / `--MODEL_PATH` - Save/load models
- `--DATA_OUTPUT_FILE` - Save JSONL run summary (one JSON object per line with config, metrics, timing)
- `--EPISODE_DATA_OUTPUT_FILE` - Save per-episode metrics to CSV (formerly `DATA_OUTPUT_FILE`)
- `--DOWNSAMPLE_FACTOR` - Block-average N data points before uploading to W&B
- `--ENHANCED_LOGGING` - Detailed PPO diagnostics (valid_frac, clip_frac, ratio stats, etc.)
- `--LOG_LOSS_INFO` - Log loss metrics (default: True)

## Available Topologies

Topology files live in `xlron/data/topologies/`. The `--topology_name` is the filename without `.json`. Topologies come in directed and undirected variants (directed = each fibre direction is a separate link).

Key topologies: `nsfnet_deeprmsa_directed/undirected`, `nsfnet_nevin_undirected`, `cost239_deeprmsa_directed/undirected`, `cost239_ptrnet_real_directed/undirected`, `usnet_gcnrnn_directed/undirected`, `usnet_ptrnet_directed/undirected`, `jpn48_directed/undirected`, `german17_directed/undirected`, `conus_directed/undirected`, `5node_directed/undirected`.

Additionally, all 119 real-world topologies from [TopologyBench](https://github.com/TopologyBench/Real-Topologies) are included in both directed and undirected variants (e.g. `coronet_directed`, `geant_undirected`, `abilene_directed`, `germany50_undirected`). Run `python xlron/data/topologies/topology_bench_to_xlron_conversion.py --list` to see all available names.

## Working with Differentiable Operations

The `diff_utils.py` module provides differentiable versions of discrete operations:

```python
# Pattern: all functions accept differentiable parameter
result = differentiable_where(condition, true_val, false_val,
                               threshold=0.5, temperature=1.0,
                               differentiable=params.differentiable)

# When differentiable=False: returns hard result directly
# When differentiable=True: uses straight-through estimator
```

Functions available:
- `differentiable_where` - Soft conditional selection
- `differentiable_compare` - Soft comparison (==, >=, <=, >, <, !=)
- `differentiable_argmax` - Soft argmax
- `differentiable_round`, `differentiable_ceil`, `differentiable_floor`
- `differentiable_index`, `differentiable_indexing` - Soft array indexing
- `differentiable_one_hot_index_update` - Soft one-hot update
- `differentiable_cond` - Soft conditional execution

## Common Patterns

### Adding a New Parameter

1. Add flag in `parameter_flags.py`
2. Add field to appropriate dataclass in `dataclasses.py`
3. Wire config to params in `make_env.py`
4. Use `params.field_name` in environment functions
5. Add to the GUI: add default to `DEFAULTS` dict in `xlron/gui/widgets.py` and add a widget in the appropriate section function (e.g. `physical_layer_section()` for GN model params, `environment_section()` for env params)
6. Document in the relevant `docs/*.md` file and add to the appropriate configuration table

### Adding New Loss/Diagnostic Metrics

When adding new metrics to the PPO loss function (`ppo.py`):

1. Add the metric to the return tuple in `_loss_fn`
2. Add the metric key to `loss_info` dict in `_update_step`
3. **Register the metric in wandb**: Add to `loss_metrics` or `diagnostics_metrics` list in `train_utils.py`
4. If conditional (like `ENHANCED_LOGGING`), update `setup_wandb()` to register metrics when the flag is enabled

Forgetting step 3 will cause metrics to not appear in wandb dashboards.

### Static vs Dynamic Arguments

- Use `pytree_node=False` in dataclasses for values that don't change during training
- Use `static_argnames` in JIT decorators for string/bool arguments
- Never use traced values as static arguments (causes unhashable error)
- `HashableArrayWrapper` (in `dataclasses.py`) wraps arrays that need to be static JIT args

### How the Training Loop Works

1. `train.py:train()` calls `process_config()` → `make()` to create env/params
2. `experiment_data_setup()` creates the model, optimizer, schedules, and `TrainState`
3. Warmup runs `ENV_WARMUP_STEPS` using heuristic actions to fill the network
4. `get_learner_fn()` returns a function that does `NUM_UPDATES` steps of: rollout → advantages → epochs of minibatch updates
5. The outer loop in `train()` runs `NUM_INCREMENTS` times, logging metrics each time
6. `TOTAL_TIMESTEPS = NUM_INCREMENTS * STEPS_PER_INCREMENT = NUM_INCREMENTS * NUM_UPDATES * ROLLOUT_LENGTH * NUM_ENVS`

### How Heuristic Evaluation Works

1. Same env setup as training, but `model = None`
2. `get_eval_fn()` replaces the learner; action selection dispatches to heuristic functions in `xlron/heuristics/heuristics.py`
3. Metrics (blocking probability, utilisation, etc.) are computed and reported identically to training

### How make_env.py Processes Config

- `process_config()` normalizes the raw flags into a consistent `Box` config
- `make()` creates the environment and params. When `continuous_operation=True`, `max_requests` is set to `TOTAL_TIMESTEPS` (the user-provided value is ignored)
- `TOTAL_TIMESTEPS` is rounded down to be divisible by `ROLLOUT_LENGTH * NUM_ENVS`
- The function returns a `LogWrapper`-wrapped env; unwrap via `env._env` to get the raw environment

### Environment Step Flow (RSA/RMSA)

1. `implement_path_action` → tentatively places the request on `link_slot_array` (`+= affected_slots_mask`)
2. `check_action_rsa` → validates (any slot at +2 means collision)
3. `complete_step_rsa` → if invalid, undoes the placement; if valid, finalises departure times
4. `generate_request_rsa` → generates next request (source, dest, datarate) and removes expired services

## Testing

```bash
# Run all tests
pytest . -v

# Run specific test
pytest xlron/environments/env_funcs_test.py::test_function_name -v

# Run with coverage
pytest . --cov=xlron
```

## Dtype Configuration

`dtype_config.py` provides device-aware dtypes resolved by `initialize_dtypes(config)` (called from `process_config`, so every entry point picks them up):
- `COMPUTE_DTYPE`, `PARAMS_DTYPE` - Neural network compute/params/optimizer (always float32 for stability; observations are cast up to `COMPUTE_DTYPE` at each model's `__call__` boundary)
- `LARGE_FLOAT_DTYPE` - Precision floats: accumulators (bitrate sums), physical SNR/power, importance weights (stays float32 under mixed precision)
- `SMALL_FLOAT_DTYPE` - Bulk floats: spectrum occupancy (`link_slot_array`), normalised features (float16 under mixed precision)
- `TIME_DTYPE` - Time/departure arrays (`current_time`, departures). float16 when `relative_arrival_times` (the bounded default), else float32. Auto-falls back to float32 under absolute time or `incremental_loading`.
- `LARGE_INT_DTYPE` - Counters & large indices: `total_requests`/`total_timesteps`/`accepted_services` (stays int32)
- `SMALL_INT_DTYPE` - Bounded indices, e.g. `required_slots` (int16 under mixed precision)
- `BINARY_DTYPE` - Binary arrays, e.g. path–link incidence (int8 under mixed precision)
- `INDEX_DTYPE` - Always int32 for JAX array indexing

`--mixed_precision` shrinks the bulk/bounded tiers to cut env-state memory (~49%) while keeping NN compute/params and precision-sensitive arrays at float32; per-tier `--*_dtype` flags override the defaults. **Reclassification rule:** arrays that are *carried and incrementally mutated* (e.g. `link_slot_array`, departure) can be reclassified at their init site and stay consistent across the scan; arrays *recomputed from scratch each step* (action masks via `mask_slots`/select_action, graph node/edge features via `init_graph_tuple`/`update_graph_tuple`) take their dtype from the recompute, so **every** site that writes them back into the carried state must cast to the narrow dtype to match the init — otherwise the `lax.scan` carry dtype mismatches. The largest single array is `graph.edges` (E×S). `--differentiable` forces everything to float32 and takes precedence over `--mixed_precision`. See `xlron/dtype_config_test.py`.
