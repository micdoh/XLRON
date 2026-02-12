# Heuristic Evaluation

XLRON includes implementations of classical heuristic algorithms for optical network resource allocation. These heuristics serve as baselines for comparison with RL-trained agents, and can also be used standalone to evaluate network performance under different traffic and topology configurations.

Heuristic evaluation is accessed through the same `train.py` script used for RL training, with the `--EVAL_HEURISTIC` flag. When this flag is set, no neural network is created or trained — instead, the selected heuristic algorithm is used to make resource allocation decisions.

## Building a Command

A heuristic evaluation command is constructed from several groups of flags:

1. **The heuristic flag** (`--EVAL_HEURISTIC`) — required to enable heuristic mode
2. **Problem environment** — topology, link resources, environment type
3. **Traffic characteristics** — load, holding time, arrival patterns
4. **Heuristic behaviour** — which algorithm to use, how to sort paths
5. **Execution control** — parallelism, timesteps, logging

The minimal command is:

```bash
python -m xlron.train.train --EVAL_HEURISTIC --path_heuristic=ksp_ff
```

This uses all default values (4-node topology, 5 link resources, etc.). In practice you will want to configure each group of flags for your specific scenario.


## 1. Problem Environment Flags

These flags define the network topology and its resources.

### `--env_type`

The environment type. For heuristic evaluation, the most commonly used types are:

| Value | Description |
|-------|-------------|
| `rsa` | Routing and Spectrum Assignment |
| `rmsa` | Routing, Modulation and Spectrum Assignment |
| `rwa` | Routing and Wavelength Assignment. A convenience wrapper around `rsa` that defaults to requested datarate of 1, slot size of 1, and spectral efficiency of 1 — i.e. each service occupies exactly one slot. |
| `deeprmsa` | DeepRMSA-compatible (same as `rmsa` but with a specific observation/action space) |
| `rsa_gn_model` | RSA with GN model physical layer impairments (see [GN Model section](#gn-model-specific-flags)) |
| `rmsa_gn_model` | RMSA with GN model physical layer impairments (see [GN Model section](#gn-model-specific-flags)) |
| `rwa_lightpath_reuse` | RWA with lightpath reuse (see [RWA-LR section](#rwa-lightpath-reuse-specific-flags)) |

### `--topology_name`

The network topology to simulate. XLRON loads topologies from JSON files in `xlron/data/topologies/`. The topology name corresponds to the filename without the `.json` extension.

Available topologies:

| Topology | Directed | Undirected |
|----------|----------|------------|
| **NSFNET (DeepRMSA)** | `nsfnet_deeprmsa_directed` | `nsfnet_deeprmsa_undirected` |
| **NSFNET (Nevin)** | — | `nsfnet_nevin_undirected` |
| **COST239 (DeepRMSA)** | `cost239_deeprmsa_directed` | `cost239_deeprmsa_undirected` |
| **COST239 (Nevin)** | — | `cost239_nevin_undirected` |
| **COST239 (PtrNet published)** | `cost239_ptrnet_published_directed` | `cost239_ptrnet_published_undirected` |
| **COST239 (PtrNet real)** | `cost239_ptrnet_real_directed` | `cost239_ptrnet_real_undirected` |
| **USNET (GCN-RNN)** | `usnet_gcnrnn_directed` | `usnet_gcnrnn_undirected` |
| **USNET (PtrNet)** | `usnet_ptrnet_directed` | `usnet_ptrnet_undirected` |
| **JPN48** | `jpn48_directed` | `jpn48_undirected` |
| **German17** | `german17_directed` | `german17_undirected` |
| **CONUS** | `conus_directed` | `conus_undirected` |
| **5-node** | `5node_directed` | `5node_undirected` |

!!! note "Directed vs Undirected Topologies"
    **Directed** topologies treat each fibre direction as a separate link (e.g. A→B and B→A are distinct links with independent spectrum). **Undirected** topologies share spectrum between both directions of a link. The choice affects the number of links in the network, the k-shortest paths computed, and the resulting blocking probability. Use the variant that matches the assumptions of the paper or scenario you are reproducing.

You can also supply a custom topology directory with `--topology_directory`.

### `--link_resources`

Number of frequency slot units (FSUs) per link. Common values: `40`, `80`, `100`.

### `--k`

Number of k-shortest paths to pre-compute between each node pair. A higher value (e.g. `50`) gives heuristics more routing options but increases computation. Default: `5`.

### `--slot_size`

Spectral width of each frequency slot in GHz. Default: `12.5`.

### `--guardband`

Number of guard band slots between adjacent services. Default: `1`. Set to `0` for scenarios that don't use guard bands.

### `--modulations_csv_filepath`

Path to a CSV file defining available modulation formats and their reach/spectral efficiency. Default: `./xlron/data/modulations/modulations_deeprmsa.csv`. Required for RMSA environments.

### `--disjoint_paths`

Use link-disjoint paths instead of standard k-shortest paths.

### `--values_bw`

Comma-separated list of possible bandwidth (more precisely, data-rate) request values (in Gbps). For example, `--values_bw=1` for unit bandwidth or `--values_bw=1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4` for a distribution of bandwidth classes. When set, bandwidth requests are sampled uniformly from this list.

### `--min_bw` / `--max_bw` / `--step_bw`

Alternative to `--values_bw`. Define a range of bandwidth request values. Default: `25` to `100` Gbps in steps of `1`.


## 2. Traffic Characteristics Flags

These flags control how traffic is generated in the simulation.

### `--load`

The offered traffic load in Erlangs. This is the primary traffic parameter — it equals `arrival_rate * mean_service_holding_time`. The arrival rate is derived from the load and holding time. Higher load means more congestion and higher blocking probability.

### `--mean_service_holding_time`

Mean holding time for connection requests. Default: `25`. The arrival rate is calculated as `load / mean_service_holding_time`. You can set this to match the assumptions of a specific paper (e.g. `20` for DeepRMSA), but the load in Erlangs is what determines network congestion.

### `--continuous_operation`

When enabled, the environment does not reset between episodes. Services arrive and depart continuously, and the network state carries over. This is the standard mode for steady-state blocking probability measurements. **Recommended for heuristic evaluation.**

### `--ENV_WARMUP_STEPS`

Number of initial environment steps to run before collecting statistics. This allows the network to reach a steady state before measurements begin. A typical value is `3000` but higher loads may require more warmup steps to stabilize. Only meaningful when `--continuous_operation` is enabled.

### `--max_requests`

Maximum number of connection requests per episode. This only applies to **episodic** (non-continuous) operation, where it defines the episode length. In `--continuous_operation` mode, this flag is ignored — `max_requests` is automatically set to `TOTAL_TIMESTEPS`. Default: `4`.

### `--incremental_loading`

Non-expiring requests — services never depart once established. Used for incremental loading scenarios where you measure how many services can be packed into the network before blocking occurs. Typically used together with `--end_first_blocking`.

### `--end_first_blocking`

End the episode as soon as the first blocking event occurs. Useful in combination with `--incremental_loading` to measure exactly how many services can be successfully established before the network runs out of resources.

### `--truncate_holding_time`

Truncate sampled holding times to be less than `2 * mean_service_holding_time`. This is used specifically to match the DeepRMSA paper's traffic model. Not needed for most other scenarios.

### `--relative_arrival_times`

Track relative inter-arrival times rather than absolute timestamps. Default: `True`. This is the standard approach.

### Traffic Matrix Options

By default, traffic is uniformly distributed across all source-destination pairs. You can change this with:

- `--random_traffic` — Generate a random traffic matrix on each environment reset
- `--custom_traffic_matrix_csv_filepath` — Load a specific traffic matrix from a CSV file


## 3. Heuristic Behaviour Flags

### `--path_heuristic`

The heuristic algorithm to use. Available options:

| Heuristic | Description |
|-----------|-------------|
| `ksp_ff` | **K-Shortest Path, First-Fit.** Try paths in order (shortest first); on each path, allocate the first available contiguous slot block. The most common baseline. |
| `ksp_lf` | **K-Shortest Path, Last-Fit.** Try paths in order; allocate the last available slot block on each path. |
| `ksp_bf` | **K-Shortest Path, Best-Fit.** Try paths in order; allocate the slot block that leaves the smallest remaining gap. |
| `ksp_mu` | **K-Shortest Path, Most-Used.** Try paths in order; prefer slots in the most congested region of the spectrum. |
| `ff_ksp` | **First-Fit across K-Shortest Paths.** Search all k paths simultaneously for the globally first available slot. |
| `lf_ksp` | **Last-Fit across K-Shortest Paths.** Search all k paths for the globally last available slot. |
| `bf_ksp` | **Best-Fit across K-Shortest Paths.** Search all k paths for the globally best-fit slot. |
| `mu_ksp` | **Most-Used across K-Shortest Paths.** Search all k paths; prefer globally most-used slots. |
| `kmc_ff` | **K-Minimum Cut, First-Fit.** Select path that minimises cut metric, then first-fit. |
| `kmf_ff` | **K-Minimum Fragmentation, First-Fit.** Select path that minimises fragmentation, then first-fit. |
| `kme_ff` | **K-Minimum Entropy, First-Fit.** Select path that minimises spectrum entropy, then first-fit. |
| `kca_ff` | **Congestion-Aware, First-Fit.** Select path considering link congestion, then first-fit. |

Default: `ksp_ff`.

### `--path_sort_criteria`

Controls how the k-shortest paths are sorted before being presented to the heuristic. This affects which path is "first" in KSP-based heuristics.

| Value | Description |
|-------|-------------|
| `spectral_resources` | Sort by ratio of hops to spectral efficiency (default). Balances path length against modulation efficiency. |
| `hops` | Sort by number of hops (fewest first). |
| `distance` | Sort by physical distance (shortest first). |
| `hops_distance` | Sort by hops first, then by distance as a tiebreaker. |
| `capacity` | Sort by path capacity (for RWA-LR environments only). |


## 4. Execution Control Flags

### `--NUM_ENVS`

Number of parallel environment instances. Each environment runs an independent simulation with its own random seed. More environments give better statistical estimates. Typical values: `100` to `2000`.

### `--TOTAL_TIMESTEPS`

Total number of environment steps to simulate, divided across all environments. For example, `--TOTAL_TIMESTEPS=20000000` with `--NUM_ENVS=2000` gives 10,000 steps per environment. Higher values give more precise blocking probability estimates.

### `--STEPS_PER_INCREMENT`

Number of steps between logging/reporting intervals. Statistics are computed and printed at the end of each increment. For example, with `--TOTAL_TIMESTEPS=20000000` and `--STEPS_PER_INCREMENT=20000000`, statistics are reported once at the end. With `--STEPS_PER_INCREMENT=1000000`, you get 20 intermediate reports.

### `--SEED`

Random seed for reproducibility. Default: `42`.


## 5. Logging and Output Flags

### `--WANDB`

Enable logging to [Weights & Biases](https://wandb.ai/). See [Understanding XLRON](understanding_xlron.md#weights-biases-wandb-integration) for setup instructions.

### `--PROJECT` / `--EXPERIMENT_NAME`

Set the W&B project and run name. If unspecified, an experiment name is auto-generated from the flags.

### `--DATA_OUTPUT_FILE`

Path to a CSV file for saving per-episode metrics. For example: `--DATA_OUTPUT_FILE=results.csv`.

### `--TRAJ_DATA_OUTPUT_FILE`

Path to a CSV file for saving per-step trajectory data (actions, requests, etc.).

### `--DOWNSAMPLE_FACTOR`

Reduce the amount of data uploaded to W&B by averaging every N consecutive data points into one (a block/windowed average). For example, `--DOWNSAMPLE_FACTOR=10` averages each group of 10 data points into a single logged value. Default: `1` (no averaging).


## RWA-Lightpath Reuse Specific Flags

The `rwa_lightpath_reuse` environment models wavelength-level routing where multiple sub-wavelength services can share a lightpath. It has a few additional flags:

### `--scale_factor`

Scale factor for lightpath capacity. Multiplies the calculated capacity of each lightpath. Default: `1.0`. Increase to model higher-capacity transceivers or decrease to model more conservative capacity estimates or reduce episode length. This interacts with the `--max_requests` flag in episodic mode, effectively controlling the episode length when using incremental loading.

### `--symbol_rate`

Symbol rate in Gbaud. Used to calculate lightpath capacity. Default: `100`.

### `--path_sort_criteria=capacity`

The `capacity` sort criterion is specific to the RWA-LR environment — it sorts paths by their lightpath capacity.

When using `rwa_lightpath_reuse`, the `--max_requests` flag interacts with `--scale_factor`: the effective episode length becomes `max_requests * scale_factor`.


## GN Model Specific Flags

The `rsa_gn_model` and `rmsa_gn_model` environments include a closed-form ISRS GN model for estimating physical layer impairments. When running heuristic evaluation with these environments, the heuristic selects the path and spectrum assignment as usual, and launch power is automatically calculated (the `--launch_power_type` must not be `rl` for heuristic evaluation).

### `--launch_power_type`

How launch power is determined. For heuristic evaluation, use `fixed` (same power per transceiver, default) or `tabular` (power depends on path). The `rl` option is not compatible with `--EVAL_HEURISTIC`.

### `--snr_margin`

Required SNR margin (in dB) above the modulation format threshold for a connection to be accepted. Default: `0.5`.

### `--max_power` / `--min_power` / `--step_power`

Launch power range and step size in dBm. Default: `0.5`, `-5`, `0.1`.

### `--last_fit`

When enabled, use KSP-LF (last-fit) instead of KSP-FF (first-fit) for the path action in GN model environments. Default: `False`.

### `--noise_data_filepath`

Path to transceiver and amplifier noise data file. Default: `None` (uses analytical noise model).

### `--monitor_active_lightpaths`

Track active lightpaths for throughput calculations. Default: `False`.

### `--max_power_per_fibre`

Maximum total launch power per fibre in dBm. Default: `21.0`. The per-channel launch power is `max_power_per_fibre - 10*log10(link_resources)` in dBm. For `link_resources=150`, the default of 21 dBm gives approximately -0.76 dBm per channel, which is in the optimal range for C-band EDFA systems. If the GN model mask finds zero valid actions on an empty network, increase this value.

### Physical Layer Parameters

These control the fibre and amplifier characteristics used in the GN model:

| Flag | Description | Default |
|------|-------------|---------|
| `--alpha` | Fibre attenuation coefficient [dB/km] | `0.2` |
| `--beta_2` | Dispersion parameter [ps^2/km] | `-21.7` |
| `--gamma` | Nonlinear coefficient | `1.2` |
| `--span_length` | Span length [km] | `100` |
| `--amplifier_noise_figure` | Amplifier noise figure [dB] | `4.5` |
| `--ref_lambda` | Reference wavelength [m] | `1564e-9` |
| `--coherent` | Add NLI contribution coherently per span | `False` |
| `--uniform_spans` | Use uniform spans (simplifies calculations) | `True` |


## Examples

### Example 1: DeepRMSA-Style Evaluation on NSFNET

Reproduce the KSP-FF baseline from the DeepRMSA paper on the directed NSFNET topology:

```bash
python -m xlron.train.train \
    --EVAL_HEURISTIC \
    --path_heuristic=ksp_ff \
    --env_type=rmsa \
    --topology_name=nsfnet_deeprmsa_directed \
    --link_resources=100 \
    --k=50 \
    --mean_service_holding_time=20 \
    --truncate_holding_time \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --modulations_csv_filepath="./xlron/data/modulations/modulations_deeprmsa.csv" \
    --TOTAL_TIMESTEPS=20000000 \
    --NUM_ENVS=2000 \
    --load=250
```

Key points:

- `--truncate_holding_time` and `--mean_service_holding_time=20` match the DeepRMSA paper's traffic model.
- `--k=50` provides a large set of candidate paths.
- `--ENV_WARMUP_STEPS=3000` lets the network reach steady state before measuring.
- `--NUM_ENVS=2000` runs 2000 parallel simulations for statistical confidence.
- Vary `--load` (e.g. from 150 to 300) to sweep the traffic load and plot blocking probability curves.

### Example 2: Sweep Over Multiple Loads

To generate a blocking probability curve, run the same command for multiple load values:

```bash
for load in 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300; do
    python -m xlron.train.train \
        --EVAL_HEURISTIC \
        --path_heuristic=ksp_ff \
        --env_type=rmsa \
        --topology_name=nsfnet_deeprmsa_directed \
        --link_resources=100 \
        --k=50 \
        --mean_service_holding_time=20 \
        --truncate_holding_time \
        --continuous_operation \
        --ENV_WARMUP_STEPS=3000 \
        --TOTAL_TIMESTEPS=20000000 \
        --NUM_ENVS=2000 \
        --load=$load
done
```

### Example 3: RSA with Unit Bandwidth

Evaluate KSP-FF on an undirected NSFNET with 40 FSUs and unit-bandwidth requests (no modulation-dependent slot sizing):

```bash
python -m xlron.train.train \
    --EVAL_HEURISTIC \
    --path_heuristic=ksp_ff \
    --env_type=rsa \
    --topology_name=nsfnet_deeprmsa_undirected \
    --link_resources=40 \
    --k=50 \
    --slot_size=1 \
    --guardband=0 \
    --mean_service_holding_time=10 \
    --values_bw=1 \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --TOTAL_TIMESTEPS=20000000 \
    --NUM_ENVS=2000 \
    --load=250
```

### Example 4: Incremental Loading on RWA-LR

Evaluate incremental loading (non-expiring requests) on the NSFNET with the `rwa_lightpath_reuse` environment:

```bash
python -m xlron.train.train \
    --EVAL_HEURISTIC \
    --path_heuristic=ksp_ff \
    --env_type=rwa_lightpath_reuse \
    --topology_name=nsfnet_nevin_undirected \
    --link_resources=80 \
    --k=5 \
    --scale_factor=1.0 \
    --incremental_loading \
    --path_sort_criteria=capacity \
    --TOTAL_TIMESTEPS=1000000 \
    --NUM_ENVS=100
```

Key points:

- `--incremental_loading` means services never expire — the network fills up until blocking.
- `--scale_factor` controls lightpath capacity scaling.
- `--path_sort_criteria=capacity` sorts candidate paths by their lightpath capacity, which is specific to the RWA-LR environment.
- No `--ENV_WARMUP_STEPS` or `--continuous_operation` needed for incremental loading.

### Example 5: First-Fit Across All Paths (FF-KSP) on JPN48

Use the `ff_ksp` heuristic which searches all k paths for the globally first available slot:

```bash
python -m xlron.train.train \
    --EVAL_HEURISTIC \
    --path_heuristic=ff_ksp \
    --env_type=rmsa \
    --topology_name=jpn48_undirected \
    --link_resources=80 \
    --k=50 \
    --max_bw=50 \
    --guardband=0 \
    --slot_size=12.5 \
    --mean_service_holding_time=12 \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --TOTAL_TIMESTEPS=20000000 \
    --NUM_ENVS=2000 \
    --load=200
```

### Example 6: Logging Results to File

Save results to a CSV file for post-processing:

```bash
python -m xlron.train.train \
    --EVAL_HEURISTIC \
    --path_heuristic=ksp_ff \
    --env_type=rmsa \
    --topology_name=nsfnet_deeprmsa_directed \
    --link_resources=100 \
    --k=50 \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --TOTAL_TIMESTEPS=20000000 \
    --NUM_ENVS=2000 \
    --load=250 \
    --DATA_OUTPUT_FILE=results.csv
```

Or log to Weights & Biases:

```bash
python -m xlron.train.train \
    --EVAL_HEURISTIC \
    --path_heuristic=ksp_ff \
    --env_type=rmsa \
    --topology_name=nsfnet_deeprmsa_directed \
    --link_resources=100 \
    --k=50 \
    --continuous_operation \
    --ENV_WARMUP_STEPS=3000 \
    --TOTAL_TIMESTEPS=20000000 \
    --NUM_ENVS=2000 \
    --load=250 \
    --WANDB \
    --PROJECT="heuristic_baselines" \
    --EXPERIMENT_NAME="ksp_ff_nsfnet_250E"
```
