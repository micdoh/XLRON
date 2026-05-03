# Capacity Bound Estimation

XLRON provides two methods for estimating capacity bounds on optical networks: **cut-set bounds** and **reconfigurable routing bounds** (also known as resource-prioritized defragmentation). These methods give lower bounds on the achievable blocking probability — i.e. they estimate the best performance any algorithm could achieve on a given network and traffic scenario.

Both methods are standalone scripts in `xlron/bounds/` and share the same environment configuration flags as [heuristic evaluation](./heuristic_evaluation.md). This page focuses on the flags specific to each bounds method — refer to the heuristic evaluation guide for the common environment, traffic, and resource flags (`--env_type`, `--topology_name`, `--link_resources`, `--load`, `--k`, `--continuous_operation`, etc.).

!!! tip "Related paper — submitted to ECOC 2026"
    A systematic comparison of these two estimators (CS, RPD) against KSP heuristics across all 118 TopologyBench topologies appears in *Comparison of Dynamic Elastic Optical Network Capacity Bound Estimation Methods* (Doherty\*, Deng\*, Beghelli, Toni, Bayvel — submitted to ECOC 2026). At 0.1 % blocking, the median gap over KSP is **5.8–9.4 % for the cut-set bound** (top-100 / top-10 cuts) and **17.0 % for resource-prioritized defragmentation**, showing that fragmentation is the larger source of lost capacity than routing inefficiency. See [Reproduce ECOC 2026 capacity bounds paper](reproduce_ecoc2026.md) for the full sweep.


## Cut-Set Bounds

The cut-set method estimates a capacity bound by identifying the most congested "bottlenecks" in the network and simulating resource allocation on those bottlenecks only.

### How It Works

1. **Cut-set discovery**: The network graph is partitioned into two sets of nodes. The links crossing each partition form a "cut-set". The method finds the most congested cut-sets — those where the ratio of traffic crossing the cut to available capacity is highest. Discovery can use exhaustive enumeration (for small networks) or a shortest-paths heuristic (for larger networks).

2. **Capacity bound simulation**: For each traffic load, multiple trials are run. Each request is checked against the cut-sets it traverses. The method finds feasible contiguous slot blocks across traversed cut-sets and uses greedy set-cover to assign physical links. Because the method has freedom in path selection (it only enforces cut-set constraints, not full path constraints), the resulting blocking probability is a lower bound on what any algorithm could achieve.

### Invocation

```bash
python -m xlron.bounds.cutsets_bounds \
    --topology_name=nsfnet_deeprmsa_directed \
    --env_type=rmsa \
    --link_resources=100 \
    --k=50 \
    --load=250 \
    --continuous_operation \
    --truncate_holding_time \
    --mean_service_holding_time=20 \
    --modulations_csv_filepath="./xlron/data/modulations/modulations_deeprmsa.csv" \
    --max_requests=100000 \
    --num_trials=10 \
    --CUTSET_EXHAUSTIVE \
    --CUTSET_BATCH_SIZE=512 \
    --CUTSET_ITERATIONS=32 \
    --CUTSET_TOP_K=256 \
    --cutset_link_selection_mode=least_congested
```

### Cut-Set Specific Flags

#### Simulation Flags

##### `--load`

Traffic load in Erlangs. When using a single load point, this is the load used. When using `--min_load`/`--max_load`/`--step_load` for a load sweep, this value is used for the initial compilation. Default: `250`.

##### `--min_load` / `--max_load` / `--step_load`

When all three are set, the script sweeps loads from `min_load` to `max_load` (inclusive) in steps of `step_load`. The environment is compiled once and reused across all loads without recompilation. Each load produces its own JSONL summary line. Example: `--min_load=150 --max_load=300 --step_load=10`.

##### `--max_requests`

Number of connection requests to simulate per trial. Higher values give more precise blocking probability estimates. Default: `4` (override for bounds, e.g. `100000`).

##### `--num_trials`

Number of independent random-seed trials. Statistics (mean, std, IQR) are computed across trials. Shared with reconfigurable routing bounds. Default: `10`.

#### Cut-Set Discovery Flags

##### `--CUTSET_EXHAUSTIVE`

Use exhaustive enumeration of all 2^N graph partitions to find cut-sets (where N is the number of nodes). This gives the best results but is only feasible for small networks (up to ~20 nodes). For larger networks like JPN48, omit this flag to use the faster shortest-paths heuristic instead.

##### `--CUTSET_BATCH_SIZE`

Batch size for parallel cut-set enumeration during exhaustive search. Default: `512`.

##### `--CUTSET_ITERATIONS`

Number of iterations per parallel process during exhaustive search. Default: `32`.

##### `--CUTSET_TOP_K`

Number of most-congested cut-sets to retain after discovery. Higher values consider more bottlenecks but increase computation. Default: `256`.

##### `--USE_MEAN_CONGESTION_THRESHOLD`

When enabled, additionally filter cut-sets by removing those with congestion below the mean. Default: `False`.

#### Link Selection Flag

##### `--cutset_link_selection_mode`

When multiple links could satisfy a cut-set constraint, this controls which link is preferred during greedy assignment:

| Value | Description |
|-------|-------------|
| `least_congested` | Prefer links with the most free slots overall (default) |
| `most_congested` | Prefer links with the fewest free slots |
| `best_fit` | Prefer links where the contiguous free run around the block is tightest |
| `random` | Random link selection |

### Output Metrics

The cut-set method reports per-load statistics across trials:

- **Blocking Probability** (mean, std, IQR) — fraction of requests blocked
- **Bitrate Blocking Probability** (mean, std, IQR) — blocking weighted by request bitrate
- **Accepted Count** (mean, std, IQR) — number of requests accepted per trial
- **Blocked Count** (mean, std, IQR) — number of requests blocked per trial
- **Always Accepted Count** (mean, std, IQR) — requests that don't traverse any selected cut-set (always accepted by definition)

### Examples

#### DeepRMSA-Style on NSFNET

```bash
python -m xlron.bounds.cutsets_bounds \
    --topology_name=nsfnet_deeprmsa_directed \
    --env_type=rmsa \
    --link_resources=100 \
    --k=50 \
    --load=250 \
    --continuous_operation \
    --truncate_holding_time \
    --mean_service_holding_time=20 \
    --modulations_csv_filepath="./xlron/data/modulations/modulations_deeprmsa.csv" \
    --max_requests=100000 \
    --num_trials=10 \
    --CUTSET_EXHAUSTIVE \
    --CUTSET_BATCH_SIZE=512 \
    --CUTSET_ITERATIONS=32 \
    --CUTSET_TOP_K=256 \
    --cutset_link_selection_mode=least_congested
```

#### Load Sweep on NSFNET

Use `--min_load`, `--max_load`, and `--step_load` to sweep loads in a single invocation. The environment is compiled once and reused across all loads:

```bash
python -m xlron.bounds.cutsets_bounds \
    --topology_name=nsfnet_deeprmsa_directed \
    --env_type=rmsa \
    --link_resources=100 \
    --k=50 \
    --load=300 \
    --min_load=150 --max_load=300 --step_load=10 \
    --continuous_operation \
    --truncate_holding_time \
    --mean_service_holding_time=20 \
    --modulations_csv_filepath="./xlron/data/modulations/modulations_deeprmsa.csv" \
    --max_requests=100000 \
    --num_trials=10 \
    --CUTSET_EXHAUSTIVE \
    --CUTSET_BATCH_SIZE=512 \
    --CUTSET_ITERATIONS=32 \
    --CUTSET_TOP_K=256 \
    --cutset_link_selection_mode=least_congested
```

Note: `--load` should be set to the maximum load in the sweep range, as it is used for the initial environment setup and compilation.

#### Large Network (JPN48) Without Exhaustive Search

For larger networks, omit `--CUTSET_EXHAUSTIVE` to use the shortest-paths heuristic for cut-set discovery:

```bash
python -m xlron.bounds.cutsets_bounds \
    --topology_name=jpn48_undirected \
    --env_type=rmsa \
    --link_resources=80 \
    --k=50 \
    --load=200 \
    --max_bw=50 \
    --guardband=0 \
    --slot_size=12.5 \
    --mean_service_holding_time=12 \
    --continuous_operation \
    --modulations_csv_filepath="./xlron/data/modulations/modulations_deeprmsa.csv" \
    --max_requests=100000 \
    --num_trials=10 \
    --CUTSET_TOP_K=256 \
    --cutset_link_selection_mode=least_congested
```


## Reconfigurable Routing Bounds

The reconfigurable routing method (also known as resource-prioritized defragmentation) estimates a capacity bound by allowing the network to reroute existing services when a new request would otherwise be blocked.

### How It Works

1. **Request sorting**: Connection requests are generated and sorted by their weighted resource requirements (slots x hops across candidate paths). Requests requiring fewer resources get higher priority for defragmentation.

2. **Main simulation loop**: Requests are processed sequentially using a standard heuristic (KSP-FF or FF-KSP). When a request blocks:
    - The method identifies all concurrently active services that overlap in time with the blocked request.
    - It attempts to **defragment** the network by replaying the active services in priority order, allowing path rerouting.
    - If defragmentation frees enough resources for the blocked request, the fix is committed.
    - Otherwise, the request remains blocked.

3. **Bound interpretation**: Because real networks cannot freely reroute active services, the blocking probability achieved by this method represents a lower bound — real performance will be equal or worse.

### Invocation

The reconfigurable routing bounds script is run directly (not as a module):

```bash
python xlron/bounds/reconfigurable_routing_bounds.py \
    --topology_name=nsfnet_deeprmsa_directed \
    --env_type=rmsa \
    --link_resources=100 \
    --k=50 \
    --load=250 \
    --continuous_operation \
    --truncate_holding_time \
    --mean_service_holding_time=20 \
    --modulations_csv_filepath="./xlron/data/modulations/modulations_deeprmsa.csv" \
    --path_heuristic=ksp_ff \
    --TOTAL_TIMESTEPS=13000 \
    --NUM_ENVS=1 \
    --COMPILE_RR_BOUNDS
```

### Reconfigurable Routing Specific Flags

##### `--TOTAL_TIMESTEPS`

Number of requests to process in the simulation. Default: `1000000`. Note that this method is sequential per environment, so large values take longer than the parallelised heuristic evaluation.

##### `--NUM_ENVS`

Number of parallel environments. Typically set to `1` for bounds estimation, since each environment runs a sequential simulation. Multiple environments run independent seeds for statistical averaging.

##### `--path_heuristic`

The heuristic used for initial resource allocation and during defragmentation replays. Supported values: `ksp_ff`, `ff_ksp`. Default: `ksp_ff`.

##### `--COMPILE_RR_BOUNDS`

When enabled, the full simulation loop is JIT-compiled as a `jax.lax.scan`, which is significantly faster but requires fixed array sizes for active request tracking. When disabled, the simulation runs as a Python loop with variable-length filtering (slower but more flexible). Default: `False`.

!!! note "Flags set automatically"
    The reconfigurable routing script internally forces `--relative_arrival_times=False` (absolute times are needed for request lifetime tracking) and sets `--max_requests` equal to `--TOTAL_TIMESTEPS`. You do not need to set these manually.

### Output Metrics

The method reports statistics across multiple seeds (typically 3):

- **Blocking Probability** (mean, std, IQR) — overall request blocking rate
- **Block Count** (mean, std, IQR) — number of requests that initially blocked
- **Fix Count** (mean, std, IQR) — number of blocks resolved by defragmentation
- **Fix Ratio** (mean, std, IQR) — fraction of blocks successfully fixed (`fix_count / block_count`). This is a key metric: a high fix ratio means most blocking events were due to suboptimal routing rather than true capacity exhaustion.

Timing breakdowns are also reported showing the fraction of time spent in the main simulation loop vs. defragmentation.

### Examples

#### DeepRMSA-Style on NSFNET

```bash
python xlron/bounds/reconfigurable_routing_bounds.py \
    --topology_name=nsfnet_deeprmsa_directed \
    --env_type=rmsa \
    --link_resources=100 \
    --k=50 \
    --load=250 \
    --continuous_operation \
    --truncate_holding_time \
    --mean_service_holding_time=20 \
    --modulations_csv_filepath="./xlron/data/modulations/modulations_deeprmsa.csv" \
    --path_heuristic=ksp_ff \
    --TOTAL_TIMESTEPS=13000 \
    --NUM_ENVS=1 \
    --COMPILE_RR_BOUNDS
```

#### Load Sweep

Use `--min_load`, `--max_load`, and `--step_load` to sweep loads in a single invocation:

```bash
python xlron/bounds/reconfigurable_routing_bounds.py \
    --topology_name=nsfnet_deeprmsa_directed \
    --env_type=rmsa \
    --link_resources=100 \
    --k=50 \
    --load=300 \
    --min_load=150 --max_load=300 --step_load=10 \
    --continuous_operation \
    --truncate_holding_time \
    --mean_service_holding_time=20 \
    --modulations_csv_filepath="./xlron/data/modulations/modulations_deeprmsa.csv" \
    --path_heuristic=ksp_ff \
    --TOTAL_TIMESTEPS=13000 \
    --NUM_ENVS=1 \
    --COMPILE_RR_BOUNDS
```

Note: `--load` should be set to the maximum load in the sweep range, as it is used for the initial environment setup (including array sizing for active request tracking).

#### RSA with Unit Bandwidth

```bash
python xlron/bounds/reconfigurable_routing_bounds.py \
    --topology_name=nsfnet_deeprmsa_undirected \
    --env_type=rsa \
    --link_resources=40 \
    --k=50 \
    --load=250 \
    --slot_size=1 \
    --guardband=0 \
    --mean_service_holding_time=10 \
    --values_bw=1 \
    --continuous_operation \
    --path_heuristic=ksp_ff \
    --TOTAL_TIMESTEPS=13000 \
    --NUM_ENVS=1 \
    --COMPILE_RR_BOUNDS
```
