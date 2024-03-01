# Commandline Options

```commandline
xlron.train.parameter_flags:
  --ACTION_MASKING: Use invalid action masking
    (default: 'false')
  --ACTIVATION: Activation function
    (default: 'tanh')
  --CLIP_EPS: PPO clipping parameter
    (default: '0.2')
    (a number)
  --DEBUG: Debug mode
    (default: 'false')
  --DEBUG_NANS: Debug NaNs
    (default: 'false')
  --DISABLE_JIT: Disable JIT compilation
    (default: 'false')
  --DOWNSAMPLE_FACTOR: Downsample factor to reduce data uploaded to wandb
    (default: '1')
    (an integer)
  --ENABLE_X64: Enable x64 floating point precision
    (default: 'false')
  --ENT_COEF: Entropy coefficient
    (default: '0.0')
    (a number)
  --EVAL_HEURISTIC: Evaluate heuristic
    (default: 'false')
  --EVAL_MODEL: Evaluate model
    (default: 'false')
  --EXPERIMENT_NAME: Name of experiment (equivalent to run name in wandb) (auto-generated based on other flags if unspecified)
    (default: '')
  --GAE_LAMBDA: GAE lambda parameter
    (default: '0.95')
    (a number)
  --GAMMA: Discount factor
    (default: '0.99')
    (a number)
  --LR: Learning rate
    (default: '0.0005')
    (a number)
  --LR_SCHEDULE: Learning rate schedule
    (default: 'warmup_cosine')
  --MAX_GRAD_NORM: Maximum gradient norm
    (default: '0.5')
    (a number)
  --MODEL_PATH: Path to save/load model
  --NUM_ENVS: Number of environments
    (default: '1')
    (an integer)
  --NUM_LAYERS: Number of layers in actor and critic networks
    (default: '2')
    (an integer)
  --NUM_MINIBATCHES: Number of minibatches per update
    (default: '1')
    (an integer)
  --NUM_SEEDS: Number of seeds
    (default: '1')
    (an integer)
  --NUM_STEPS: Number of steps per environment
    (default: '150')
    (an integer)
  --NUM_UNITS: Number of hidden units in actor and critic networks
    (default: '64')
    (an integer)
  --ORDERED: Order print statements when debugging (must be false if using pmap)
    (default: 'true')
  --PREALLOCATE_MEM: Preallocate GPU memory
    (default: 'true')
  --PREALLOCATE_MEM_FRACTION: Fraction of GPU memory to preallocate
    (default: '0.95')
  --PRINT_MEMORY_USE: Print memory usage
    (default: 'false')
  --PROJECT: Name of project (same as experiment name if unspecified)
    (default: '')
  --SAVE_MODEL: Save model (will be saved to --MODEL_PATH locally and uploaded to wandb if --WANDB is True)
    (default: 'false')
  --SCHEDULE_MULTIPLIER: Increase the learning rate schedule horizon by this factor (to keep schedule for longer final training runs consistent with that from tuning runs)
    (default: '1')
    (an integer)
  --SEED: Random seed
    (default: '42')
    (an integer)
  --TOTAL_TIMESTEPS: Total number of timesteps
    (default: '1000000.0')
    (a number)
  --UPDATE_EPOCHS: Number of epochs per update
    (default: '10')
    (an integer)
  --USE_GNN: Use GNN
    (default: 'false')
  --USE_PMAP: Use pmap
    (default: 'false')
  --VF_COEF: Value function coefficient
    (default: '0.5')
    (a number)
  --VISIBLE_DEVICES: Comma-separated indices of (desired) visible GPUs e.g. 1,2,3
    (default: '0')
  --WANDB: Use wandb
    (default: 'false')
  --WARMUP_END_FRACTION: Fraction of init LR that is final LR
    (default: '0.1')
    (a number)
  --WARMUP_PEAK_MULTIPLIER: Increase the learning rate warmup peak compared to init
    (default: '1.0')
    (a number)
  --WARMUP_STEPS_FRACTION: Fraction of total timesteps to use for warmup
    (default: '0.2')
    (a number)
  --aggregate_slots: Number of slots to aggregate into a single action
    (default: '1')
    (an integer)
  --continuous_operation: If True, do not reset the environment at the end of an episode
    (default: 'false')
  --custom_traffic_matrix_csv_filepath: Path to custom traffic matrix CSV file
  --deterministic: Deterministic evaluation (use mode of action distribution)
    (default: 'False')
  --disjoint_paths: Use disjoint paths (k paths still considered)
    (default: 'false')
  --end_first_blocking: End episode on first blocking event
    (default: 'false')
  --env_type: Environment type
    (default: 'vone')
  --gnn_latent: GNN latent size
    (default: '64')
    (an integer)
  --gnn_mlp_layers: Number of MLP layers
    (default: '2')
    (an integer)
  --guardband: Guard band in slots
    (default: '1')
    (an integer)
  --incremental_loading: Incremental increase in traffic load (non-expiring requests)
    (default: 'false')
  --k: Number of paths
    (default: '5')
    (an integer)
  --link_resources: Number of link resources
    (default: '5')
    (an integer)
  --load: Load
    (default: '150')
    (an integer)
  --max_bw: Maximum requested bandwidth
    (default: '100')
    (an integer)
  --max_node_resources: Maximum number of node resources
    (default: '1')
    (an integer)
  --max_requests: Maximum number of requests in an episode
    (default: '4.0')
    (a number)
  --max_timesteps: Maximum number of timesteps in an episode
    (default: '30.0')
    (a number)
  --max_traffic: Maximum traffic
    (default: '1.0')
  --mean_service_holding_time: Mean service holding time
    (default: '15')
    (an integer)
  --message_passing_steps: Number of message passing steps
    (default: '3')
    (an integer)
  --min_bw: Minimum requested bandwidth
    (default: '25')
    (an integer)
  --min_node_resources: Minimum number of node resources
    (default: '1')
    (an integer)
  --min_traffic: Minimum traffic
    (default: '0.0')
  --model: Used to hold model parameters
    (a comma separated list)
  --modulations_csv_filepath: (no help available)
  --multiple_topologies_directory: Directory containing JSON definitions of network topologies that will be alternated per episode
  --node_heuristic: Node heuristic to be evaluated
    (default: 'random')
  --node_probs: List of node probabilities for selection
    (a comma separated list)
  --node_resources: Number of node resources
    (default: '4')
    (an integer)
  --normalize_by_link_length: Normalize by link length
    (default: 'false')
  --output_edges_size: Output edges size (not used)
    (default: '64')
    (an integer)
  --output_globals_size: Output globals size
    (default: '64')
    (an integer)
  --output_nodes_size: Output nodes size
    (default: '64')
    (an integer)
  --path_heuristic: Path heuristic to be evaluated
    (default: 'ksp_ff')
  --random_traffic: Random traffic matrix for RSA on each reset (else uniform or custom)
    (default: 'false')
  --scale_factor: Scale factor for link capacity (only used in RWA with lightpath reuse)
    (default: '1.0')
    (a number)
  --slot_size: Spectral width of frequency slot in GHz
    (default: '12.5')
    (a number)
  --step_bw: Step size for requested bandwidth values between min and max
    (default: '1')
    (an integer)
  --step_traffic: Step size for traffic values between min and max
    (default: '0.1')
  --symbol_rate: Symbol rate in Gbaud (only used in RWA with lightpath reuse
    (default: '100')
    (an integer)
  --topology_directory: Directory containing JSON definitions of network topologies
  --topology_name: Topology name
    (default: '4node')
  --traffic_requests_csv_filepath: Path to traffic request CSV file
  --values_bw: List of requested bandwidth values
    (a comma separated list)
  --virtual_topologies: Virtual topologies
    (default: '3_ring')
    (a comma separated list)
  --weight: Edge attribute name for ordering k-shortest paths

absl.flags:
  --flagfile: Insert flag definitions from the given file into the command line.
    (default: '')
  --undefok: comma-separated list of flag names that it is okay to specify on the command line even if the program does not define a flag with that name.  IMPORTANT: flags in this list that have arguments
    MUST use the --flag=value format.
    (default: '')
```