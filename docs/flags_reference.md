# Commandline Options

```commandline
xlron.train.parameter_flags:
  --[no]ACTION_MASKING: Use invalid action masking
    (default: 'false')
  --ACTIVATION: Activation function
    (default: 'tanh')
  --ADAM_BETA1: Adam beta1
    (default: '0.9')
    (a number)
  --ADAM_BETA2: Adam beta2
    (default: '0.999')
    (a number)
  --ADAM_EPS: Adam epsilon
    (default: '1e-05')
    (a number)
  --CLIP_EPS: PPO clipping parameter
    (default: '0.2')
    (a number)
  --DATA_OUTPUT_FILE: Path to save data output
  --[no]DEBUG: Debug mode
    (default: 'false')
  --[no]DEBUG_NANS: Debug NaNs
    (default: 'false')
  --[no]DISABLE_JIT: Disable JIT compilation
    (default: 'false')
  --DOWNSAMPLE_FACTOR: Downsample factor to reduce data uploaded to wandb
    (default: '1')
    (an integer)
  --EMULATED_DEVICES: Number of devices to emulate
    (an integer)
  --[no]ENABLE_X64: Enable x64 floating point precision
    (default: 'false')
  --ENT_COEF: Entropy coefficient
    (default: '0.0')
    (a number)
  --ENV_WARMUP_STEPS: Number of warmup steps before training or eval
    (default: '0')
    (an integer)
  --[no]EVAL_HEURISTIC: Evaluate heuristic
    (default: 'false')
  --[no]EVAL_MODEL: Evaluate model
    (default: 'false')
  --EXPERIMENT_NAME: Name of experiment (equivalent to run name in wandb) (auto-generated based on other flags if unspecified)
    (default: '')
  --GAE_LAMBDA: GAE lambda parameter
    (default: '0.95')
    (a number)
  --GAMMA: Discount factor
    (default: '0.999')
    (a number)
  --[no]LAYER_NORM: Use layer normalization
    (default: 'false')
  --[no]LOAD_MODEL: Load model for retraining or evaluation
    (default: 'false')
  --LR: Learning rate
    (default: '0.0005')
    (a number)
  --LR_SCHEDULE: Learning rate schedule
    (default: 'warmup_cosine')
  --MAX_GRAD_NORM: Maximum gradient norm
    (default: '0.5')
    (a number)
  --MINIBATCH_SIZE: Minibatch size
    (default: '1')
    (an integer)
  --MODEL_PATH: Path to save/load model
  --[no]NO_PRINT_FLAGS: Do not print flags
    (default: 'false')
  --NUM_DEVICES: Number of devices
    (default: '1')
    (an integer)
  --NUM_ENVS: Number of environments per device
    (default: '1')
    (an integer)
  --NUM_LAYERS: Number of layers in actor and critic networks
    (default: '2')
    (an integer)
  --NUM_LEARNERS: Number of independent learners i.e. how many independent experiments to run with a unique set of learned parameters each
    (default: '1')
    (an integer)
  --NUM_MINIBATCHES: Number of minibatches per update
    (default: '1')
    (an integer)
  --NUM_UNITS: Number of hidden units in actor and critic networks
    (default: '64')
    (an integer)
  --NUM_UPDATES: Number of rollouts per environment
    (default: '1')
    (an integer)
  --[no]ORDERED: Order print statements when debugging (must be false if using pmap)
    (default: 'true')
  --[no]PLOTTING: Plotting
    (default: 'false')
  --[no]PREALLOCATE_MEM: Preallocate GPU memory
    (default: 'true')
  --PREALLOCATE_MEM_FRACTION: Fraction of GPU memory to preallocate
    (default: '0.95')
  --[no]PRINT_MEMORY_USE: Print memory usage
    (default: 'false')
  --PROJECT: Name of project (same as experiment name if unspecified)
    (default: '')
  --ROLLOUT_LENGTH: Number of steps per rollout per environment
    (default: '150')
    (an integer)
  --[no]SAVE_MODEL: Save model (will be saved to --MODEL_PATH locally and uploaded to wandb if --WANDB is True)
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
  --[no]USE_GNN: Use GNN
    (default: 'false')
  --VF_COEF: Value function coefficient
    (default: '0.5')
    (a number)
  --VISIBLE_DEVICES: Comma-separated indices of (desired) visible GPUs e.g. 1,2,3
    (default: '0')
  --[no]WANDB: Use wandb
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
  --alpha: Fibre attenuation coefficient, alpha [dB/km]
    (default: '0.2')
    (a number)
  --amplifier_noise_figure: Amplifier noise figure [dB]
    (default: '4.5')
    (a number)
  --beta_2: Dispersion parameter [ps^2/km]
    (default: '-21.7')
    (a number)
  --[no]continuous_operation: If True, do not reset the environment at the end of an episode
    (default: 'false')
  --custom_traffic_matrix_csv_filepath: Path to custom traffic matrix CSV file
  --[no]deterministic: Deterministic evaluation (use mode of action distribution)
    (default: 'false')
  --[no]disjoint_paths: Use disjoint paths (k paths still considered)
    (default: 'false')
  --[no]end_first_blocking: End episode on first blocking event
    (default: 'false')
  --env_type: Environment type
    (default: 'vone')
  --gamma: Nonlinear coefficient
    (default: '0.0012')
    (a number)
  --gnn_latent: GNN latent size
    (default: '64')
    (an integer)
  --gnn_mlp_layers: Number of MLP layers
    (default: '2')
    (an integer)
  --guardband: Guard band in slots
    (default: '1')
    (an integer)
  --[no]incremental_loading: Incremental increase in traffic load (non-expiring requests)
    (default: 'false')
  --k: Number of paths
    (default: '5')
    (an integer)
  --lambda0: Wavelength [nm]
    (default: '1550.0')
    (a number)
  --link_resources: Number of link resources
    (default: '5')
    (an integer)
  --load: Load
    (default: '250.0')
    (a number)
  --[no]log_actions: Log actions taken and other details
    (default: 'false')
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
    (default: '25.0')
    (a number)
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
  --modulations_csv_filepath: Modulation format definitions for RSA environment
    (default: './examples/modulations.csv')
  --multiple_topologies_directory: Directory containing JSON definitions of network topologies that will be alternated per episode
  --node_heuristic: Node heuristic to be evaluated
    (default: 'random')
  --node_probs: List of node probabilities for selection
    (a comma separated list)
  --node_resources: Number of node resources
    (default: '4')
    (an integer)
  --[no]normalize_by_link_length: Normalize by link length
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
  --[no]random_traffic: Random traffic matrix for RSA on each reset (else uniform or custom)
    (default: 'false')
  --reward_type: Reward type
    (default: 'service')
  --scale_factor: Scale factor for link capacity (only used in RWA with lightpath reuse)
    (default: '1.0')
    (a number)
  --slot_size: Spectral width of frequency slot in GHz
    (default: '12.5')
    (a number)
  --span_length: Span length [km]
    (default: '100.0')
    (a number)
  --step_bw: Step size for requested bandwidth values between min and max
    (default: '1')
    (an integer)
  --step_traffic: Step size for traffic values between min and max
    (default: '0.1')
  --symbol_rate: Symbol rate [Gbaud]
    (default: '100.0')
    (a number)
  --topology_directory: Directory containing JSON definitions of network topologies
  --topology_name: Topology name
    (default: '4node')
  --traffic_requests_csv_filepath: Path to traffic request CSV file
  --[no]truncate_holding_time: Truncate holding time to less than 2*mean_service_holding_time
    (default: 'false')
  --values_bw: List of requested bandwidth values
    (a comma separated list)
  --virtual_topologies: Virtual topologies
    (default: '3_ring')
    (a comma separated list)
  --weight: Edge attribute name for ordering k-shortest paths

absl.flags:
  --flagfile: Insert flag definitions from the given file into the command line.
    (default: '')
  --undefok: comma-separated list of flag names that it is okay to specify on the command line even if the program does not define a flag with that name.  IMPORTANT: flags in this list that have arguments MUST use the --flag=value
    format.
    (default: '')
```