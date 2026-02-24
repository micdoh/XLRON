from absl import flags

# N.B. Use can pass the flag --flagfile=PATH_TO_FLAGFILE to add flags without typing them out

# Training hyperparameters
flags.DEFINE_integer("SEED", 42, "Random seed")
flags.DEFINE_integer(
    "NUM_LEARNERS",
    1,
    "Number of independent learners i.e. how many independent experiments to run "
    "with a unique set of learned parameters each",
)
flags.DEFINE_integer("NUM_DEVICES", 1, "Number of devices")
flags.DEFINE_integer("NUM_ENVS", 1, "Number of environments per device")
flags.DEFINE_integer("ROLLOUT_LENGTH", 150, "Number of steps per rollout per environment")
flags.DEFINE_integer(
    "NUM_UPDATES",
    1,
    "Number of parameter updates (not including multiple epochs) (calculated in train.py "
    "but included here so that it can be passed to the model)",
)
flags.DEFINE_integer("MINIBATCH_SIZE", 1, "Minibatch size")
flags.DEFINE_float(
    "TOTAL_TIMESTEPS",
    1e6,
    "Total number of timesteps. For reconfigurable routing bounds: total requests per trial.",
)
flags.DEFINE_integer("STEPS_PER_INCREMENT", 100000, "Number of steps per logging increment")
flags.DEFINE_integer("NUM_INCREMENTS", 1, "Number of increments to log")
flags.DEFINE_integer("UPDATE_EPOCHS", 1, "Number of epochs per update")
flags.DEFINE_integer("NUM_MINIBATCHES", 1, "Number of minibatches per update")
flags.DEFINE_integer("ACTION_DIM", 2, "Dimension of action space")
flags.DEFINE_integer("INPUT_DIM", 2, "Dimension of input space")

flags.DEFINE_float("LR", 5e-4, "Learning rate")
flags.DEFINE_float("GAMMA", 0.999, "Discount factor")
flags.DEFINE_float("GAE_LAMBDA", None, "GAE lambda parameter")
flags.DEFINE_float("INITIAL_LAMBDA", 0.9, "Initial lambda parameter for GAE")
flags.DEFINE_float("FINAL_LAMBDA", 0.98, "Final lambda parameter for GAE")
flags.DEFINE_float("CLIP_EPS", 0.2, "PPO clipping parameter")
flags.DEFINE_float("ENT_COEF", 0.0, "Entropy coefficient")
flags.DEFINE_float("VF_COEF", 0.5, "Value function coefficient")
flags.DEFINE_float("ADAM_EPS", 1e-5, "Adam epsilon")
flags.DEFINE_float("ADAM_BETA1", 0.9, "Adam beta1")
flags.DEFINE_float("ADAM_BETA2", 0.999, "Adam beta2")
flags.DEFINE_float("MAX_GRAD_NORM", 0.5, "Maximum gradient norm")
flags.DEFINE_string("ACTIVATION", "tanh", "Activation function")
flags.DEFINE_float("LOGR_CLIP", 10.0, "Log ratio clip to range +/- this value")
flags.DEFINE_float("ADV_CLIP", 10.0, "Advantage clip to range +/- this value")
flags.DEFINE_string("LR_SCHEDULE", "cosine", "Learning rate schedule")
flags.DEFINE_float(
    "LR_SCHEDULE_MULTIPLIER",
    1.0,
    "Multiply the LR schedule horizon by this factor",
)
flags.DEFINE_float(
    "LAMBDA_SCHEDULE_MULTIPLIER",
    1.0,
    "Multiply the GAE-lambda schedule horizon by this factor",
)
flags.DEFINE_float(
    "ENT_SCHEDULE_MULTIPLIER",
    1.0,
    "Multiply the entropy schedule horizon by this factor",
)
flags.DEFINE_float(
    "VML_SCHEDULE_MULTIPLIER",
    1.0,
    "Multiply the valid mass loss schedule horizon by this factor",
)
flags.DEFINE_float(
    "VF_SCHEDULE_MULTIPLIER",
    1.0,
    "Multiply the value function LR schedule horizon by this factor",
)
flags.DEFINE_float(
    "WARMUP_MULTIPLIER", 1, "Increase the learning rate warmup peak compared to init"
)
flags.DEFINE_float("WARMUP_STEPS_FRACTION", 0.2, "Fraction of total timesteps to use for warmup")
flags.DEFINE_float("LR_END_FRACTION", 0.1, "Fraction of init LR that is final LR")
flags.DEFINE_integer("NUM_LAYERS", 2, "Number of layers in actor and critic networks")
flags.DEFINE_integer("NUM_UNITS", 64, "Number of hidden units in actor and critic networks")
flags.DEFINE_float(
    "TEMPERATURE",
    1.0,
    "Temperature for softmax action selection "
    "(high temperature, more exploration) or (low temperature, more exploitation)",
)
flags.DEFINE_float("EPSILON", 1e-8, "Small number to prevent div zero")
# Additional training parameters
flags.DEFINE_string(
    "VISIBLE_DEVICES",
    None,
    "Comma-separated indices of (desired) visible GPUs e.g. 1,2,3",
)
flags.DEFINE_boolean("DETERMINISTIC_OPS", False, "Use deterministic GPU operations")
flags.DEFINE_boolean("PREALLOCATE_MEM", True, "Preallocate GPU memory")
flags.DEFINE_string("PREALLOCATE_MEM_FRACTION", "0.95", "Fraction of GPU memory to preallocate")
flags.DEFINE_boolean("PRINT_MEMORY_USE", False, "Print memory usage")
flags.DEFINE_boolean("WANDB", False, "Use wandb")
flags.DEFINE_boolean(
    "SAVE_MODEL",
    False,
    "Save model (will be saved to --MODEL_PATH locally and uploaded to wandb if --WANDB is True)",
)
flags.DEFINE_boolean("RETRAIN_MODEL", False, "Load model for retraining")
flags.DEFINE_boolean("OVERWRITE_MODEL", True, "Overwrite model saved at MODEL_PATH")
flags.DEFINE_boolean("DEBUG", False, "Debug mode")
flags.DEFINE_boolean("DEBUG_NANS", False, "Debug NaNs")
flags.DEFINE_boolean("NO_TRUNCATE", False, "Do not truncate printed arrays")
flags.DEFINE_boolean(
    "ORDERED",
    True,
    "Order print statements when debugging (must be false if using pmap)",
)
flags.DEFINE_boolean("PRINT_FLAGS", False, "Print flags")
flags.DEFINE_string("MODEL_PATH", None, "Path to save/load model")
flags.DEFINE_string("PROJECT", "", "Name of project (same as experiment name if unspecified)")
flags.DEFINE_string(
    "EXPERIMENT_NAME",
    "",
    "Name of experiment (equivalent to run name in wandb) "
    "(auto-generated based on other flags if unspecified)",
)
flags.DEFINE_integer("DOWNSAMPLE_FACTOR", 1, "Downsample factor to reduce data uploaded to wandb")
flags.DEFINE_boolean("DISABLE_JIT", False, "Disable JIT compilation")
flags.DEFINE_boolean("ENABLE_X64", False, "Enable x64 floating point precision")
flags.DEFINE_boolean("ACTION_MASKING", True, "Use invalid action masking")
flags.DEFINE_string(
    "DATA_OUTPUT_FILE", None, "Path to save JSONL run summary (one JSON object per line)"
)
flags.DEFINE_string("EPISODE_DATA_OUTPUT_FILE", None, "Path to save per-episode CSV data output")
flags.DEFINE_string(
    "TRAJ_DATA_OUTPUT_FILE", None, "Path to save trajectory (actions, etc.) data output"
)
flags.DEFINE_boolean("PLOTTING", False, "Plotting")
flags.DEFINE_string(
    "RENDER_EVAL_MODE",
    "off",
    "Eval-only rendering mode: off, save, or human. "
    "Only used for EVAL_HEURISTIC/EVAL_MODEL runs.",
)
flags.DEFINE_float("RENDER_FPS", 2.0, "Playback/recording frame rate for eval rendering")
flags.DEFINE_float(
    "RENDER_SCALE",
    0.6,
    "Scale factor for render figure resolution. Lower values render faster.",
)
flags.DEFINE_string(
    "RENDER_OUTPUT_FILE",
    None,
    "Path to save eval render recording (.gif/.mp4). "
    "If unset and mode includes save, defaults inside the run directory when available.",
)
flags.DEFINE_integer(
    "RENDER_MAX_STEPS",
    100,
    "Max requests/steps to render during eval.",
)
flags.DEFINE_boolean(
    "RENDER_CLICK_THROUGH",
    False,
    "If true and using human render mode, wait for key press between rendered requests.",
)
flags.DEFINE_integer("EMULATED_DEVICES", None, "Number of devices to emulate")
flags.DEFINE_boolean("log_actions", False, "Log actions taken and other details")
flags.DEFINE_boolean("log_path_lengths", False, "Log path length statistics")
flags.DEFINE_boolean("log_wrapper", True, "Wrap Env in LogEnvWrapper")
flags.DEFINE_boolean(
    "PROFILE",
    False,
    "Enable wall-clock profiling of environment and training components. "
    "On CPU, records fine-grained per-call section timings via host callbacks. "
    "On GPU, records first-call (compilation + first execution) latency per section. "
    "Prints a summary table of call counts, total/mean time, and percentage breakdown.",
)
flags.DEFINE_boolean(
    "COMPILE_RR_BOUNDS",
    False,
    "AOT-compile run_defrag in reconfigurable routing bounds (default: interpreted)",
)
flags.DEFINE_boolean("LOG_LOSS_INFO", True, "Log loss metrics")
flags.DEFINE_boolean("LOG_ALL_INFO", True, "Log every metric")
flags.DEFINE_boolean(
    "ENHANCED_LOGGING",
    False,
    "Enable enhanced diagnostic logging for PPO loss function "
    "(valid_frac, clip_frac, ratio stats, valid_mass stats, n_valid stats, adv stats, gate_frac)",
)
flags.DEFINE_boolean("DEBUG_LOSS", False, "Debug loss calculation")
flags.DEFINE_boolean("REWARD_CENTERING", False, "Use reward centering")
flags.DEFINE_float(
    "INITIAL_AVERAGE_REWARD",
    0.0,
    "Initial average reward estimate for reward centering",
)
flags.DEFINE_float(
    "REWARD_STEPSIZE",
    0.001,
    "Initial step size for reward centering average reward update",
)

# Prioritized Experience Replay flags
flags.DEFINE_float(
    "PRIO_ALPHA",
    0.0,
    "Priority exponent for prioritized experience replay "
    "(0.0 = uniform sampling, 1.0 = fully prioritized)",
)
flags.DEFINE_float(
    "PRIO_BETA0",
    1.0,
    "Initial importance sampling correction exponent "
    "(annealed to 1.0 over training, set to 1.0 to disable)",
)

# VTrace / Puffer Advantage flags
flags.DEFINE_float(
    "RHO_CLIP",
    -1.0,
    "Clip importance ratio for TD error in VTrace-style advantage calculation "
    "(set <= 0 for standard GAE without clipping)",
)
flags.DEFINE_float(
    "C_CLIP",
    -1.0,
    "Clip importance ratio for GAE accumulation in VTrace-style advantage calculation "
    "(set <= 0 for standard GAE without clipping)",
)
flags.DEFINE_boolean(
    "USE_RNN",
    False,
    "Use RNN-based policy (affects prioritization: trajectory-level vs sample-level)",
)
flags.DEFINE_boolean("KEEP_VF", "False", "Load the pre-trained value function")

# Entropy scheduling flags
flags.DEFINE_string("ENT_SCHEDULE", "constant", "Enable entropy coefficient scheduling")
flags.DEFINE_float(
    "ENT_END_FRACTION",
    0.1,
    "Fraction of initial entropy coefficient that is final entropy coefficient",
)
flags.DEFINE_float("WEIGHT_DECAY", 0.0, "Weight decay for optimizer")

# Separate value function optimizer flags
flags.DEFINE_boolean(
    "SEPARATE_VF_OPTIMIZER",
    False,
    "Use a separate optimizer for the value function (critic) with its own hyperparameters",
)
flags.DEFINE_float("VF_LR", None, "Learning rate for value function optimizer (default: LR / 3)")
flags.DEFINE_string(
    "VF_LR_SCHEDULE",
    None,
    "Learning rate schedule for VF optimizer (default: same as LR_SCHEDULE)",
)
flags.DEFINE_float(
    "VF_LR_END_FRACTION",
    None,
    "Fraction of init VF LR that is final VF LR (default: same as LR_END_FRACTION)",
)
flags.DEFINE_float(
    "VF_WARMUP_MULTIPLIER",
    None,
    "VF LR warmup peak multiplier (default: same as WARMUP_MULTIPLIER)",
)
flags.DEFINE_float(
    "VF_WARMUP_STEPS_FRACTION",
    None,
    "Fraction of total timesteps for VF LR warmup (default: same as WARMUP_STEPS_FRACTION)",
)
flags.DEFINE_float("VF_ADAM_EPS", None, "Adam epsilon for VF optimizer (default: same as ADAM_EPS)")
flags.DEFINE_float(
    "VF_ADAM_BETA1", None, "Adam beta1 for VF optimizer (default: same as ADAM_BETA1)"
)
flags.DEFINE_float(
    "VF_ADAM_BETA2", None, "Adam beta2 for VF optimizer (default: same as ADAM_BETA2)"
)
flags.DEFINE_float(
    "VF_MAX_GRAD_NORM", None, "Max gradient norm for VF optimizer (default: same as MAX_GRAD_NORM)"
)
flags.DEFINE_float(
    "VF_WEIGHT_DECAY", None, "Weight decay for VF optimizer (default: same as WEIGHT_DECAY)"
)

flags.DEFINE_boolean(
    "STEP_ON_GRADIENT", False, "Whether to step schedule on gradient update or per update loop"
)

# Reward scaling flag
flags.DEFINE_float(
    "REWARD_SCALE", 1.0, "Reward scaling factor (multiply all rewards by this value)"
)
# Include "no op"
flags.DEFINE_boolean("include_no_op", False, "Whether to include a NO OP action.")
flags.DEFINE_boolean(
    "OFF_POLICY_IAM",
    False,
    "Use off-policy invalid action masking i.e. log prob ratio is unmasked policy / masked",
)
flags.DEFINE_float(
    "VALID_MASS_TARGET",
    0.05,
    "Target valid mass threshold for soft damping of actor and entropy losses "
    "(valid mass below this value linearly damps the loss contribution)",
)
flags.DEFINE_float(
    "VALID_MASS_LOSS_COEF",
    0.0,
    "Coefficient for valid mass loss term that encourages the unmasked policy to place "
    "probability on valid actions (0.0 = disabled)",
)
flags.DEFINE_string(
    "VML_SCHEDULE", "constant", "Valid mass loss coefficient schedule (constant, linear, cosine)"
)
flags.DEFINE_float(
    "VML_END_FRACTION",
    10.0,
    "Fraction of initial VALID_MASS_LOSS_COEF that is final coefficient "
    "(>1.0 anneals upward, e.g. 10.0 means final = 10x initial)",
)

# Flags for mixed precision
flags.DEFINE_string("compute_dtype", None, "Compute precision dtype (float32, bfloat16)")
flags.DEFINE_string("params_dtype", None, "Parameter storage dtype (float32, bfloat16)")
flags.DEFINE_string("float_dtype", None, "Float dtype (float32, bfloat16)")
flags.DEFINE_string("large_float_dtype", None, "Large float dtype (float32, bfloat16)")
flags.DEFINE_string("small_float_dtype", None, "Small float dtype (float32, float16)")
flags.DEFINE_string("int_dtype", None, "Integer dtype (int32)")
flags.DEFINE_string("large_int_dtype", None, "Large integer dtype (int32)")
flags.DEFINE_string("small_int_dtype", None, "Small integer dtype (int32, int8)")
flags.DEFINE_string("binary_dtype", None, "Binary dtype (bool)")
flags.DEFINE_string("reward_dtype", None, "Reward dtype (float32)")
flags.DEFINE_string("action_dtype", None, "Action dtype (int32)")

# Environment parameters
flags.DEFINE_string("env_type", "rmsa", "Environment type")
flags.DEFINE_float(
    "load",
    250,
    "Traffic load in Erlangs (arrival_rate = load / mean_service_holding_time). "
    "Also used as the single traffic load point for capacity bound estimation.",
)
flags.DEFINE_float("mean_service_holding_time", 25, "Mean service holding time")
flags.DEFINE_integer("k", 5, "Number of paths")
flags.DEFINE_string("topology_name", "4node", "Topology name")
flags.DEFINE_integer("link_resources", 5, "Number of link resources")
flags.DEFINE_float(
    "max_requests",
    4,
    "Maximum number of requests in an episode. "
    "For cut-set capacity bounds: number of requests per trial.",
)
flags.DEFINE_integer("min_bw", 25, "Minimum requested bandwidth")
flags.DEFINE_integer("max_bw", 100, "Maximum requested bandwidth")
flags.DEFINE_integer("step_bw", 1, "Step size for requested bandwidth values between min and max")
flags.DEFINE_string("values_bw", None, "List of requested bandwidth values")
flags.DEFINE_float("slot_size", 12.5, "Spectral width of frequency slot in GHz")
flags.DEFINE_boolean(
    "incremental_loading",
    False,
    "Incremental increase in traffic load (non-expiring requests)",
)
flags.DEFINE_boolean("end_first_blocking", False, "End episode on first blocking event")
flags.DEFINE_boolean(
    "terminate_on_episode_end",
    False,
    "If True, treat max_requests reached as terminated (not truncated). "
    "When False (default), reaching max_requests is treated as truncation.",
)
flags.DEFINE_boolean(
    "continuous_operation",
    False,
    "If True, do not reset the environment at the end of an episode",
)
flags.DEFINE_integer("aggregate_slots", 1, "Number of slots to aggregate into a single action")
flags.DEFINE_boolean("disjoint_paths", False, "Use disjoint paths (k paths still considered)")
flags.DEFINE_integer("guardband", 1, "Guard band in slots")
flags.DEFINE_integer(
    "symbol_rate", 100, "Symbol rate in Gbaud (only used in RWA with lightpath reuse"
)
flags.DEFINE_float(
    "scale_factor",
    1.0,
    "Scale factor for link capacity (only used in RWA with lightpath reuse)",
)
flags.DEFINE_string(
    "path_sort_criteria",
    "spectral_resources",
    "How paths should be sorted. Must be one of 'spectral_resources' (default), 'hops', 'distance', 'hops_distance', 'capacity' (for RWA-LR only)",
)
flags.DEFINE_string(
    "modulations_csv_filepath",
    "./xlron/data/modulations/modulations_deeprmsa.csv",
    "Modulation format definitions for RSA environment",
)
flags.DEFINE_boolean(
    "calc_minimum_osnr",
    True,
    "Calculate minimum OSNR column from spectral efficiency using GSNR threshold formula "
    "instead of reading pre-specified values from the modulations CSV",
)
flags.DEFINE_float(
    "beta_fec",
    1.5e-2,
    "Pre-FEC BER target for GSNR threshold calculation (used with --calc_minimum_osnr)",
)
flags.DEFINE_float(
    "fec_rate",
    0.8,
    "FEC code rate (e.g. 0.8 = 20%% overhead). In rsa_gn_model it sets "
    "fec_threshold = 1 - fec_rate for the Shannon capacity formula; in rmsa_gn_model it scales "
    "accepted bitrate (effective_bitrate = requested_bitrate * fec_rate).",
)
flags.DEFINE_string("traffic_requests_csv_filepath", None, "Path to traffic request CSV file")
flags.DEFINE_string(
    "topology_directory",
    None,
    "Directory containing JSON definitions of network topologies",
)
flags.DEFINE_string(
    "multiple_topologies_directory",
    None,
    "Directory containing JSON definitions of network topologies that will be alternated per episode",
)
flags.DEFINE_float("traffic_intensity", 0, "Traffic intensity (arrival rate * mean holding time)")
flags.DEFINE_boolean(
    "maximise_throughout",
    False,
    "Maximise throughput instead of minimising blocking probability",
)
flags.DEFINE_string("reward_type", "service", "Reward type")
flags.DEFINE_boolean(
    "truncate_holding_time",
    False,
    "Truncate holding time to less than 2*mean_service_holding_time",
)
flags.DEFINE_integer("ENV_WARMUP_STEPS", 0, "Number of warmup steps before training or eval")
flags.DEFINE_boolean(
    "pack_path_bits",
    False,
    "Pack path bits to save memory, then unpack when path row is selected",
)
flags.DEFINE_boolean(
    "relative_arrival_times",
    True,
    "Don't track the absolute current time, just the relative time since the last request",
)
# RSA-specific environment parameters
flags.DEFINE_boolean(
    "random_traffic",
    False,
    "Generate a new random traffic matrix on each episode reset, so source-destination "
    "pair probabilities vary between episodes. When False, the traffic matrix is fixed "
    "for the entire run (uniform by default, or loaded from a CSV via "
    "--custom_traffic_matrix_csv_filepath).",
)
flags.DEFINE_string(
    "custom_traffic_matrix_csv_filepath",
    None,
    "Path to a CSV file specifying a custom traffic matrix. The CSV should be an NxN "
    "matrix (N = number of nodes) where entry (i, j) gives the relative traffic demand "
    "from node i to node j. Leave blank to use uniform traffic (equal probability for "
    "all source-destination pairs). Ignored when --random_traffic is enabled.",
)
flags.DEFINE_float(
    "alpha",
    0.2,
    "Legacy fibre attenuation alpha [dB/km] used by the path-capacity approximation "
    "(e.g., rwa_lightpath_reuse). GN-model environments use attenuation/attenuation_bar in [1/m].",
)
flags.DEFINE_float(
    "beta_2",
    -21.7,
    "Second-order dispersion parameter [ps^2/km] for the legacy path-capacity calculation "
    "(e.g., rwa_lightpath_reuse). GN-model environments use dispersion_coeff/dispersion_slope instead.",
)
flags.DEFINE_float(
    "gamma",
    1.2,
    "Legacy nonlinear coefficient used by the path-capacity approximation "
    "(e.g., rwa_lightpath_reuse). GN-model environments use nonlinear_coefficient instead.",
)
flags.DEFINE_float("span_length", 100, "Span length [km]")
flags.DEFINE_float(
    "span_lumped_loss_db",
    None,
    "Optional per-span lumped loss [dB] compensated by inline EDFAs in GN model ASE calculations. "
    "If unset, no additional lumped span loss is applied.",
)
flags.DEFINE_float("lambda0", 1550, "Wavelength [nm]")
# VONE-specific environment parameters
flags.DEFINE_integer("node_resources", 4, "Number of node resources")
flags.DEFINE_string("virtual_topologies", "3_ring", "Virtual topologies")
flags.DEFINE_integer("min_node_resources", 1, "Minimum number of node resources")
flags.DEFINE_integer("max_node_resources", 1, "Maximum number of node resources")
flags.DEFINE_string("node_probs", None, "List of node probabilities for selection")
# Heuristic-specific parameters
flags.DEFINE_boolean("EVAL_HEURISTIC", False, "Evaluate heuristic")
flags.DEFINE_string("path_heuristic", "ksp_ff", "Path heuristic to be evaluated")
flags.DEFINE_string("node_heuristic", "random", "Node heuristic to be evaluated")
# GNN-specific parameters
flags.DEFINE_boolean("USE_GNN", False, "Use GNN")
flags.DEFINE_integer("num_spectral_features", 8, "No. of spectral features")
flags.DEFINE_boolean(
    "DISABLE_NODE_FEATURES",
    False,
    "Use node features in the GNN (spectral and source-dest)",
)
flags.DEFINE_integer("message_passing_steps", 3, "Number of message passing steps")
flags.DEFINE_integer("mlp_layers", None, "Number of MLP layers")
flags.DEFINE_integer("mlp_latent", None, "Size of MLP latent dimension")
flags.DEFINE_integer("edge_embedding_size", 128, "Size of edge embeddings")
flags.DEFINE_integer("edge_mlp_layers", 2, "Number of edge MLP layers")
flags.DEFINE_integer("edge_mlp_latent", 128, "Size of edge MLP latent dimension")
flags.DEFINE_integer("edge_output_size_actor", 0, "Size of edge output for actor (not used)")
flags.DEFINE_integer("edge_output_size_critic", 1, "Size of edge output for critic")
flags.DEFINE_integer("global_embedding_size", 8, "Size of global embeddings")
flags.DEFINE_integer("global_mlp_layers", 1, "Number of global MLP layers")
flags.DEFINE_integer("global_mlp_latent", 16, "Size of global MLP latent dimension")
flags.DEFINE_integer("global_output_size_actor", 0, "Size of global output for actor")
flags.DEFINE_integer("global_output_size_critic", 1, "Size of global output for critic")
flags.DEFINE_integer("node_embedding_size", 16, "Size of node embeddings")
flags.DEFINE_integer("node_mlp_layers", 2, "Number of node MLP layers")
flags.DEFINE_integer("node_mlp_latent", 128, "Size of node MLP latent dimension")
flags.DEFINE_integer("node_output_size_actor", 0, "Size of node output for actor")
flags.DEFINE_integer("node_output_size_critic", 0, "Size of node output for critic")
flags.DEFINE_integer("attn_mlp_layers", 1, "Number of attention MLP layers")
flags.DEFINE_integer("attn_mlp_latent", 64, "Size of attention MLP latent dimension")
flags.DEFINE_boolean("normalize_by_link_length", False, "Normalize by link length")
flags.DEFINE_boolean("gnn_layer_norm", True, "Use layer normalization in GNN")
flags.DEFINE_boolean("mlp_layer_norm", False, "Use layer normalization in MLPs of GNN")

# Transformer-specific parameters
flags.DEFINE_boolean("USE_TRANSFORMER", False, "Use Transformer architecture")
flags.DEFINE_string(
    "transformer_obs_type",
    "departure",
    "Type of observation to feed to transformer. \
    1. 'departure': use departure times. \
    2. 'occupancy': use link occupancy (binary). \
    3. 'capacity': use remaining capacity array (for use with RWA-LR environment).",
)
flags.DEFINE_integer("transformer_embedding_size", 128, "Size of transformer token embeddings")
flags.DEFINE_integer(
    "transformer_intermediate_size",
    256,
    "Size of intermediate layer in transformer feed-forward blocks",
)
flags.DEFINE_integer("transformer_num_layers", 1, "Number of transformer encoder layers")
flags.DEFINE_integer("transformer_num_heads", 4, "Number of attention heads in transformer")
flags.DEFINE_integer(
    "num_wire_features",
    8,
    "Number of spectral features for WiRE positional encodings (computed from line graph Laplacian)",
)
flags.DEFINE_float("transformer_dropout_rate", 0.1, "Dropout rate for transformer layers")
flags.DEFINE_float(
    "transformer_attention_dropout_rate",
    0.1,
    "Dropout rate for transformer attention layers",
)
flags.DEFINE_boolean(
    "transformer_share_layers",
    False,
    "Share encoder layers between actor and critic in transformer",
)
flags.DEFINE_integer("transformer_actor_mlp_width", 128, "Width of actor MLP head in transformer")
flags.DEFINE_integer("transformer_critic_mlp_width", 128, "Width of critic MLP head in transformer")
flags.DEFINE_integer("transformer_actor_mlp_depth", 1, "Depth of actor MLP head in transformer")
flags.DEFINE_integer("transformer_critic_mlp_depth", 2, "Depth of critic MLP head in transformer")
flags.DEFINE_boolean(
    "transformer_enable_dropout", False, "Enable dropout during training in transformer"
)

# Eval during training parameters
flags.DEFINE_boolean(
    "EVAL_DURING_TRAINING", False, "Run periodic evaluation during training to track best model"
)
flags.DEFINE_integer(
    "EVAL_TIMESTEPS", 10000, "Number of timesteps per eval run (0 = use STEPS_PER_INCREMENT)"
)
flags.DEFINE_integer("EVAL_FREQUENCY", 1, "Run eval every N increments")

# Model evaluation parameters
flags.DEFINE_boolean("EVAL_MODEL", False, "Load model for evaluation")
flags.DEFINE_string("min_traffic", "0.0", "Minimum traffic")
flags.DEFINE_string("max_traffic", "1.0", "Maximum traffic")
flags.DEFINE_string("step_traffic", "0.1", "Step size for traffic values between min and max")
flags.DEFINE_boolean(
    "deterministic", False, "Deterministic evaluation (use mode of action distribution)"
)

# GN model parameters
flags.DEFINE_float("ref_lambda", 1564e-9, "Reference wavelength [m]")
flags.DEFINE_float("max_power_per_fibre", 23.0, "Max launch power per fibre [dBm]")
flags.DEFINE_float(
    "power_per_channel",
    None,
    "Per-channel launch power [dBm]. If None, defaults to max_power_per_fibre divided equally among slots.",
)
flags.DEFINE_string(
    "power_per_channel_per_band",
    None,
    "Comma-separated per-channel launch power values [dBm], one per band in "
    "band_preference order (e.g. '2.3,2.5' for C,L). Overrides --power_per_channel.",
)
flags.DEFINE_string(
    "launch_power_csv",
    None,
    "Path to a CSV file specifying per-slot launch power. "
    "Expected columns: slot_index (int), freq_ghz (float), power_dbm (float). "
    "Slots not present in the file keep the default launch power. "
    "Overrides --power_per_channel and --power_per_channel_per_band.",
)
flags.DEFINE_string(
    "launch_power_type",
    "fixed",
    "Can be fixed (same power per transceiver), "
    "tabular (power depends on path), or rl (power selected by agent).",
)
flags.DEFINE_float(
    "nonlinear_coefficient",
    1.2e-3,
    "GN-model nonlinear coefficient [1/W^2] used in ISRS/GN NLI calculations "
    "(rsa_gn_model, rmsa_gn_model).",
)
flags.DEFINE_float(
    "raman_gain_slope",
    2.8e-17,
    "Raman gain slope [1/(W*m*Hz)]. Typical value ~0.028 1/(W*km*THz) = 2.8e-17 in SI.",
)
flags.DEFINE_float(
    "attenuation",
    4.605111673e-5,
    "GN-model attenuation coefficient a [1/m] used in ISRS/GN NLI and ASE calculations.",
)
flags.DEFINE_float(
    "attenuation_bar",
    4.605111673e-5,
    "GN-model attenuation coefficient a_bar [1/m] used alongside attenuation in ISRS/GN calculations.",
)
flags.DEFINE_float(
    "dispersion_coeff",
    17e-6,
    "Dispersion coefficient D [s/m^2] for GN-model/ISRS calculations; used to derive beta2 "
    "as a function of wavelength.",
)
flags.DEFINE_float(
    "dispersion_slope",
    60.7,
    "Dispersion slope dD/dlambda [s/m^3] for GN-model/ISRS calculations; used with dispersion_coeff "
    "for frequency-dependent beta2/beta3 terms.",
)
flags.DEFINE_boolean("coherent", True, "Add NLI contribution coherently per span")
flags.DEFINE_boolean(
    "mod_format_correction", False, "Apply non-Gaussian modulation format correction"
)
flags.DEFINE_multi_integer("interband_gap_width", None, "Gap between bands [GHz]")
flags.DEFINE_multi_integer("interband_gap_start", None, "Start index of gap between bands [GHz]")
flags.DEFINE_boolean(
    "enforce_band_gaps",
    True,
    "Compute and enforce band boundary gaps from CSV data (non-GN-model multiband envs). "
    "GN model envs always enforce band gaps when band_preference is set.",
)
flags.DEFINE_float(
    "snr_margin",
    0.5,
    "Margin required for estimated SNR for mod. format selection [dB]",
)
flags.DEFINE_float(
    "max_snr",
    30.0,
    "Maximum SNR that a link can take (just used for normalization purposes)",
)
flags.DEFINE_float(
    "min_snr",
    7.0,
    "Minimum SNR for any data transmission",
)
flags.DEFINE_float("max_power", 0.5, "Maximum launch power [dBm]")
flags.DEFINE_float("min_power", -5, "Minimum launch power [dBm]")
flags.DEFINE_float("step_power", 0.1, "Step size for launch power values between min and max")
flags.DEFINE_boolean("discrete_launch_power", False, "Discrete launch power values")
flags.DEFINE_float("min_concentration", 0.1, "For continuous launch power dist.")
flags.DEFINE_float("max_concentration", 20.0, "For continuous launch power dist.")
flags.DEFINE_boolean("last_fit", False, "Use KSP-FF for path_action, else KSP-LF")
flags.DEFINE_boolean(
    "GNN_OUTPUT_LP",
    False,
    "Use GNN for launch power optimization in RSA GN Model environment",
)
flags.DEFINE_boolean("GNN_OUTPUT_RSA", False, "Use GNN for RSA in RSA GN Model environment")
flags.DEFINE_string(
    "noise_data_filepath",
    None,
    "Path to GN-model noise CSV. This file provides transceiver back-to-back SNR and "
    "EDFA noise figure values, and can specify them per band or per sub-band. During "
    "GN-model calculations these values set the transceiver B2B SNR and amplifier noise "
    "figure used for each selected band/sub-band; if omitted, built-in defaults are used.",
)
flags.DEFINE_string(
    "band_data_filepath",
    None,
    "Path to band definition CSV file for computing band gaps in GN model",
)
flags.DEFINE_string(
    "band_preference",
    None,
    "Comma-separated band preference order for first-fit/last-fit heuristics in GN model "
    "environments (e.g. 'C,L,S,U,E,O'). First-fit exhausts slots in preferred band order.",
)
flags.DEFINE_string(
    "slots_per_band",
    None,
    "Comma-separated number of slots per band (e.g. '45,45'). Must match number of bands in "
    "band_preference. Overrides the default behaviour of filling entire band width with slots. "
    "Gap slots between bands are still added as normal.",
)
flags.DEFINE_float(
    "inter_band_gap_ghz",
    25.0,
    "Spectral width of inter-band gap in GHz (~0.2nm at 1550nm). "
    "Used in GN model envs to set the physical gap width between bands.",
)
flags.DEFINE_boolean(
    "uniform_spans", True, "Use uniform spans (on by default: simplifies calculations)"
)
flags.DEFINE_integer(
    "num_subchannels",
    1,
    "Number of Nyquist subchannels per frequency slot for GN model NLI calculation. "
    "Divides each slot's bandwidth into N subchannels with effective baud rate "
    "B_eff = slot_size / num_subchannels, reducing SPM. XPM and ASE are unchanged. "
    "Best used with slot_size equal to the desired channel bandwidth "
    "(e.g. slot_size=100, num_subchannels=8 models 8x12.5 GHz subcarriers).",
)
# Distributed Raman Amplification parameters
flags.DEFINE_boolean(
    "use_raman_amp", False, "Enable Distributed Raman Amplification model for NLI calculation"
)
flags.DEFINE_string(
    "raman_pump_power_fw",
    None,
    "Forward Raman pump powers in Watts (comma-separated, one per pump)",
)
flags.DEFINE_string(
    "raman_pump_power_bw",
    None,
    "Backward Raman pump powers in Watts (comma-separated, one per pump)",
)
flags.DEFINE_string(
    "raman_pump_freq_fw",
    None,
    "Forward Raman pump frequencies in Hz (comma-separated, one per pump)",
)
flags.DEFINE_string(
    "raman_pump_freq_bw",
    None,
    "Backward Raman pump frequencies in Hz (comma-separated, one per pump)",
)
flags.DEFINE_float(
    "raman_max_bandwidth_thz",
    15.0,
    "Maximum modulated bandwidth in THz for DRA triangular Raman approximation validity. "
    "When DRA is enabled, bands are trimmed to fit within this limit.",
)

# Flags for optimize_launch_power.py
flags.DEFINE_boolean(
    "optimise_launch_power",
    False,
    "Use deteministic requests from list_of_requests to optimise launch power",
)
flags.DEFINE_integer("EVAL_STEPS", 100, "Number of steps to run in each evaluation")
flags.DEFINE_integer("OPTIMIZATION_ITERATIONS", 1000, "Number of optimization iterations")
flags.DEFINE_boolean("traffic_array", False, "Use traffic array")
flags.DEFINE_list("list_of_requests", None, "Traffic request list")
# Flags for finding cut-sets only
flags.DEFINE_boolean(
    "CUTSET_EXHAUSTIVE",
    False,
    "Use exhaustive search method to find cut-sets (else shortest paths method). The exhaustive method is recommended unless the graph is large (40+ nodes), which requires either parallelization of the search on GPU or use the approximate shortest paths method.",
)
flags.DEFINE_integer(
    "CUTSET_PARALLEL_PROCESSES",
    1,
    "How many parallel processes to launch in exhaustive cut-sets search.",
)
flags.DEFINE_integer(
    "CUTSET_BATCH_SIZE",
    512,
    "Batch size for cut-set generation (only relevant when finding cutsets exhaustively for larger networks (40+ nodes) on GPU)",
)
flags.DEFINE_integer(
    "CUTSET_ITERATIONS",
    32,
    "Number of iterations per parallel process (only relevant when finding cutsets exhaustively for larger networks (40+ nodes) on GPU)",
)
flags.DEFINE_integer("CUTSET_TOP_K", 256, "Number of top congested cutsets to return")
# Shared capacity bound estimation flags
flags.DEFINE_integer(
    "num_trials",
    10,
    "Number of independent random-seed trials for capacity bound estimation "
    "(used by both cut-set and reconfigurable routing bounds)",
)
flags.DEFINE_string(
    "cutset_link_selection_mode",
    "least_congested",
    "Link selection heuristic for cut-set capacity bound simulation: "
    "least_congested, most_congested, best_fit, random",
)
# Flags for capacity estimation with Baroni (reconfigurable routing / resource-prioritized defragmentation) method
flags.DEFINE_boolean("deterministic_requests", False, "Use deterministic requests")
flags.DEFINE_boolean(
    "sort_requests", True, "Sort requests in descending order of required resources"
)
# Flags for defining approximation parameters for differentiable functions and optimization
flags.DEFINE_boolean("INITIALIZE_ACTIONS_HEURISTIC", False, "Initialize actions with heuristic")
flags.DEFINE_boolean("INITIALIZE_ACTIONS_RANDOM", False, "Initialize actions randomly")
flags.DEFINE_boolean("INITIALIZE_ACTIONS_ASCENDING", False, "Initialize actions in ascending order")
flags.DEFINE_boolean(
    "INITIALIZE_ACTIONS_DESCENDING", False, "Initialize actions in descending order"
)
flags.DEFINE_boolean("INITIALIZE_ACTIONS_MAX", False, "Initialize actions at max. value")
flags.DEFINE_float(
    "temperature",
    5.0,
    "Temperature for differentiable function approximations (higher temp. = closer to original function)",
)
flags.DEFINE_boolean(
    "differentiable",
    False,
    "Enable differentiable mode for gradient-based optimization (uses straight-through estimators and temperature approximations)",
)
# Add differentiable optimization-specific flags
flags.DEFINE_boolean(
    "ACTION_OPTIMIZATION",
    False,
    "Directly optimise rollout actions using first-order gradients from differentiable environment",
)
flags.DEFINE_float(
    "OPTIMIZATION_LEARNING_RATE", 0.05, "Learning rate for gradient-based action optimization"
)
flags.DEFINE_boolean("PATH_SLOT_ACTIONS", False, "Use 2-part path-slot actions for optimization")
