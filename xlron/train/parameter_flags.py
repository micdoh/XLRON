from absl import flags

# N.B. Use can pass the flag --flagfile=PATH_TO_FLAGFILE to add flags without typing them out

# Training hyperparameters
flags.DEFINE_integer("SEED", 42, "Random seed")
flags.DEFINE_integer("NUM_LEARNERS", 1, "Number of independent learners i.e. how many independent experiments to run "
                                        "with a unique set of learned parameters each")
flags.DEFINE_integer("NUM_DEVICES", 1, "Number of devices")
flags.DEFINE_integer("NUM_ENVS", 1, "Number of environments per device")
flags.DEFINE_integer("ROLLOUT_LENGTH", 150, "Number of steps per rollout per environment")
flags.DEFINE_integer("NUM_UPDATES", 1, "Number of rollouts per environment")
flags.DEFINE_integer("MINIBATCH_SIZE", 1, "Minibatch size")
flags.DEFINE_float("TOTAL_TIMESTEPS", 1e6, "Total number of timesteps")
flags.DEFINE_integer("UPDATE_EPOCHS", 10, "Number of epochs per update")
flags.DEFINE_integer("NUM_MINIBATCHES", 1, "Number of minibatches per update")

flags.DEFINE_float("LR", 5e-4, "Learning rate")
flags.DEFINE_float("GAMMA", 0.999, "Discount factor")
flags.DEFINE_float("GAE_LAMBDA", 0.95, "GAE lambda parameter")
flags.DEFINE_float("CLIP_EPS", 0.2, "PPO clipping parameter")
flags.DEFINE_float("ENT_COEF", 0.0, "Entropy coefficient")
flags.DEFINE_float("VF_COEF", 0.5, "Value function coefficient")
flags.DEFINE_float("ADAM_EPS", 1e-5, "Adam epsilon")
flags.DEFINE_float("ADAM_BETA1", 0.9, "Adam beta1")
flags.DEFINE_float("ADAM_BETA2", 0.999, "Adam beta2")
flags.DEFINE_boolean("LAYER_NORM", False, "Use layer normalization")
flags.DEFINE_float("MAX_GRAD_NORM", 0.5, "Maximum gradient norm")
flags.DEFINE_string("ACTIVATION", "tanh", "Activation function")
flags.DEFINE_string("LR_SCHEDULE", "warmup_cosine", "Learning rate schedule")
flags.DEFINE_float("SCHEDULE_MULTIPLIER", 1, "Increase the learning rate schedule horizon "
                                               "by this factor (to keep schedule for longer final training runs "
                                               "consistent with that from tuning runs)")
flags.DEFINE_float("WARMUP_PEAK_MULTIPLIER", 1, "Increase the learning rate warmup peak compared to init")
flags.DEFINE_float("WARMUP_STEPS_FRACTION", 0.2, "Fraction of total timesteps to use for warmup")
flags.DEFINE_float("WARMUP_END_FRACTION", 0.1, "Fraction of init LR that is final LR")
flags.DEFINE_integer("NUM_LAYERS", 2, "Number of layers in actor and critic networks")
flags.DEFINE_integer("NUM_UNITS", 64, "Number of hidden units in actor and critic networks")
# Additional training parameters
flags.DEFINE_string("VISIBLE_DEVICES", "0", "Comma-separated indices of (desired) visible GPUs e.g. 1,2,3")
flags.DEFINE_boolean("PREALLOCATE_MEM", True, "Preallocate GPU memory")
flags.DEFINE_string("PREALLOCATE_MEM_FRACTION", "0.95", "Fraction of GPU memory to preallocate")
flags.DEFINE_boolean("PRINT_MEMORY_USE", False, "Print memory usage")
flags.DEFINE_boolean("WANDB", False, "Use wandb")
flags.DEFINE_boolean("SAVE_MODEL", False, "Save model (will be saved to --MODEL_PATH locally and uploaded to wandb if --WANDB is True)")
flags.DEFINE_boolean("DEBUG", False, "Debug mode")
flags.DEFINE_boolean("DEBUG_NANS", False, "Debug NaNs")
flags.DEFINE_boolean("NO_TRUNCATE", False, "Do not truncate printed arrays")
flags.DEFINE_boolean("ORDERED", True, "Order print statements when debugging "
                                      "(must be false if using pmap)")
flags.DEFINE_boolean("NO_PRINT_FLAGS", False, "Do not print flags")
flags.DEFINE_string("MODEL_PATH", None, "Path to save/load model")
flags.DEFINE_string("PROJECT", "", "Name of project (same as experiment name if unspecified)")
flags.DEFINE_string("EXPERIMENT_NAME", "", "Name of experiment (equivalent to run name in wandb) "
                                           "(auto-generated based on other flags if unspecified)")
flags.DEFINE_integer("DOWNSAMPLE_FACTOR", 1, "Downsample factor to reduce data uploaded to wandb")
flags.DEFINE_boolean("DISABLE_JIT", False, "Disable JIT compilation")
flags.DEFINE_boolean("ENABLE_X64", False, "Enable x64 floating point precision")
flags.DEFINE_boolean("ACTION_MASKING", False, "Use invalid action masking")
flags.DEFINE_boolean("LOAD_MODEL", False, "Load model for retraining or evaluation")
flags.DEFINE_string("DATA_OUTPUT_FILE", None, "Path to save data output")
flags.DEFINE_boolean("PLOTTING", False, "Plotting")
flags.DEFINE_integer("EMULATED_DEVICES", None, "Number of devices to emulate")
flags.DEFINE_boolean("log_actions", False, "Log actions taken and other details")
flags.DEFINE_boolean("PROFILE", False, "Profile programme with perfetto")
# Environment parameters
flags.DEFINE_string("env_type", "vone", "Environment type")
flags.DEFINE_float("load", 250, "Load")
flags.DEFINE_float("mean_service_holding_time", 25, "Mean service holding time")
flags.DEFINE_integer("k", 5, "Number of paths")
flags.DEFINE_string("topology_name", "4node", "Topology name")
flags.DEFINE_integer("link_resources", 5, "Number of link resources")
flags.DEFINE_float("max_requests", 4, "Maximum number of requests in an episode")
flags.DEFINE_float("max_timesteps", 30, "Maximum number of timesteps in an episode")
flags.DEFINE_integer("min_bw", 25, "Minimum requested bandwidth")
flags.DEFINE_integer("max_bw", 100, "Maximum requested bandwidth")
flags.DEFINE_integer("step_bw", 1, "Step size for requested bandwidth values between min and max")
flags.DEFINE_list("values_bw", None, "List of requested bandwidth values")
flags.DEFINE_float("slot_size", 12.5, "Spectral width of frequency slot in GHz")
flags.DEFINE_boolean("incremental_loading", False, "Incremental increase in traffic load (non-expiring requests)")
flags.DEFINE_boolean("end_first_blocking", False, "End episode on first blocking event")
flags.DEFINE_boolean("continuous_operation", False, "If True, do not reset the environment at the end of an episode")
flags.DEFINE_integer("aggregate_slots", 1, "Number of slots to aggregate into a single action")
flags.DEFINE_boolean("disjoint_paths", False, "Use disjoint paths (k paths still considered)")
flags.DEFINE_integer("guardband", 1, "Guard band in slots")
flags.DEFINE_integer("symbol_rate", 100, "Symbol rate in Gbaud (only used in RWA with lightpath reuse")
flags.DEFINE_float("scale_factor", 1.0, "Scale factor for link capacity (only used in RWA with lightpath reuse)")
flags.DEFINE_string("weight", None, "Edge attribute name for ordering k-shortest paths")
flags.DEFINE_string("modulations_csv_filepath", "./examples/modulations.csv", "Modulation format definitions for RSA environment")
flags.DEFINE_string("traffic_requests_csv_filepath", None, "Path to traffic request CSV file")
flags.DEFINE_string("topology_directory", None, "Directory containing JSON definitions of network topologies")
flags.DEFINE_string("multiple_topologies_directory", None,
                    "Directory containing JSON definitions of network topologies that will be alternated per episode")
flags.DEFINE_float("traffic_intensity", 0, "Traffic intensity (arrival rate * mean holding time)")
flags.DEFINE_boolean("maximise_throughout", False, "Maximise throughput instead of minimising blocking probability")
flags.DEFINE_boolean("use_gn_model", False, "Use GN model for link SNR estimation")
flags.DEFINE_boolean("include_isrs", False, "Include ISRS in the GN model or just ASE noise")
flags.DEFINE_string("reward_type", "service", "Reward type")
flags.DEFINE_boolean("truncate_holding_time", False, "Truncate holding time to less than 2*mean_service_holding_time")
flags.DEFINE_integer("ENV_WARMUP_STEPS", 0, "Number of warmup steps before training or eval")
# RSA-specific environment parameters
flags.DEFINE_boolean("random_traffic", False, "Random traffic matrix for RSA on each reset (else uniform or custom)")
flags.DEFINE_string("custom_traffic_matrix_csv_filepath", None, "Path to custom traffic matrix CSV file")
flags.DEFINE_float("alpha", 0.2, "Fibre attenuation coefficient, alpha [dB/km]")
flags.DEFINE_float("amplifier_noise_figure", 4.5, "Amplifier noise figure [dB]")
flags.DEFINE_float("beta_2", -21.7, "Dispersion parameter [ps^2/km]")
flags.DEFINE_float("gamma", 1.2e-3, "Nonlinear coefficient")
flags.DEFINE_float("span_length", 100, "Span length [km]")
flags.DEFINE_float("lambda0", 1550, "Wavelength [nm]")
# VONE-specific environment parameters
flags.DEFINE_integer("node_resources", 4, "Number of node resources")
flags.DEFINE_list("virtual_topologies", "3_ring", "Virtual topologies")
flags.DEFINE_integer("min_node_resources", 1, "Minimum number of node resources")
flags.DEFINE_integer("max_node_resources", 1, "Maximum number of node resources")
flags.DEFINE_list("node_probs", None, "List of node probabilities for selection")
# Heuristic-specific parameters
flags.DEFINE_boolean("EVAL_HEURISTIC", False, "Evaluate heuristic")
flags.DEFINE_string("path_heuristic", "ksp_ff", "Path heuristic to be evaluated")
flags.DEFINE_string("node_heuristic", "random", "Node heuristic to be evaluated")
# GNN-specific parameters
flags.DEFINE_boolean("USE_GNN", False, "Use GNN")
flags.DEFINE_integer("gnn_latent", 64, "GNN latent size")
flags.DEFINE_integer("message_passing_steps", 3, "Number of message passing steps")
flags.DEFINE_integer("output_edges_size", 64, "Output edges size (not used)")
flags.DEFINE_integer("output_nodes_size", 64, "Output nodes size")
flags.DEFINE_integer("output_globals_size", 64, "Output globals size")
flags.DEFINE_integer("gnn_mlp_layers", 2, "Number of MLP layers")
flags.DEFINE_boolean("normalize_by_link_length", False, "Normalize by link length")
# Model evaluation parameters
flags.DEFINE_boolean("EVAL_MODEL", False, "Evaluate model")
flags.DEFINE_list("model", None, "Used to hold model parameters")
flags.DEFINE_string("min_traffic", "0.0", "Minimum traffic")
flags.DEFINE_string("max_traffic", "1.0", "Maximum traffic")
flags.DEFINE_string("step_traffic", "0.1", "Step size for traffic values between min and max")
flags.DEFINE_boolean("deterministic", False, "Deterministic evaluation (use mode of action distribution)")
# GN model parameters
flags.DEFINE_float("ref_lambda", 1577.5e-9, "Reference wavelength [m]")
flags.DEFINE_float("launch_power", -2.0, "Launch power [dBm]")
flags.DEFINE_string("launch_power_type", "fixed", "Can be fixed (same power per transceiver), "
                                                  "tabular (power depends on path), or rl (power selected by agent).")
flags.DEFINE_float("nonlinear_coefficient", 1.2e-3, "Nonlinear coefficient [1/W^2]")
flags.DEFINE_float("raman_gain_slope", 0.028 / 1e3 / 1e12, "Raman gain slope [1/m/W]")
flags.DEFINE_float("attenuation", 0.2 / 4.343 / 1e3, "Attenuation [1/m]")
flags.DEFINE_float("attenuation_bar", 0.2 / 4.343 / 1e3, "Attenuation [1/m]")
flags.DEFINE_float("dispersion_coeff", 17 * 1e-12 / 1e-9 / 1e3, "Dispersion [s/m^2]")
flags.DEFINE_float("dispersion_slope", 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9, "Dispersion slope [s/m^3]")
flags.DEFINE_float("noise_figure", 4, "Amplifier noise figure [dB]")
flags.DEFINE_float("num_roadms", 1, "Consider ROADM loss in SNR calculation (1 ROADM per link)")
flags.DEFINE_float("roadm_loss", 18, "ROADM losses [dB]")
flags.DEFINE_boolean("coherent", False, "Add NLI contribution coherently per span")
flags.DEFINE_boolean("mod_format_correction", False, "Apply non-Gaussian modulation format correction")
flags.DEFINE_float("interband_gap", 500, "Gap between C+L bands [GHz]")
flags.DEFINE_integer("gap_start", 44, "Start index of gap between C+L bands [slots]")
flags.DEFINE_float("snr_margin", 0.5, "Margin required for estimated SNR for mod. format selection [dB]")
flags.DEFINE_float("max_power", 9, "Maximum launch power [dBm]")
flags.DEFINE_float("min_power", -5, "Minimum launch power [dBm]")
flags.DEFINE_boolean("first_fit", False, "Use KSP-FF for path_action, else KSP-LF")
# Flags for optimize_launch_power.py
flags.DEFINE_float("num_spans", 10, "Number of spans")
