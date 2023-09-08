from absl import flags

# N.B. Use can pass the flag --flagfile=PATH_TO_FLAGFILE to add flags without typing them out

# Training hyperparameters
flags.DEFINE_float("LR", 5e-4, "Learning rate")
flags.DEFINE_integer("NUM_ENVS", 1, "Number of environments")
flags.DEFINE_integer("NUM_STEPS", 150, "Number of steps per environment")
flags.DEFINE_float("TOTAL_TIMESTEPS", 1e6, "Total number of timesteps")
flags.DEFINE_integer("UPDATE_EPOCHS", 1, "Number of epochs per update")
flags.DEFINE_integer("NUM_MINIBATCHES", 1, "Number of minibatches per update")
flags.DEFINE_float("GAMMA", 0.99, "Discount factor")
flags.DEFINE_float("GAE_LAMBDA", 0.95, "GAE lambda parameter")
flags.DEFINE_float("CLIP_EPS", 0.2, "PPO clipping parameter")
flags.DEFINE_float("ENT_COEF", 0.01, "Entropy coefficient")
flags.DEFINE_float("VF_COEF", 0.5, "Value function coefficient")
flags.DEFINE_float("MAX_GRAD_NORM", 0.5, "Maximum gradient norm")
flags.DEFINE_string("ACTIVATION", "tanh", "Activation function")
flags.DEFINE_boolean("ANNEAL_LR", True, "Anneal learning rate")
flags.DEFINE_integer("SEED", 42, "Random seed")
flags.DEFINE_integer("NUM_SEEDS", 1, "Number of seeds")
# Additional training parameters
flags.DEFINE_integer("DEFAULT_DEVICE", None, "Default device index")
flags.DEFINE_boolean("USE_PMAP", False, "Use pmap")
flags.DEFINE_boolean("WANDB", False, "Use wandb")
flags.DEFINE_boolean("SAVE_MODEL", False, "Save model")
flags.DEFINE_integer("NUM_DEVICES", None, "Number of devices to emulate")
flags.DEFINE_boolean("DEBUG", False, "Debug mode")
flags.DEFINE_boolean("ORDERED", True, "Order print statements when debugging "
                                      "(must be false if using pmap)")
flags.DEFINE_string("MODEL_PATH", ".", "Path to save/load model")
flags.DEFINE_string("EXPERIMENT_NAME", "experiment", "Name of experiment")
# Environment parameters
flags.DEFINE_string("env_type", "vone", "Environment type")
flags.DEFINE_integer("load", 100, "Load")
flags.DEFINE_integer("k", 5, "Number of paths")
flags.DEFINE_string("topology_name", "4node", "Topology name")
flags.DEFINE_integer("link_resources", 4, "Number of link resources")
flags.DEFINE_float("max_requests", 4, "Maximum number of requests in an episode")
flags.DEFINE_float("max_timesteps", 30, "Maximum number of timesteps in an episode")
flags.DEFINE_integer("min_slots", 1, "Minimum number of slots")
flags.DEFINE_integer("max_slots", 1, "Maximum number of slots")
# VONE-specific environment parameters
flags.DEFINE_integer("mean_service_holding_time", 10, "Mean service holding time")
flags.DEFINE_integer("node_resources", 4, "Number of node resources")
flags.DEFINE_list("virtual_topologies", "3_ring", "Virtual topologies")
flags.DEFINE_integer("min_node_resources", 1, "Minimum number of node resources")
flags.DEFINE_integer("max_node_resources", 1, "Maximum number of node resources")
