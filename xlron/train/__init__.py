"""Training utilities and PPO implementation for reinforcement learning."""

# Core PPO training
from xlron.train.ppo import (
    get_learner_fn,
    compute_trajectory_priority_weights,
    compute_sample_priority_weights,
    sample_prioritized_batch,
)

# Training utilities and state management
from xlron.train.train_utils import (
    # Core training state
    TrainState,
    # Model initialization
    init_network,
    experiment_data_setup,
    # Action selection
    select_action,
    select_action_eval,
    # Warmup and scheduling
    get_warmup_fn,
    make_lr_schedule,
    make_ent_schedule,
    # Model management
    save_model,
    # Utility functions
    scale_gradient,
    count_parameters,
    ndim_at_least,
    merge_leading_dims,
    unreplicate_n_dims,
    unreplicate_batch_dim,
    moving_average,
    reshape_keys,
    # Metrics processing
    get_mean_std_iqr,
    get_episode_end_mean_std_iqr,
    process_metrics,
    plot_metrics,
    log_actions,
    print_metrics,
    log_metrics,
    # Weights and Biases
    setup_wandb,
    # Metric definitions
    metrics,
    loss_metrics,
)

__all__ = [
    # PPO
    "get_learner_fn",
    "compute_trajectory_priority_weights",
    "compute_sample_priority_weights",
    "sample_prioritized_batch",
    # Training state
    "TrainState",
    # Model setup
    "init_network",
    "experiment_data_setup",
    # Action selection
    "select_action",
    "select_action_eval",
    # Warmup and scheduling
    "get_warmup_fn",
    "make_lr_schedule",
    "make_ent_schedule",
    # Model management
    "save_model",
    # Utilities
    "scale_gradient",
    "count_parameters",
    "ndim_at_least",
    "merge_leading_dims",
    "unreplicate_n_dims",
    "unreplicate_batch_dim",
    "moving_average",
    "reshape_keys",
    # Metrics
    "get_mean_std_iqr",
    "get_episode_end_mean_std_iqr",
    "process_metrics",
    "plot_metrics",
    "log_actions",
    "print_metrics",
    "log_metrics",
    "setup_wandb",
    "metrics",
    "loss_metrics",
]