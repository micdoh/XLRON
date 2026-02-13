"""Training utilities and PPO implementation for reinforcement learning."""

_PPO_NAMES = {
    "_sample_prioritized_batch",
    "compute_sample_priority_weights",
    "compute_trajectory_priority_weights",
    "get_learner_fn",
}

_TRAIN_UTILS_NAMES = {
    "TrainState",
    "count_parameters",
    "experiment_data_setup",
    "get_episode_end_mean_std_iqr",
    "get_mean_std_iqr",
    "get_warmup_fn",
    "init_network",
    "log_actions",
    "log_metrics",
    "loss_metrics",
    "make_ent_schedule",
    "make_lr_schedule",
    "merge_leading_dims",
    "metrics",
    "moving_average",
    "ndim_at_least",
    "plot_metrics",
    "print_metrics",
    "process_metrics",
    "reshape_keys",
    "save_model",
    "scale_gradient",
    "select_action",
    "select_action_eval",
    "setup_wandb",
    "unreplicate_batch_dim",
    "unreplicate_n_dims",
}


def __getattr__(name):
    if name in _PPO_NAMES:
        from xlron.train import ppo

        return getattr(ppo, name)
    if name in _TRAIN_UTILS_NAMES:
        from xlron.train import train_utils

        return getattr(train_utils, name)
    raise AttributeError(f"module 'xlron.train' has no attribute {name}")


__all__ = [
    # PPO
    "get_learner_fn",
    "compute_trajectory_priority_weights",
    "compute_sample_priority_weights",
    "_sample_prioritized_batch",
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
