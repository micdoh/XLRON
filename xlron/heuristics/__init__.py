"""Heuristic algorithms for routing and spectrum allocation."""

# Main heuristic functions
from xlron.heuristics.heuristics import (
    # K-Shortest Path with First/Last/Best Fit
    ksp_ff,
    ksp_lf,
    ksp_bf,
    ff_ksp,
    lf_ksp,
    bf_ksp,
    # Most-Used heuristics
    ksp_mu,
    mu_ksp,
    # Advanced heuristics
    kmc_ff,
    kmf_ff,
    kme_ff,
    kca_ff,
    # Utility functions
    get_link_weights,
    get_action_mask,
    best_fit,
    first_fit,
    last_fit,
    most_used,
)

# Evaluation utilities
from xlron.heuristics.eval_heuristic import (
    get_eval_fn,
)

__all__ = [
    # Main heuristics
    "ksp_ff",
    "ksp_lf",
    "ksp_bf",
    "ff_ksp",
    "lf_ksp",
    "bf_ksp",
    "ksp_mu",
    "mu_ksp",
    "kmc_ff",
    "kmf_ff",
    "kme_ff",
    "kca_ff",
    # Utility functions
    "get_link_weights",
    "get_action_mask",
    "best_fit",
    "first_fit",
    "last_fit",
    "most_used",
    # Evaluation
    "get_eval_fn",
]