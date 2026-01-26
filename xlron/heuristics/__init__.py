"""Heuristic algorithms for routing and spectrum allocation."""

# Main heuristic functions
# Evaluation utilities
from xlron.heuristics.eval_heuristic import get_eval_fn
from xlron.heuristics.heuristics import (  # K-Shortest Path with First/Last/Best Fit; Most-Used heuristics; Advanced heuristics; Utility functions
    best_fit,
    bf_ksp,
    ff_ksp,
    first_fit,
    get_action_mask,
    get_link_weights,
    kca_ff,
    kmc_ff,
    kme_ff,
    kmf_ff,
    ksp_bf,
    ksp_ff,
    ksp_lf,
    ksp_mu,
    last_fit,
    lf_ksp,
    most_used,
    mu_ksp,
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