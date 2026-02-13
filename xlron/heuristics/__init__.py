"""Heuristic algorithms for routing and spectrum allocation."""


def __getattr__(name):
    _heuristic_names = {
        "best_fit",
        "bf_ksp",
        "ff_ksp",
        "first_fit",
        "get_action_mask",
        "get_link_weights",
        "kca_ff",
        "kmc_ff",
        "kme_ff",
        "kmf_ff",
        "ksp_bf",
        "ksp_ff",
        "ksp_lf",
        "ksp_mu",
        "last_fit",
        "lf_ksp",
        "most_used",
        "mu_ksp",
    }
    if name in _heuristic_names:
        from xlron.heuristics import heuristics

        return getattr(heuristics, name)
    if name == "get_eval_fn":
        from xlron.heuristics.eval_heuristic import get_eval_fn

        return get_eval_fn
    raise AttributeError(f"module 'xlron.heuristics' has no attribute {name}")


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
