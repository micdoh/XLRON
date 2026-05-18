"""Distributed Raman Amplification (DRA) GN model -- implementation withheld.

The DRA model implementation has been removed from the public release of XLRON
pending publication of the underlying research.

The DRA module extends the ISRS GN model with Raman-pump-aware nonlinear
interference (NLI) and amplified spontaneous emission (ASE) calculations. It was
developed by Henrique Buglia and Mindaugas Jarmolovicius of the UCL Optical
Networks Group. The source code is available on request from the authors.

This module is a stub that preserves the public interface so the rest of XLRON
imports cleanly. Each function below raises ``NotImplementedError`` when called.
"""

_REMOVED_MESSAGE = (
    "The Distributed Raman Amplification (DRA) GN model implementation has been "
    "removed from the public release of XLRON pending publication of the "
    "underlying research. The source code is available on request from the "
    "authors, Henrique Buglia and Mindaugas Jarmolovicius (UCL Optical Networks "
    "Group)."
)


def fit_dra_params_triangular(*args, **kwargs):
    """Stub -- DRA implementation withheld. See module docstring."""
    raise NotImplementedError(_REMOVED_MESSAGE)


def fit_dra_params_jax(*args, **kwargs):
    """Stub -- DRA implementation withheld. See module docstring."""
    raise NotImplementedError(_REMOVED_MESSAGE)


def gn_model_dra(*args, **kwargs):
    """Stub -- DRA implementation withheld. See module docstring."""
    raise NotImplementedError(_REMOVED_MESSAGE)


def get_snr_dra(*args, **kwargs):
    """Stub -- DRA implementation withheld. See module docstring."""
    raise NotImplementedError(_REMOVED_MESSAGE)
