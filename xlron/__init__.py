"""XLRON: Reinforcement learning for optical network optimization.

A library for training and evaluating RL agents on optical network routing
and spectrum allocation problems.
"""

# Data type configuration
from xlron.dtype_config import (
    # Dtype variables
    COMPUTE_DTYPE,
    PARAMS_DTYPE,
    LARGE_FLOAT_DTYPE,
    SMALL_FLOAT_DTYPE,
    LARGE_INT_DTYPE,
    MED_INT_DTYPE,
    SMALL_INT_DTYPE,
    # Dtype utilities
    initialize_dtypes,
    get_dtype_config,
    detect_device_type,
)

# Submodules are imported for convenience
from xlron import environments
from xlron import heuristics
from xlron import models
from xlron import train

__version__ = "0.1.0"

__all__ = [
    # Submodules
    "environments",
    "heuristics",
    "models",
    "train",
    # Dtype configuration
    "COMPUTE_DTYPE",
    "PARAMS_DTYPE",
    "LARGE_FLOAT_DTYPE",
    "SMALL_FLOAT_DTYPE",
    "LARGE_INT_DTYPE",
    "MED_INT_DTYPE",
    "SMALL_INT_DTYPE",
    "initialize_dtypes",
    "get_dtype_config",
    "detect_device_type",
]