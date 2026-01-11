"""XLRON: Reinforcement learning for optical network optimization.

A library for training and evaluating RL agents on optical network routing
and spectrum allocation problems.
"""

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
]