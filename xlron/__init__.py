"""XLRON: Reinforcement learning for optical network optimization.

A library for training and evaluating RL agents on optical network routing
and spectrum allocation problems.
"""

__version__ = "0.1.0"

_SUBMODULES = {"environments", "heuristics", "bounds", "models", "train"}


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"xlron.{name}")
    raise AttributeError(f"module 'xlron' has no attribute {name}")


__all__ = [
    "environments",
    "heuristics",
    "bounds",
    "models",
    "train",
]
