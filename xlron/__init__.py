"""XLRON: Reinforcement learning for optical network optimization.

A library for training and evaluating RL agents on optical network routing
and spectrum allocation problems.
"""

import os

# Pass closed-over device arrays (path tables, physical-layer tables) to compiled
# programs as arguments instead of embedding them as constants. On USA100 (k=70)
# that drops ~950 MB of constants from every program: ksp_ff XLA compile 8.3 s ->
# 2.1 s and peak host RSS 11.7 -> 4.9 GB, with unchanged steps/s (issue #80).
# JAX reads this once at import, so it must be set before jax is imported; set
# JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=0 to opt out.
# See https://docs.jax.dev/en/latest/internals/constants.html
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")

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
