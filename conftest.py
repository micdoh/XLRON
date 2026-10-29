"""Root pytest configuration.

Test modules often import jax before xlron, so set xlron's JAX runtime default
here, before any test module is collected (see xlron/__init__.py).

The suite's time is dominated by XLA compilation, so tests also default to a
persistent compilation cache in ``.jax_cache/`` at the repository root: a
repeat run reuses every executable whose program, flags and jax version are
unchanged. Set ``JAX_COMPILATION_CACHE_DIR`` to move it, or
``JAX_ENABLE_COMPILATION_CACHE=false`` to switch it off.
"""

import os
import pathlib

os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(pathlib.Path(__file__).parent / ".jax_cache")
)
# Test programs are small; most compile in well under jax's 1 s default.
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")


def pytest_collection_modifyitems(config, items):
    """Start the long training smoke tests first.

    Under ``pytest -n`` they otherwise start last (``xlron/train`` sorts late)
    and leave the other workers idle while they finish.
    """
    items.sort(key=lambda item: item.get_closest_marker("slow") is None)
