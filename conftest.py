"""Root pytest configuration.

Test modules often import jax before xlron, so set xlron's JAX runtime default
here, before any test module is collected (see xlron/__init__.py).
"""

import os

os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")
