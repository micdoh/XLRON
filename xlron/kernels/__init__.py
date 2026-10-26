"""Optional Pallas kernels for the GN model (selected with --gn_kernel_backend=pallas)."""

import jax


def interpret_default() -> bool:
    """Run kernels in the Pallas interpreter unless a GPU backend is active.

    The interpreter evaluates the same kernel program on CPU, so CPU tests cover
    the kernel logic; it is far slower than compiled GPU execution.
    """
    return jax.default_backend() != "gpu"
