"""Single-kernel stream compaction of a boolean vector (Pallas).

`jnp.nonzero(mask, size=n, fill_value=0)` lowers to a blocked cumulative sum,
a scatter and a broadcast fill on GPU: several launches that dominate the many
small repacks of an all_slots GN mask. One program scans the whole vector here.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from xlron.kernels import interpret_default

# One program holds the whole vector; beyond this the scan would spill registers.
MAX_ELEMENTS = 1 << 16


def compact_true_indices(mask, *, num_warps=8, interpret=None):
    """(indices, count) of the True entries of a 1-D mask.

    indices matches jnp.nonzero(mask, size=mask.size, fill_value=0)[0]: ascending
    positions of True entries, padded with 0; count equals jnp.sum(mask).
    """
    if mask.ndim != 1:
        raise ValueError("A 1-D mask is required")
    size = mask.shape[0]
    if size > MAX_ELEMENTS:
        raise ValueError(f"Masks longer than {MAX_ELEMENTS} are not supported")
    interpret = interpret_default() if interpret is None else interpret
    block = pl.next_power_of_2(size)

    def kernel(mask_ref, _zeros_ref, out_ref, count_ref):
        positions = jnp.arange(block)
        flags = plt.load(mask_ref.at[positions], mask=positions < size, other=0)
        # Masked lanes get distinct slots in a discarded tail: the interpreter
        # stores masked scatters as read-modify-writes, so a masked lane sharing a
        # live lane's destination could restore the old value.
        destinations = jnp.where(flags != 0, jnp.cumsum(flags) - 1, block + positions)
        # out_ref aliases a zero buffer, so entries past the count stay 0.
        plt.store(out_ref.at[destinations], positions, mask=flags != 0)
        plt.store(count_ref.at[jnp.int32(0)], jnp.sum(flags))

    indices, count = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((2 * block,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        input_output_aliases={1: 0},
        compiler_params=plt.CompilerParams(num_warps=num_warps),
        interpret=interpret,
        name="compact_true_indices",
    )(mask.astype(jnp.int32), jnp.zeros((2 * block,), jnp.int32))
    return indices[:size], count[0]
