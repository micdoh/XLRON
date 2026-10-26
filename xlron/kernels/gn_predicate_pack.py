"""Combine identical old path/slot SNR predicates without dropping thresholds."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from xlron.kernels import interpret_default


# Each program compares its row against every link for a tile of slots, holding a
# (next_pow2(links), block_slots) tile; keep that tile bounded on large topologies.
MAX_TILE_ELEMENTS = 8192


def pack_old_snr_predicates(path_index, required, num_paths, *, block_slots=None, interpret=None):
    """Keep one predicate per valid (path ID, slot), at its strongest threshold.

    Path SNR is a function of path ID and slot, independent of the referring
    link row. NaN rejects unconditionally and is preserved explicitly. Invalid
    path IDs retain their original predicates. -inf encodes an inactive check.
    Intended for spectrum-valid insertions, which never overwrite old slots.
    """
    if path_index.shape != required.shape or required.ndim != 2:
        raise ValueError("matching link/slot matrices are required")
    if required.dtype != jnp.float32:
        raise ValueError("float32 thresholds are required")
    interpret = interpret_default() if interpret is None else interpret
    links, slots = required.shape
    padded_links = pl.next_power_of_2(links)
    if block_slots is None:
        # Work is O(links^2 * slots) either way; this only bounds register use.
        block_slots = max(1, min(128, MAX_TILE_ELEMENTS // padded_links))

    def kernel(ids_ref, required_ref, out_ref):
        row, tile = pl.program_id(0), pl.program_id(1)
        cols = tile * block_slots + jnp.arange(block_slots)
        live_cols = cols < slots
        current = plt.load(ids_ref.at[row, cols], mask=live_cols, other=-1)
        original = plt.load(required_ref.at[row, cols], mask=live_cols, other=-jnp.inf)
        others = jnp.arange(padded_links)
        live = (others[:, None] < links) & live_cols[None, :]
        ids = plt.load(ids_ref.at[others[:, None], cols[None, :]], mask=live, other=-1)
        thresholds = plt.load(
            required_ref.at[others[:, None], cols[None, :]], mask=live, other=-jnp.inf
        )
        same = (ids == current[None, :]) & live & (thresholds != -jnp.inf)
        first = jnp.min(jnp.where(same, others[:, None], links), axis=0)
        nan = jnp.sum((same & jnp.isnan(thresholds)).astype(jnp.int32), axis=0) > 0
        maximum = jnp.max(jnp.where(same & ~jnp.isnan(thresholds), thresholds, -jnp.inf), axis=0)
        maximum = jnp.where(nan, jnp.nan, maximum)
        merged = jnp.where(row == first, maximum, -jnp.inf)
        result = jnp.where((current >= 0) & (current < num_paths), merged, original)
        plt.store(out_ref.at[row, cols], result, mask=live_cols)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(required.shape, jnp.float32),
        grid=(links, pl.cdiv(slots, block_slots)),
        compiler_params=plt.CompilerParams(num_warps=4),
        interpret=interpret,
        name="gn_pack_old_predicates",
    )(path_index, required)
