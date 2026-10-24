"""Supported-backend XPM experiment; validate on CUDA before enabling it.

Uses Mosaic GPU with approximate math disabled. ``atan2(x, 1)`` expresses the
same arctangent as the existing formula using Mosaic's supported primitive.
Inputs are padded and interfering channels packed outside the kernel so every
device load has a contiguous in-bounds slice, including the final tile.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mg

from xlron.kernels.gn_xpm import _xpm_pairs


def xpm_mosaic(power, frequency, bandwidth, *, block_i=128, block_k=128, **physics):
    """Compute the current XPM formula with a per-link dynamic reduction bound."""
    if power.ndim != 1 or power.shape != frequency.shape or power.shape != bandwidth.shape:
        raise ValueError("power, frequency and bandwidth must have the same 1-D shape")
    if not power.size or any(x.dtype != jnp.float32 for x in (power, frequency, bandwidth)):
        raise ValueError("nonempty float32 inputs are required")
    if any(x < 1 or x & (x - 1) for x in (block_i, block_k)):
        raise ValueError("tile sizes must be positive powers of two")
    n = power.size
    padded = pl.cdiv(n, max(block_i, block_k)) * max(block_i, block_k)
    pad = lambda x: jnp.pad(x, (0, padded - n))
    live = (power != 0) & (bandwidth != 0)
    count = jnp.sum(live)
    indices = jnp.nonzero(live, size=n, fill_value=0)[0]
    positions = jnp.arange(n)
    packed = [
        pad(jnp.where(positions < count, x[indices], 0.0)) for x in (power, frequency, bandwidth)
    ]

    @mg.kernel(
        out_type=jax.ShapeDtypeStruct((padded,), jnp.float32),
        grid=(pl.cdiv(padded, block_i),),
        grid_names=("row",),
        compiler_params=mg.CompilerParams(
            approx_math=False, lowering_semantics=mg.LoweringSemantics.Lane
        ),
    )
    def kernel(p_ref, f_ref, bw_ref, pk_ref, fk_ref, bk_ref, total_ref, count_ref, out_ref):
        rows = pl.ds(jax.lax.axis_index("row") * block_i, block_i)
        row_layout = mg.Layout.WGMMA.reduce(1)
        column_layout = mg.Layout.WGMMA.reduce(0)
        pi = mg.load(p_ref.at[rows], optimized=False, layout=row_layout)
        fi = mg.load(f_ref.at[rows], optimized=False, layout=row_layout)
        bi = mg.load(bw_ref.at[rows], optimized=False, layout=row_layout)
        total = total_ref[()]
        live_count = count_ref[()]

        def accumulate(tile, acc):
            columns = pl.ds(tile * block_k, block_k)
            pk = mg.load(pk_ref.at[columns], optimized=False, layout=column_layout)
            fk = mg.load(fk_ref.at[columns], optimized=False, layout=column_layout)
            bk = mg.load(bk_ref.at[columns], optimized=False, layout=column_layout)

            def broadcast(x, axis):
                return mg.layout_cast(
                    jax.lax.broadcast_in_dim(x, (block_i, block_k), (axis,)),
                    mg.Layout.WGMMA,
                )

            pairs = _xpm_pairs(
                broadcast(pi, 0),
                broadcast(fi, 0),
                broadcast(bi, 0),
                broadcast(pk, 1),
                broadcast(fk, 1),
                broadcast(bk, 1),
                total,
                **physics,
                atan_fn=lambda x: jax.lax.atan2(x, jnp.ones_like(x)),
            )
            return acc + jnp.sum(pairs, axis=1)

        result = jax.lax.fori_loop(
            0,
            (live_count + block_k - 1) // block_k,
            accumulate,
            mg.layout_cast(jnp.zeros((block_i,), jnp.float32), row_layout),
        )
        mg.store(out_ref.at[rows], result, optimized=False)

    return kernel(pad(power), pad(frequency), pad(bandwidth), *packed, jnp.sum(power), count)[:n]
