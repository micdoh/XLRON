"""Float32 tiled XPM proof of concept for the public uniform-span ISRS GN model.

Matches the formula in ``isrs_gn_model.get_snr_fused``; this is an implementation
equivalence experiment, not a correction of the physical model. No SPM, ASE or
modulation-correction formula is changed here. Arrays describe one link and can
be vmapped over links/candidates/environments.

This A100 proof of concept explicitly selects Pallas' deprecated Triton backend.
Mosaic GPU has added Ampere primitives; a supported-backend port still needs its
own implementation and validation. CPU tests use Pallas interpret mode.
Reverse-mode derivatives are not provided; normal PPO does not differentiate
through the environment.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

EPS = 1e-12


def _xpm_pairs(
    pi,
    fi,
    bi,
    pk,
    fk,
    bk,
    total,
    *,
    a,
    a_bar,
    gamma,
    cr,
    beta2,
    beta3,
    beta4,
    atan_fn=jnp.arctan,
):
    """Pair contributions with explicitly broadcast row/column operands."""
    phi = (
        2
        * jnp.pi**2
        * (fi - fk)
        * (
            beta2
            + jnp.pi * beta3 * (fi + fk)
            + 2 / 3 * jnp.pi**2 * beta4 * (fi**2 + fi * fk + fk**2)
        )
    )
    ap = a + a_bar
    tilt = (ap - fk * total * cr) ** 2
    denom = bk * phi * a_bar * (2 * a + a_bar)
    ratio = (pk / jnp.maximum(pi, EPS)) ** 2
    pairs = (
        32
        / 27
        * ratio
        * gamma**2
        / jnp.where(denom != 0, denom, 1.0)
        * ((tilt - a**2) / a * atan_fn(phi * bi / a) + (ap**2 - tilt) / ap * atan_fn(phi * bi / ap))
    )
    return jnp.where(denom != 0, pairs, 0.0)


def _xpm_rows(pi, fi, bi, pk, fk, bk, total, *, multiplicity=None, **physics):
    """Sum one output-row tile against one interfering-channel tile."""
    pairs = _xpm_pairs(
        pi[:, None],
        fi[:, None],
        bi[:, None],
        pk[None, :],
        fk[None, :],
        bk[None, :],
        total,
        **physics,
    )
    if multiplicity is not None:
        pairs = pairs * multiplicity[None, :]
    return jnp.sum(pairs, axis=1)


def xpm_reference(power, frequency, bandwidth, **physics):
    """The existing fused model's dense XPM coefficient before span accumulation."""
    return _xpm_rows(
        power, frequency, bandwidth, power, frequency, bandwidth, jnp.sum(power), **physics
    )


def xpm_compact_jax(power, frequency, bandwidth, *, block_k=128, **physics):
    """Pure-JAX experiment: visit only nonzero interfering channels in tiles.

    All output rows are retained. Zero power or zero bandwidth gives an exactly
    zero pair contribution in the original formula. Total power still uses every
    input channel, so packing does not change the Raman tilt calculation.
    Dynamic loops are intended for simulation inference, not reverse-mode AD.
    """
    n = power.shape[0]
    live = (power != 0) & (bandwidth != 0)
    indices = jnp.nonzero(live, size=n, fill_value=0)[0]
    count = jnp.sum(live)
    total = jnp.sum(power)

    def accumulate(tile, acc):
        positions = tile * block_k + jnp.arange(block_k)
        selected = indices[jnp.minimum(positions, n - 1)]
        pk = jnp.where(positions < count, power[selected], 0.0)
        bk = jnp.where(positions < count, bandwidth[selected], 0.0)
        return acc + _xpm_rows(
            power, frequency, bandwidth, pk, frequency[selected], bk, total, **physics
        )

    return jax.lax.fori_loop(0, (count + block_k - 1) // block_k, accumulate, jnp.zeros_like(power))


def xpm_pallas(
    power,
    frequency,
    bandwidth,
    *,
    a,
    a_bar,
    gamma,
    cr,
    beta2,
    beta3,
    beta4,
    block_i=8,
    block_k=128,
    num_warps=4,
    compact=False,
    compact_rows=False,
    reuse_rows=False,
    group_interferers=False,
    skip_empty_tiles=False,
    skip_empty_columns=False,
    interpret=False,
):
    """Evaluate exact dense XPM in tiles, without writing a channel-pair matrix.

    ``compact`` packs nonzero interfering channels and shortens the reduction.
    ``compact_rows`` evaluates only rows with nonzero bandwidth, whose complement
    has exactly zero XPM. It unpacks the results back to the original slot order.
    Zero-power rows with positive bandwidth are retained for coefficient parity.
    ``reuse_rows`` evaluates consecutive identical (power, frequency, bandwidth)
    rows once. It retains all interferers and the original total power, including
    repeated entries; restoring a row never changes its reduction operands.
    ``group_interferers`` reduces consecutive identical interfering entries once,
    weighted by their exact multiplicity. Total power uses the original array.
    ``skip_empty_tiles`` avoids row packing by branching inside the kernel when
    a contiguous tile has no nonzero-bandwidth outputs.
    The only arithmetic change is float32 evaluation/reduction order.
    """
    if power.ndim != 1 or power.shape != frequency.shape or power.shape != bandwidth.shape:
        raise ValueError("power, frequency and bandwidth must have the same nonempty 1-D shape")
    if power.shape[0] == 0 or any(x.dtype != jnp.float32 for x in (power, frequency, bandwidth)):
        raise ValueError("nonempty float32 inputs are required")
    if any(x < 1 or x & (x - 1) for x in (block_i, block_k, num_warps)):
        raise ValueError("tile sizes and num_warps must be positive powers of two")
    n = power.shape[0]
    physics = {
        "a": a,
        "a_bar": a_bar,
        "gamma": gamma,
        "cr": cr,
        "beta2": beta2,
        "beta3": beta3,
        "beta4": beta4,
    }
    compact = compact or group_interferers
    live = (power != 0) & (bandwidth != 0)
    same_previous = jnp.concatenate(
        (
            jnp.array([False]),
            (power[1:] == power[:-1])
            & (frequency[1:] == frequency[:-1])
            & (bandwidth[1:] == bandwidth[:-1]),
        )
    )
    column_representatives = live & ~same_previous if group_interferers else live
    indices = (
        jnp.nonzero(column_representatives, size=n, fill_value=0)[0] if compact else jnp.arange(n)
    )
    count = jnp.sum(column_representatives) if compact else jnp.int32(n)
    multiplicity = jnp.ones(n, jnp.float32)
    if group_interferers:
        same_next = jnp.concatenate((same_previous[1:], jnp.array([False])))
        ends = jnp.nonzero(live & ~same_next, size=n, fill_value=0)[0]
        multiplicity = (ends - indices + 1).astype(jnp.float32)
    compact_rows = compact_rows or reuse_rows
    live_rows = bandwidth != 0
    representatives = live_rows
    if reuse_rows:
        representatives = live_rows & ~same_previous
    row_indices = (
        jnp.nonzero(representatives, size=n, fill_value=0)[0] if compact_rows else jnp.arange(n)
    )
    row_count = jnp.sum(representatives) if compact_rows else jnp.int32(n)

    def kernel(
        p_ref,
        f_ref,
        bw_ref,
        total_ref,
        index_ref,
        count_ref,
        row_ref,
        row_count_ref,
        weight_ref,
        out_ref,
    ):
        rows = pl.program_id(0) * block_i + jnp.arange(block_i)
        valid_rows = rows < row_count_ref[()] if compact_rows else rows < n
        selected_rows = (
            plt.load(row_ref.at[rows], mask=valid_rows, other=0) if compact_rows else rows
        )
        pi = plt.load(p_ref.at[selected_rows], mask=valid_rows, other=0.0)
        fi = plt.load(f_ref.at[selected_rows], mask=valid_rows, other=0.0)
        bi = plt.load(bw_ref.at[selected_rows], mask=valid_rows, other=0.0)
        total = total_ref[()]
        live_count = count_ref[()]

        def accumulate(tile, acc):
            columns = tile * block_k + jnp.arange(block_k)
            selected = (
                plt.load(index_ref.at[columns], mask=columns < live_count, other=0)
                if compact
                else columns
            )
            pk = plt.load(p_ref.at[selected], mask=columns < live_count, other=0.0)
            fk = plt.load(f_ref.at[selected], mask=columns < live_count, other=0.0)
            bk = plt.load(bw_ref.at[selected], mask=columns < live_count, other=0.0)
            weights = (
                plt.load(weight_ref.at[columns], mask=columns < live_count, other=0.0)
                if group_interferers
                else None
            )

            def add_pairs():
                return acc + _xpm_rows(
                    pi, fi, bi, pk, fk, bk, total, multiplicity=weights, **physics
                )

            if skip_empty_columns:
                return jax.lax.cond(
                    jnp.sum(((pk != 0) & (bk != 0)).astype(jnp.int32)) > 0,
                    add_pairs,
                    lambda: acc,
                )
            return add_pairs()

        def reduce_rows():
            return jax.lax.fori_loop(
                0,
                (live_count + block_k - 1) // block_k,
                accumulate,
                jnp.zeros((block_i,), jnp.float32),
            )

        tile_has_outputs = (
            pl.program_id(0) * block_i < row_count_ref[()]
            if compact_rows
            else jnp.sum((bi != 0).astype(jnp.int32)) > 0
        )
        result = (
            jax.lax.cond(
                tile_has_outputs,
                reduce_rows,
                lambda: jnp.zeros((block_i,), jnp.float32),
            )
            if compact_rows or skip_empty_tiles
            else reduce_rows()
        )
        plt.store(out_ref.at[rows], result, mask=rows < n)

    packed_result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(power.shape, jnp.float32),
        grid=(pl.cdiv(n, block_i),),
        compiler_params=plt.CompilerParams(num_warps=num_warps),
        interpret=interpret,
        name="gn_xpm_tiled",
    )(
        power,
        frequency,
        bandwidth,
        jnp.sum(power),
        indices,
        count,
        row_indices,
        row_count,
        multiplicity,
    )
    if compact_rows:
        inverse = jnp.maximum(jnp.cumsum(representatives, dtype=jnp.int32) - 1, 0)
        return jnp.where(live_rows, packed_result[inverse], 0.0)
    return packed_result
