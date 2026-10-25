"""Tiled equivalent of the generic uniform GN model's current 1-D broadcasting.

The generic model passes 1-D power/bandwidth arrays and their transposes to
``_xpm``. Transposing 1-D arrays does not introduce a row axis: both operands
broadcast over columns. This kernel intentionally preserves that existing
behavior. It is not the separate mask model's row/column XPM formulation and
does not correct the physical-model discrepancy documented in the audit.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from xlron.kernels.gn_xpm import _xpm_pairs


def xpm_selected_pallas(
    power,
    frequency,
    bandwidth,
    output_indices,
    *,
    a,
    a_bar,
    gamma,
    cr,
    beta2,
    beta3,
    beta4,
    num_warps=4,
    interpret=False,
):
    """Reduce only requested output rows, retaining all interfering slots.

    Endpoint screens need two rows. One program per row avoids the packing,
    prefix scans and scatter needed by the all-output kernel. The Raman sum
    deliberately includes powers whose bandwidth is zero, as in ordinary JAX.
    """
    if power.ndim != 1 or power.shape != frequency.shape or power.shape != bandwidth.shape:
        raise ValueError("matching nonempty 1-D inputs are required")
    if not power.size or any(x.dtype != jnp.float32 for x in (power, frequency, bandwidth)):
        raise ValueError("nonempty float32 inputs are required")
    parameters = (a, a_bar, gamma, cr, beta2, beta3, beta4)
    if any(jnp.ndim(x) != 0 for x in parameters):
        raise ValueError("selected XPM requires scalar uniform-span parameters")
    if output_indices.ndim != 1 or not output_indices.size:
        raise ValueError("nonempty 1-D output indices are required")
    n = power.size
    columns = pl.next_power_of_2(n)
    physics = jnp.stack([jnp.asarray(x, jnp.float32) for x in parameters])

    def kernel(p_ref, f_ref, b_ref, indices_ref, total_ref, physics_ref, out_ref):
        row = pl.program_id(0)
        fi = f_ref[indices_ref[row]]
        k = jnp.arange(columns)
        pk = plt.load(p_ref.at[k], mask=k < n, other=0.0)
        fk = plt.load(f_ref.at[k], mask=k < n, other=0.0)
        bk = plt.load(b_ref.at[k], mask=k < n, other=0.0)
        physical = dict(
            zip(
                ("a", "a_bar", "gamma", "cr", "beta2", "beta3", "beta4"),
                (physics_ref[i] for i in range(7)),
                strict=True,
            )
        )
        pairs = _xpm_pairs(pk, fi, bk, pk, fk, bk, total_ref[()], **physical)
        out_ref[row] = jnp.sum(pairs)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_indices.shape, jnp.float32),
        grid=(output_indices.size,),
        compiler_params=plt.CompilerParams(num_warps=num_warps),
        interpret=interpret,
        name="gn_xpm_selected",
    )(power, frequency, bandwidth, output_indices, jnp.sum(power), physics)


def xpm_legacy_pallas(
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
    block_i=16,
    block_k=64,
    reuse_rows=True,
    group_interferers=False,
    persistent_rows=None,
    interpret=False,
):
    """Preserve every output and nonzero column of the generic scalar-parameter model.

    Physics scalars are explicit inputs because generic GN parameters can be
    traced JAX scalars. All powers still enter the Raman tilt sum. Frequency-only
    row reuse is exact for this particular broadcasting convention.
    """
    if power.ndim != 1 or power.shape != frequency.shape or power.shape != bandwidth.shape:
        raise ValueError("matching nonempty 1-D inputs are required")
    if not power.size or any(x.dtype != jnp.float32 for x in (power, frequency, bandwidth)):
        raise ValueError("nonempty float32 inputs are required")
    parameters = (a, a_bar, gamma, cr, beta2, beta3, beta4)
    if any(jnp.ndim(x) != 0 for x in parameters):
        raise ValueError("the generic XPM kernel requires scalar uniform-span parameters")
    if any(x < 1 or x & (x - 1) for x in (block_i, block_k)):
        raise ValueError("tile dimensions must be positive powers of two")
    n = power.shape[0]
    live = (power != 0) & (bandwidth != 0)
    same_previous = jnp.concatenate(
        (
            jnp.array([False]),
            (power[1:] == power[:-1])
            & (frequency[1:] == frequency[:-1])
            & (bandwidth[1:] == bandwidth[:-1]),
        )
    )
    representatives = live & ~same_previous if group_interferers else live
    columns = jnp.nonzero(representatives, size=n, fill_value=0)[0]
    count = jnp.sum(representatives)
    multiplicity = jnp.ones(n, jnp.float32)
    if group_interferers:
        same_next = jnp.concatenate((same_previous[1:], jnp.array([False])))
        ends = jnp.nonzero(live & ~same_next, size=n, fill_value=0)[0]
        multiplicity = (ends - columns + 1).astype(jnp.float32)
    unique = jnp.concatenate((jnp.array([True]), frequency[1:] != frequency[:-1]))
    rows = jnp.nonzero(unique, size=n, fill_value=0)[0] if reuse_rows else jnp.arange(n)
    row_count = jnp.sum(unique) if reuse_rows else jnp.int32(n)
    physics = jnp.stack([jnp.asarray(x, jnp.float32) for x in parameters])

    if persistent_rows is not None and persistent_rows < 1:
        raise ValueError("persistent row programs must be positive")
    programs = min(pl.cdiv(n, block_i), persistent_rows or pl.cdiv(n, block_i))

    def kernel(
        p_ref,
        f_ref,
        b_ref,
        total_ref,
        column_ref,
        count_ref,
        row_ref,
        row_count_ref,
        physics_ref,
        weight_ref,
        out_ref,
    ):
        def row_tile(tile_i):
            positions = tile_i * block_i + jnp.arange(block_i)
            valid = positions < row_count_ref[()]
            selected = plt.load(row_ref.at[positions], mask=valid, other=0)
            fi = plt.load(f_ref.at[selected], mask=valid, other=0.0)
            physical = dict(
                zip(
                    ("a", "a_bar", "gamma", "cr", "beta2", "beta3", "beta4"),
                    (physics_ref[i] for i in range(7)),
                    strict=True,
                )
            )

            def accumulate(tile, acc):
                indices = tile * block_k + jnp.arange(block_k)
                mask = indices < count_ref[()]
                selected = plt.load(column_ref.at[indices], mask=mask, other=0)
                pk = plt.load(p_ref.at[selected], mask=mask, other=0.0)
                fk = plt.load(f_ref.at[selected], mask=mask, other=0.0)
                bk = plt.load(b_ref.at[selected], mask=mask, other=0.0)
                pairs = _xpm_pairs(
                    pk[None, :],
                    fi[:, None],
                    bk[None, :],
                    pk[None, :],
                    fk[None, :],
                    bk[None, :],
                    total_ref[()],
                    **physical,
                )
                if group_interferers:
                    weight = plt.load(weight_ref.at[indices], mask=mask, other=0.0)
                    pairs = pairs * weight[None, :]
                return acc + jnp.sum(pairs, axis=1)

            result = jax.lax.cond(
                tile_i * block_i < row_count_ref[()],
                lambda: jax.lax.fori_loop(
                    0,
                    (count_ref[()] + block_k - 1) // block_k,
                    accumulate,
                    jnp.zeros(block_i, jnp.float32),
                ),
                lambda: jnp.zeros(block_i, jnp.float32),
            )
            plt.store(out_ref.at[positions], result, mask=positions < n)

        if persistent_rows is None:
            row_tile(pl.program_id(0))
        else:

            def visit(tile_i):
                row_tile(tile_i)
                return tile_i + programs

            jax.lax.while_loop(
                lambda tile_i: tile_i * block_i < row_count_ref[()], visit, pl.program_id(0)
            )

    packed = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(power.shape, jnp.float32),
        grid=(programs,),
        compiler_params=plt.CompilerParams(num_warps=4),
        interpret=interpret,
        name="gn_xpm_generic_persistent" if persistent_rows is not None else "gn_xpm_generic",
    )(
        power,
        frequency,
        bandwidth,
        jnp.sum(power),
        columns,
        count,
        rows,
        row_count,
        physics,
        multiplicity,
    )
    if reuse_rows:
        return packed[jnp.cumsum(unique, dtype=jnp.int32) - 1]
    return packed
