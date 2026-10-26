"""Pallas XPM reduction for the uniform-span ISRS GN model.

Evaluates exactly the pair formula `isrs_gn_model.xpm_pairs` (Ref. [1, Eq. (11)])
and sums it over interferers, without materialising the channel-pair matrix.
Two exact reductions make wide, mostly-empty or multi-slot spectra cheap:

- interferers: only live (power and bandwidth nonzero) channels are visited,
  and a run of identical consecutive (power, frequency, bandwidth) entries, as
  produced by one multi-slot lightpath, is evaluated once with its multiplicity;
- outputs: rows with zero bandwidth have zero XPM, and a run of identical rows
  shares one result.

Total power for the Raman tilt always uses every input entry. The only
arithmetic difference from ordinary JAX is float32 evaluation/reduction order.
Uses the Pallas Triton backend; reverse-mode derivatives are not provided, so
the environment selects it only for non-differentiable simulation.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from xlron.environments.gn_model.isrs_gn_model import xpm_pairs
from xlron.kernels import interpret_default

PHYSICS = ("a", "a_bar", "gamma", "cr", "beta2", "beta3", "beta4")


def _runs(power, frequency, bandwidth):
    """True where an entry repeats its predecessor's (power, frequency, bandwidth)."""
    return jnp.concatenate(
        (
            jnp.array([False]),
            (power[1:] == power[:-1])
            & (frequency[1:] == frequency[:-1])
            & (bandwidth[1:] == bandwidth[:-1]),
        )
    )


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
    block_i=16,
    block_k=32,
    num_warps=4,
    persistent_rows=16,
    interpret=None,
):
    """XPM coefficient per channel of one link (before span accumulation).

    Args:
        power, frequency, bandwidth: float32 (N,) channel arrays [W, Hz, Hz].
        a ... beta4: scalar uniform-span physics (Python or 0-d JAX values).
        block_i, block_k: output-row and interferer tile sizes (powers of two).
        persistent_rows: number of programs that loop over row tiles; None
            launches one program per row tile.
        interpret: run in the Pallas interpreter (default: when not on a GPU).
    """
    if power.ndim != 1 or power.shape != frequency.shape or power.shape != bandwidth.shape:
        raise ValueError("power, frequency and bandwidth must have the same nonempty 1-D shape")
    if not power.size or any(x.dtype != jnp.float32 for x in (power, frequency, bandwidth)):
        raise ValueError("nonempty float32 inputs are required")
    if any(x < 1 or x & (x - 1) for x in (block_i, block_k, num_warps)):
        raise ValueError("tile sizes and num_warps must be positive powers of two")
    if persistent_rows is not None and persistent_rows < 1:
        raise ValueError("persistent_rows must be positive")
    values = (a, a_bar, gamma, cr, beta2, beta3, beta4)
    if any(jnp.ndim(x) != 0 for x in values):
        raise ValueError("the Pallas XPM kernel requires scalar uniform-span physics")
    interpret = interpret_default() if interpret is None else interpret
    n = power.shape[0]
    physics = jnp.stack([jnp.asarray(x, jnp.float32) for x in values])

    repeat = _runs(power, frequency, bandwidth)
    live = (power != 0) & (bandwidth != 0)
    columns = live & ~repeat
    column_index = jnp.nonzero(columns, size=n, fill_value=0)[0]
    column_count = jnp.sum(columns)
    # A run starting at column_index[j] ends at the next live entry that is not
    # repeated by its successor; its length is the multiplicity.
    run_end = jnp.nonzero(live & ~jnp.concatenate((repeat[1:], jnp.array([False]))), size=n)[0]
    multiplicity = (run_end - column_index + 1).astype(jnp.float32)

    outputs = bandwidth != 0
    rows = outputs & ~repeat
    row_index = jnp.nonzero(rows, size=n, fill_value=0)[0]
    row_count = jnp.sum(rows)
    row_tiles = pl.cdiv(n, block_i)
    programs = row_tiles if persistent_rows is None else min(row_tiles, persistent_rows)

    def kernel(p_ref, f_ref, b_ref, total_ref, col_ref, col_count_ref, weight_ref, row_ref,
               row_count_ref, physics_ref, out_ref):  # fmt: skip
        coefficients = {name: physics_ref[i] for i, name in enumerate(PHYSICS)}
        program = pl.program_id(0)

        def row_tile(tile_i):
            positions = tile_i * block_i + jnp.arange(block_i)
            valid = positions < row_count_ref[()]
            selected = plt.load(row_ref.at[positions], mask=valid, other=0)
            pi = plt.load(p_ref.at[selected], mask=valid, other=0.0)
            fi = plt.load(f_ref.at[selected], mask=valid, other=0.0)
            bi = plt.load(b_ref.at[selected], mask=valid, other=0.0)

            def accumulate(tile_k, acc):
                k = tile_k * block_k + jnp.arange(block_k)
                mask = k < col_count_ref[()]
                chosen = plt.load(col_ref.at[k], mask=mask, other=0)
                pk = plt.load(p_ref.at[chosen], mask=mask, other=0.0)
                fk = plt.load(f_ref.at[chosen], mask=mask, other=0.0)
                bk = plt.load(b_ref.at[chosen], mask=mask, other=0.0)
                weight = plt.load(weight_ref.at[k], mask=mask, other=0.0)
                pairs = xpm_pairs(
                    pi[:, None],
                    fi[:, None],
                    bi[:, None],
                    pk[None, :],
                    fk[None, :],
                    bk[None, :],
                    total_ref[()],
                    **coefficients,
                )
                return acc + jnp.sum(pairs * weight[None, :], axis=1)

            result = jax.lax.fori_loop(
                0,
                (col_count_ref[()] + block_k - 1) // block_k,
                accumulate,
                jnp.zeros(block_i, jnp.float32),
            )
            plt.store(out_ref.at[positions], result, mask=positions < n)

        if persistent_rows is None:

            @pl.when(program * block_i < row_count_ref[()])
            def _():
                row_tile(program)

        else:

            def visit(tile_i):
                row_tile(tile_i)
                return tile_i + programs

            jax.lax.while_loop(
                lambda tile_i: tile_i * block_i < row_count_ref[()], visit, program
            )

    packed = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(power.shape, jnp.float32),
        grid=(programs,),
        compiler_params=plt.CompilerParams(num_warps=num_warps),
        interpret=interpret,
        name="gn_xpm",
    )(
        power,
        frequency,
        bandwidth,
        jnp.sum(power),
        column_index,
        column_count,
        multiplicity,
        row_index,
        row_count,
        physics,
    )
    # Packed slot r holds the r-th run of output rows; unused tail slots are
    # never read because every output row maps to a run that was evaluated.
    run = jnp.maximum(jnp.cumsum(rows, dtype=jnp.int32) - 1, 0)
    return jnp.where(outputs, packed[run], 0.0)
