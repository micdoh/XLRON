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


def _xpm_rows(pi, fi, bi, pk, fk, bk, total, *, a, a_bar, gamma, cr, beta2, beta3, beta4):
    """Sum one output-row tile against one interfering-channel tile."""
    fi, fk = fi[:, None], fk[None, :]
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
    denom = bk[None, :] * phi * a_bar * (2 * a + a_bar)
    ratio = (pk[None, :] / jnp.maximum(pi[:, None], EPS)) ** 2
    pairs = (
        32
        / 27
        * ratio
        * gamma**2
        / jnp.where(denom != 0, denom, 1.0)
        * (
            (tilt - a**2) / a * jnp.arctan(phi * bi[:, None] / a)
            + (ap**2 - tilt) / ap * jnp.arctan(phi * bi[:, None] / ap)
        )
    )
    return jnp.sum(jnp.where(denom != 0, pairs, 0.0), axis=1)


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
    interpret=False,
):
    """Evaluate exact dense XPM in tiles, without writing a channel-pair matrix.

    All slot rows are retained. ``compact`` packs nonzero interfering channels
    and shortens the kernel's reduction loop; it does not approximate the physics.
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
    live = (power != 0) & (bandwidth != 0)
    indices = jnp.nonzero(live, size=n, fill_value=0)[0] if compact else jnp.arange(n)
    count = jnp.sum(live) if compact else jnp.int32(n)

    def kernel(p_ref, f_ref, bw_ref, total_ref, index_ref, count_ref, out_ref):
        rows = pl.program_id(0) * block_i + jnp.arange(block_i)
        pi = plt.load(p_ref.at[rows], mask=rows < n, other=0.0)
        fi = plt.load(f_ref.at[rows], mask=rows < n, other=0.0)
        bi = plt.load(bw_ref.at[rows], mask=rows < n, other=0.0)
        total = total_ref[()]
        live_count = count_ref[()]

        def accumulate(tile, acc):
            columns = tile * block_k + jnp.arange(block_k)
            selected = plt.load(index_ref.at[columns], mask=columns < live_count, other=0)
            pk = plt.load(p_ref.at[selected], mask=columns < live_count, other=0.0)
            fk = plt.load(f_ref.at[selected], mask=columns < live_count, other=0.0)
            bk = plt.load(bw_ref.at[selected], mask=columns < live_count, other=0.0)
            return acc + _xpm_rows(pi, fi, bi, pk, fk, bk, total, **physics)

        result = jax.lax.fori_loop(
            0,
            (live_count + block_k - 1) // block_k,
            accumulate,
            jnp.zeros((block_i,), jnp.float32),
        )
        plt.store(out_ref.at[rows], result, mask=rows < n)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(power.shape, jnp.float32),
        grid=(pl.cdiv(n, block_i),),
        compiler_params=plt.CompilerParams(num_warps=num_warps),
        interpret=interpret,
        name="gn_xpm_tiled",
    )(power, frequency, bandwidth, jnp.sum(power), indices, count)
