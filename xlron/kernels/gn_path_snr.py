"""Fused sparse path-NSR gather, avoiding batched multidimensional index tensors."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from scipy.constants import c, h

from xlron.environments.gn_model.isrs_gn_model import calculate_roadm_ase


def _path_reduce(snr_ref, index_ref, paths_ref, num_links, slots, num_paths, hops, block_slots):
    row = pl.program_id(0)
    columns = pl.program_id(1) * block_slots + jnp.arange(block_slots)
    mask = columns < slots
    pid = plt.load(index_ref.at[row, columns], mask=mask, other=-1)
    pid = jnp.clip(jnp.where(pid < 0, pid + num_paths, pid), 0, num_paths - 1)
    nsr = jnp.zeros((block_slots,), jnp.float32)
    count = jnp.zeros((block_slots,), jnp.int32)
    first = jnp.zeros((block_slots,), jnp.int32)
    for hop in range(hops):
        link = plt.load(paths_ref.at[pid, hop], mask=mask, other=num_links)
        valid = mask & (link < num_links)
        snr = plt.load(snr_ref.at[jnp.minimum(link, num_links - 1), columns], mask=valid, other=1.0)
        nsr = nsr + jnp.where(valid, 1.0 / snr, 0.0)
        count = count + valid.astype(jnp.int32)
        if hop == 0:
            first = jnp.where(valid, link, 0)
    return columns, mask, nsr, count, first


def path_nsr_pallas(link_snr, path_index, path_links, *, block_slots=128, interpret=False):
    """Sum inverse link SNR along every referenced path and slot in float32.

    ``path_links`` contains ascending link IDs, padded with ``num_links``. Negative
    path IDs use JAX's wrapping convention (the -1 dummy path is the last row).
    Padding is masked before the reciprocal, including zero/NaN padded-link SNR.
    No path, output slot or on-path contribution is omitted.
    """
    if link_snr.ndim != 2 or path_index.shape != link_snr.shape or path_links.ndim != 2:
        raise ValueError("Expected (links, slots) SNR/path IDs and (paths, hops) link IDs")
    if link_snr.dtype != jnp.float32:
        raise ValueError("float32 link SNR is required")
    num_links, slots = link_snr.shape
    num_paths, hops = path_links.shape

    def kernel(snr_ref, index_ref, paths_ref, out_ref):
        columns, mask, nsr, _, _ = _path_reduce(
            snr_ref, index_ref, paths_ref, num_links, slots, num_paths, hops, block_slots
        )
        plt.store(out_ref.at[pl.program_id(0), columns], nsr, mask=mask)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(link_snr.shape, jnp.float32),
        grid=(num_links, pl.cdiv(slots, block_slots)),
        compiler_params=plt.CompilerParams(num_warps=4),
        interpret=interpret,
        name="gn_path_nsr",
    )(link_snr, path_index, path_links)


def path_snr_pallas(
    link_snr,
    path_index,
    path_links,
    channel_power,
    centre_ghz,
    bandwidth_ghz,
    roadm_express_loss,
    roadm_add_drop_loss,
    roadm_noise_figure,
    transceiver_snr,
    *,
    ref_lambda,
    block_slots=128,
    precompute=False,
    interpret=False,
):
    """Fuse the exact path gather, ROADM ASE, transceiver noise and dB conversion.

    Calls the shared ROADM formula inside the kernel. All per-slot parameter
    arrays and the original per-link channel metadata are retained.
    """
    num_links, slots = link_snr.shape
    num_paths, hops = path_links.shape
    per_slot = tuple(
        jnp.broadcast_to(x, (slots,))
        for x in (roadm_express_loss, roadm_add_drop_loss, roadm_noise_figure, transceiver_snr)
    )
    if precompute:
        express, ad, nf, trx = per_slot
        nf_linear = 10 ** (nf / 10)
        express_gain = 10 ** (express / 10)
        express_m1 = express_gain - 1
        nsp_express = nf_linear * express_gain / (2.0 * express_m1)
        ad_gain = 10 ** (ad / 10)
        ad_m1 = ad_gain - 1
        nsp_ad = nf_linear * ad_gain / (2.0 * ad_m1)
        ad_coefficient = 2 * 2 * nsp_ad * ad_m1 * h
        trx_linear = 10 ** (trx / 10)
        trx_nsr = jnp.where(trx_linear > 1.0, 1.0 / trx_linear, 0.0)
        per_slot = (nsp_express, express_m1, ad_coefficient, trx_nsr)

    def kernel(
        snr_ref,
        index_ref,
        paths_ref,
        power_ref,
        centre_ref,
        bandwidth_ref,
        express_ref,
        ad_ref,
        nf_ref,
        trx_ref,
        out_ref,
    ):
        columns, mask, nsr, count, first = _path_reduce(
            snr_ref, index_ref, paths_ref, num_links, slots, num_paths, hops, block_slots
        )
        power = plt.load(power_ref.at[first, columns], mask=mask, other=0.0)
        centre = plt.load(centre_ref.at[first, columns], mask=mask, other=0.0) * 1e9
        bandwidth = plt.load(bandwidth_ref.at[first, columns], mask=mask, other=0.0) * 1e9
        express = plt.load(express_ref.at[columns], mask=mask, other=1.0)
        ad = plt.load(ad_ref.at[columns], mask=mask, other=1.0)
        nf = plt.load(nf_ref.at[columns], mask=mask, other=1.0)
        trx = plt.load(trx_ref.at[columns], mask=mask, other=0.0)
        if precompute:
            frequency = c / ref_lambda + centre
            ase = (
                jnp.maximum(count - 1, 0) * 2 * express * ad * h * frequency * bandwidth
                + nf * frequency * bandwidth
            )
            trx_nsr = trx
        else:
            ase = calculate_roadm_ase(
                express, ad, nf, jnp.maximum(count - 1, 0), ref_lambda, centre, bandwidth
            )
            trx_linear = 10 ** (trx / 10)
            trx_nsr = jnp.where(trx_linear > 1.0, 1.0 / trx_linear, 0.0)
        nsr = nsr + jnp.where(power > 0, ase / power, 0.0)
        nsr = nsr + trx_nsr
        result = jnp.nan_to_num(10 * jnp.log10(1 / nsr), nan=-50, neginf=-50, posinf=50)
        plt.store(out_ref.at[pl.program_id(0), columns], result, mask=mask)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(link_snr.shape, jnp.float32),
        grid=(num_links, pl.cdiv(slots, block_slots)),
        compiler_params=plt.CompilerParams(num_warps=4),
        interpret=interpret,
        name="gn_path_snr_fused",
    )(link_snr, path_index, path_links, channel_power, centre_ghz, bandwidth_ghz, *per_slot)
