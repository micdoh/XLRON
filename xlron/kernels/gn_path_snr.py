"""Fused sparse path-NSR gather, avoiding batched multidimensional index tensors."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from scipy.constants import c, h

from xlron.environments.gn_model.isrs_gn_model import calculate_roadm_ase


def _path_reduce(
    snr_ref,
    index_ref,
    paths_ref,
    num_links,
    slots,
    num_paths,
    hops,
    block_slots,
    wanted=None,
    row=None,
    columns=None,
    updated_refs=None,
):
    row = pl.program_id(0) if row is None else row
    columns = (
        pl.program_id(1) * block_slots + jnp.arange(block_slots) if columns is None else columns
    )
    mask = columns < slots
    if wanted is not None:
        mask = mask & wanted
    pid = plt.load(index_ref.at[row, columns], mask=mask, other=-1)
    pid = jnp.clip(jnp.where(pid < 0, pid + num_paths, pid), 0, num_paths - 1)
    nsr = jnp.zeros((block_slots,), jnp.float32)
    count = jnp.zeros((block_slots,), jnp.int32)
    first = jnp.zeros((block_slots,), jnp.int32)
    for hop in range(hops):
        link = plt.load(paths_ref.at[pid, hop], mask=mask, other=num_links)
        valid = mask & (link < num_links)
        safe_link = jnp.minimum(link, num_links - 1)
        if updated_refs is None:
            snr = plt.load(snr_ref.at[safe_link, columns], mask=valid, other=1.0)
        else:
            updates_ref, lookup_ref = updated_refs
            source = plt.load(lookup_ref.at[safe_link], mask=valid, other=-1)
            baseline = plt.load(
                snr_ref.at[safe_link, columns], mask=valid & (source < 0), other=1.0
            )
            updated = plt.load(
                updates_ref.at[jnp.maximum(source, 0), columns],
                mask=valid & (source >= 0),
                other=1.0,
            )
            snr = jnp.where(source >= 0, updated, baseline)
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
    num_warps=4,
    precompute=False,
    evidence=None,
    unchanged=None,
    link_updates=None,
    interpret=False,
):
    """Fuse the exact path gather, ROADM ASE, transceiver noise and dB conversion.

    Calls the shared ROADM formula inside the kernel. All per-slot parameter
    arrays and the original per-link channel metadata are retained.
    With ``evidence=(encoded_old_required_snr, affected)``, reduce directly
    to (minimum new-path SNR, all existing paths sufficient). Only slots excluded
    by both admission predicates are skipped. Inactive old predicates are encoded
    as -inf, which every finite nan_to_num result satisfies. NaN/+inf required
    thresholds remain active and retain their original comparison behavior.
    ``unchanged=(baseline_path_snr, overlap_by_path_id)`` may reuse baseline
    results only for paths with no changed link; their old predicates still
    participate in the verdict. Intersecting paths gather every hop again.
    ``link_updates=(updated_rows, row_lookup)`` reads changed rows directly,
    avoiding a full candidate SNR scatter. A negative lookup keeps baseline SNR.
    """
    if unchanged is not None and evidence is None:
        raise ValueError("Unchanged path results require admission evidence")
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
        *tail_refs,
    ):
        row, tile = pl.program_id(0), pl.program_id(1)
        columns = tile * block_slots + jnp.arange(block_slots)
        mask = columns < slots
        unchanged_ok = jnp.bool_(True)
        out_ref = tail_refs[-1]
        offset = (2 if evidence is not None else 0) + (2 if unchanged is not None else 0)
        updated_refs = tail_refs[offset : offset + 2] if link_updates is not None else None
        if evidence is not None:
            required_ref, affected_ref = tail_refs[:2]
            required = plt.load(required_ref.at[row, columns], mask=mask, other=-jnp.inf)
            old_live = required != -jnp.inf
            affected = plt.load(affected_ref.at[row, columns], mask=mask, other=0) != 0
            if unchanged is not None:
                baseline_ref, overlap_ref = tail_refs[2:4]
                ids = plt.load(index_ref.at[row, columns], mask=mask, other=-1)
                ids = jnp.clip(jnp.where(ids < 0, ids + num_paths, ids), 0, num_paths - 1)
                overlaps = plt.load(overlap_ref.at[ids], mask=mask, other=True)
                retained = old_live & ~overlaps
                baseline = plt.load(baseline_ref.at[row, columns], mask=retained, other=50.0)
                unchanged_ok = jnp.sum((retained & ~(baseline >= required)).astype(jnp.int32)) == 0
                old_live = old_live & overlaps
            wanted = old_live | affected
        else:
            wanted = None

        def calculate():
            _columns, mask, nsr, count, first = _path_reduce(
                snr_ref,
                index_ref,
                paths_ref,
                num_links,
                slots,
                num_paths,
                hops,
                block_slots,
                wanted,
                row,
                columns,
                updated_refs,
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
            if evidence is not None:
                floor = jnp.min(jnp.where(affected, result, jnp.inf))
                ok = jnp.sum((old_live & ~(result >= required)).astype(jnp.int32)) == 0
                return floor, ok.astype(jnp.float32)
            return result

        if evidence is not None:
            assert wanted is not None
            floor, ok = jax.lax.cond(
                jnp.sum(wanted.astype(jnp.int32)) > 0,
                calculate,
                lambda: (
                    jnp.min(jnp.full((block_slots,), jnp.inf, jnp.float32)),
                    jnp.max(jnp.ones((block_slots,), jnp.float32)),
                ),
            )
            if unchanged is not None:
                ok = ok * unchanged_ok.astype(jnp.float32)
            plt.store(out_ref.at[jnp.int32(0), row, tile], floor)
            plt.store(out_ref.at[jnp.int32(1), row, tile], ok)
        else:
            plt.store(out_ref.at[row, columns], calculate(), mask=mask)

    shape = (2, num_links, pl.cdiv(slots, block_slots)) if evidence is not None else link_snr.shape
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(num_links, pl.cdiv(slots, block_slots)),
        compiler_params=plt.CompilerParams(num_warps=num_warps),
        interpret=interpret,
        name="gn_path_snr_evidence" if evidence is not None else "gn_path_snr_fused",
    )(
        link_snr,
        path_index,
        path_links,
        channel_power,
        centre_ghz,
        bandwidth_ghz,
        *per_slot,
        *(evidence or ()),
        *(unchanged or ()),
        *(link_updates or ()),
    )
    if evidence is not None:
        return jnp.min(result[0]), jnp.all(result[1] != 0)
    return result
