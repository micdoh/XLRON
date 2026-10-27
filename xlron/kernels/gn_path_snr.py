"""Fused path-SNR gather for the GN environment (Pallas).

For every (link, slot) that carries a lightpath, sums inverse link SNR along
that lightpath's path, adds ROADM ASE and transceiver noise, and converts to dB:
the same quantity as `env_funcs.get_lightpath_snr`, without materialising the
(links, slots, hops) gather. ROADM ASE uses the shared
`roadm_ase_coefficients` / `roadm_ase_from_coefficients` helpers.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from xlron.environments.gn_model.isrs_gn_model import (
    roadm_ase_coefficients,
    roadm_ase_from_coefficients,
)
from xlron.kernels import interpret_default


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
    evidence=None,
    unchanged=None,
    delta=None,
    block_slots=128,
    num_warps=4,
    interpret=None,
):
    """Lightpath SNR [dB] per (link, slot), or an admission verdict.

    Args:
        link_snr: (L, S) float32 linear link SNR.
        path_index: (L, S) path ID per slot; negative IDs wrap (the -1 dummy
            path is the last row of path_links).
        path_links: (P, H) ascending link IDs per path, padded with L.
        channel_power, centre_ghz, bandwidth_ghz: (L, S) channel metadata.
        roadm_* / transceiver_snr: per-slot (or scalar) parameters in dB.
        evidence: optional (required_db, affected). Instead of the SNR array,
            return (minimum SNR over affected slots, all live predicates pass).
            required_db is -inf for inactive predicates; NaN/+inf thresholds keep
            their rejecting comparison.
        unchanged: optional (baseline_db, overlap_by_path) with evidence: paths
            sharing no link with the candidate reuse their baseline SNR instead
            of being gathered again; their predicates still count.
        delta: optional (links, changed_snr, slots, values) with evidence. The
            (L, S) inputs then describe the network before one candidate lightpath
            is inserted, and the candidate is applied on the fly instead of being
            materialised: links (H,) are its path links padded with L,
            changed_snr (H, S) their re-solved link SNR, slots = [first, last,
            path ID] (int32) and values = [power, centre GHz, bandwidth GHz]
            (float32) for the new channel. evidence's affected must be None; the
            affected cells are the candidate's links over its slot range.
    """
    if unchanged is not None and evidence is None:
        raise ValueError("Unchanged path results require admission evidence")
    if delta is not None and (evidence is None or evidence[1] is not None):
        raise ValueError("Candidate deltas require evidence without an affected mask")
    if link_snr.ndim != 2 or path_index.shape != link_snr.shape or path_links.ndim != 2:
        raise ValueError("Expected (links, slots) SNR/path IDs and (paths, hops) link IDs")
    interpret = interpret_default() if interpret is None else interpret
    num_links, slots = link_snr.shape
    num_paths, hops = path_links.shape
    express, add_drop = (
        jnp.broadcast_to(x, (slots,)).astype(jnp.float32)
        for x in roadm_ase_coefficients(roadm_express_loss, roadm_add_drop_loss, roadm_noise_figure)
    )
    trx_linear = 10 ** (jnp.broadcast_to(transceiver_snr, (slots,)) / 10)
    trx_nsr = jnp.where(trx_linear > 1.0, 1.0 / trx_linear, 0.0).astype(jnp.float32)

    def kernel(snr_ref, index_ref, paths_ref, power_ref, centre_ref, bandwidth_ref, express_ref,
               add_drop_ref, trx_ref, *tail_refs):  # fmt: skip
        row, tile = pl.program_id(0), pl.program_id(1)
        columns = tile * block_slots + jnp.arange(block_slots)
        mask = columns < slots
        out_ref = tail_refs[-1]
        unchanged_ok = jnp.bool_(True)
        wanted = mask
        if delta is not None:
            links_ref, changed_ref, slots_ref, values_ref = tail_refs[-5:-1]
            changed_links = [links_ref[h] for h in range(changed_hops)]
            new_id = slots_ref[2]
            # Candidate cells: its path links over its slot range.
            inserted = (columns >= slots_ref[0]) & (columns <= slots_ref[1]) & mask

        def path_ids(cells):
            ids = plt.load(index_ref.at[row, columns], mask=cells, other=-1)
            if delta is None:
                return ids
            on_path = functools.reduce(jnp.logical_or, [row == x for x in changed_links])
            return jnp.where(on_path & inserted, new_id, ids)

        if evidence is not None:
            required_ref = tail_refs[0]
            required = plt.load(required_ref.at[row, columns], mask=mask, other=-jnp.inf)
            old_live = required != -jnp.inf
            if delta is None:
                affected_ref = tail_refs[1]
                affected = plt.load(affected_ref.at[row, columns], mask=mask, other=0) != 0
            else:
                on_path = functools.reduce(jnp.logical_or, [row == x for x in changed_links])
                affected = inserted & on_path
            if unchanged is not None:
                offset = 1 if delta is not None else 2  # no affected input with deltas
                baseline_ref, overlap_ref = tail_refs[offset : offset + 2]
                ids = path_ids(mask)
                ids = jnp.clip(jnp.where(ids < 0, ids + num_paths, ids), 0, num_paths - 1)
                overlaps = plt.load(overlap_ref.at[ids], mask=mask, other=True)
                retained = old_live & ~overlaps
                baseline = plt.load(baseline_ref.at[row, columns], mask=retained, other=50.0)
                unchanged_ok = jnp.sum((retained & ~(baseline >= required)).astype(jnp.int32)) == 0
                old_live = old_live & overlaps
            wanted = mask & (old_live | affected)

        def calculate():
            pid = path_ids(wanted)
            new_channel = (pid == new_id) & inserted if delta is not None else None
            pid = jnp.clip(jnp.where(pid < 0, pid + num_paths, pid), 0, num_paths - 1)
            nsr = jnp.zeros((block_slots,), jnp.float32)
            count = jnp.zeros((block_slots,), jnp.int32)
            first = jnp.zeros((block_slots,), jnp.int32)
            for hop in range(hops):
                link = plt.load(paths_ref.at[pid, hop], mask=wanted, other=num_links)
                valid = wanted & (link < num_links)
                safe_link = jnp.minimum(link, num_links - 1)
                if delta is None:
                    snr = plt.load(snr_ref.at[safe_link, columns], mask=valid, other=1.0)
                else:
                    # Changed links read the candidate's re-solved rows.
                    hop_row = jnp.full((block_slots,), -1, jnp.int32)
                    for h, x in enumerate(changed_links):
                        hop_row = jnp.where(link == x, h, hop_row)
                    fresh = valid & (hop_row >= 0)
                    snr = jnp.where(
                        fresh,
                        plt.load(changed_ref.at[jnp.maximum(hop_row, 0), columns], mask=fresh,
                                 other=1.0),
                        plt.load(snr_ref.at[safe_link, columns], mask=valid & ~fresh, other=1.0),
                    )  # fmt: skip
                # Padding is masked before the reciprocal: no 0 * inf NaNs.
                nsr = nsr + jnp.where(valid, 1.0 / snr, 0.0)
                count = count + valid.astype(jnp.int32)
                if hop == 0:
                    first = jnp.where(valid, link, 0)
            power = plt.load(power_ref.at[first, columns], mask=wanted, other=0.0)
            centre = plt.load(centre_ref.at[first, columns], mask=wanted, other=0.0)
            bandwidth = plt.load(bandwidth_ref.at[first, columns], mask=wanted, other=0.0)
            if delta is not None:
                power = jnp.where(new_channel, values_ref[0], power)
                centre = jnp.where(new_channel, values_ref[1], centre)
                bandwidth = jnp.where(new_channel, values_ref[2], bandwidth)
            centre, bandwidth = centre * 1e9, bandwidth * 1e9
            ase = roadm_ase_from_coefficients(
                plt.load(express_ref.at[columns], mask=wanted, other=0.0),
                plt.load(add_drop_ref.at[columns], mask=wanted, other=0.0),
                jnp.maximum(count - 1, 0),
                ref_lambda,
                centre,
                bandwidth,
            )
            nsr = nsr + jnp.where(power > 0, ase / power, 0.0)
            nsr = nsr + plt.load(trx_ref.at[columns], mask=wanted, other=0.0)
            result = jnp.nan_to_num(10 * jnp.log10(1 / nsr), nan=-50, neginf=-50, posinf=50)
            if evidence is None:
                return result
            floor = jnp.min(jnp.where(affected & mask, result, jnp.inf))
            ok = jnp.sum((old_live & mask & ~(result >= required)).astype(jnp.int32)) == 0
            return floor, ok.astype(jnp.float32)

        if evidence is None:
            plt.store(out_ref.at[row, columns], calculate(), mask=mask)
            return
        # Triton needs both branches to yield IR values, not Python constants,
        # so build the fallback scalars by reducing vectors.
        floor, ok = jax.lax.cond(
            jnp.sum(wanted.astype(jnp.int32)) > 0,
            calculate,
            lambda: (
                jnp.min(jnp.full((block_slots,), jnp.inf, jnp.float32)),
                jnp.max(jnp.ones((block_slots,), jnp.float32)),
            ),
        )
        plt.store(out_ref.at[jnp.int32(0), row, tile], floor)
        plt.store(out_ref.at[jnp.int32(1), row, tile], ok * unchanged_ok.astype(jnp.float32))

    changed_hops = 0 if delta is None else delta[0].shape[0]
    tiles = pl.cdiv(slots, block_slots)
    shape = (2, num_links, tiles) if evidence is not None else link_snr.shape
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(num_links, tiles),
        compiler_params=plt.CompilerParams(num_warps=num_warps),
        interpret=interpret,
        name="gn_path_snr_evidence" if evidence is not None else "gn_path_snr",
    )(
        link_snr,
        path_index,
        path_links,
        channel_power,
        centre_ghz,
        bandwidth_ghz,
        express,
        add_drop,
        trx_nsr,
        *(x for x in evidence or () if x is not None),
        *(unchanged or ()),
        *(delta or ()),
    )
    if evidence is not None:
        return jnp.min(result[0]), jnp.all(result[1] != 0)
    return result


def path_snr_pallas_from_state(state, params, **kwargs):
    """path_snr_pallas for a GN environment state and its parameters."""
    return path_snr_pallas(
        state.link_snr_array,
        state.path_index_array,
        params.path_link_indices.val,
        state.channel_power_array,
        state.channel_centre_freq_array,
        state.channel_centre_bw_array,
        params.roadm_express_loss.val,
        params.roadm_add_drop_loss.val,
        params.roadm_noise_figure.val,
        params.transceiver_snr.val,
        ref_lambda=params.ref_lambda,
        **kwargs,
    )
