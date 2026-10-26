"""Pallas path SNR: every stored hop/slot, empty paths, padding, ROADM/TRX parity.

Runs the kernels in the Pallas interpreter off-GPU and natively on a GPU.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments.dataclasses import (
    HashableArrayWrapper,
    RMSAGNModelEnvParams,
)
from xlron.environments.env_funcs import (
    get_candidate_snr_evidence,
    get_lightpath_snr,
    get_required_snr_se_kurtosis_array,
)
from xlron.environments.make_env import make
from xlron.kernels.gn_predicate_pack import pack_old_snr_predicates


@pytest.mark.parametrize("reuse", [False, True])
def test_packed_old_path_still_rejects_degradation_on_another_shared_link(reuse):
    env, params = make(
        dict(
            env_type="rmsa_gn_model",
            topology_name="5node_directed",
            link_resources=17,
            k=2,
            values_bw=[100],
        ),
        log_wrapper=False,
    )
    params = cast(RMSAGNModelEnvParams, params)
    _, empty = env.reset(jax.random.PRNGKey(0), params)
    paths = np.asarray(params.path_link_indices.val)
    pid = int(np.flatnonzero(np.sum(paths < params.num_links, axis=1) >= 2)[0])
    links = jnp.asarray(paths[pid][paths[pid] < params.num_links])
    old_slot, new_slot = 3, 9
    state = empty.replace(
        # Existing lightpaths depart in the future (none is held for release).
        link_slot_departure_array=jnp.full_like(empty.link_slot_departure_array, 100.0),
        link_snr_array=jnp.full_like(empty.link_snr_array, 1000),
        path_index_array=empty.path_index_array.at[links, old_slot].set(pid),
        channel_power_array=empty.channel_power_array.at[links, old_slot].set(1e-4),
        channel_centre_bw_array=empty.channel_centre_bw_array.at[links, old_slot].set(12.5),
        modulation_format_index_array=empty.modulation_format_index_array.at[links, old_slot].set(
            0
        ),
    )
    candidate = state.replace(
        path_index_array=state.path_index_array.at[links, new_slot].set(pid),
        channel_power_array=state.channel_power_array.at[links, new_slot].set(1e-4),
        channel_centre_bw_array=state.channel_centre_bw_array.at[links, new_slot].set(12.5),
    )
    affected = jnp.zeros_like(state.path_index_array, dtype=bool).at[links, new_slot].set(True)
    required = get_required_snr_se_kurtosis_array(state.modulation_format_index_array, 2, params)
    packed = pack_old_snr_predicates(
        state.path_index_array,
        jnp.where(state.channel_centre_bw_array != 0, required, -jnp.inf),
        len(paths),
    )
    assert np.isfinite(packed[links[0], old_slot])
    assert packed[links[-1], old_slot] == -jnp.inf

    kernel_params = params.replace(gn_kernel_backend="pallas")
    evaluate = jax.jit(
        lambda c, r: get_candidate_snr_evidence(
            state,
            c,
            affected,
            kernel_params,
            old_required=r,
            unchanged=(get_lightpath_snr(state, params), jnp.ones(len(paths), bool))
            if reuse
            else None,
        )
    )
    before_floor, before_ok = evaluate(candidate, packed)
    assert before_ok
    # The checked predicate lives on the first link row, but the SNR must still
    # gather the changed last link of that existing path at its original slot.
    degraded = candidate.replace(
        link_snr_array=candidate.link_snr_array.at[links[-1], old_slot].set(0.01)
    )
    after_floor, after_ok = evaluate(degraded, packed)
    _, exhaustive_ok = get_candidate_snr_evidence(state, degraded, affected, params)
    np.testing.assert_allclose(after_floor, before_floor)
    assert after_floor > 1.9 and not after_ok and not exhaustive_ok


@pytest.mark.parametrize("bad_baseline", [False, True])
def test_unaffected_path_reuse_preserves_existing_rejection(bad_baseline):
    env, params = make(
        dict(
            env_type="rmsa_gn_model",
            topology_name="5node_directed",
            link_resources=17,
            k=2,
            values_bw=[100],
        ),
        log_wrapper=False,
    )
    params = cast(RMSAGNModelEnvParams, params)
    _, empty = env.reset(jax.random.PRNGKey(0), params)
    paths = np.asarray(params.path_link_indices.val)
    live_paths = [i for i, row in enumerate(paths) if np.sum(row < params.num_links) >= 2]
    old, new = next(
        (a, b)
        for a in live_paths
        for b in live_paths
        if not set(paths[a][paths[a] < params.num_links])
        & set(paths[b][paths[b] < params.num_links])
    )
    old_links = jnp.asarray(paths[old][paths[old] < params.num_links])
    new_links = jnp.asarray(paths[new][paths[new] < params.num_links])
    state = empty.replace(
        # Existing lightpaths depart in the future (none is held for release).
        link_slot_departure_array=jnp.full_like(empty.link_slot_departure_array, 100.0),
        link_snr_array=jnp.full_like(empty.link_snr_array, 1000)
        .at[old_links[-1], 3]
        .set(0.01 if bad_baseline else 1000),
        path_index_array=empty.path_index_array.at[old_links, 3].set(old),
        channel_power_array=empty.channel_power_array.at[old_links, 3].set(1e-4),
        channel_centre_bw_array=empty.channel_centre_bw_array.at[old_links, 3].set(12.5),
    )
    candidate = state.replace(
        link_snr_array=state.link_snr_array.at[new_links].multiply(0.5),
        path_index_array=state.path_index_array.at[new_links, 9].set(new),
        channel_power_array=state.channel_power_array.at[new_links, 9].set(1e-4),
        channel_centre_bw_array=state.channel_centre_bw_array.at[new_links, 9].set(12.5),
    )
    affected = jnp.zeros_like(state.path_index_array, bool).at[new_links, 9].set(True)
    overlap = jnp.any(jnp.isin(params.path_link_indices.val, new_links), axis=-1)
    baseline = get_lightpath_snr(state, params)
    expected = get_candidate_snr_evidence(state, candidate, affected, params)
    actual = jax.jit(
        lambda c: get_candidate_snr_evidence(
            state,
            c,
            affected,
            params.replace(gn_kernel_backend="pallas"),
            unchanged=(baseline, overlap),
        )
    )(candidate)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-5)
    assert bool(actual[1]) == bool(expected[1]) == (not bad_baseline)


@pytest.mark.parametrize("express_loss", [None, 0.0])
def test_full_path_snr_with_roadm_and_transceivers(express_loss):
    env, params = make(
        {
            "env_type": "rmsa_gn_model",
            "topology_name": "5node_directed",
            "link_resources": 17,
            "k": 3,
            "values_bw": [100],
            "slot_size": 12.5,
            "guardband": 0,
        },
        log_wrapper=False,
    )
    _, state = env.reset(jax.random.PRNGKey(0), params)
    params = cast(RMSAGNModelEnvParams, params)
    if express_loss is not None:
        # 0 dB express loss divides by (G - 1) = 0 in both implementations.
        params = params.replace(
            roadm_express_loss=HashableArrayWrapper(
                jnp.full_like(params.roadm_express_loss.val, express_loss)
            )
        )
    rng = np.random.default_rng(7)
    shape = state.link_snr_array.shape
    state = state.replace(
        link_snr_array=jnp.asarray(rng.uniform(1, 100, shape).astype(np.float32)),
        path_index_array=jnp.asarray(
            rng.integers(-1, params.path_link_indices.val.shape[0] - 1, shape, dtype=np.int32)
        ),
        channel_power_array=jnp.full(shape, 1e-4),
        channel_centre_bw_array=jnp.full(shape, 12.5),
        channel_centre_freq_array=jnp.broadcast_to(jnp.linspace(-1000, 1000, shape[1]), shape),
    )
    expected = jax.jit(lambda s: get_lightpath_snr(s, params))(state)
    kernel_params = params.replace(gn_kernel_backend="pallas")
    actual = jax.jit(lambda s: get_lightpath_snr(s, kernel_params))(state)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-5)


@pytest.mark.parametrize(
    "mode", ["empty", "sparse", "accept", "reject", "nan", "positive_inf", "negative_inf"]
)
@pytest.mark.parametrize("mask_dtype", [jnp.bool_, jnp.int32])
def test_reduced_candidate_evidence_keeps_all_original_predicates(mode, mask_dtype):
    env, params = make(
        {
            "env_type": "rmsa_gn_model",
            "topology_name": "5node_directed",
            "link_resources": 140,
            "k": 2,
            "values_bw": [100],
        },
        log_wrapper=False,
    )
    params = cast(RMSAGNModelEnvParams, params)
    _, state = env.reset(jax.random.PRNGKey(0), params)
    shape = state.link_snr_array.shape
    rng = np.random.default_rng(37)
    old_bw = rng.random(shape) < 0.15
    affected = rng.random(shape) < 0.03
    old_bw[::2, :] = False
    affected[::2, :] = False  # Exercise completely empty tiles and padded tails.
    if mode == "empty":
        old_bw[:] = affected[:] = False
    mods = params.modulations_array.val
    thresholds = {
        "accept": -1000,
        "reject": 1000,
        "nan": jnp.nan,
        "positive_inf": jnp.inf,
        "negative_inf": -jnp.inf,
    }
    if mode in thresholds:
        params = params.replace(
            modulations_array=HashableArrayWrapper(mods.at[:, 2].set(thresholds[mode]))
        )
    state = state.replace(
        # Every existing lightpath departs in the future (none is held for release).
        link_slot_departure_array=jnp.full(shape, 100.0, state.link_slot_departure_array.dtype),
        channel_centre_bw_array=jnp.asarray(old_bw.astype(np.float32) * 12.5),
        modulation_format_index_array=jnp.asarray(
            rng.integers(0, len(mods), shape, dtype=np.int32)
        ),
    )
    candidate = state.replace(
        link_snr_array=jnp.asarray(rng.uniform(1, 100, shape).astype(np.float32)).at[0, 0].set(0),
        path_index_array=jnp.asarray(
            rng.integers(-1, params.path_link_indices.val.shape[0] - 1, shape, dtype=np.int32)
        ),
        channel_power_array=jnp.full(shape, 1e-4),
        channel_centre_bw_array=jnp.full(shape, 12.5),
        channel_centre_freq_array=jnp.broadcast_to(jnp.linspace(-1000, 1000, shape[1]), shape),
    )
    full = get_lightpath_snr(candidate, params)
    required = get_required_snr_se_kurtosis_array(state.modulation_format_index_array, 2, params)
    expected_floor = jnp.min(jnp.where(affected, full, jnp.inf))
    expected_ok = jnp.all(jnp.where(old_bw, full >= required, True))
    floor, ok = jax.jit(
        lambda s, c, a: get_candidate_snr_evidence(
            s, c, a, params.replace(gn_kernel_backend="pallas")
        )
    )(state, candidate, jnp.asarray(affected, mask_dtype))
    np.testing.assert_allclose(floor, expected_floor, rtol=1e-6, atol=2e-5)
    np.testing.assert_array_equal(ok, expected_ok)
