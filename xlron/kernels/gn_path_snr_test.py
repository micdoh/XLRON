"""Every stored hop/slot, empty paths, padding and full ROADM/TRX SNR parity."""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments.dataclasses import (
    GNModelEnvParams,
    HashableArrayWrapper,
    RMSAGNModelEnvParams,
)
from xlron.environments.env_funcs import (
    get_candidate_baseline_snr,
    get_candidate_snr_evidence,
    get_lightpath_snr,
    get_required_snr_se_kurtosis_array,
)
from xlron.environments.make_env import make
from xlron.kernels import gn_path_snr
from xlron.kernels.gn_predicate_pack import pack_old_snr_predicates


@pytest.mark.parametrize("native", [False, True])
def test_cached_evidence_backend_is_independent_of_observation_backend(native, monkeypatch):
    from xlron.environments import env_funcs

    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    if not native:
        monkeypatch.setattr(
            gn_path_snr, "path_snr_pallas", partial(gn_path_snr.path_snr_pallas, interpret=True)
        )
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
    _, state = env.reset(jax.random.PRNGKey(0), params)
    state = state.replace(
        link_snr_array=jnp.full_like(state.link_snr_array, 1000),
        path_index_array=jnp.zeros_like(state.path_index_array),
        channel_power_array=jnp.full_like(state.channel_power_array, 1e-4),
        channel_centre_bw_array=jnp.full_like(state.channel_centre_bw_array, 12.5),
    )
    required = get_candidate_baseline_snr(state, params, backend="pallas")
    # A separately configured observation backend must not perturb admission at
    # an exact SNR threshold; every cached predicate is still evaluated.
    monkeypatch.setattr(
        env_funcs, "get_lightpath_snr", lambda s, p: jnp.full_like(s.link_snr_array, -123)
    )
    baseline = get_candidate_baseline_snr(state, params, backend="pallas")
    np.testing.assert_array_equal(baseline, required)
    _, ok = get_candidate_snr_evidence(
        state,
        state,
        jnp.zeros_like(state.path_index_array, bool),
        params,
        backend="pallas",
        old_required=required,
        unchanged=(baseline, jnp.zeros(params.path_link_indices.val.shape[0], bool)),
    )
    assert ok


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_packed_old_path_still_rejects_degradation_on_another_shared_link(
    native, reuse, monkeypatch
):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    if not native:
        monkeypatch.setattr(
            gn_path_snr, "path_snr_pallas", partial(gn_path_snr.path_snr_pallas, interpret=True)
        )
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
        interpret=not native,
    )
    assert np.isfinite(packed[links[0], old_slot])
    assert packed[links[-1], old_slot] == -jnp.inf

    evaluate = jax.jit(
        lambda c, r: get_candidate_snr_evidence(
            state,
            c,
            affected,
            params,
            backend="pallas",
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


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bad_baseline", [False, True])
def test_unaffected_path_reuse_preserves_existing_rejection(native, bad_baseline, monkeypatch):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    if not native:
        monkeypatch.setattr(
            gn_path_snr, "path_snr_pallas", partial(gn_path_snr.path_snr_pallas, interpret=True)
        )
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
            state, c, affected, params, backend="pallas", unchanged=(baseline, overlap)
        )
    )(candidate)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-5)
    assert bool(actual[1]) == bool(expected[1]) == (not bad_baseline)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("slots", [1, 17, 140])
def test_path_nsr_matches_independent_hop_sum(native, slots):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(13)
    links, paths, hops = 8, 19, 6
    table = np.full((paths, hops), links, np.int32)
    for row in table[:-1]:
        count = rng.integers(1, hops + 1)
        row[:count] = np.sort(rng.choice(links, count, replace=False))
    snr = rng.uniform(1, 100, (2, links, slots)).astype(np.float32)
    index = rng.integers(-1, paths - 1, snr.shape, dtype=np.int32)
    # Zero on-path SNR must give infinity; padded zero-SNR links must be ignored.
    snr[0, -1] = 0
    expected = np.zeros_like(snr)
    with np.errstate(divide="ignore"):
        for batch, link, slot in np.ndindex(snr.shape):
            for hop in table[index[batch, link, slot]]:
                if hop < links:
                    expected[batch, link, slot] += 1 / snr[batch, hop, slot]
    actual = jax.jit(
        jax.vmap(
            lambda s, p: gn_path_snr.path_nsr_pallas(s, p, jnp.asarray(table), interpret=not native)
        )
    )(jnp.asarray(snr), jnp.asarray(index))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("native", [False, True])
def test_compact_link_updates_match_materialized_path_snr(native):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(93)
    links, slots, paths = 5, 19, 4
    snr = jnp.asarray(rng.uniform(1, 100, (links, slots)), jnp.float32)
    updates = snr[jnp.array([3, 0])] * jnp.linspace(0.1, 1.1, slots)[None, :]
    updates = updates.at[0, 5].set(0.0).at[1, 9].set(jnp.nan)
    lookup = jnp.array([1, -1, -1, 0, -1])
    table = jnp.array([[0, 3, links], [0, 2, 4], [1, 3, links], [links] * 3])
    ids = jnp.asarray(rng.integers(-1, paths - 1, (links, slots)), jnp.int32)
    power = jnp.full_like(snr, 1e-4)
    centres = jnp.broadcast_to(jnp.linspace(-100, 100, slots), snr.shape)
    bandwidth = jnp.full_like(snr, 25)
    full = snr.at[jnp.array([3, 0])].set(updates)

    def evaluate(s, changed=None, evidence=None):
        return gn_path_snr.path_snr_pallas(
            s,
            ids,
            table,
            power,
            centres,
            bandwidth,
            jnp.full(slots, 5.0),
            jnp.full(slots, 8.0),
            jnp.full(slots, 5.0),
            jnp.full(slots, 21.0),
            ref_lambda=1.564e-6,
            precompute=True,
            interpret=not native,
            link_updates=changed,
            evidence=evidence,
        )

    expected = jax.jit(evaluate)(full)
    actual = jax.jit(lambda s, u: evaluate(s, (u, lookup)))(snr, updates)
    np.testing.assert_array_equal(actual, expected)
    required = jnp.where(jnp.arange(links)[:, None] % 2, 10.0, -jnp.inf)
    required = jnp.broadcast_to(required, snr.shape)
    affected = jnp.broadcast_to(jnp.arange(slots)[None, :] % 5 == 0, snr.shape)
    expected_evidence = jax.jit(lambda s: evaluate(s, evidence=(required, affected)))(full)
    actual_evidence = jax.jit(lambda s, u: evaluate(s, (u, lookup), (required, affected)))(
        snr, updates
    )
    np.testing.assert_array_equal(actual_evidence, expected_evidence)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("backend", ["pallas", "pallas_fused", "pallas_precomputed"])
def test_full_path_snr_with_roadm_and_transceivers(native, backend, monkeypatch):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    if not native:
        monkeypatch.setattr(
            gn_path_snr, "path_nsr_pallas", partial(gn_path_snr.path_nsr_pallas, interpret=True)
        )
        monkeypatch.setattr(
            gn_path_snr, "path_snr_pallas", partial(gn_path_snr.path_snr_pallas, interpret=True)
        )
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
    params = cast(GNModelEnvParams, params)
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
    actual = jax.jit(lambda s: get_lightpath_snr(s, params, nsr_backend=backend))(state)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-5)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "mode", ["empty", "sparse", "accept", "reject", "nan", "positive_inf", "negative_inf"]
)
@pytest.mark.parametrize("mask_dtype", [jnp.bool_, jnp.int32])
def test_reduced_candidate_evidence_keeps_all_original_predicates(
    native, mode, mask_dtype, monkeypatch
):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
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
    if not native:
        monkeypatch.setattr(
            gn_path_snr, "path_snr_pallas", partial(gn_path_snr.path_snr_pallas, interpret=True)
        )
    floor, ok = jax.jit(
        lambda s, c, a: get_candidate_snr_evidence(s, c, a, params, backend="pallas")
    )(state, candidate, jnp.asarray(affected, mask_dtype))
    np.testing.assert_allclose(floor, expected_floor, rtol=1e-6, atol=2e-5)
    np.testing.assert_array_equal(ok, expected_ok)
