"""Unrestricted GN policies retain interior slots and exact admission checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments.dataclasses import HashableArrayWrapper
from xlron.environments.env_funcs import get_snr_link_array, mask_slots_rmsa_gn_model
from xlron.environments.transformer_snr_test import setup


def fixture(mode):
    c, env, p, s = setup("rmsa_gn_model", "lifetime_snr")
    mods = p.modulations_array.val.at[:, 2].set(-100.0)
    p = p.replace(gn_mask_mode=mode, modulations_array=HashableArrayWrapper(mods))
    return c, env, p, s


def with_mask(env, p, s):
    mask, full, mods = env.action_mask(s, p)
    return s.replace(link_slot_mask=mask, full_link_slot_mask=full, mod_format_mask=mods)


def test_all_slots_exposes_interior_starts_and_respects_power():
    _, env, p, s = fixture("all_slots")
    full = with_mask(env, p, s)
    restricted = with_mask(env, p.replace(gn_mask_mode="ff_lf"), s)
    assert full.link_slot_mask.shape == (p.k_paths * p.link_resources,)
    assert full.link_slot_mask[3] and not restricted.link_slot_mask[3]
    assert full.link_slot_mask.sum() > restricted.link_slot_mask.sum()
    rejected = with_mask(env, p.replace(max_power_per_fibre=1e-12), s)
    assert not rejected.link_slot_mask.any()


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_full_format_pruning_preserves_mask_and_selected_format(batch_size):
    _, env, p, s = fixture("all_slots")
    # Keep tied efficiencies and force the last/highest format to fail. Both
    # traversal orders must select the first valid format on an SE tie.
    mods = p.modulations_array.val.at[-1, 2].set(1000.0)
    mods = mods.at[1, 1].set(mods[0, 1])
    p = p.replace(gn_mask_batch_size=batch_size, modulations_array=HashableArrayWrapper(mods))
    s = s.replace(link_slot_array=s.link_slot_array.at[:, 4].set(-1))
    reference = mask_slots_rmsa_gn_model(s, p, s.request_array, prune_formats=False)
    actual = mask_slots_rmsa_gn_model(s, p, s.request_array)
    np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
    np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)
    assert actual.link_slot_mask.any()


def test_full_mask_valid_actions_pass_actual_admission():
    # Real thresholds: compare the mask to a separate full-network admission
    # solve, including existing connections, after a first accepted request.
    _, env, p, s = setup("rmsa_gn_model", "lifetime_snr")
    p = p.replace(gn_mask_mode="all_slots")
    for key in (21, 22):
        s = with_mask(env, p, s)
        indices = np.flatnonzero(np.asarray(s.link_slot_mask))
        assert len(indices)
        # Exercise an interior start where possible.
        action = indices[len(indices) // 2]
        before = int(s.accepted_services)
        _, s, reward, *_ = env.step(jax.random.PRNGKey(key), s, jnp.int32(action), p)
        assert reward >= 0 and int(s.accepted_services) == before + 1


@pytest.mark.parametrize("backend", ["jax", "pallas"])
def test_selected_gn_output_rows_retain_full_interferer_sum(backend, monkeypatch):
    if backend == "pallas" and jax.default_backend() != "gpu":
        from functools import partial
        from xlron.kernels import gn_xpm_legacy

        monkeypatch.setattr(
            gn_xpm_legacy,
            "xpm_selected_pallas",
            partial(
                gn_xpm_legacy.xpm_selected_pallas,
                interpret=True,
            ),
        )
    _, _, p, s = setup("rmsa_gn_model", "lifetime_snr")
    f = jnp.linspace(-1000.0, 1000.0, p.link_resources)
    power = jnp.array([1e-4, 2e-4, 0, 1e-4, 3e-4, 0, 1e-4, 2e-4])
    s = s.replace(
        channel_power_array=jnp.broadcast_to(power, s.channel_power_array.shape),
        channel_centre_bw_array=jnp.full_like(s.channel_centre_bw_array, 25.0),
        channel_centre_freq_array=jnp.broadcast_to(f, s.channel_centre_freq_array.shape),
    )
    indices = jnp.array([6, 0, 6, 2])
    actual = get_snr_link_array(s, p, output_indices=indices, output_xpm_backend=backend)
    expected = get_snr_link_array(s, p)[:, indices]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    links = jnp.array([3, 1, 3])
    rows = (
        s.channel_power_array[links],
        s.channel_centre_freq_array[links],
        s.channel_centre_bw_array[links],
    )
    selected = get_snr_link_array(
        s, p, links, output_indices=indices, channel_rows=rows, output_xpm_backend=backend
    )
    np.testing.assert_allclose(selected, expected[links], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("screen_batch", [None, 1, 8])
def test_new_endpoint_screen_preserves_full_mask(narrow, screen_batch):
    _, _, p, s = setup("rmsa_gn_model", "lifetime_snr")
    p = p.replace(gn_mask_mode="all_slots", gn_mask_batch_size=3)
    reference = mask_slots_rmsa_gn_model(s, p, s.request_array, prune_formats=False)
    actual = mask_slots_rmsa_gn_model(
        s,
        p,
        s.request_array,
        prefilter_new=True,
        narrow_new_snr=narrow,
        screen_batch_size=screen_batch,
    )
    np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
    np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)


@pytest.mark.parametrize("bandwidth", [100, 200, 400])
@pytest.mark.parametrize("thresholds", ["real", "highest_reject", "tie"])
def test_width_grouping_preserves_every_format_and_admission(bandwidth, thresholds):
    _, env, p, state = setup("rmsa_gn_model", "lifetime_snr")
    mods = p.modulations_array.val
    if thresholds == "highest_reject":
        mods = mods.at[-1, 2].set(1000.0)
    elif thresholds == "tie":
        mods = mods.at[:, 2].set(-100.0).at[-1, 1].set(mods[-2, 1])
    p = p.replace(
        gn_mask_mode="all_slots", gn_mask_batch_size=3, modulations_array=HashableArrayWrapper(mods)
    )
    for seed in (71, 72):
        state = state.replace(request_array=state.request_array.at[1].set(bandwidth))
        reference = mask_slots_rmsa_gn_model(state, p, state.request_array, prune_formats=False)
        actual = mask_slots_rmsa_gn_model(
            state,
            p,
            state.request_array,
            prefilter_new=True,
            group_widths=True,
            reuse_screened_power=True,
            narrow_new_snr=True,
            reuse_unaffected_snr=True,
            screen_batch_size=8,
        )
        np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
        np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)
        if thresholds in ("highest_reject", "tie"):
            assert not np.any(np.asarray(actual.mod_format_mask) == len(mods) - 1)
        if actual.link_slot_mask.any():
            _, state, *_ = env.step(
                jax.random.PRNGKey(seed), actual, jnp.argmax(actual.link_slot_mask), p
            )


@pytest.mark.parametrize("field", ["mod_format_correction", "use_raman_amp"])
def test_width_grouping_rejects_format_dependent_physics(field):
    _, _, p, state = fixture("all_slots")
    with pytest.raises(ValueError, match="Width grouping requires"):
        mask_slots_rmsa_gn_model(
            state, p.replace(**{field: True}), state.request_array, group_widths=True
        )


@pytest.mark.parametrize("screening,pruning", [(True, True), (False, True), (True, False)])
def test_reusing_screened_power_preserves_thresholds_and_fallback(screening, pruning):
    _, _, p, state = fixture("all_slots")
    p = p.replace(gn_mask_batch_size=3)
    power = p.slot_launch_power_array.val[0]
    # Include both sides of the exact one-channel limit and the boundary itself.
    for limit in (jnp.nextafter(power, jnp.float32(0)), power, jnp.nextafter(power, jnp.inf)):
        q = p.replace(max_power_per_fibre=float(limit))
        reference = mask_slots_rmsa_gn_model(state, q, state.request_array, prune_formats=False)
        actual = mask_slots_rmsa_gn_model(
            state,
            q,
            state.request_array,
            prune_formats=pruning,
            prefilter_new=screening,
            reuse_screened_power=True,
        )
        np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
        np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)


def test_reusing_screened_power_checks_unaffected_overloaded_links():
    _, _, p, state = fixture("all_slots")
    p = p.replace(gn_mask_batch_size=3)
    state = state.replace(
        channel_power_array=state.channel_power_array.at[0, 0].set(1.0),
        channel_centre_bw_array=state.channel_centre_bw_array.at[0, 0].set(p.slot_size),
        path_index_array=state.path_index_array.at[0, 0].set(0),
        link_slot_array=state.link_slot_array.at[0, 0].set(1),
    )
    p = p.replace(max_power_per_fibre=0.1)
    reference = mask_slots_rmsa_gn_model(state, p, state.request_array, prune_formats=False)
    actual = mask_slots_rmsa_gn_model(
        state, p, state.request_array, prefilter_new=True, reuse_screened_power=True
    )
    assert not actual.link_slot_mask.any()
    np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
    np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)


@pytest.mark.parametrize("narrow", [False, True, "compact"])
def test_compacted_old_path_predicates_preserve_full_mask(monkeypatch, narrow):
    from functools import partial
    from xlron.kernels import gn_predicate_pack

    if jax.default_backend() != "gpu":
        monkeypatch.setattr(
            gn_predicate_pack,
            "pack_old_snr_predicates",
            partial(gn_predicate_pack.pack_old_snr_predicates, interpret=True),
        )
    _, env, p, state = setup("rmsa_gn_model", "lifetime_snr")
    p = p.replace(gn_mask_mode="all_slots", gn_mask_batch_size=3)
    for seed in (28, 29):
        reference = mask_slots_rmsa_gn_model(state, p, state.request_array, prefilter_new=True)
        actual = mask_slots_rmsa_gn_model(
            state,
            p,
            state.request_array,
            prefilter_new=True,
            pack_old=True,
            reuse_screened_power=bool(narrow),
            narrow_new_snr=bool(narrow),
            reuse_unaffected_snr=bool(narrow),
            compact_snr_updates=narrow == "compact",
        )
        np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
        np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)
        state = actual
        _, state, *_ = env.step(
            jax.random.PRNGKey(seed), state, jnp.argmax(actual.link_slot_mask), p
        )


@pytest.mark.parametrize("chunk", [1, 2])
def test_mask_environment_chunks_preserve_outer_vmap(chunk):
    _, _, p, s = fixture("all_slots")
    p = p.replace(gn_mask_batch_size=3)
    # Different candidate counts, plus a non-divisible final environment chunk.
    states = [s.replace(link_slot_array=s.link_slot_array.at[:, :i].set(-1)) for i in (0, 3, 7)]
    states = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    reference = jax.vmap(
        lambda st: mask_slots_rmsa_gn_model(st, p, st.request_array, prefilter_new=True)
    )(states)
    q = p.replace(gn_mask_env_batch_size=chunk)
    actual = jax.vmap(
        lambda st: mask_slots_rmsa_gn_model(st, q, st.request_array, prefilter_new=True)
    )(states)
    np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
    np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)


def test_selected_slot_falls_back_to_physically_valid_modulation():
    _, env, p, s = fixture("selected_slot")
    mods = p.modulations_array.val.at[1:, 2].set(1000.0)
    p = p.replace(modulations_array=HashableArrayWrapper(mods))
    s = with_mask(env, p, s)
    assert s.link_slot_mask[3]
    _, next_state, reward, *_ = env.step(jax.random.PRNGKey(8), s, jnp.array(3), p)
    assert reward >= 0 and next_state.accepted_services == 1
    occupied = np.asarray(next_state.channel_power_array) > 0
    assert occupied.any()
    assert np.all(np.asarray(next_state.modulation_format_index_array)[occupied] == 0)


def test_selected_slot_physical_rejection_rolls_back_and_counts_blocking():
    _, env, p, s = fixture("selected_slot")
    mods = p.modulations_array.val.at[:, 2].set(1000.0)
    p = p.replace(modulations_array=HashableArrayWrapper(mods))
    s = with_mask(env, p, s)
    assert s.link_slot_mask[3]  # Physical feasibility is intentionally deferred.
    _, next_state, reward, *_ = env.step(jax.random.PRNGKey(9), s, jnp.array(3), p)
    assert reward < 0 and next_state.accepted_services == 0
    assert next_state.blocked_snr == 1 and next_state.blocked_spectrum == 0
    np.testing.assert_array_equal(next_state.link_slot_array, s.link_slot_array)
    np.testing.assert_array_equal(next_state.channel_power_array, s.channel_power_array)


def test_selected_slot_rejects_forced_band_gap_action():
    _, env, p, s = fixture("selected_slot")
    s = s.replace(link_slot_array=s.link_slot_array.at[:, 3].set(-1))
    s = with_mask(env, p, s)
    assert not s.link_slot_mask[3]
    _, ns, _, *_ = env.step(jax.random.PRNGKey(10), s, jnp.array(3), p)
    assert ns.blocked_spectrum == 1 and ns.accepted_services == 0
    np.testing.assert_array_equal(ns.link_slot_array, s.link_slot_array)
