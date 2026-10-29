"""Accuracy checks for sparse link refresh and bounded candidate evaluation."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments import env_funcs as ef
from xlron.environments.gn_model import isrs_gn_model
from xlron.environments.make_env import make


@cache
def setup(mode="uniform", packed=False, slots=12):
    env, params = make(
        {
            "env_type": "rmsa_gn_model",
            "topology_name": "5node_directed",
            "k": 2,
            "link_resources": slots,
            "slot_size": 25,
            "guardband": 0,
            "values_bw": [100],
            "max_requests": 100,
            "incremental_loading": True,
            "coherent": False,
            "max_power_per_fibre": 21,
            "power_per_channel": -10,
            "include_no_op": False,
            "uniform_spans": mode != "nonuniform",
            "mod_format_correction": mode == "corrected",
            "pack_path_bits": packed,
        },
        log_wrapper=False,
    )
    _, state = env.reset(jax.random.PRNGKey(3), params)
    return env, params, state


def exhaustive_mask(state, params):
    """Small-network oracle: independently evaluate FF/LF using full-network SNR.

    No reuse of cached link SNR or candidate streaming; all existing lightpaths,
    endpoint noise, modulation thresholds and fibre power budgets are checked.
    """
    nodes, datarate = ef.read_rsa_request(state.request_array)
    paths = ef.get_paths(params, nodes)
    mods = np.asarray(params.modulations_array)
    mask = np.zeros((params.k_paths, params.link_resources), dtype=np.float32)
    mod_mask = np.full(mask.shape, -1, dtype=np.float32)
    best_se = np.full(mask.shape, -np.inf)
    snr_fn = (
        ef.get_snr_link_array_fused
        if params.uniform_spans and not params.mod_format_correction
        else ef.get_snr_link_array
    )
    for path_idx, path in enumerate(paths):
        if not np.any(path):
            continue
        occupied = np.any(
            np.asarray(state.link_slot_array)[np.asarray(path, dtype=bool)] != 0, axis=0
        )
        for mod_idx, mod in enumerate(mods):
            width = int(ef.required_slots(datarate, mod[1], params.slot_size, params.guardband))
            starts = [
                i
                for i in range(params.link_resources - width + 1)
                if not np.any(occupied[i : i + width])
            ]
            if not starts:
                continue
            for start in sorted({starts[0], starts[-1]}):
                affected = ef.get_affected_slots_mask(start, width, path, params)
                pid = ef.get_lightpath_index(params, nodes, path_idx)
                power = params.slot_launch_power_array[start]
                temp = state.replace(
                    channel_power_array=ef.set_path_links(
                        state.channel_power_array.copy(), affected, power
                    ),
                    channel_centre_bw_array=ef.set_path_links(
                        state.channel_centre_bw_array.copy(), affected, params.slot_size
                    ),
                    channel_centre_freq_array=ef.set_path_links(
                        state.channel_centre_freq_array.copy(),
                        affected,
                        ef.get_centre_frequency(start, width, params),
                    ),
                    path_index_array=ef.set_path_links(
                        state.path_index_array.copy(), affected, pid
                    ),
                    modulation_format_index_array=ef.set_path_links(
                        state.modulation_format_index_array.copy(),
                        affected,
                        jnp.asarray(mod_idx, dtype=state.modulation_format_index_array.dtype),
                    ),
                )
                temp = temp.replace(link_snr_array=snr_fn(temp, params))
                snr = ef.get_snr_for_path(path, temp.link_snr_array, params, temp)
                new_ok = (
                    float(jnp.minimum(snr[start], snr[start + width - 1]))
                    >= mod[2] + params.snr_margin
                )
                existing_ok = not bool(ef.check_snr_sufficient(temp, params))
                powers = ef.compute_total_power_per_link(
                    temp.channel_power_array, temp.path_index_array, temp.channel_centre_freq_array
                )
                if (
                    new_ok
                    and existing_ok
                    and bool(jnp.all(powers <= params.max_power_per_fibre))
                    and mod[1] > best_se[path_idx, start]
                ):
                    best_se[path_idx, start] = mod[1]
                    mod_mask[path_idx, start] = mod_idx
                    mask[path_idx, start] = 1
    return mask.reshape(-1), mod_mask.reshape(-1)


@pytest.mark.parametrize("mode", ["uniform", "corrected", "nonuniform"])
def test_selected_link_snr_matches_full_refresh(mode):
    _, params, state = setup(mode)
    rng = np.random.default_rng(0)
    power = jnp.asarray(rng.uniform(1e-4, 8e-4, state.channel_power_array.shape), dtype=jnp.float32)
    freqs = jnp.broadcast_to(params.slot_centre_freq_array, power.shape)
    state = state.replace(
        channel_power_array=power,
        channel_centre_bw_array=jnp.full_like(power, 25),
        channel_centre_freq_array=freqs,
        modulation_format_index_array=jnp.zeros_like(state.modulation_format_index_array),
    )
    indices = jnp.array([params.num_links - 1, 0, 2, 0])
    funcs = [ef.get_snr_link_array]
    if mode == "uniform":
        funcs.append(ef.get_snr_link_array_fused)
    for fn in funcs:
        np.testing.assert_allclose(
            fn(state, params, indices), fn(state, params)[indices], rtol=1e-6
        )


@pytest.mark.parametrize("mode", ["uniform", "corrected", "nonuniform"])
@pytest.mark.parametrize("slots", [12, 140])
def test_streamed_mask_matches_full_network_oracle_with_stale_snr(mode, slots):
    env, params, state = setup(mode, slots=slots)
    # Empty network cache is intentionally wrong: candidates must refresh the
    # baseline rather than trust cached SNR left by departure/removal operations.
    state = state.replace(link_snr_array=jnp.zeros_like(state.link_snr_array))
    for iteration in range(2):
        expected_mask, expected_mods = exhaustive_mask(state, params)
        mask, _, mods = env.action_mask(state, params)
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(mods, expected_mods)
        state = state.replace(link_slot_mask=mask, mod_format_mask=mods)
        action = jnp.array([jnp.argmax(mask), 0], dtype=jnp.int32)
        _, state, *_ = env.step(jax.random.PRNGKey(iteration), state, action, params)
        state = state.replace(link_snr_array=jnp.zeros_like(state.link_snr_array))


@pytest.mark.parametrize("packed", [False, True])
def test_sparse_paths_match_per_path_reference(packed):
    _, params, state = setup(packed=packed)
    paths = params.path_link_array
    if packed:
        paths = jnp.unpackbits(paths, axis=1)[:, : params.num_links]
    rng = np.random.default_rng(5)
    snr = rng.uniform(1, 1000, state.link_snr_array.shape).astype(np.float32)
    snr[0, 0] = 0  # On-path inf and off-path zeros must retain select semantics.
    state = state.replace(
        link_snr_array=jnp.asarray(snr),
        path_index_array=jnp.asarray(
            rng.integers(-1, len(paths), state.path_index_array.shape),
            dtype=state.path_index_array.dtype,
        ),
    )
    reference = jax.vmap(lambda p: ef.get_snr_for_path(p, state.link_snr_array, params, state))(
        paths
    )
    expected = reference[state.path_index_array, jnp.arange(params.link_resources)[None, :]]
    np.testing.assert_allclose(ef.get_lightpath_snr(state, params), expected, atol=1e-4)


def test_single_slot_sparse_snr_keeps_shape():
    _, params, state = setup()
    params = params.replace(link_resources=1)
    names = [
        "link_snr_array",
        "path_index_array",
        "channel_power_array",
        "channel_centre_bw_array",
        "channel_centre_freq_array",
    ]
    state = state.replace(**{n: getattr(state, n)[:, :1] for n in names})
    for name in [
        "transceiver_snr",
        "roadm_express_loss",
        "roadm_add_drop_loss",
        "roadm_noise_figure",
    ]:
        params = params.replace(**{name: getattr(params, name)[:1]})
    result = ef.get_lightpath_snr(state, params)
    assert result.shape == (params.num_links, 1)
    assert bool(jnp.all(jnp.isfinite(result)))


@pytest.mark.parametrize("include_mod, expected_count", [(False, 4), (True, 5)])
def test_candidate_groups_preserve_physics_keys(include_mod, expected_count):
    representatives, packed, count = ef._gn_candidate_groups(
        jnp.array([0, 0, 0, 0, 1, 1]),
        jnp.array([0, 0, 0, 1, 0, 0]),
        jnp.array([2, 2, 2, 2, 2, 2]),
        jnp.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0]),
        jnp.array([0, 1, 0, 0, 0, 0]),
        jnp.array([True, True, True, True, True, False]),
        include_mod,
    )
    assert int(count) == expected_count
    assert int(representatives[1]) == (1 if include_mod else 0)
    assert int(representatives[-1]) == 4
    np.testing.assert_array_equal(packed[:count], [0, 1, 2, 3, 4] if include_mod else [0, 2, 3, 4])


def test_reused_footprint_keeps_distinct_modulation_thresholds():
    env, params, state = setup(slots=140)
    mods = params.modulations_array.at[:, 2].set(1000).at[1, 2].set(-1000)
    params = params.replace(modulations_array=mods)
    expected_mask, expected_mods = exhaustive_mask(state, params)
    mask, _, selected = env.action_mask(state, params)
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(selected, expected_mods)
    assert np.any(np.asarray(mask))
    assert np.all(np.asarray(selected)[np.asarray(mask, dtype=bool)] == 1)


def test_batched_compaction_handles_empty_and_live_candidate_sets():
    env, params, state = setup(slots=140)
    full = state.replace(link_slot_array=jnp.full_like(state.link_slot_array, -1))
    batched = jax.tree.map(lambda x, y: jnp.stack([x, y]), state, full)
    masks, _, mods = jax.jit(jax.vmap(lambda s: env.action_mask(s, params)))(batched)
    expected_mask, _, expected_mods = env.action_mask(state, params)
    np.testing.assert_array_equal(masks[0], expected_mask)
    np.testing.assert_array_equal(mods[0], expected_mods)
    assert np.any(np.asarray(masks[0]))
    assert not np.any(np.asarray(masks[1]))
    assert np.all(np.asarray(mods[1]) == -1)


def test_compact_xpm_full_snr_matches_dense_with_different_live_counts():
    _, params, _ = setup(slots=140)
    rng = np.random.default_rng(13)
    power = rng.uniform(1e-5, 3e-4, (3, 140)).astype(np.float32)
    live = rng.random(power.shape) < 0.35
    live[0] = False
    live[2] = True
    power *= live
    bandwidth = rng.choice([12.5e9, 25e9, 50e9], power.shape).astype(np.float32) * live
    frequency = np.broadcast_to(np.linspace(-8e12, 8e12, 140, dtype=np.float32), power.shape).copy()
    frequency[:, 2] = frequency[:, 1]  # Same-centre terms remain excluded.
    physics = {
        name: getattr(params, name)
        for name in (
            "ref_lambda",
            "attenuation",
            "attenuation_bar",
            "nonlinear_coeff",
            "raman_gain_slope",
            "dispersion_coeff",
            "dispersion_slope",
            "coherent",
            "num_subchannels",
            "span_lumped_loss_db",
        )
    }

    def evaluate(p, f, bw, compact):
        return isrs_gn_model.get_snr_fused(
            p,
            f,
            bw,
            num_spans=3,
            span_length=80_000,
            num_channels=140,
            amplifier_noise_figure=params.amplifier_noise_figure,
            transceiver_snr=params.transceiver_snr,
            compact_xpm=compact,
            **physics,
        )

    args = tuple(jnp.asarray(x) for x in (power, frequency, bandwidth))
    expected = jax.jit(jax.vmap(lambda *x: evaluate(*x, False)))(*args)
    actual = jax.jit(jax.vmap(lambda *x: evaluate(*x, True)))(*args)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert np.all(np.asarray(actual)[~live] == -1e5)
