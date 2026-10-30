"""The fused endpoint screen computes the generic GN model at the inserted channel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.gn_model import setup
from xlron.environments import env_funcs
from xlron.kernels import gn_endpoint_snr


def _signal(params, start, count):
    columns = jnp.arange(params.link_resources)
    return (columns >= start) & (columns < start + count)


def _clear(state, params, start, count):
    """The state with the candidate's slots free on every link."""
    inside = _signal(params, start, count)
    return state.replace(
        channel_power_array=jnp.where(inside, 0.0, state.channel_power_array),
        channel_centre_freq_array=jnp.where(inside, 0.0, state.channel_centre_freq_array),
        channel_centre_bw_array=jnp.where(inside, 0.0, state.channel_centre_bw_array),
    )


def _insert(state, params, links, start, count, power, centre):
    """Full-network state with a channel of count signal slots on the given links."""
    inside = _signal(params, start, count)
    rows = links[links < params.num_links]

    def put(array, value):
        return array.at[rows].set(jnp.where(inside, value, array[rows]))

    return state.replace(
        channel_power_array=put(state.channel_power_array, power),
        channel_centre_freq_array=put(state.channel_centre_freq_array, centre),
        channel_centre_bw_array=put(state.channel_centre_bw_array, count * params.slot_size),
    )


@pytest.mark.parametrize("slots", [8, 1668])
@pytest.mark.parametrize("coherent,subchannels,loss", [(False, 1, None), (True, 3, 1.7)])
def test_endpoint_snr_matches_generic_model(coherent, subchannels, loss, slots):
    _, _, p, s = setup("rmsa_gn_model", "lifetime_snr")
    p = p.replace(
        link_resources=slots,
        coherent=coherent,
        num_subchannels=subchannels,
        span_lumped_loss_db=loss,
        amplifier_noise_figure=jnp.linspace(4, 7, slots),
        transceiver_snr=jnp.full((slots,), 30.0),
        slot_centre_freq_array=(jnp.arange(slots) - (slots - 1) / 2) * p.slot_size,
    )
    span = 1000 if slots == 8 else 10000
    f = jnp.linspace(-span, span, slots)
    # Mixed powers exercise the channel/interferer power ratio (issue #66).
    power = jnp.resize(jnp.array([1e-4, 2e-4, 0, 1e-4, 3e-4, 0, 1e-4, 2e-4]), (slots,))
    shape = (p.num_links, slots)
    s = s.replace(
        channel_power_array=jnp.broadcast_to(power, shape),
        channel_centre_freq_array=jnp.broadcast_to(f, shape),
        channel_centre_bw_array=jnp.full(shape, 25.0),
    )
    links = jnp.array([3, 1, p.num_links])  # The padded entry returns ones.
    for start, count in [(0, 1), (2, 3), (slots - 2, 2)]:
        # Channels are centred on their signal slots of the slot grid.
        centre = float(env_funcs.get_centre_frequency(start, count, p))
        free = _clear(s, p, start, count)
        inserted = _insert(free, p, links, start, count, 1.5e-4, centre)
        full = env_funcs.get_snr_link_array(inserted, p, links[:-1])
        expected = full[:, start]
        # The channel's SNR is broadcast to all its signal slots.
        np.testing.assert_array_equal(
            full[:, start : start + count], jnp.broadcast_to(expected[:, None], (2, count))
        )
        actual = jax.jit(
            lambda s, start=start, count=count, centre=centre: gn_endpoint_snr.endpoint_snr_pallas(
                s,
                p,
                links,
                jnp.int32(start),
                jnp.float32(1.5e-4),
                jnp.float32(centre),
                jnp.float32(count * p.slot_size),
            )
        )(free)
        np.testing.assert_allclose(actual[:-1], expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(actual[-1], 1.0)


def test_pallas_full_mask_matches_reference_and_is_admitted():
    _, env, p, s = setup("rmsa_gn_model", "lifetime_snr")
    p = p.replace(gn_mask_mode="all_slots", gn_mask_batch_size=3, gn_mask_screen_batch_size=8)
    kernel = p.replace(gn_kernel_backend="pallas")
    for seed in (81, 82):
        expected = env_funcs.mask_slots_rmsa_gn_model(s, p, s.request_array)
        actual = env_funcs.mask_slots_rmsa_gn_model(s, kernel, s.request_array)
        np.testing.assert_array_equal(actual.link_slot_mask, expected.link_slot_mask)
        np.testing.assert_array_equal(actual.mod_format_mask, expected.mod_format_mask)
        assert actual.link_slot_mask.any()
        before = int(s.accepted_services)
        _, s, reward, *_ = env.step(
            jax.random.PRNGKey(seed), actual, jnp.argmax(actual.link_slot_mask), kernel
        )
        assert reward >= 0 and int(s.accepted_services) == before + 1


def test_screen_slack_covers_screen_versus_full_solve_difference():
    """Near-threshold candidates: the screen must never reject what the full solve admits."""
    _, _, p, s = setup("rmsa_gn_model", "lifetime_snr")
    p = p.replace(gn_mask_mode="all_slots", gn_kernel_backend="pallas")
    nodes, _ = env_funcs.read_rsa_request(s.request_array)
    path = env_funcs.get_paths(p, nodes)[0]
    links = jnp.nonzero(path, size=p.path_link_indices.shape[1], fill_value=p.num_links)[0]
    worst = 0.0
    for start in range(0, p.link_resources - 1, 3):
        count, width, centre = env_funcs.get_channel_values(start, 2 + p.guardband, p)
        power = p.slot_launch_power_array[start] * count
        sampled = gn_endpoint_snr.endpoint_snr_pallas(s, p, links, start, power, centre, width)
        inserted = _insert(s, p, links, start, int(count), power, centre)
        full = env_funcs.get_snr_link_array(inserted, p)
        screened_db = 10 * np.log10(np.asarray(sampled[links < p.num_links]))[:, None]
        full_db = 10 * np.log10(
            np.asarray(full[links[links < p.num_links]][:, start : start + int(count)])
        )
        worst = max(worst, float(np.max(np.abs(screened_db - full_db))))
    print(f"max |screen - full| = {worst:.2e} dB")
    assert worst < env_funcs.ENDPOINT_SCREEN_SLACK_DB / 10
