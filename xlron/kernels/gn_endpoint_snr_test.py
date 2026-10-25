"""Fused screening retains exact-model interference, SPM and ISRS ASE."""

from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments import env_funcs
from xlron.environments.dataclasses import HashableArrayWrapper
from tests.helpers.gn_model import setup
from xlron.kernels import gn_endpoint_snr, gn_xpm_legacy


@pytest.fixture(autouse=True)
def isolate_mask_backend_overrides():
    for fn in (env_funcs.mask_slots_rmsa_gn_model, env_funcs._mask_slots_rmsa_gn_model):
        cast(Any, fn).clear_cache()
    yield
    for fn in (env_funcs.mask_slots_rmsa_gn_model, env_funcs._mask_slots_rmsa_gn_model):
        cast(Any, fn).clear_cache()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("slots", [8, 1668])
@pytest.mark.parametrize("coherent,subchannels,loss", [(False, 1, None), (True, 3, 1.7)])
def test_fused_candidate_optical_snr(native, coherent, subchannels, loss, slots):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    _, _, p, s = setup("rmsa_gn_model", "lifetime_snr")
    p = p.replace(
        link_resources=slots,
        coherent=coherent,
        num_subchannels=subchannels,
        span_lumped_loss_db=loss,
        amplifier_noise_figure=HashableArrayWrapper(jnp.linspace(4, 7, slots)),
        transceiver_snr=HashableArrayWrapper(jnp.full((slots,), 30.0)),
    )
    f = jnp.linspace(-1000 if slots == 8 else -10000, 1000 if slots == 8 else 10000, slots)
    power = jnp.resize(jnp.array([1e-4, 2e-4, 0, 1e-4, 3e-4, 0, 1e-4, 2e-4]), (slots,))
    shape = (p.num_links, slots)
    s = s.replace(
        channel_power_array=jnp.broadcast_to(power, shape),
        channel_centre_freq_array=jnp.broadcast_to(f, shape),
        channel_centre_bw_array=jnp.full(shape, 25.0),
    )
    links = jnp.array([3, 1, 3, p.num_links])
    for start, end, centre in [(0, 0, -975.0), (2, 4, 120.0), (slots - 2, slots - 1, 990.0)]:
        columns = jnp.arange(p.link_resources)
        inside = (columns >= start) & (columns <= end)
        rows = (
            jnp.where(inside, 1.5e-4, s.channel_power_array[links]),
            jnp.where(inside, centre, s.channel_centre_freq_array[links]),
            jnp.where(inside, p.slot_size, s.channel_centre_bw_array[links]),
        )
        expected = env_funcs.get_snr_link_array(
            s, p, links, output_indices=jnp.array([start, end]), channel_rows=rows
        )
        actual = jax.jit(
            lambda s, start=start, end=end, centre=centre: gn_endpoint_snr.endpoint_snr_pallas(
                s,
                p,
                links,
                jnp.int32(start),
                jnp.int32(end),
                jnp.float32(1.5e-4),
                jnp.float32(centre),
                interpret=not native,
            )
        )(s)
        np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("screen", ["fused", "selected"])
def test_endpoint_full_mask_and_admission(native, screen, monkeypatch):
    if native and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    if not native:
        monkeypatch.setattr(
            gn_endpoint_snr,
            "endpoint_snr_pallas",
            partial(gn_endpoint_snr.endpoint_snr_pallas, interpret=True),
        )
        monkeypatch.setattr(
            gn_xpm_legacy,
            "xpm_selected_pallas",
            partial(gn_xpm_legacy.xpm_selected_pallas, interpret=True),
        )
    if screen == "selected":
        monkeypatch.setattr(
            env_funcs,
            "get_snr_link_array",
            partial(env_funcs.get_snr_link_array, output_xpm_backend="pallas"),
        )
    _, env, p, s = setup("rmsa_gn_model", "lifetime_snr")
    p = p.replace(gn_mask_mode="all_slots", gn_mask_batch_size=3)
    for seed in (81, 82):
        expected = env_funcs.mask_slots_rmsa_gn_model(s, p, s.request_array, prune_formats=False)
        actual = env_funcs.mask_slots_rmsa_gn_model(
            s,
            p,
            s.request_array,
            prefilter_new=True,
            fused_endpoint_snr=screen == "fused",
            reuse_screened_power=True,
            narrow_new_snr=True,
            reuse_unaffected_snr=True,
            screen_batch_size=8,
        )
        np.testing.assert_array_equal(actual.link_slot_mask, expected.link_slot_mask)
        np.testing.assert_array_equal(actual.mod_format_mask, expected.mod_format_mask)
        assert actual.link_slot_mask.any()
        before = int(s.accepted_services)
        _, s, reward, *_ = env.step(
            jax.random.PRNGKey(seed), actual, jnp.argmax(actual.link_slot_mask), p
        )
        assert reward >= 0 and int(s.accepted_services) == before + 1
