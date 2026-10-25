"""Compact observations must retain status, time and full-resolution action detail."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments.env_funcs import get_obs_transformer
from xlron.environments.transformer_snr_test import setup
from xlron.models.spectrum_transformer import band_convolution, ordered_spectrum_summary
from xlron.train.train_utils import _make_param_label_fn, init_network


@pytest.fixture(autouse=True)
def strict_float32_matmul():
    # Validate the contraction/gradient rewrite in FP32. A100's default GEMM
    # policy can select TF32 differently for the fused and offset shapes.
    with jax.default_matmul_precision("highest"):
        yield


def test_compact_lifetime_status_and_unsaturated_snr():
    _, _, p, s = setup("rmsa_gn_model", "lifetime_snr")
    s = s.replace(
        link_slot_array=s.link_slot_array.at[0, :5].set(jnp.array([-1, 0, 1, 1, 1])),
        link_slot_departure_array=s.link_slot_departure_array.at[0, :5].set(
            jnp.array([0, 0, 0, 1, 3]) * s.mean_service_holding_time
        ),
        channel_power_array=s.channel_power_array.at[0, 2:5].set(1e-4),
        link_snr_array=s.link_snr_array.at[0, 2:5].set(jnp.array([0.1, 1e3, 1e4])),
    )
    x = get_obs_transformer(s, p)
    i = p.line_graph_spectral_features.val.shape[1]
    np.testing.assert_allclose(x[0, i : i + 5], [-1, 0, 1, 1 + np.log(2), 1 + np.log(4)], rtol=1e-6)
    q = x[0, i + p.link_resources : i + p.link_resources + 5]
    assert q[2] < 0 and q[4] > q[3] > 0
    assert np.isfinite(x).all()
    absolute = s.replace(
        current_time=jnp.array(10.0, s.current_time.dtype),
        link_slot_departure_array=s.link_slot_departure_array + 10,
    )
    np.testing.assert_allclose(
        get_obs_transformer(absolute, p.replace(relative_arrival_times=False)), x
    )


def test_local_convolution_cannot_cross_band_gap():
    conv = eqx.nn.Conv1d(1, 1, 5, use_bias=False, key=jax.random.PRNGKey(0))
    conv = eqx.tree_at(lambda m: m.weight, conv, jnp.ones_like(conv.weight))
    status = jnp.array([0.0, 0.0, -1.0, 0.0, 0.0])
    x = jnp.array([[0.0], [5.0], [100.0], [0.0], [0.0]])
    result = band_convolution(conv, x, status)
    assert result[3, 0] == 0 and result[2, 0] == 0 and result[0, 0] == 5
    assert np.isfinite(ordered_spectrum_summary(result)).all()


def test_fused_convolution_value_and_gradient_match_offset_reference():
    conv = eqx.nn.Conv1d(3, 8, 5, key=jax.random.PRNGKey(11))
    x = jax.random.normal(jax.random.PRNGKey(12), (19, 3))
    status = jnp.zeros(19).at[7:9].set(-1)

    def reference(m):
        segment = jnp.cumsum(status < 0)
        pad = jnp.pad(x, ((2, 2), (0, 0)))
        ids = jnp.pad(segment, (2, 2), constant_values=-1)
        usable = jnp.pad(status >= 0, (2, 2))
        out = jnp.zeros((19, 8))
        for o in range(5):
            mask = (ids[o : o + 19] == segment) & usable[o : o + 19]
            out = out + (pad[o : o + 19] * mask[:, None]) @ m.weight[:, :, o].T
        return jnp.where((status >= 0)[:, None], out + m.bias[:, 0], 0)

    np.testing.assert_allclose(
        band_convolution(conv, x, status), reference(conv), rtol=2e-5, atol=1e-6
    )
    g1 = eqx.filter_grad(lambda m: jnp.sum(band_convolution(m, x, status) ** 2))(conv)
    g2 = eqx.filter_grad(lambda m: jnp.sum(reference(m) ** 2))(conv)
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("encoder,width", [("dense", 8), ("conv", 8), ("conv", 16)])
def test_full_resolution_forward_gradient_checkpoint_and_optimizer_labels(tmp_path, encoder, width):
    c, _, p, s = setup("rmsa_gn_model", "lifetime_snr")
    c.transformer_spectrum_encoder = encoder
    c.transformer_embedding_size = width
    c.transformer_actor_pooling = "sum"
    model = init_network(c, jax.random.PRNGKey(1))

    def forward(m):
        pi, value = m(s, p)
        return pi.logits, value

    logits, value = eqx.filter_jit(forward)(model)
    assert np.isfinite(value)
    assert logits.shape == (p.k_paths * p.link_resources,)
    assert c.aggregate_slots == 1

    @eqx.filter_value_and_grad
    def loss(m):
        policy, v = m(s, p)
        return -policy.log_prob(jnp.array(3)) + (v - 1.0) ** 2

    val, grad = loss(model)
    assert np.isfinite(val) and all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
    labels = _make_param_label_fn(model)(eqx.filter(model, eqx.is_inexact_array))
    for path, label in jax.tree_util.tree_flatten_with_path(labels)[0]:
        key = jax.tree_util.keystr(path)
        if ".critic" in key or ".actor_critic[1]" in key:
            assert label == "critic"
        else:
            assert label == "actor"
    path = tmp_path / "model.eqx"
    eqx.tree_serialise_leaves(path, model)
    restored = eqx.tree_deserialise_leaves(path, model)
    np.testing.assert_array_equal(restored(s, p)[0].logits, model(s, p)[0].logits)
    if encoder == "conv":
        # Retain different adjacent-slot scores even though summary features pool.
        s2 = s.replace(
            link_slot_array=s.link_slot_array.at[:, 3].set(1),
            link_slot_departure_array=s.link_slot_departure_array.at[:, 3].set(25),
        )
        scores = model(s2, p)[0].logits.reshape(p.k_paths, p.link_resources)
        assert not np.allclose(scores[:, 3], scores[:, 4])
