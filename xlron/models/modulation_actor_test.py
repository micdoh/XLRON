"""Candidate format/width features must agree during acting and PPO fitting."""

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from tests.helpers.gn_model import setup
from xlron.environments.dataclasses import RMSAGNModelEnvState
from xlron.environments.wrappers import LogWrapper
from xlron.models.modulation_actor import ModulationAwareActorCritic, candidate_modulation_features
from xlron.train.ppo import _env_step
from xlron.train.train_utils import TrainState, init_network


@pytest.mark.parametrize("encoder", ["dense", "conv"])
def test_candidate_features_head_gradients_and_base_initialization(encoder):
    c, env, p, s = setup("rmsa_gn_model", "lifetime_snr")
    c.transformer_spectrum_encoder = encoder
    c.transformer_actor_pooling = "sum"
    base = init_network(c, jax.random.PRNGKey(5))
    c.transformer_modulation_features = True
    network = init_network(c, jax.random.PRNGKey(5))
    p = p.replace(gn_mask_mode="all_slots")
    mask, full, mods = env.action_mask(s, p)
    s = s.replace(link_slot_mask=mask, full_link_slot_mask=full, mod_format_mask=mods)
    features = candidate_modulation_features(s, p)
    assert features.shape == (p.k_paths * p.link_resources, 4)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features[:, 0], mask)
    np.testing.assert_allclose(network(s, p)[0].logits, base(s, p)[0].logits, atol=1e-6)
    action = jnp.argmax(mask)
    grad = eqx.filter_grad(lambda m: -m(s, p)[0].log_prob(action))(network)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
    assert np.any(np.asarray(grad.actor_candidate.layers[-1].weight) != 0)
    # These are candidate-specific features rather than link-token columns.
    alternate = s.replace(mod_format_mask=s.mod_format_mask.at[action].set(0))
    assert not np.array_equal(candidate_modulation_features(alternate, p)[action], features[action])


def test_rollout_stores_fresh_features_and_reproduces_behaviour_log_probability():
    c, raw_env, p, s = setup("rmsa_gn_model", "lifetime_snr")
    c.transformer_modulation_features = True
    c.transformer_actor_pooling = "sum"
    p = p.replace(gn_mask_mode="all_slots")
    model = init_network(c, jax.random.PRNGKey(10))
    assert isinstance(model, ModulationAwareActorCritic)
    # A nonzero head exposes stale features in the behaviour probability.
    model = eqx.tree_at(
        lambda m: m.actor_candidate.layers[-1].weight,
        model,
        jnp.ones_like(model.actor_candidate.layers[-1].weight),
    )
    train = TrainState.create(model, optax.adam(1e-4))
    env = LogWrapper(raw_env)
    _, outer = env.reset(jax.random.PRNGKey(11), p)
    outer = outer.replace(env_state=s)
    key = jax.random.PRNGKey(12)
    runner = (train, outer, (s, p), key, key)
    _, transition = _env_step(runner, None, env, p, c)
    acting, _ = transition.obs
    assert isinstance(acting, RMSAGNModelEnvState)
    assert np.any(np.asarray(acting.mod_format_mask) >= 0)
    policy, _ = model(acting, p)
    masked = distrax.Categorical(
        logits=policy.logits - 1e8 * (1 - transition.action_mask.astype(jnp.float32))
    )
    np.testing.assert_allclose(masked.log_prob(transition.action), transition.log_prob, atol=1e-6)
