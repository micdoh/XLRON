"""Unit tests for the off-policy IAM recentered PPO clip (IAM_RECENTER_CLIP).

These check the ratio identity that the recentering in ``_loss_fn`` relies on, using the
same distrax masking/log-prob ops as ``select_action_batched`` (rollout-time storage) and
``_loss_fn`` (loss-time ratio). At no update (theta_new == theta_old):

* unit mode:       ratio == valid_mass (mu_old) for every valid action (not 1)
* recentered mode: ratio == 1 for every valid action (log ratio == 0)
"""

import chex
import distrax
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

LOGR_CLIP = 10.0


def _make_state(key, n_actions, n_valid):
    """Random unmasked logits plus a mask with exactly n_valid valid actions."""
    klog, kmask = jax.random.split(key)
    logits = jax.random.normal(klog, (n_actions,))
    idx = jax.random.permutation(kmask, n_actions)[:n_valid]
    mask = jnp.zeros((n_actions,)).at[idx].set(1.0)
    return logits, mask


def _behaviour(logits, mask):
    """Rollout-time storage, mirrors select_action_batched (masked sample policy)."""
    pi_masked = distrax.Categorical(logits=logits + (-1e8 * (1 - mask)))
    valid_mass = jnp.sum(jax.nn.softmax(logits, axis=-1) * mask, axis=-1)
    return pi_masked, valid_mass


def _ratio(logits, action, behaviour_log_prob, valid_mass, recenter_clip):
    """Loss-time ratio, mirrors _loss_fn (unmasked new policy / masked behaviour)."""
    log_prob = distrax.Categorical(logits=logits).log_prob(action)  # OFF_POLICY_IAM: unmasked
    log_ratio = log_prob - behaviour_log_prob
    if recenter_clip:
        log_ratio = log_ratio - jnp.log(valid_mass + 1e-8)
    log_ratio = jnp.clip(log_ratio, -LOGR_CLIP, LOGR_CLIP)
    return jnp.exp(log_ratio)


class RecenterClipTest(chex.TestCase):
    @parameterized.named_parameters(
        ("dense", 16, 12, 0),
        ("medium", 32, 8, 1),
        ("sparse", 64, 3, 2),
        ("single_valid", 16, 1, 3),
    )
    def test_ratio_identities_at_no_update(self, n_actions, n_valid, seed):
        logits, mask = _make_state(jax.random.PRNGKey(seed), n_actions, n_valid)
        pi_masked, valid_mass = _behaviour(logits, mask)
        for a in jnp.where(mask > 0)[0]:
            behaviour_log_prob = pi_masked.log_prob(a)
            unit = _ratio(logits, a, behaviour_log_prob, valid_mass, recenter_clip=False)
            recentered = _ratio(logits, a, behaviour_log_prob, valid_mass, recenter_clip=True)
            # Unit mode: ratio is the (state-constant) valid mass mu_old, the same for every
            # valid action, not 1.
            chex.assert_trees_all_close(unit, valid_mass, atol=1e-4)
            # Recentered mode: ratio is exactly 1.
            chex.assert_trees_all_close(recentered, jnp.array(1.0), atol=1e-4)

    def test_negative_advantage_clipping(self):
        # CLIP_EPS as tight as the transformer runs use; mu_old (~0.5) < 1 - eps.
        clip_eps = 0.04
        logits, mask = _make_state(jax.random.PRNGKey(7), 32, 6)
        pi_masked, valid_mass = _behaviour(logits, mask)
        a = jnp.where(mask > 0)[0][0]
        behaviour_log_prob = pi_masked.log_prob(a)
        adv = jnp.array(-1.0)  # negative advantage: a bad action that should be demoted

        def actor_clipped(ratio):
            la1 = ratio * adv
            la2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            return bool(la2 < la1)  # min picks the clipped (flat) branch -> no gradient

        unit = _ratio(logits, a, behaviour_log_prob, valid_mass, recenter_clip=False)
        recentered = _ratio(logits, a, behaviour_log_prob, valid_mass, recenter_clip=True)
        self.assertLess(float(valid_mass), 1.0 - clip_eps)  # precondition for the artefact
        self.assertTrue(actor_clipped(unit))  # unit mode: negative-advantage gradient killed
        self.assertFalse(actor_clipped(recentered))  # recentered: gradient flows


if __name__ == "__main__":
    absltest.main()
