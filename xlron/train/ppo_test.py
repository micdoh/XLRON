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


class BatchedMaskedPolicyTest(chex.TestCase):
    """Regression for the batched masked-policy construction in ``_loss_fn``.

    The models return a bare ``distrax.Categorical``; after vmap its logits are (B, A).
    ``_loss_fn`` must build the masked policy from ``pi._logits`` (per-sample rows).
    ``pi[0]._logits`` is distrax batch-indexing — sample 0's logits broadcast against every
    sample's mask — which silently corrupts entropy (off-policy IAM) and log_prob (on-policy).
    """

    def test_masked_log_prob_uses_per_sample_logits(self):
        key = jax.random.PRNGKey(0)
        b, a = 4, 8
        logits = jax.random.normal(key, (b, a))
        masks = jnp.stack([_make_state(jax.random.PRNGKey(s), a, 3)[1] for s in range(b)])
        actions = jnp.array([jnp.argmax(m) for m in masks])

        pi = distrax.Categorical(logits=logits)  # batched bare dist, as returned by vmap(model)
        # _loss_fn construction
        pi_masked = distrax.Categorical(logits=pi._logits + (-1e8 * (1 - masks)))
        # per-sample reference
        expected = jnp.stack(
            [
                distrax.Categorical(logits=logits[i] + (-1e8 * (1 - masks[i]))).log_prob(actions[i])
                for i in range(b)
            ]
        )
        chex.assert_trees_all_close(pi_masked.log_prob(actions), expected, atol=1e-5)
        chex.assert_tree_all_finite(pi_masked.entropy())

        # The buggy construction (sample 0's logits for every row) must NOT match.
        pi_masked_buggy = distrax.Categorical(logits=pi[0]._logits + (-1e8 * (1 - masks)))
        self.assertGreater(float(jnp.abs(pi_masked_buggy.log_prob(actions) - expected).max()), 1e-3)


class ActorGradientEquivalenceTest(chex.TestCase):
    """The emergent non-recentered clip and the explicit REC+PO+MU stack are the same estimator.

    At no-update (theta == theta_old, one epoch / one minibatch) with valid_mass < 1 - eps:

    * emergent:  ratio = mu_old; Adv>0 flows unclipped (grad mu * Adv * dlogpi), Adv<0 clipped flat
    * explicit:  recentered ratio = 1; POSITIVE_ADV_ONLY zeroes Adv<0; MU_WEIGHT_ACTOR restores mu

    With the actor loss normalized by w_sum in BOTH modes (not the positive-only count), the
    actor gradients must be identical.
    """

    def test_gradients_match_at_no_update(self):
        b, a = 6, 12
        clip_eps_emergent, clip_eps_explicit = 0.04, 0.2
        key = jax.random.PRNGKey(3)
        theta_old = jax.random.normal(key, (b, a))
        masks = jnp.stack([_make_state(jax.random.PRNGKey(s + 10), a, 4)[1] for s in range(b)])
        adv = jnp.array([1.3, -0.7, 0.2, -1.5, 0.9, -0.1])
        w = jnp.ones((b,))
        w_sum = w.sum()

        # Behaviour policy (stopped gradient): masked sample, stored log_prob and valid mass
        pi_m_old = distrax.Categorical(logits=theta_old + (-1e8 * (1 - masks)))
        actions = pi_m_old.sample(seed=jax.random.PRNGKey(1))
        behaviour_log_prob = pi_m_old.log_prob(actions)
        mu_old = jnp.sum(jax.nn.softmax(theta_old, axis=-1) * masks, axis=-1)
        assert bool((mu_old < 1.0 - clip_eps_emergent).all()), "precondition"

        def surrogate(ratio, eps):
            la1 = ratio * adv
            la2 = jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
            return jnp.minimum(la1, la2)

        def emergent_loss(theta):
            log_prob = distrax.Categorical(logits=theta).log_prob(actions)  # unmasked
            ratio = jnp.exp(log_prob - behaviour_log_prob)
            return -(surrogate(ratio, clip_eps_emergent) * w).sum() / w_sum

        def explicit_loss(theta):
            log_prob = distrax.Categorical(logits=theta).log_prob(actions)
            log_ratio = log_prob - behaviour_log_prob - jnp.log(mu_old + 1e-8)  # recentred
            ratio = jnp.exp(log_ratio)
            actor_w = w * (adv > 0)  # POSITIVE_ADV_ONLY
            actor_num = actor_w * mu_old  # MU_WEIGHT_ACTOR
            return -(surrogate(ratio, clip_eps_explicit) * actor_num).sum() / w_sum

        g_emergent = jax.grad(emergent_loss)(theta_old)
        g_explicit = jax.grad(explicit_loss)(theta_old)
        chex.assert_trees_all_close(g_emergent, g_explicit, atol=1e-5)

    def test_mu_gated_po_completes_identity_for_mixed_lobes(self):
        """With PO_MU_GATE, the identity extends to states at mu >= 1 - eps.

        The emergent clip only blocks negative advantages where the ratio (= mu) sits below
        the clip floor; in the uncongested mu ~ 1 lobe the ratio is inside the band and BOTH
        advantage signs flow (two-sided PPO, with the implicit mu-scaling ~ 1). Gating the
        explicit filter by mu < 1 - eps therefore makes the explicit REC+PO+MU gradient
        identical to the emergent one for every sample — including negative advantages in
        high-mu states, which plain POSITIVE_ADV_ONLY (previous test's precondition) filters.
        """
        b, a = 8, 12
        clip_eps_emergent, clip_eps_explicit = 0.04, 0.2
        key = jax.random.PRNGKey(5)
        theta_old = jax.random.normal(key, (b, a))
        masks = jnp.stack([_make_state(jax.random.PRNGKey(s + 20), a, 4)[1] for s in range(b)])
        # Push the last 4 samples into the high-mu lobe: bury their invalid logits so the
        # unmasked policy places ~all mass on valid actions (mu ~ 1)
        high = jnp.arange(b) >= b // 2
        theta_old = jnp.where(high[:, None] * (1 - masks) > 0, theta_old - 30.0, theta_old)
        # Negative advantages present in BOTH lobes (the high-lobe negatives are the case
        # plain POSITIVE_ADV_ONLY gets wrong)
        adv = jnp.array([1.3, -0.7, 0.2, -1.5, 0.9, -0.6, -1.1, 0.4])
        w = jnp.ones((b,))
        w_sum = w.sum()

        pi_m_old = distrax.Categorical(logits=theta_old + (-1e8 * (1 - masks)))
        actions = pi_m_old.sample(seed=jax.random.PRNGKey(2))
        behaviour_log_prob = pi_m_old.log_prob(actions)
        mu_old = jnp.sum(jax.nn.softmax(theta_old, axis=-1) * masks, axis=-1)
        # Mixed-lobe precondition: both lobes populated
        assert bool((mu_old[~high] < 1.0 - clip_eps_emergent).all()), "low lobe precondition"
        assert bool((mu_old[high] >= 1.0 - clip_eps_emergent).all()), "high lobe precondition"

        def surrogate(ratio, eps):
            la1 = ratio * adv
            la2 = jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
            return jnp.minimum(la1, la2)

        def emergent_loss(theta):
            log_prob = distrax.Categorical(logits=theta).log_prob(actions)  # unmasked
            ratio = jnp.exp(log_prob - behaviour_log_prob)
            return -(surrogate(ratio, clip_eps_emergent) * w).sum() / w_sum

        def explicit_gated_loss(theta):
            log_prob = distrax.Categorical(logits=theta).log_prob(actions)
            log_ratio = log_prob - behaviour_log_prob - jnp.log(mu_old + 1e-8)  # recentred
            ratio = jnp.exp(log_ratio)
            keep = (adv > 0) | (mu_old >= 1.0 - clip_eps_emergent)  # PO_MU_GATE rule
            actor_num = w * keep * mu_old  # + MU_WEIGHT_ACTOR
            return -(surrogate(ratio, clip_eps_explicit) * actor_num).sum() / w_sum

        g_emergent = jax.grad(emergent_loss)(theta_old)
        g_explicit = jax.grad(explicit_gated_loss)(theta_old)
        chex.assert_trees_all_close(g_emergent, g_explicit, atol=1e-5)

        # Sanity: WITHOUT the gate the identity must break on high-lobe negatives
        def explicit_ungated_loss(theta):
            log_prob = distrax.Categorical(logits=theta).log_prob(actions)
            ratio = jnp.exp(log_prob - behaviour_log_prob - jnp.log(mu_old + 1e-8))
            actor_num = w * (adv > 0) * mu_old
            return -(surrogate(ratio, clip_eps_explicit) * actor_num).sum() / w_sum

        g_ungated = jax.grad(explicit_ungated_loss)(theta_old)
        self.assertGreater(float(jnp.abs(g_emergent - g_ungated).max()), 1e-3)


if __name__ == "__main__":
    absltest.main()
