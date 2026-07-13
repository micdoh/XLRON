"""Unit tests for the off-policy IAM recentered PPO clip (IAM_RECENTER_CLIP).

These check the ratio identity that the recentering in ``_loss_fn`` relies on, using the
same distrax masking/log-prob ops as ``select_action_batched`` (rollout-time storage) and
``_loss_fn`` (loss-time ratio). At no update (theta_new == theta_old):

* unit mode:       ratio == valid_mass (mu_old) for every valid action (not 1)
* recentered mode: ratio == 1 for every valid action (log ratio == 0)
"""

from types import SimpleNamespace
from typing import NamedTuple

import chex
import distrax
import jax
import jax.numpy as jnp
import optax
from absl.testing import absltest, parameterized
from box import Box

from xlron.environments.dataclasses import RSATransition
from xlron.environments.make_env import make
from xlron.models.mlp import ActorCriticMLP
from xlron.train.ppo import (
    _calculate_puffer_advantage,
    _env_step,
    _loss_fn,
    _recompute_vtrace_advantages,
    _sample_prioritized_batch,
    compute_sample_priority_weights,
    compute_trajectory_priority_weights,
)
from xlron.train.train_utils import TrainState

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


class PrioritizedReplayWeightsTest(chex.TestCase):
    """Regression for the PER importance weights in ``_sample_prioritized_batch``.

    Standard PER (Schaul et al. 2016) uses w_i = (N * P(i))^-beta / max_i w_i. The old code
    computed P(i)^-beta with no N factor and no max-normalization, so near-uniform priorities
    gave weights ~N^beta (e.g. ~150x for N=300k, beta=0.4) instead of ~1, inflating the actor
    surrogate relative to the uniform path (weights exactly 1.0) and to VF_COEF/ENT_COEF.
    """

    def _config(self, **overrides):
        base = dict(
            ROLLOUT_LENGTH=8,
            NUM_ENVS=4,
            NUM_MINIBATCHES=2,
            MINIBATCH_SIZE=16,
            NUM_LEARNERS=1,
            PRIO_ALPHA=0.6,
            PRIO_BETA0=0.4,
            USE_RNN=False,
            RHO_CLIP=0.0,
            C_CLIP=0.0,
        )
        base.update(overrides)
        return Box(base)

    def _batch(self, config):
        n = config.ROLLOUT_LENGTH * config.NUM_ENVS
        arr = jnp.arange(n, dtype=jnp.float32).reshape(config.ROLLOUT_LENGTH, config.NUM_ENVS)
        return (arr, arr + 100.0, arr - 100.0)

    def test_uniform_priorities_give_unit_weights(self):
        config = self._config()
        priorities = jnp.ones((config.ROLLOUT_LENGTH, config.NUM_ENVS))
        _, weights = _sample_prioritized_batch(
            self._batch(config), priorities, jnp.array(0.4), jax.random.PRNGKey(0), config
        )
        # Uniform priorities must match the PRIO_ALPHA=0 path's scale (weights of exactly 1).
        chex.assert_trees_all_close(weights, jnp.ones_like(weights), atol=1e-6)

    def test_nonuniform_priorities_max_normalized(self):
        config = self._config()
        priorities = (
            jnp.arange(1, config.ROLLOUT_LENGTH * config.NUM_ENVS + 1)
            .astype(jnp.float32)
            .reshape(config.ROLLOUT_LENGTH, config.NUM_ENVS)
        )
        _, weights = _sample_prioritized_batch(
            self._batch(config), priorities, jnp.array(0.4), jax.random.PRNGKey(0), config
        )
        self.assertLessEqual(float(weights.max()), 1.0 + 1e-6)
        # The rarest (lowest-priority) samples carry the largest correction weight = 1.
        self.assertGreater(float(weights.max()), float(weights.min()))

    def test_trajectory_branch_uniform_priorities_give_unit_weights(self):
        config = self._config(USE_RNN=True, RHO_CLIP=1.0, C_CLIP=1.0)
        priorities = jnp.ones((config.NUM_ENVS,))
        _, weights = _sample_prioritized_batch(
            self._batch(config), priorities, jnp.array(0.4), jax.random.PRNGKey(0), config
        )
        chex.assert_trees_all_close(weights, jnp.ones_like(weights), atol=1e-6)


class VTracePrioritizedReplayTest(chex.TestCase):
    """Regression tests for VTrace + prioritized replay ordering.

    VTrace advantages/targets must be computed over the temporally-ordered rollout with
    the true bootstrap value V(s_{T+1}) BEFORE prioritized resampling / shuffling can
    permute the batch. The old code re-ran the reverse-time scan inside ``_loss_fn`` over
    (potentially resampled) minibatches — chaining randomly resampled transitions as if
    consecutive whenever PRIO_ALPHA > 0 — and bootstrapped each chunk's last row on its
    own value instead of V(s_{T+1}).
    """

    T, E, OBS_DIM, N_ACTIONS = 6, 3, 6, 5

    def _config(self, **overrides):
        base = dict(
            env_type="rwa",
            USE_GNN=False,
            USE_TRANSFORMER=False,
            USE_RNN=False,
            OFF_POLICY_IAM=False,
            LOGR_CLIP=10.0,
            ROLLOUT_LENGTH=self.T,
            NUM_ENVS=self.E,
            NUM_MINIBATCHES=2,
            MINIBATCH_SIZE=self.T * self.E // 2,
            NUM_LEARNERS=1,
            GAMMA=0.99,
            GAE_LAMBDA=0.9,
            REWARD_CENTERING=False,
            RHO_CLIP=1.0,
            C_CLIP=1.0,
            PRIO_ALPHA=0.6,
            PRIO_BETA0=0.4,
            PROFILE=False,
            DEBUG=False,
            DEBUG_LOSS=False,
            ENHANCED_LOGGING=False,
            IAM_GATING=False,
            IAM_DAMPING=False,
            ADV_CLIP=10.0,
            CLIP_EPS=0.2,
            VF_COEF=0.5,
            VALID_MASS_LOSS_COEF=0.0,
        )
        base.update(overrides)
        return Box(base)

    def _rollout(self, seed=0):
        """Synthetic ordered (T, E) rollout whose stored log-probs come from the same
        model, so the recomputed VTrace importance ratio is exactly 1 at no-update."""
        t, e, obs_dim, n_actions = self.T, self.E, self.OBS_DIM, self.N_ACTIONS
        kmodel, kobs, kact, krew, klast = jax.random.split(jax.random.PRNGKey(seed), 5)
        model = ActorCriticMLP(n_actions, obs_dim, num_layers=1, num_units=16, key=kmodel)
        train_state = TrainState.create(model, optax.adam(1e-3))
        obs = jax.random.normal(kobs, (t, e, obs_dim))
        pi, value = jax.vmap(model)(obs.reshape(t * e, obs_dim))
        action = pi.sample(seed=kact)
        # All-ones mask: the masked behaviour policy equals the unmasked policy
        log_prob = pi.log_prob(action)
        traj = RSATransition(
            terminal=jnp.zeros((t, e), dtype=bool),
            truncated=jnp.zeros((t, e), dtype=bool),
            action=action.reshape(t, e),
            value=value.reshape(t, e),
            reward=jax.random.normal(krew, (t, e)),
            log_prob=log_prob.reshape(t, e),
            obs=(obs,),
            # Unique per-sample id rides through resampling to identify each transition
            info={"idx": jnp.arange(t * e, dtype=jnp.float32).reshape(t, e)},
            action_mask=jnp.ones((t, e, n_actions)),
            valid_mass=jnp.ones((t, e)),
        )
        last_val = jax.random.normal(klast, (e,))
        return model, train_state, traj, last_val

    def test_recompute_matches_ordered_reference_at_no_update(self):
        _, train_state, traj, last_val = self._rollout()
        config = self._config()
        adv, targets = _recompute_vtrace_advantages(train_state, traj, last_val, config)
        # At no-update the importance ratio is 1 everywhere, so the VTrace recompute must
        # equal the plain advantage scan over the ordered rollout with unit ratios.
        ref_adv, ref_targets, _ = _calculate_puffer_advantage(
            train_state, traj, last_val, jnp.ones_like(traj.reward), config
        )
        self.assertEqual(adv.shape, (self.T, self.E))
        chex.assert_trees_all_close(adv, ref_adv, atol=1e-5)
        chex.assert_trees_all_close(targets, ref_targets, atol=1e-5)

    def test_last_row_bootstraps_on_supplied_last_value(self):
        _, train_state, traj, last_val = self._rollout()
        config = self._config()
        adv, _ = _recompute_vtrace_advantages(train_state, traj, last_val, config)
        # Final-row advantage: delta_T = rho * (r_T + gamma * V(s_{T+1}) - V(s_T)), rho == 1
        expected_last = traj.reward[-1] + config.GAMMA * last_val - traj.value[-1]
        chex.assert_trees_all_close(adv[-1], expected_last, atol=1e-5)
        # Regression: the old in-loss recompute bootstrapped each chunk on its own last
        # value (delta_T = r_T + gamma * V(s_T) - V(s_T)) instead of V(s_{T+1}).
        self_bootstrap = traj.reward[-1] + config.GAMMA * traj.value[-1] - traj.value[-1]
        self.assertGreater(float(jnp.abs(adv[-1] - self_bootstrap).max()), 1e-4)

    def test_advantages_ride_with_samples_through_prioritized_resampling(self):
        """Order-invariance: with PRIO_ALPHA > 0 + VTrace clipping, every resampled sample
        must carry exactly the advantage/target computed for it on the ordered rollout,
        regardless of the resampling permutation (different RNG keys)."""
        _, train_state, traj, last_val = self._rollout()
        config = self._config()
        adv, targets = _recompute_vtrace_advantages(train_state, traj, last_val, config)
        priorities = compute_sample_priority_weights(adv, jnp.array(config.PRIO_ALPHA))
        flat_adv, flat_targets = adv.reshape(-1), targets.reshape(-1)
        for seed in (0, 1):
            minibatches, _ = _sample_prioritized_batch(
                (traj, adv, targets),
                priorities,
                jnp.array(config.PRIO_BETA0),
                jax.random.PRNGKey(seed),
                config,
            )
            mb_traj, mb_adv, mb_targets = minibatches
            idx = mb_traj.info["idx"].reshape(-1).astype(jnp.int32)
            # Resampling with replacement must permute (traj, adv, targets) consistently
            chex.assert_trees_all_close(mb_adv.reshape(-1), flat_adv[idx], atol=1e-6)
            chex.assert_trees_all_close(mb_targets.reshape(-1), flat_targets[idx], atol=1e-6)

    def test_loss_uses_precomputed_advantages_not_in_loss_scan(self):
        """With VTrace advantages precomputed on ordered data, ``_loss_fn`` must consume
        them as-is: identical batch_info gives identical loss/grads whether the VTrace
        clipping flags are on or off (the old code re-ran the scan in-loss when on).

        The loss is evaluated with a *drifted* policy (different weights from the
        behaviour model that produced the stored log-probs) so ratio != 1: at no-update
        the actor term is -mean(normalized adv) == 0 for any advantages, which would make
        this check vacuous.
        """
        _, train_state, traj, last_val = self._rollout()
        drifted = ActorCriticMLP(
            self.N_ACTIONS, self.OBS_DIM, num_layers=1, num_units=16, key=jax.random.PRNGKey(99)
        )
        n = self.T * self.E
        config_vtrace = self._config(NUM_MINIBATCHES=1, MINIBATCH_SIZE=n)
        config_gae = self._config(NUM_MINIBATCHES=1, MINIBATCH_SIZE=n, RHO_CLIP=0.0, C_CLIP=0.0)
        adv, targets = _recompute_vtrace_advantages(train_state, traj, last_val, config_vtrace)
        flat = jax.tree.map(lambda x: x.reshape((n,) + x.shape[2:]), (traj, adv, targets))
        batch_info = (*flat, jnp.ones((n,)))
        (loss_v, _), grads_v = _loss_fn(drifted, train_state, batch_info, config_vtrace)
        (loss_g, _), grads_g = _loss_fn(drifted, train_state, batch_info, config_gae)
        chex.assert_trees_all_close(loss_v, loss_g, atol=1e-6)
        chex.assert_trees_all_close(grads_v, grads_g, atol=1e-6)
        # The passed advantages must actually be consumed under the VTrace config: a
        # non-affine perturbation (invariant neither to the loss's mean/std advantage
        # normalization nor to the old in-loss recompute, which ignored passed adv
        # entirely) must change the loss.
        perturbed_info = (flat[0], flat[1] + flat[1] ** 2, flat[2], jnp.ones((n,)))
        (loss_p, _), _ = _loss_fn(drifted, train_state, perturbed_info, config_vtrace)
        self.assertGreater(abs(float(loss_p - loss_v)), 1e-6)


class SingleEnvPrioritizedReplayTest(chex.TestCase):
    """Regression tests for ``_sample_prioritized_batch`` with the unvmapped NUM_ENVS=1 rollout.

    With NUM_ENVS=1 the rollout is not vmapped, so transition leaves are
    (ROLLOUT_LENGTH, ...) with no env axis. The per-sample resampling assumed a
    (T, E, ...) layout: ``x.reshape((-1, *x.shape[2:]))`` flattened a real feature
    dimension (e.g. an obs leaf (32, 4403) -> (140896,)) and the reshape back to
    x.shape crashed with "cannot reshape array of shape (32,) (size 32) into shape
    (32, 4403)". The trajectory (RNN) branch likewise took along a nonexistent env
    axis 1 and tiled importance weights to (T, 1) instead of (T,).
    """

    T, OBS_DIM, N_ACTIONS = 8, 6, 5

    def _config(self, **overrides):
        base = dict(
            env_type="rwa",
            USE_GNN=False,
            USE_TRANSFORMER=False,
            USE_RNN=False,
            OFF_POLICY_IAM=False,
            LOGR_CLIP=10.0,
            ROLLOUT_LENGTH=self.T,
            NUM_ENVS=1,
            NUM_MINIBATCHES=2,
            MINIBATCH_SIZE=self.T // 2,
            NUM_LEARNERS=1,
            GAMMA=0.99,
            GAE_LAMBDA=0.9,
            REWARD_CENTERING=False,
            RHO_CLIP=1.0,
            C_CLIP=1.0,
            PRIO_ALPHA=0.6,
            PRIO_BETA0=0.4,
            PROFILE=False,
            DEBUG=False,
            DEBUG_LOSS=False,
            ENHANCED_LOGGING=False,
            IAM_GATING=False,
            IAM_DAMPING=False,
            ADV_CLIP=10.0,
            CLIP_EPS=0.2,
            VF_COEF=0.5,
            VALID_MASS_LOSS_COEF=0.0,
        )
        base.update(overrides)
        return Box(base)

    def _rollout(self, seed=0):
        """Synthetic ordered NUM_ENVS=1 rollout: leaves are (T, ...) with no env axis."""
        t, obs_dim, n_actions = self.T, self.OBS_DIM, self.N_ACTIONS
        kmodel, kobs, kact, krew, klast = jax.random.split(jax.random.PRNGKey(seed), 5)
        model = ActorCriticMLP(n_actions, obs_dim, num_layers=1, num_units=16, key=kmodel)
        train_state = TrainState.create(model, optax.adam(1e-3))
        obs = jax.random.normal(kobs, (t, obs_dim))
        pi, value = jax.vmap(model)(obs)
        action = pi.sample(seed=kact)
        # All-ones mask: the masked behaviour policy equals the unmasked policy
        log_prob = pi.log_prob(action)
        traj = RSATransition(
            terminal=jnp.zeros((t,), dtype=bool),
            truncated=jnp.zeros((t,), dtype=bool),
            action=action.reshape(t),
            value=value.reshape(t),
            reward=jax.random.normal(krew, (t,)),
            log_prob=log_prob.reshape(t),
            obs=(obs,),
            # Unique per-sample id rides through resampling to identify each transition
            info={"idx": jnp.arange(t, dtype=jnp.float32)},
            action_mask=jnp.ones((t, n_actions)),
            valid_mass=jnp.ones((t,)),
        )
        last_val = jax.random.normal(klast, ())
        return model, train_state, traj, last_val

    def test_per_sample_resampling_carries_matching_rows(self):
        """NUM_ENVS=1 + PRIO_ALPHA>0 (+ PRIO_BETA0 != 1) must not crash, and every
        resampled sample must carry its own obs row / advantage / target."""
        _, train_state, traj, last_val = self._rollout()
        config = self._config()
        adv, targets = _recompute_vtrace_advantages(train_state, traj, last_val, config)
        self.assertEqual(adv.shape, (self.T,))
        priorities = compute_sample_priority_weights(adv, jnp.array(config.PRIO_ALPHA))
        for seed in (0, 1):
            minibatches, weights = _sample_prioritized_batch(
                (traj, adv, targets),
                priorities,
                jnp.array(config.PRIO_BETA0),
                jax.random.PRNGKey(seed),
                config,
            )
            mb_traj, mb_adv, mb_targets = minibatches
            idx = mb_traj.info["idx"].reshape(-1).astype(jnp.int32)
            # Resampling with replacement must permute (traj, adv, targets) consistently
            chex.assert_trees_all_close(mb_adv.reshape(-1), adv[idx], atol=1e-6)
            chex.assert_trees_all_close(mb_targets.reshape(-1), targets[idx], atol=1e-6)
            # Feature-bearing leaves (the crash case) keep whole per-sample rows intact
            mb_obs = mb_traj.obs[0].reshape(-1, self.OBS_DIM)
            chex.assert_trees_all_close(mb_obs, traj.obs[0][idx], atol=1e-6)
            # PER importance weights are per-sample and max-normalized
            self.assertEqual(weights.shape, (config.NUM_MINIBATCHES, config.MINIBATCH_SIZE))
            self.assertLessEqual(float(weights.max()), 1.0 + 1e-6)

    def test_trajectory_branch_single_env_identity(self):
        """USE_RNN trajectory-level sampling with a single env: sampling one trajectory
        out of one is the identity, with unit importance weights of shape (T,)."""
        _, train_state, traj, last_val = self._rollout()
        config = self._config(USE_RNN=True)
        adv, targets = _recompute_vtrace_advantages(train_state, traj, last_val, config)
        priorities = compute_trajectory_priority_weights(adv, jnp.array(config.PRIO_ALPHA))
        minibatches, weights = _sample_prioritized_batch(
            (traj, adv, targets),
            priorities,
            jnp.array(config.PRIO_BETA0),
            jax.random.PRNGKey(0),
            config,
        )
        mb_traj, mb_adv, _ = minibatches
        # No shuffle with RNN and identity sampling: original temporal order is preserved
        chex.assert_trees_all_close(mb_adv.reshape(-1), adv, atol=1e-6)
        chex.assert_trees_all_close(
            mb_traj.obs[0].reshape(-1, self.OBS_DIM), traj.obs[0], atol=1e-6
        )
        # The single trajectory has probability 1, so its PER weight is exactly 1
        self.assertEqual(weights.shape, (config.NUM_MINIBATCHES, config.MINIBATCH_SIZE))
        chex.assert_trees_all_close(weights, jnp.ones_like(weights), atol=1e-6)


class _Traj(NamedTuple):
    terminal: jnp.ndarray
    truncated: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray


class AdvantageTruncationTest(chex.TestCase):
    """Regression for GAE leakage across auto-reset (truncation) boundaries.

    env.step auto-resets on terminal OR truncated, so at a truncated step value[t+1] is
    V(post-reset state). The advantage recursion must therefore mask with done = terminal
    OR truncated; masking with terminal alone bootstraps on the post-reset value and
    propagates next-episode TD errors into the previous episode.
    """

    def _adv(self, rewards, values, truncated, last_value):
        t = rewards.shape[0]
        traj = _Traj(
            terminal=jnp.zeros((t,), dtype=bool),
            truncated=truncated,
            value=values,
            reward=rewards,
        )
        config = Box(
            dict(GAE_LAMBDA=0.9, GAMMA=0.99, REWARD_CENTERING=False, RHO_CLIP=0.0, C_CLIP=0.0)
        )
        train_state = SimpleNamespace(step=jnp.array(0), avg_reward=jnp.array(0.0))
        adv, _, _ = _calculate_puffer_advantage(
            train_state, traj, last_value, jnp.ones_like(rewards), config
        )
        return adv

    def test_no_credit_flows_across_truncation(self):
        truncated = jnp.array([False, False, True, False, False, False])
        rewards_a = jnp.array([1.0, 0.5, -1.0, 2.0, 3.0, 1.0])
        values_a = jnp.array([0.3, 0.2, 0.1, 5.0, 4.0, 3.0])
        # Same trajectory before the boundary, wildly different afterwards
        rewards_b = rewards_a.at[3:].set(jnp.array([-7.0, 11.0, 0.0]))
        values_b = values_a.at[3:].set(jnp.array([-2.0, 9.0, 1.0]))

        adv_a = self._adv(rewards_a, values_a, truncated, jnp.array(2.0))
        adv_b = self._adv(rewards_b, values_b, truncated, jnp.array(-5.0))

        # Advantages up to and including the truncated step are independent of the next episode
        chex.assert_trees_all_close(adv_a[:3], adv_b[:3], atol=1e-6)
        # The truncated step must not bootstrap on the (post-reset) next value
        chex.assert_trees_all_close(adv_a[2], rewards_a[2] - values_a[2], atol=1e-6)


class TransitionMaskTest(chex.TestCase):
    """Regression for the transition action_mask/valid_mass stored by ``_env_step``.

    env.step auto-resets on done, replacing link_slot_mask with the initial all-ones mask
    and valid_mass with 1.0. The transition must store the acting state's mask/valid mass
    (as used by select_action), captured before env.step.
    """

    def test_truncated_step_stores_acting_mask(self):
        settings = dict(
            load=100,
            k=2,
            topology_name="4node",
            link_resources=4,
            max_requests=10,
            mean_service_holding_time=10,
            env_type="rwa",
            values_bw=[1],
            slot_size=1,
            guardband=0,
            continuous_operation=False,
        )
        env, params = make(settings, log_wrapper=True)
        rng = jax.random.PRNGKey(0)
        reset_key, step_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key, params)

        # Occupy all slots except slot 0 on every link (so the acting mask has zeros) and
        # make this the final request of the episode (so env.step truncates + auto-resets).
        inner = env_state.env_state
        occupied = jnp.ones_like(inner.link_slot_array).at[:, 0].set(0)
        inner = inner.replace(
            link_slot_array=occupied,
            total_requests=jnp.array(params.max_requests - 1, dtype=inner.total_requests.dtype),
        )
        env_state = env_state.replace(env_state=inner)
        expected_mask = env.action_mask(inner, params)[0]  # ty: ignore[unresolved-attribute]
        # Precondition: the acting mask is not the all-ones initial mask
        self.assertFalse(bool(jnp.all(expected_mask == 1)))

        model = ActorCriticMLP(
            int(expected_mask.shape[-1]),
            int(obs.shape[-1]),
            num_layers=1,
            num_units=16,
            key=jax.random.PRNGKey(1),
        )
        train_state = TrainState.create(model, optax.adam(1e-3))
        config = Box(
            dict(
                env_type="rwa",
                USE_GNN=False,
                USE_TRANSFORMER=False,
                deterministic=False,
                DEBUG=False,
                REWARD_SCALE=1.0,
            )
        )

        runner_state = (train_state, env_state, tuple([obs]), step_key, step_key)
        runner_state, transition = _env_step(runner_state, None, env, params, config)

        self.assertTrue(bool(transition.truncated))
        # The stored mask is the acting mask, not the post-reset all-ones mask
        chex.assert_trees_all_close(
            transition.action_mask, expected_mask.astype(transition.action_mask.dtype)
        )
        self.assertFalse(bool(jnp.all(transition.action_mask == 1)))
        # ... even though the post-step (reset) state carries the all-ones initial mask
        self.assertTrue(bool(jnp.all(runner_state[1].env_state.link_slot_mask == 1)))


class PowerHeadShapeTest(chex.TestCase):
    """Regression for the chosen-path power log-prob/entropy selection in ``_loss_fn``.

    The GNN power head is per-path: vmapped over the minibatch, power_dist has batch shape
    (B, k). ``_loss_fn`` must reduce log_prob/entropy to shape (B,) for the chosen path.
    The old vmapped ``dynamic_slice`` left log_prob at (B, 1) — silently broadcasting the
    log-ratio to (B, B) — and never reduced entropy from (B, k).
    """

    def test_chosen_path_selection_shapes_and_values(self):
        b, k, levels = 5, 3, 4
        key = jax.random.PRNGKey(0)
        logits = jax.random.normal(key, (b, k, levels))
        power_dist = distrax.Categorical(logits=logits)
        power_actions = jnp.tile(jnp.array([0, 1, 2, 3, 0])[:, None], (1, k))  # (B, k)
        path_indices = jnp.array([0, 1, 2, 0, 1])
        stored_log_prob = jnp.zeros((b,))  # rollout-time log_prob is (B,)

        power_log_prob = power_dist.log_prob(power_actions)  # (B, k)
        # The fixed selection used in _loss_fn
        selected_lp = jnp.take_along_axis(power_log_prob, path_indices[:, None], axis=1)[:, 0]
        selected_ent = jnp.take_along_axis(power_dist.entropy(), path_indices[:, None], axis=1)[
            :, 0
        ]
        self.assertEqual(selected_lp.shape, (b,))
        self.assertEqual(selected_ent.shape, (b,))
        self.assertEqual((selected_lp - stored_log_prob).shape, (b,))  # log-ratio stays (B,)
        expected = jnp.stack(
            [power_dist.log_prob(power_actions)[i, path_indices[i]] for i in range(b)]
        )
        chex.assert_trees_all_close(selected_lp, expected, atol=1e-6)

        # The old construction yields (B, 1): the log-ratio silently broadcasts to (B, B).
        old = jax.vmap(lambda x, i: jax.lax.dynamic_slice(x, (i,), (1,)))(
            power_log_prob, path_indices
        )
        self.assertEqual(old.shape, (b, 1))
        self.assertEqual((old - stored_log_prob).shape, (b, b))


if __name__ == "__main__":
    absltest.main()
