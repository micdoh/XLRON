"""The log-barrier must keep a corrective gradient after valid mass collapses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.train.ppo import _log_valid_mass
from xlron.train.train_utils import diagnostics_metrics


@pytest.mark.parametrize("mask", [[1.0, 0.0, 1.0], [0.5, 0.0, 0.25], [1.0, 1.0, 1.0]])
def test_log_valid_mass_value_and_gradient_match_probability_definition(mask):
    logits = jnp.array([0.2, -0.4, 1.1])
    mask = jnp.array(mask)
    reference = lambda x: -jnp.log(jnp.sum(jax.nn.softmax(x) * mask))
    actual = lambda x: -_log_valid_mass(x, mask)
    expected_value, expected_grad = jax.value_and_grad(reference)(logits)
    value, grad = jax.jit(jax.value_and_grad(actual))(logits)
    np.testing.assert_allclose(value, expected_value, atol=2e-7)
    np.testing.assert_allclose(grad, expected_grad, atol=2e-7)


@pytest.mark.parametrize("gap", [30.0, 100.0, 1000.0])
def test_collapsed_valid_logits_receive_full_corrective_gradient(gap):
    # Two actions, only the very unlikely second action is feasible.
    # -log P(second) = softplus(gap), with gradients [sigmoid(gap), -sigmoid(gap)].
    logits = jnp.array([0.0, -gap])
    mask = jnp.array([0.0, 1.0])
    value, grad = jax.jit(jax.value_and_grad(lambda x: -_log_valid_mass(x, mask)))(logits)
    np.testing.assert_allclose(value, gap, atol=1e-5)
    np.testing.assert_allclose(grad, [1.0, -1.0], atol=1e-6)


def test_empty_masks_have_zero_loss_and_finite_zero_gradient_in_mixed_batch():
    logits = jnp.array([[0.0, -1000.0], [1000.0, -1000.0], [0.0, -1000.0]])
    mask = jnp.array([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    value, grad = jax.jit(jax.value_and_grad(lambda x: -_log_valid_mass(x, mask).mean()))(logits)
    np.testing.assert_allclose(value, 1000.0 / 3, rtol=1e-6)
    np.testing.assert_allclose(grad, [[1 / 3, -1 / 3], [0, 0], [0, 0]], atol=1e-6)
    assert np.isfinite(grad).all()


def test_log_barrier_is_invariant_to_common_logit_shift():
    mask = jnp.array([0.0, 1.0, 1.0])
    logits = jnp.array([10.0, -30.0, -20.0])
    np.testing.assert_allclose(
        _log_valid_mass(logits, mask), _log_valid_mass(logits + 2000, mask), atol=2e-4
    )


def test_new_diagnostics_are_registered_in_wandb():
    assert "diagnostics/current_log_valid_mass_mean" in diagnostics_metrics
    assert "diagnostics/current_valid_mass_below_floor_frac" in diagnostics_metrics
