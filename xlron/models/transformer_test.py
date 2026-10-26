"""Moving the final affine actor projection preserves outputs and gradients."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.models.transformer import pooled_actor_projection


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_pooled_projection_preserves_logits_and_all_gradients(depth, bias):
    mlp = eqx.nn.MLP(
        8, 140, 16, depth, use_bias=bias, use_final_bias=bias, key=jax.random.PRNGKey(1)
    )
    embeddings = jax.random.normal(jax.random.PRNGKey(2), (5, 8))
    # Includes a dummy path and nonbinary weights to check the bias multiplier.
    paths = jnp.array([[1, 1, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0.5, 0, 1, 0.25]])

    def control(model, x, p):
        return jnp.sum(p[:, :, None] * jax.vmap(model)(x)[None, :, :], axis=1)

    expected = control(mlp, embeddings, paths)
    actual = pooled_actor_projection(mlp, embeddings, paths)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(actual[1], jnp.zeros(140))

    def objective(arguments, fn):
        model, x, p = arguments
        logits = fn(model, x, p)
        return jnp.mean(logits**2)

    arguments = (mlp, embeddings, paths)
    grad = eqx.filter_grad(objective)(arguments, pooled_actor_projection)
    reference = eqx.filter_grad(objective)(arguments, control)
    for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(reference), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
