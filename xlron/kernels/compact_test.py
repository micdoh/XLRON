import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.kernels.compact import compact_true_indices


@pytest.mark.parametrize("size", [1, 7, 128, 8340])
@pytest.mark.parametrize("density", [0.0, 0.03, 0.5, 1.0])
def test_matches_nonzero(size, density):
    mask = jax.random.uniform(jax.random.PRNGKey(size), (size,)) < density
    indices, count = jax.jit(compact_true_indices)(mask)
    np.testing.assert_array_equal(indices, jnp.nonzero(mask, size=size, fill_value=0)[0])
    assert int(count) == int(jnp.sum(mask))


def test_runs_unbatched_under_lax_map():
    masks = jax.random.uniform(jax.random.PRNGKey(0), (3, 50)) < 0.3
    indices, counts = jax.lax.map(compact_true_indices, masks)
    for mask, got, count in zip(masks, indices, counts):
        np.testing.assert_array_equal(got, jnp.nonzero(mask, size=50, fill_value=0)[0])
        assert int(count) == int(jnp.sum(mask))
