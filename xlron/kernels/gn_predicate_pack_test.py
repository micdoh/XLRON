"""Identical path/slot checks must retain strict, infinite and NaN thresholds."""

import jax
import jax.numpy as jnp
import numpy as np

from xlron.kernels.gn_predicate_pack import pack_old_snr_predicates


def test_duplicate_predicates_match_all_original_comparisons():
    ids = np.array([[2, 2, 1, 1, -1, 7], [2, 2, 1, 3, -1, 7], [2, 4, 1, 3, 2, 7]], np.int32)
    required = np.array(
        [[4, 2, 3, -np.inf, 1, 2], [7, np.nan, -np.inf, 5, 3, 6], [6, 8, np.inf, 4, 7, np.nan]],
        np.float32,
    )
    actual = jax.jit(lambda a, b: pack_old_snr_predicates(a, b, 5, block_slots=8))(
        jnp.array(ids), jnp.array(required)
    )
    expected = required.copy()
    for col in range(ids.shape[1]):
        for path in range(5):
            rows = np.flatnonzero((ids[:, col] == path) & (required[:, col] != -np.inf))
            if len(rows):
                expected[rows, col] = -np.inf
                expected[rows[0], col] = np.max(required[rows, col])
    np.testing.assert_array_equal(actual, expected)
    # The same SNR for every duplicate referrer, including slots with empty
    # old predicates; test below, at and above the finite thresholds.
    for snr in (-50.0, 3.0, 7.0, 50.0):
        np.testing.assert_array_equal(
            np.all(snr >= required, axis=0), np.all(snr >= np.asarray(actual), axis=0)
        )
    rng = np.random.default_rng(17)
    for _ in range(100):
        path_snr = rng.uniform(-10, 20, (5, ids.shape[1]))
        values = np.where(
            (ids >= 0) & (ids < 5),
            path_snr[np.clip(ids, 0, 4), np.arange(ids.shape[1])],
            rng.uniform(-10, 20, ids.shape),
        )
        np.testing.assert_array_equal(
            np.all(values >= required, axis=0), np.all(values >= np.asarray(actual), axis=0)
        )
