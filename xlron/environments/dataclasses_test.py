"""Tests for xlron/environments/dataclasses.py."""

import numpy as np

from xlron.environments.dataclasses import HashableArrayWrapper


class _CountingArray(np.ndarray):
    calls = 0

    def tobytes(self, *args, **kwargs):
        type(self).calls += 1
        return super().tobytes(*args, **kwargs)


class TestHashableArrayWrapper:
    def test_hash_is_computed_once(self):
        arr = np.arange(6).view(_CountingArray)
        _CountingArray.calls = 0
        wrapper = HashableArrayWrapper(arr)
        assert hash(wrapper) == hash(wrapper) == hash(np.arange(6).tobytes())
        assert _CountingArray.calls == 1

    def test_hash_follows_mutation_and_reassignment(self):
        wrapper = HashableArrayWrapper(np.arange(4))
        before = hash(wrapper)
        wrapper[0] = 9
        assert hash(wrapper) != before
        assert hash(wrapper) == hash(np.array([9, 1, 2, 3]).tobytes())
        wrapper.val = np.arange(4)
        assert hash(wrapper) == before

    def test_equal_contents_compare_equal(self):
        assert HashableArrayWrapper(np.arange(3)) == HashableArrayWrapper(np.arange(3))
        assert HashableArrayWrapper(np.arange(3)) != HashableArrayWrapper(np.arange(1, 4))
