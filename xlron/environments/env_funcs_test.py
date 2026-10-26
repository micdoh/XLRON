"""
Unit tests for `env_funcs.py`.
See `chex` github and docs for info on test framework.

Key points:
There is a class for each function under test.
chex.all_variants() decorator runs the test once for each variant (e.g. jitted, non-jitted, pmapped, etc.) of the function under test.
parameterized.named_parameters() decorator runs the test once for each set of parameters passed to the function under test.
"""

import json
from typing import cast
from functools import partial
import pathlib
import tempfile

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from tests.helpers.gn_model import setup
from tests.helpers.rsa import (
    rsa_4node_3_slot_request_test_setup,
    rsa_nsfnet_16_mod_test_setup,
    rsa_nsfnet_16_test_setup,
    rwa_4node_agg_slots_test_setup,
    rwa_4node_test_setup,
    settings_rwa_4node,
)
from xlron.environments import env_funcs as ef
from xlron.environments.dataclasses import *
from xlron.environments.dataclasses import HashableArrayWrapper
from xlron.environments.env_funcs import *
from xlron.environments.env_funcs import (
    generate_request_rsa,
    get_obs_transformer,
    get_snr_link_array,
    mask_slots_rmsa_gn_model,
)
from xlron.environments.make_env import make
from xlron.environments.rsa.rsa import *
from xlron.environments.wrappers import *
from xlron.train.train_utils import init_network


class InitPathLinkArrayTest(chex.TestCase):
    # N.B. jit not required here as this function is
    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        (
            "case_base",
            "4node",
            2,
            jnp.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 1, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 1, 0, 0],
                    [1, 0, 1, 1],
                    [0, 0, 1, 0],
                    [1, 1, 0, 1],
                    [0, 0, 1, 1],
                    [1, 1, 0, 0],
                    [0, 0, 0, 1],
                    [1, 1, 1, 0],
                ]
            ),
        ),
    )
    def test_init_path_link_array(self, topology_name, k, expected):
        graph = make_graph(topology_name)
        path_link_array = self.variant(init_path_link_array)(graph, k, path_sort_criteria="hops")
        print(path_link_array)
        chex.assert_trees_all_close(path_link_array, expected)


class GetPathIndicesTest(chex.TestCase):
    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base", 0, 3, 2, 4, 4),
        ("case_conus", 73, 74, 2, 75, 5548),
        ("case_nsfnet", 1, 2, 5, 14, 65),
    )
    def test_get_path_indices(self, s, d, k, N, expected):
        # get_path_indices now takes params as first arg
        _, _, _, _, params = rwa_4node_test_setup()
        paths_start_index = self.variant(get_path_indices)(params, s, d, k, N)
        chex.assert_trees_all_close(paths_start_index, expected)


class GetPathsTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base", jnp.array([0, 3]), jnp.array([[0, 1, 0, 0], [1, 0, 1, 1]])),
        ("case_switched", jnp.array([3, 0]), jnp.array([[0, 1, 0, 0], [1, 0, 1, 1]])),
    )
    def test_get_paths(self, nodes, expected):
        paths = self.variant(get_paths, static_argnums=(0,))(self.params, nodes)
        chex.assert_trees_all_close(paths, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_01",
            jnp.array([0, 1]),
            jnp.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            "case_12",
            jnp.array([1, 2]),
            jnp.array(
                [
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                ]
            ),
        ),
    )
    def test_get_paths_nsfnet(self, nodes, expected):
        self.key, self.env, self.obs, self.state, self.params = rsa_nsfnet_16_test_setup()
        paths = self.variant(get_paths, static_argnums=(0,))(self.params, nodes)
        print(paths)
        chex.assert_trees_all_close(paths, expected)


class GenerateArrivalHoldingTimesTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base", (jnp.array([0.02910362]), jnp.array([7.173531]))),
    )
    def test_generate_arrival_times(self, expected):
        min_arr, min_hold = 0, 0
        arrival_time, holding_time = 0, 0
        rng = jax.random.PRNGKey(0)
        for i in range(1, 10000):
            rng, key = jax.random.split(rng)
            arrival_time, holding_time = self.variant(generate_arrival_holding_times)(
                key,
                self.params,
                jnp.array(self.params.arrival_rate),
                jnp.array(self.params.mean_service_holding_time),
            )
            min_arr = jnp.minimum(min_arr, arrival_time)
            min_hold = jnp.minimum(min_hold, holding_time)
            print(i, arrival_time, holding_time)
        print(min_arr, min_hold)
        chex.assert_trees_all_close(arrival_time, expected[0])
        chex.assert_trees_all_close(holding_time, expected[1])

    def test_truncated_holding_time_independent_of_arrival_time(self):
        """Regression test: with truncate_holding_time, the candidate holding-time keys
        were re-split from the parent key, so (split being prefix-stable) candidate 0
        reused key_arrival. When candidates 1-4 were all truncated, the holding time was
        the same exponential draw as the arrival time (perfectly correlated).
        """
        _, _, _, _, params = rwa_4node_test_setup(truncate_holding_time=True)
        keys = jax.random.split(jax.random.PRNGKey(42), 50000)
        # With rate == mean == 1, an aliased draw makes holding_time == arrival_time exactly
        arrival, holding = jax.vmap(generate_arrival_holding_times, in_axes=(0, None, None, None))(
            keys, params, jnp.array(1.0), jnp.array(1.0)
        )
        arrival, holding = arrival.reshape(-1), holding.reshape(-1)
        aliased = jnp.sum((holding == arrival) & (holding > 0))
        self.assertEqual(int(aliased), 0)


class SetPathLinksTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    def _make_action_info(self, link_array, path, initial_slot, num_slots):
        initial_slot_index = jnp.array(initial_slot)
        num_slots_arr = jnp.array(num_slots)
        slot_indices = jnp.arange(link_array.shape[1])
        slot_mask = (slot_indices >= initial_slot_index) & (
            slot_indices < initial_slot_index + num_slots_arr
        )
        combined_mask = path[:, None] * slot_mask[None, :].astype(jnp.float32)
        return ActionInfo(
            action=jnp.array(0),
            path_index=jnp.array(0),
            initial_slot_index=initial_slot_index,
            num_slots=num_slots_arr,
            path=path,
            se=jnp.array(1.0),
            requested_datarate=jnp.array(1),
            nodes_sd=jnp.array([0, 1]),
            affected_slots_mask=combined_mask,
            power_action=jnp.float32(0.0),
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_base",
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [1, 0, 0, 0],
            3,
            2,
            -1,
            [[0, 0, 0, -1, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ),
        (
            "case_overwrite",
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]],
            [1, 0, 1, 0],
            1,
            3,
            0,
            [[1, 0, 0, 0, 1], [2, 2, 2, 2, 2], [3, 0, 0, 0, 3], [4, 4, 4, 4, 4]],
        ),
        (
            "case_multilink",
            [[5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]],
            [1, 0, 1, 1],
            0,
            4,
            -99,
            [
                [-99, -99, -99, -99, 5],
                [5, 5, 5, 5, 5],
                [-99, -99, -99, -99, 5],
                [-99, -99, -99, -99, 5],
            ],
        ),
        (
            "case_no_path",
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [0, 0, 0, 0],
            3,
            2,
            -1,
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        ),
    )
    def test_set_path_links(self, link_array, path, initial_slot, num_slots, value, expected):
        link_array = jnp.array(link_array, dtype=jnp.float32)
        path = jnp.array(path, dtype=jnp.float32)
        expected = jnp.array(expected, dtype=jnp.float32)
        action_info = self._make_action_info(link_array, path, initial_slot, num_slots)
        updated_link_array = self.variant(set_path_links)(
            link_array, action_info.affected_slots_mask, value
        )
        chex.assert_trees_all_close(updated_link_array, expected)


class UpdatePathLinksTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    def _make_action_info(self, link_array, path, initial_slot, num_slots):
        # Build a minimal state-like object to compute the mask
        initial_slot_index = jnp.array(initial_slot)
        num_slots_arr = jnp.array(num_slots)
        # Compute affected_slots_mask manually: (num_links, num_slots_per_link) mask
        slot_indices = jnp.arange(link_array.shape[1])
        slot_mask = (slot_indices >= initial_slot_index) & (
            slot_indices < initial_slot_index + num_slots_arr
        )
        combined_mask = path[:, None] * slot_mask[None, :].astype(jnp.float32)
        return ActionInfo(
            action=jnp.array(0),
            path_index=jnp.array(0),
            initial_slot_index=initial_slot_index,
            num_slots=num_slots_arr,
            path=path,
            se=jnp.array(1.0),
            requested_datarate=jnp.array(1),
            nodes_sd=jnp.array([0, 1]),
            affected_slots_mask=combined_mask,
            power_action=jnp.float32(0.0),
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_base",
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [1, 0, 0, 0],
            3,
            2,
            -1,
            [[0, 0, 0, -1, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ),
        (
            "case_value",
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [1, 0, 1, 1],
            0,
            4,
            -99,
            [
                [-99, -99, -99, -99, 0],
                [0, 0, 0, 0, 0],
                [-99, -99, -99, -99, 0],
                [-99, -99, -99, -99, 0],
            ],
        ),
        (
            "case_no_path",
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [0, 0, 0, 0],
            3,
            2,
            -1,
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ),
    )
    def test_update_path_links(self, link_array, path, initial_slot, num_slots, value, expected):
        link_array = jnp.array(link_array)
        path = jnp.array(path)
        expected = jnp.array(expected)
        action_info = self._make_action_info(link_array, path, initial_slot, num_slots)
        updated_link_array = self.variant(update_path_links)(link_array, action_info, value)
        chex.assert_trees_all_close(updated_link_array, expected)


class RemoveExpiredSlotRequestsTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        # This test sets absolute departure/current times, so pin the absolute-time
        # mode (the flag default, now applied to dict configs too, is relative).
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup(
            relative_arrival_times=False
        )
        # Set link 0, slots 0-1 as occupied (path=[1,0,0,0], initial_slot=0, num_slots=2)
        link_slot = self.state.link_slot_array.at[0, 0].set(1).at[0, 1].set(1)
        # Departure is now positive (time when service expires)
        departure_val = float(jnp.squeeze(self.state.current_time + self.state.holding_time))
        link_dept = (
            self.state.link_slot_departure_array.at[0, 0]
            .set(departure_val)
            .at[0, 1]
            .set(departure_val)
        )
        self.state = self.state.replace(
            link_slot_array=link_slot,
            link_slot_departure_array=link_dept,
        )

    @chex.all_variants()
    @parameterized.named_parameters(("case_base", jnp.full((4, 4), 0)))
    def test_remove_expired_slot_requests(self, expected):
        state = self.state.replace(current_time=10e4)
        jax.debug.print("departure {}", state.link_slot_departure_array, ordered=True)
        jax.debug.print("state.link_slot_array {}", state.link_slot_array, ordered=True)
        updated_state = self.variant(remove_expired_services_rsa)(state, self.params)
        jax.debug.print(
            "updated_state.link_slot_departure_array {}",
            updated_state.link_slot_departure_array,
            ordered=True,
        )
        jax.debug.print(
            "updated_state.link_slot_array {}", updated_state.link_slot_array, ordered=True
        )
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(("case_base", jnp.full((4, 4), 0)))
    def test_remove_expired_slot_requests_departure(self, expected):
        state = self.state.replace(current_time=10e4)
        updated_state = self.variant(remove_expired_services_rsa)(state, self.params)
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class UndoLinkSlotActionTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()
        # Set link 0, slots 0-1 as occupied (path=[1,0,0,0], initial_slot=0, num_slots=2)
        link_slot = self.state.link_slot_array.at[0, 0].set(1).at[0, 1].set(1)
        departure_val = float(jnp.squeeze(self.state.current_time + self.state.holding_time))
        link_dept = (
            self.state.link_slot_departure_array.at[0, 0]
            .set(departure_val)
            .at[0, 1]
            .set(departure_val)
        )
        self.state = self.state.replace(
            link_slot_array=link_slot,
            link_slot_departure_array=link_dept,
        )
        # Build ActionInfo matching the manual allocation above
        path = jnp.array([1, 0, 0, 0])
        initial_slot_index = jnp.array(0)
        num_slots = jnp.array(2)
        mask = get_affected_slots_mask(initial_slot_index, num_slots, path, self.params)
        self.action_info = ActionInfo(
            action=jnp.array(0),
            path_index=jnp.array(0),
            initial_slot_index=initial_slot_index,
            num_slots=num_slots,
            path=path,
            se=jnp.array(1.0),
            requested_datarate=jnp.array([0]),
            nodes_sd=jnp.array([0, 1]),
            affected_slots_mask=mask,
            power_action=jnp.float32(0.0),
        )


class ImplementPathActionTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    def _make_action_info(self, path, initial_slot_index, num_slots):
        initial_slot_index = jnp.array(initial_slot_index)
        num_slots = jnp.array(num_slots)
        mask = get_affected_slots_mask(initial_slot_index, num_slots, path, self.params)
        return ActionInfo(
            action=jnp.array(0),
            path_index=jnp.array(0),
            initial_slot_index=initial_slot_index,
            num_slots=num_slots,
            path=path,
            se=jnp.array(1.0),
            requested_datarate=jnp.array(1),
            nodes_sd=jnp.array([0, 1]),
            affected_slots_mask=mask,
            power_action=jnp.float32(0.0),
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_base",
            jnp.array([1, 0, 0, 0]),
            0,
            2,
            jnp.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ),
        (
            "case_extra",
            jnp.array([1, 0, 1, 0]),
            2,
            2,
            jnp.array([[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]]),
        ),
    )
    def test_implement_path_action(self, path, initial_slot_index, num_slots, expected):
        action_info = self._make_action_info(path, initial_slot_index, num_slots)
        updated_state = self.variant(implement_path_action, static_argnums=(2,))(
            self.state, action_info, self.params
        )
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_base",
            jnp.array([1, 0, 0, 0]),
            0,
            2,
            jnp.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ),
    )
    def test_implement_path_action_departure(self, path, initial_slot_index, num_slots, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        action_info = self._make_action_info(path, initial_slot_index, num_slots)
        updated_state = self.variant(implement_path_action, static_argnums=(2,))(
            state, action_info, self.params
        )
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class CheckNoSpectrumReuseTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    def _make_action_info(self, state, path, initial_slot_index, num_slots):
        initial_slot_index = jnp.array(initial_slot_index)
        num_slots = jnp.array(num_slots)
        mask = get_affected_slots_mask(initial_slot_index, num_slots, path, self.params)
        return ActionInfo(
            action=jnp.array(0),
            path_index=jnp.array(0),
            initial_slot_index=initial_slot_index,
            num_slots=num_slots,
            path=path,
            se=jnp.array(1.0),
            requested_datarate=jnp.array(1),
            nodes_sd=jnp.array([0, 1]),
            affected_slots_mask=mask,
            power_action=jnp.float32(0.0),
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base", jnp.array([1, 0, 0, 0]), 0, 2, jnp.array(False)),
        ("case_extra", jnp.array([1, 0, 1, 0]), 2, 2, jnp.array(False)),
    )
    def test_check_no_spectrum_reuse(self, path, initial_slot_index, num_slots, expected):
        action_info = self._make_action_info(self.state, path, initial_slot_index, num_slots)
        updated_state = self.variant(implement_path_action, static_argnums=(2,))(
            self.state, action_info, self.params
        )
        actual = self.variant(check_no_spectrum_reuse)(updated_state, action_info, self.params)
        chex.assert_trees_all_close(actual, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base", jnp.array([1, 0, 0, 0]), 0, 2, jnp.array(True)),
        ("case_extra", jnp.array([1, 0, 1, 0]), 2, 2, jnp.array(True)),
    )
    def test_check_no_spectrum_reuse_fail(self, path, initial_slot_index, num_slots, expected):
        action_info = self._make_action_info(self.state, path, initial_slot_index, num_slots)
        updated_state = self.variant(implement_path_action, static_argnums=(2,))(
            self.state, action_info, self.params
        )
        updated_state = self.variant(implement_path_action, static_argnums=(2,))(
            updated_state, action_info, self.params
        )
        actual = self.variant(check_no_spectrum_reuse)(updated_state, action_info, self.params)
        chex.assert_trees_all_close(actual, expected)


class InitPathLengthArrayTest(chex.TestCase):
    def setUp(self):
        super().setUp()

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        (
            "case_nsfnet",
            "nsfnet_deeprmsa_undirected",
            1,
            jnp.array([1050.0, 1500.0, 1800.0, 2400.0]),
        ),
        ("case_4node", "4node", 1, jnp.array([4, 7, 1, 3])),
    )
    def test_init_path_length_array(self, topology_name, k, expected):
        graph = make_graph(topology_name)
        path_link_array = init_path_link_array(graph, k, path_sort_criteria="hops")
        path_length_array = self.variant(init_path_length_array)(path_link_array, graph)
        chex.assert_trees_all_close(path_length_array[:4], expected)


class InitModulationsArrayTest(chex.TestCase):
    def setUp(self):
        super().setUp()

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        (
            "case_base",
            jnp.array(
                [
                    [100000.0, 1.0, 12.6, -14.0, -1.0],
                    [2500.0, 2.0, 12.6, -17.0, -1.0],
                    [1250.0, 3.0, 18.6, -20.0, -0.82],
                    [625.0, 4.0, 22.4, -23.0, -0.68],
                ]
            ),
        ),
    )
    def test_init_modulations_array(self, expected):
        modulations_filepath = str(
            pathlib.Path(__file__).parents[1].absolute()
            / "data"
            / "modulations"
            / "modulations_deeprmsa.csv"
        )
        modulations_array = self.variant(init_modulations_array)(modulations_filepath)
        chex.assert_trees_all_close(modulations_array, expected)


class InitPathSEArrayTest(chex.TestCase):
    def setUp(self):
        super().setUp()

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("case_nsfnet", "nsfnet_deeprmsa_undirected", 1, jnp.array([3.0, 2.0, 2.0, 2.0])),
        ("case_4node", "4node", 1, jnp.array([4.0, 4.0, 4.0, 4.0])),
    )
    def test_init_path_se_array(self, topology_name, k, expected):
        graph = make_graph(topology_name)
        path_link_array = init_path_link_array(graph, k, path_sort_criteria="hops")
        path_length_array = init_path_length_array(path_link_array, graph)
        modulations_filepath = str(
            pathlib.Path(__file__).parents[1].absolute()
            / "data"
            / "modulations"
            / "modulations_deeprmsa.csv"
        )
        modulations_array = init_modulations_array(modulations_filepath)
        path_se_array = self.variant(init_path_se_array)(path_length_array, modulations_array)
        chex.assert_trees_all_close(path_se_array[:4], expected)


class RequiredSlotsTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_3_slot_request_test_setup(
            slot_size=12.5
        )

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("case_empty", 100, 2, jnp.array([5])),
    )
    def test_required_slots(self, bit_rate, se, expected):
        slots = self.variant(required_slots)(bit_rate, se, 12.5)
        chex.assert_trees_all_close(slots, expected)


class MaskSlotsBitRateModFormat(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_nsfnet_16_mod_test_setup(
            env_type="rmsa"
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_empty",
            jnp.array([0, 100, 1]),
            jnp.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            jnp.array(
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ).astype(jnp.float32),
        ),
        (
            "case_full",
            jnp.array([0, 49, 1]),
            jnp.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            jnp.array(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ).astype(jnp.float32),
        ),
        (
            "case_middle",
            jnp.array([0, 100, 1]),
            jnp.array(
                [
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                ]
            ),
            jnp.array(
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ).astype(jnp.float32),
        ),
    )
    def test_mask_slots_bit_rate_mod_format(self, request, link_slot_array, expected):
        self.state = self.state.replace(link_slot_array=link_slot_array, request_array=request)
        jax.debug.print("state.link_slot_mask {}", self.state.link_slot_mask, ordered=True)
        link_slot_mask, full_link_slot_mask = self.variant(
            self.env.action_mask, static_argnums=(1,)
        )(self.state, self.params)
        # Strip trailing no-op action if present
        if link_slot_mask.shape[0] > expected.shape[0]:
            link_slot_mask = link_slot_mask[: expected.shape[0]]
        jax.debug.print("link_slot_mask {}", link_slot_mask, ordered=True)
        jax.debug.print("expected {}", expected, ordered=True)
        chex.assert_trees_all_close(link_slot_mask, expected)


class AggregateSlotsTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_agg_slots_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_all_invalid",
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "case_all_valid",
            jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            "case_first_edge_valid",
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        (
            "case_last_edge_valid",
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            "case_middle_valid",
            jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
        ),
    )
    def test_aggregate_slots(self, mask, expected):
        result = self.variant(aggregate_slots, static_argnums=(1,))(mask, self.params)
        chex.assert_trees_all_close(result, expected)


class ProcessPathActionTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_agg_slots_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_last_fit",
            jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            jnp.array(4),
            jnp.array(2),
        ),
        (
            "case_first_fit",
            jnp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            jnp.array(1),
            jnp.array(2),
        ),
    )
    def test_process_path_action(self, full_link_slot_mask, path_action, expected):
        state = self.state.replace(full_link_slot_mask=full_link_slot_mask)
        result_path, result_slot = self.variant(process_path_action, static_argnums=(1,))(
            state, self.params, path_action
        )
        jax.debug.print("result path {}", result_path, ordered=True)
        jax.debug.print("result slot {}", result_slot, ordered=True)
        chex.assert_trees_all_close(result_slot, expected)


class FindBlockStartsTest(chex.TestCase):
    def setUp(self):
        super().setUp()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_full", jnp.array([1, 1, 1, 1, 1, 1, 1]), jnp.array([0, 0, 0, 0, 0, 0, 0])),
        ("case_empty", jnp.array([0, 0, 0, 0, 0, 0, 0]), jnp.array([1, 0, 0, 0, 0, 0, 0])),
        ("case_few", jnp.array([0, 1, 0, 0, 1, 0, 0]), jnp.array([1, 0, 1, 0, 0, 1, 0])),
        ("case_many", jnp.array([0, 1, 1, 0, 1, 1, 1]), jnp.array([1, 0, 0, 1, 0, 0, 0])),
    )
    def test_find_block_starts(self, slots, expected):
        actual = self.variant(find_block_starts)(slots)
        chex.assert_trees_all_close(actual, expected)


class FindBlockEndsTest(chex.TestCase):
    def setUp(self):
        super().setUp()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_full", jnp.array([1, 1, 1, 1, 1, 1, 1]), jnp.array([0, 0, 0, 0, 0, 0, 0])),
        ("case_empty", jnp.array([0, 0, 0, 0, 0, 0, 0]), jnp.array([0, 0, 0, 0, 0, 0, 1])),
        ("case_few", jnp.array([0, 1, 0, 0, 1, 0, 0]), jnp.array([1, 0, 0, 1, 0, 0, 1])),
        ("case_many", jnp.array([0, 1, 1, 0, 1, 1, 1]), jnp.array([1, 0, 0, 1, 0, 0, 0])),
    )
    def test_find_block_ends(self, slots, expected):
        actual = self.variant(find_block_ends)(slots)
        chex.assert_trees_all_close(actual, expected)


class FindBlockSizesTest(chex.TestCase):
    def setUp(self):
        super().setUp()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_full", jnp.array([1, 1, 1, 1, 1, 1, 1]), jnp.array([0, 0, 0, 0, 0, 0, 0])),
        ("case_empty", jnp.array([0, 0, 0, 0, 0, 0, 0]), jnp.array([7, 0, 0, 0, 0, 0, 0])),
        ("case_few", jnp.array([0, 1, 0, 0, 1, 0, 0]), jnp.array([1, 0, 2, 0, 0, 2, 0])),
        ("case_many", jnp.array([0, 1, 1, 0, 1, 1, 1]), jnp.array([1, 0, 0, 1, 0, 0, 0])),
        ("case_gap", jnp.array([1, 0, 0, 0, 0, 0, 1]), jnp.array([0, 5, 0, 0, 0, 0, 0])),
    )
    def test_find_block_sizes(self, slots, expected):
        actual = self.variant(find_block_sizes)(slots)
        chex.assert_trees_all_close(actual, expected)

    # Don't test on device as it causes static_argnames to be traced...
    @chex.variants(without_device=True)
    @parameterized.named_parameters(
        ("case_full", jnp.array([1, 1, 1, 1, 1, 1, 1]), jnp.array([0, 0, 0, 0, 0, 0, 0])),
        ("case_empty", jnp.array([0, 0, 0, 0, 0, 0, 0]), jnp.array([1, 2, 3, 4, 5, 6, 7])),
        ("case_few", jnp.array([0, 1, 0, 0, 1, 0, 0]), jnp.array([1, 0, 1, 2, 0, 1, 2])),
        ("case_many", jnp.array([0, 1, 1, 0, 1, 1, 1]), jnp.array([1, 0, 0, 1, 0, 0, 0])),
        ("case_gap", jnp.array([1, 0, 0, 0, 0, 0, 1]), jnp.array([0, 1, 2, 3, 4, 5, 0])),
        (
            "case_tricky",
            jnp.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1]),
            jnp.array([1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 1, 0]),
        ),
    )
    def test_find_block_sizes_reverse(self, slots, expected):
        actual = self.variant(find_block_sizes, static_argnames=("starts_only", "reverse"))(
            slots, starts_only=False, reverse=True
        )
        chex.assert_trees_all_close(actual, expected)

    @chex.variants(without_device=True)
    @parameterized.named_parameters(
        ("case_full", jnp.array([1, 1, 1, 1, 1, 1, 1]), jnp.array([0, 0, 0, 0, 0, 0, 0])),
        ("case_empty", jnp.array([0, 0, 0, 0, 0, 0, 0]), jnp.array([7, 6, 5, 4, 3, 2, 1])),
        ("case_few", jnp.array([0, 1, 0, 0, 1, 0, 0]), jnp.array([1, 0, 2, 1, 0, 2, 1])),
        ("case_many", jnp.array([0, 1, 1, 0, 1, 1, 1]), jnp.array([1, 0, 0, 1, 0, 0, 0])),
        ("case_gap", jnp.array([1, 0, 0, 0, 0, 0, 1]), jnp.array([0, 5, 4, 3, 2, 1, 0])),
    )
    def test_find_block_sizes_not_starts_only(self, slots, expected):
        actual = self.variant(find_block_sizes, static_argnames=("starts_only", "reverse"))(
            slots, starts_only=False
        )
        chex.assert_trees_all_close(actual, expected)

    def test_find_block_sizes_matches_bruteforce_randomised(self):
        """The O(n) run-length fast path (differentiable=False) and the O(n^2) soft path
        must both match a brute-force reference over randomised inputs and edge cases."""

        def reference(slots, starts_only, reverse):
            slots = np.asarray(slots)
            n = slots.shape[0]
            sizes = np.zeros(n)
            for i in range(n):
                if slots[i] != 0:
                    continue
                if reverse:
                    j = i
                    while j >= 0 and slots[j] == 0:
                        j -= 1
                    sizes[i] = i - j
                else:
                    j = i
                    while j < n and slots[j] == 0:
                        j += 1
                    sizes[i] = j - i
            if starts_only:
                starts = np.abs(np.clip(np.diff(slots, prepend=1), -1, 0))
                sizes = sizes * starts
            return sizes

        rng = np.random.default_rng(0)
        cases = [
            np.zeros(7),  # all free
            np.ones(7),  # all occupied
            np.zeros(1),  # single free slot
            np.ones(1),  # single occupied slot
            np.array([0.0, -1.0, 0.0, 1.0, 0.0]),  # negative occupancy values (kmf_ff)
        ]
        for n in (1, 2, 3, 7, 50, 100):
            for density in (0.2, 0.5, 0.8):
                cases.append((rng.random(n) < density).astype(np.float32))
        for slots in cases:
            slots = jnp.array(slots)
            for starts_only in (True, False):
                for reverse in (True, False):
                    expected = reference(slots, starts_only, reverse)
                    fast = find_block_sizes(
                        slots, starts_only=starts_only, reverse=reverse, differentiable=False
                    )
                    soft = find_block_sizes(
                        slots, starts_only=starts_only, reverse=reverse, differentiable=True
                    )
                    chex.assert_trees_all_close(np.asarray(fast), expected)
                    chex.assert_trees_all_close(np.asarray(soft), expected)


class MaskSlotsMatchesBruteForceTest(chex.TestCase):
    """mask_slots (int cumsum + per-path required-slots gather) must match a brute-force
    per-slot window check over randomised occupancies and requests."""

    def _reference_mask(self, state, params):
        request = np.asarray(state.request_array).ravel()
        source, datarate, dest = request[0], request[1], request[2]
        path_indices = np.asarray(get_path_index_array(params, jnp.array([source, dest]))).astype(
            int
        )
        paths = np.asarray(params.path_link_array.val)[path_indices]
        se_array = np.asarray(params.path_se_array.val)
        paths_se = se_array[np.clip(path_indices, 0, se_array.shape[0] - 1)]
        occupied = np.asarray(state.link_slot_array) != 0
        num_slots = params.link_resources
        mask = np.zeros((params.k_paths, num_slots), dtype=np.float32)
        for k in range(params.k_paths):
            path = paths[k]
            if path.max() == 0:
                continue  # dummy path row -> unselectable
            se = paths_se[k] if params.consider_modulation_format else 1
            req = (
                0
                if datarate == 0
                else int(np.ceil(datarate / (se * params.slot_size))) + params.guardband
            )
            path_occupied = occupied[path == 1].any(axis=0)
            for s in range(num_slots):
                if s + req <= num_slots and not path_occupied[s : s + req].any():
                    mask[k, s] = 1.0
        return mask.reshape(-1)

    @parameterized.named_parameters(
        ("case_mod_format", True),
        ("case_no_mod_format", False),
    )
    def test_mask_slots_matches_bruteforce_randomised(self, consider_modulation_format):
        if consider_modulation_format:
            key, env, obs, state, params = rsa_nsfnet_16_mod_test_setup(env_type="rmsa")
            datarates = [25.0, 50.0, 100.0, 200.0, 400.0]
        else:
            key, env, obs, state, params = rsa_nsfnet_16_test_setup()
            datarates = [1.0, 2.0, 3.0]
        # Outer jit so the inner mask_slots donation is not applied to the shared state buffers
        mask_fn = jax.jit(lambda s: mask_slots(s, params))
        rng = np.random.default_rng(42)
        num_links = params.num_links
        num_slots = params.link_resources
        occupancies = [
            np.zeros((num_links, num_slots)),  # empty spectrum
            np.ones((num_links, num_slots)),  # full spectrum
        ]
        for density in (0.1, 0.3, 0.6):
            for _ in range(3):
                occupancies.append(
                    (rng.random((num_links, num_slots)) < density).astype(np.float32)
                )
        for occ in occupancies:
            source = int(rng.integers(0, params.num_nodes))
            dest = int(rng.integers(0, params.num_nodes))
            while dest == source:
                dest = int(rng.integers(0, params.num_nodes))
            datarate = float(rng.choice(datarates))
            test_state = state.replace(
                link_slot_array=jnp.array(occ, dtype=state.link_slot_array.dtype),
                request_array=jnp.array([source, datarate, dest], dtype=state.request_array.dtype),
            )
            _, full_mask = mask_fn(test_state)
            expected = self._reference_mask(test_state, params)
            np.testing.assert_array_equal(np.asarray(full_mask), expected)


class CalculateFragmentationTest(chex.TestCase):
    def setUp(self):
        super().setUp()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_empty", jnp.zeros((2, 8)), 0.0),
        ("case_full", jnp.ones((2, 8)), 0.0),
        ("case_contiguous_free", jnp.array([[1, 1, 1, 0, 0, 0, 0, 0]]), 0.0),
        ("case_checkerboard", jnp.array([[0, 1, 0, 1, 0, 1, 0, 1]]), 0.75),
        # Negative values (in-service slots) count as occupied: free runs of 2 and 4
        ("case_negative_occupied", jnp.array([[-1, 0, 0, -1, 0, 0, 0, 0]]), 1 - 4 / 6),
        # Mean across links: contiguous link (0) and checkerboard link (0.75)
        (
            "case_mean_over_links",
            jnp.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 1]]),
            0.375,
        ),
    )
    def test_calculate_fragmentation(self, link_slot_array, expected):
        actual = self.variant(calculate_fragmentation)(link_slot_array)
        chex.assert_trees_all_close(actual, jnp.array(expected, dtype=actual.dtype))


class FragmentationLoggingTest(chex.TestCase):
    """LogWrapper must surface the fragmentation metric stashed by step_env."""

    def test_fragmentation_in_info_and_log_state(self):
        env, params = make(settings_rwa_4node())
        key = jax.random.PRNGKey(0)
        obs, log_state = env.reset(key, params)
        chex.assert_trees_all_close(
            log_state.fragmentation, jnp.array(0, dtype=log_state.fragmentation.dtype)
        )
        step = jax.jit(env.step, static_argnums=(3,))
        _, log_state, _, _, _, info = step(key, log_state, jnp.array(0), params)
        expected = calculate_fragmentation(log_state.env_state.link_slot_array)
        chex.assert_trees_all_close(info["fragmentation"], expected)
        chex.assert_trees_all_close(log_state.fragmentation, expected)


class TopologyNodeIdNormalisationTest(chex.TestCase):
    """make_graph must relabel mixed-base topology JSONs to 0..N-1 in sorted-id order.

    TopologyBench-derived JSONs number nodes 1..N, but node labels are used directly as row
    indices into 0-based node-feature arrays (spectral features, source/dest one-hots) and as
    GNN senders/receivers — 1-based labels silently misroute every edge's features and drop
    messages to the highest node (segment_sum with num_segments=N).
    """

    def test_one_based_json_relabelled_to_zero_based(self):
        graph = make_graph("nsfnet_deeprmsa_directed")
        self.assertEqual(list(graph.nodes), list(range(14)))

    def test_shuffled_out_of_order_json_sorted_and_attrs_preserved(self):
        topology = {
            "directed": False,
            "multigraph": False,
            "graph": {},
            "nodes": [{"id": 3}, {"id": 1}, {"id": 4}, {"id": 2}],
            "links": [
                {"source": 1, "target": 2, "distance": 100},
                {"source": 2, "target": 3, "distance": 200},
                {"source": 3, "target": 4, "distance": 300},
                {"source": 4, "target": 1, "distance": 400},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(pathlib.Path(tmpdir) / "tiny_one_based.json", "w") as f:
                json.dump(topology, f)
            graph = make_graph("tiny_one_based", topology_directory=tmpdir)
        # Iteration order must equal sorted relabelled ids regardless of JSON node order
        self.assertEqual(list(graph.nodes), [0, 1, 2, 3])
        # Old id i maps to i-1; edge attributes must follow their (relabelled) endpoints
        self.assertEqual(graph.edges[(0, 1)]["distance"], 100)
        self.assertEqual(graph.edges[(1, 2)]["distance"], 200)
        self.assertEqual(graph.edges[(2, 3)]["distance"], 300)
        self.assertEqual(graph.edges[(0, 3)]["distance"], 400)

    def test_graph_tuple_senders_receivers_are_valid_row_indices(self):
        # Full pipeline on a 1-based topology: make() -> reset -> state.graph
        _, _, _, state, params = rsa_nsfnet_16_test_setup()
        n = int(params.num_nodes)
        senders = np.asarray(state.graph.senders)
        receivers = np.asarray(state.graph.receivers)
        self.assertEqual(min(senders.min(), receivers.min()), 0)  # was 1 pre-normalisation
        self.assertLess(int(max(senders.max(), receivers.max())), n)  # id N used to clamp/drop
        # senders/receivers must equal the sorted edge list of the relabelled graph, so
        # nodes[senders] gathers each edge's true endpoint feature row
        edges = np.array(sorted(make_graph("nsfnet_deeprmsa_undirected").edges))
        e = edges.shape[0]
        np.testing.assert_array_equal(senders[:e], edges[:, 0])
        np.testing.assert_array_equal(receivers[:e], edges[:, 1])
        # Undirected graphs append the reverse direction as a second block
        np.testing.assert_array_equal(senders[e:], edges[:, 1])
        np.testing.assert_array_equal(receivers[e:], edges[:, 0])


class UndirectedEdgeFeatureLayoutTest(chex.TestCase):
    """Undirected graph tuples must block-duplicate edge features ([f; f]), matching the
    [fwd..., bwd...] senders/receivers layout and the GNN readout's edges[: E] slice.
    jnp.repeat interleaved them ([f0, f0, f1, f1, ...]), attaching link j's features to the
    wrong edges for every j > 0."""

    def test_edge_features_match_links_in_both_blocks(self):
        _, _, _, state, params = rsa_nsfnet_16_test_setup()
        e = int(params.num_links)
        # Distinctive per-link feature rows: row j = (j + 1) / mean_service_holding_time
        departures = jnp.tile(
            jnp.arange(1.0, e + 1.0)[:, None], (1, int(params.link_resources))
        ).astype(state.link_slot_departure_array.dtype)
        state = state.replace(link_slot_departure_array=departures)
        expected_rows = np.asarray(departures, dtype=np.float32) / float(
            state.mean_service_holding_time
        )

        graph = init_graph_tuple(state, params, jnp.eye(int(params.num_nodes)))
        got = np.asarray(graph.edges, dtype=np.float32)  # ty: ignore[unresolved-attribute]
        self.assertEqual(got.shape[0], 2 * e)
        np.testing.assert_allclose(got[:e], expected_rows, rtol=1e-2)  # forward block, link order
        np.testing.assert_allclose(got[e:], expected_rows, rtol=1e-2)  # reverse block, link order

        # update_graph_tuple must produce the same layout for the carried graph
        state = state.replace(graph=graph)
        state = update_graph_tuple(state, params)
        got = np.asarray(state.graph.edges, dtype=np.float32)
        np.testing.assert_allclose(got[:e], expected_rows, rtol=1e-2)
        np.testing.assert_allclose(got[e:], expected_rows, rtol=1e-2)


class DeterministicReplayOrderTest(chex.TestCase):
    """Deterministic replay must serve list_of_requests in order from row 0.

    Regression: total_requests starts at -1 and the request was read before the
    increment, so every episode began with list_of_requests[-1] (wrapped index) and ran
    [n-1, 0, 1, ...]."""

    def test_requests_served_in_order(self):
        rows = [
            [0, 12, 1, 0.5, 5.0, 0.5],
            [1, 12, 2, 1.0, 5.0, 1.0],
            [2, 12, 3, 1.5, 5.0, 1.5],
        ]
        settings = dict(
            k=2,
            topology_name="4node",
            link_resources=8,
            values_bw=[12],
            env_type="rsa",
            deterministic_requests=True,
            list_of_requests=rows,
            incremental_loading=True,
        )
        env, params = make(settings, log_wrapper=False)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        step = jax.jit(env.step, static_argnums=(3,))
        served = [np.asarray(state.request_array, dtype=np.float32)]
        for _ in range(2):
            _, state, *_ = step(key, state, jnp.array(0), params)
            served.append(np.asarray(state.request_array, dtype=np.float32))
        for got, row in zip(served, rows):
            np.testing.assert_allclose(got, np.array([row[0], row[1], row[2]], dtype=np.float32))


class GenerateRequestBwProbsTest(chex.TestCase):
    """Tests for weighted bandwidth request sampling via values_bw_probs."""

    def _empirical_bw_freqs(self, state, params, values, n=10000):
        keys = jax.random.split(jax.random.PRNGKey(1), n)

        def _gen_bw(k):
            new_state = generate_request_rsa(k, state, params)
            return new_state.request_array[1]  # ty: ignore[unresolved-attribute]

        bws = jax.vmap(_gen_bw)(keys)
        return jnp.array([jnp.mean(bws == v) for v in values])

    def test_default_is_uniform(self):
        key, env, obs, state, params = rsa_nsfnet_16_test_setup()
        self.assertIsNone(params.values_bw_probs)
        freqs = self._empirical_bw_freqs(state, params, [1, 2, 3])
        chex.assert_trees_all_close(freqs, jnp.array([1 / 3, 1 / 3, 1 / 3]), atol=0.02)

    def test_weighted_sampling_matches_probs(self):
        key, env, obs, state, params = rsa_nsfnet_16_test_setup(values_bw_probs="0.7,0.2,0.1")
        chex.assert_trees_all_close(jnp.sum(params.values_bw_probs.val), 1.0)
        freqs = self._empirical_bw_freqs(state, params, [1, 2, 3])
        chex.assert_trees_all_close(freqs, jnp.array([0.7, 0.2, 0.1]), atol=0.02)

    def test_relative_weights_are_normalised(self):
        key, env, obs, state, params = rsa_nsfnet_16_test_setup(values_bw_probs="7,2,1")
        chex.assert_trees_all_close(
            params.values_bw_probs.val, jnp.array([0.7, 0.2, 0.1], dtype=jnp.float32)
        )

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rsa_nsfnet_16_test_setup(values_bw_probs="0.5,0.5")

    def test_negative_prob_raises(self):
        with self.assertRaises(ValueError):
            rsa_nsfnet_16_test_setup(values_bw_probs="0.5,0.6,-0.1")


class GenerateRequestDistributionTest(chex.TestCase):
    """The inverse-CDF request sampler (pre-computed traffic CDF + fused uniform draw)
    must reproduce the traffic-matrix source-dest distribution, the bandwidth
    distribution, and the exponential arrival/holding time means."""

    def _sample_requests(self, state, params, n, seed=2):
        keys = jax.random.split(jax.random.PRNGKey(seed), n)

        def _gen(k):
            s = generate_request_rsa(k, state, params)
            return s.request_array, s.arrival_time, s.holding_time  # ty: ignore[unresolved-attribute]

        return jax.vmap(_gen)(keys)

    def _empirical_pair_freqs(self, requests, num_nodes):
        source = np.asarray(requests[:, 0], dtype=np.int64)
        dest = np.asarray(requests[:, 2], dtype=np.int64)
        counts = np.zeros((num_nodes, num_nodes))
        np.add.at(counts, (source, dest), 1)
        return counts / counts.sum()

    def test_source_dest_bw_and_time_distributions(self):
        key, env, obs, state, params = rsa_nsfnet_16_test_setup()
        # The fixed uniform traffic matrix must have a pre-computed CDF
        self.assertIsNotNone(params.traffic_cdf)
        n = 100_000
        requests, arrival_times, holding_times = self._sample_requests(state, params, n)

        # Source-dest frequencies vs traffic matrix probabilities (folded to
        # upper-triangular for the undirected graph, mirroring generate_request_rsa)
        tm = np.asarray(state.traffic_matrix, dtype=np.float64)
        probs = tm / tm.sum()
        if not params.directed_graph:
            probs = np.triu(probs + probs.T, k=1)
        freqs = self._empirical_pair_freqs(np.asarray(requests), params.num_nodes)
        # Per-pair binomial std at p ~ 1/91 is ~3e-4 for n=1e5; 1.5e-3 is > 4 sigma
        np.testing.assert_allclose(freqs, probs, atol=1.5e-3)
        # No self-loops or lower-triangular (unfolded) pairs
        self.assertEqual(freqs[np.isclose(probs, 0)].sum(), 0)

        # Bandwidth values sampled uniformly from values_bw
        bws = np.asarray(requests[:, 1])
        for v in [1, 2, 3]:
            np.testing.assert_allclose((bws == v).mean(), 1 / 3, atol=0.01)

        # Exponential means: arrival ~ 1/arrival_rate, holding ~ mean_service_holding_time
        np.testing.assert_allclose(
            np.asarray(arrival_times).mean(), 1 / np.asarray(state.arrival_rate), rtol=0.02
        )
        np.testing.assert_allclose(
            np.asarray(holding_times).mean(), params.mean_service_holding_time, rtol=0.02
        )

    def test_random_traffic_fallback_matches_state_matrix(self):
        # random_traffic has no pre-computed CDF: the per-step cumsum fallback must
        # sample from the (randomly initialised) state.traffic_matrix
        key, env, obs, state, params = rsa_nsfnet_16_test_setup(random_traffic=True)
        self.assertIsNone(params.traffic_cdf)
        n = 100_000
        requests, _, _ = self._sample_requests(state, params, n, seed=3)
        tm = np.asarray(state.traffic_matrix, dtype=np.float64)
        probs = tm / tm.sum()
        if not params.directed_graph:
            probs = np.triu(probs + probs.T, k=1)
        freqs = self._empirical_pair_freqs(np.asarray(requests), params.num_nodes)
        np.testing.assert_allclose(freqs, probs, atol=2e-3)


class MakeGraphUnknownTopologyTest(chex.TestCase):
    """make_graph must fail with a helpful ValueError (not a raw FileNotFoundError)
    listing close-match suggestions when --topology_name doesn't match a bundled JSON."""

    def test_typo_raises_value_error_with_suggestion(self):
        with self.assertRaisesRegex(ValueError, "Unknown topology 'nfsnet_deeprmsa_directed'"):
            make_graph("nfsnet_deeprmsa_directed")
        try:
            make_graph("nfsnet_deeprmsa_directed")
        except ValueError as e:
            self.assertIn("nsfnet_deeprmsa_directed", str(e))

    def test_no_close_match_still_helpful(self):
        with self.assertRaisesRegex(ValueError, "topologies are available in"):
            make_graph("zzzzzz_not_a_topology")

    def test_custom_topology_directory_used_for_suggestions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(pathlib.Path(tmpdir) / "mynet.json", "w") as f:
                json.dump({"nodes": [], "links": []}, f)
            with self.assertRaisesRegex(ValueError, "mynet"):
                make_graph("mynett", topology_directory=tmpdir)


def fixture(mode):
    c, env, p, s = setup("rmsa_gn_model", "lifetime_snr")
    mods = p.modulations_array.val.at[:, 2].set(-100.0)
    p = p.replace(gn_mask_mode=mode, modulations_array=HashableArrayWrapper(mods))
    return c, env, p, s


def reference_mask(s, p):
    """Unpruned ordinary-JAX all_slots mask (every format evaluated at every start)."""
    jax_params = p.replace(gn_kernel_backend="jax")
    return jax.jit(partial(ef._mask_slots_rmsa_gn_model, prune_formats=False), static_argnums=1)(
        s, jax_params, s.request_array
    )


def with_mask(env, p, s):
    mask, full, mods = env.action_mask(s, p)
    return s.replace(link_slot_mask=mask, full_link_slot_mask=full, mod_format_mask=mods)


class TestGetExistingSnrRequirements:
    @staticmethod
    def test_reserve_caps_at_fresh_headroom_and_never_relaxes_the_physical_minimum():
        _, _, p, s = setup("rmsa_gn_model", "lifetime_snr")
        p = p.replace(snr_margin=1.0, existing_snr_margin=1.0)
        s = s.replace(
            channel_centre_bw_array=jnp.ones_like(s.channel_centre_bw_array).at[:, -1].set(0),
            modulation_format_index_array=jnp.zeros_like(s.modulation_format_index_array),
            # Live lightpaths depart in the future (none is held for release).
            link_slot_departure_array=jnp.full_like(s.link_slot_departure_array, 100.0),
        )
        threshold = p.modulations_array.val[0, 2]
        headroom = jnp.array([-1.0, 0.0, 0.2, 1.0, 3.0, 50.0, 0.5, 0.0])
        baseline = jnp.broadcast_to(threshold + headroom, s.channel_centre_bw_array.shape)
        actual = get_existing_snr_requirements(s, p, baseline)
        expected = (
            jnp.broadcast_to(
                threshold + jnp.array([0.0, 0.0, 0.2, 1.0, 1.0, 1.0, 0.5, 0.0]), actual.shape
            )
            .at[:, -1]
            .set(-jnp.inf)
        )
        np.testing.assert_allclose(actual, expected)
        disabled = get_existing_snr_requirements(s, p.replace(existing_snr_margin=0), baseline)
        np.testing.assert_array_equal(
            disabled, jnp.where(s.channel_centre_bw_array > 0, threshold, -jnp.inf)
        )


def retained_reserve_fixture():
    _, env, p, s = setup("rmsa_gn_model", "lifetime_snr")
    p = p.replace(gn_mask_mode="all_slots", raman_gain_slope=0.0)
    mods = p.modulations_array.val.at[:, 2].set(-100.0)
    p = p.replace(modulations_array=HashableArrayWrapper(mods))
    s = s.replace(request_array=jnp.array([0, 100, 1]))
    s = with_mask(env, p, s)
    s = env.step(jax.random.PRNGKey(2), s, jnp.int32(0), p)[1]
    active = s.channel_centre_bw_array > 0
    s = s.replace(
        request_array=jnp.array([0, 100, 1]),
        modulation_format_index_array=jnp.where(active, 0, -1),
    )
    fresh = s.replace(link_snr_array=get_snr_link_array(s, p))
    baseline = get_lightpath_snr(fresh, p)
    old_snr = float(np.asarray(baseline)[np.asarray(active)].min())
    p = p.replace(modulations_array=HashableArrayWrapper(mods.at[0, 2].set(old_snr - 0.1)))
    # The reserve must use fresh physics, even for a heuristic with a stale cache.
    s = s.replace(link_snr_array=jnp.full_like(s.link_snr_array, 1e8))
    return env, p, s


class TestMaskSlotsRmsaGnModel:
    @staticmethod
    def test_new_insertion_cannot_break_an_existing_modulation_minimum():
        env, p, s = retained_reserve_fixture()
        active = s.channel_centre_bw_array > 0
        plain = mask_slots_rmsa_gn_model(s, p, s.request_array)
        action = jnp.int32(1)
        tentative = implement_action_rmsa_gn_model(plain, env.process_action(plain, action, p), p)

        def old_snr(state):
            fresh = state.replace(link_snr_array=get_snr_link_array(state, p))
            return float(np.asarray(get_lightpath_snr(fresh, p))[np.asarray(active)].min())

        before, after = old_snr(s), old_snr(tentative)
        assert before > after
        # Put the old format's physical threshold between its pre/post-insertion
        # SNR. The new service uses another format with an unchanged threshold.
        mods = p.modulations_array.val.at[0, 2].set((before + after) / 2)
        p = p.replace(modulations_array=HashableArrayWrapper(mods))
        physical = mask_slots_rmsa_gn_model(s, p, s.request_array)
        assert not physical.link_slot_mask[action]
        assert physical.link_slot_mask[8]  # A disjoint candidate remains feasible.
        # Admission independently rejects the harmful insertion even if passed
        # the old mask, so masking and the step both protect the hard minimum.
        rejected = env.step(jax.random.PRNGKey(4), plain, action, p)[1]
        assert rejected.accepted_services == s.accepted_services

    @staticmethod
    @pytest.mark.parametrize("backend", ["jax", "pallas"])
    def test_retained_reserve_matches_admission_and_allows_unchanged_depleted_paths(backend):
        env, p, s = retained_reserve_fixture()
        p = p.replace(gn_kernel_backend=backend, gn_mask_screen_batch_size=8)
        plain = mask_slots_rmsa_gn_model(s, p, s.request_array)
        reserve_p = p.replace(existing_snr_margin=0.5)
        protected = mask_slots_rmsa_gn_model(s, reserve_p, s.request_array)
        assert plain.link_slot_mask[1] and not protected.link_slot_mask[1]
        # Path 1 shares no links with the old lightpath. Its 0.1 dB headroom
        # is below the target reserve, yet remains above its physical minimum.
        assert protected.link_slot_mask[8]
        rejected = env.step(jax.random.PRNGKey(4), plain, jnp.int32(1), reserve_p)[1]
        accepted = env.step(jax.random.PRNGKey(4), protected, jnp.int32(8), reserve_p)[1]
        assert rejected.accepted_services == s.accepted_services
        assert accepted.accepted_services == s.accepted_services + 1
        # A physically failing old connection is still rejected globally.
        bad_mods = reserve_p.modulations_array.val.at[0, 2].add(0.2)
        failing_p = reserve_p.replace(modulations_array=HashableArrayWrapper(bad_mods))
        failing = mask_slots_rmsa_gn_model(s, failing_p, s.request_array)
        assert not failing.link_slot_mask.any()

    @staticmethod
    def test_all_slots_exposes_interior_starts_and_respects_power():
        _, env, p, s = fixture("all_slots")
        full = with_mask(env, p, s)
        restricted = with_mask(env, p.replace(gn_mask_mode="ff_lf"), s)
        assert full.link_slot_mask.shape == (p.k_paths * p.link_resources,)
        assert full.link_slot_mask[3] and not restricted.link_slot_mask[3]
        assert full.link_slot_mask.sum() > restricted.link_slot_mask.sum()
        rejected = with_mask(env, p.replace(max_power_per_fibre=1e-12), s)
        assert not rejected.link_slot_mask.any()

    @staticmethod
    @pytest.mark.parametrize("batch_size", [1, 3, 8])
    def test_full_format_pruning_preserves_mask_and_selected_format(batch_size):
        _, env, p, s = fixture("all_slots")
        # Keep tied efficiencies and force the last/highest format to fail. Both
        # traversal orders must select the first valid format on an SE tie.
        mods = p.modulations_array.val.at[-1, 2].set(1000.0)
        mods = mods.at[1, 1].set(mods[0, 1])
        p = p.replace(gn_mask_batch_size=batch_size, modulations_array=HashableArrayWrapper(mods))
        s = s.replace(link_slot_array=s.link_slot_array.at[:, 4].set(-1))
        reference = reference_mask(s, p)
        actual = mask_slots_rmsa_gn_model(s, p, s.request_array)
        np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
        np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)
        assert actual.link_slot_mask.any()

    @staticmethod
    @pytest.mark.parametrize("backend", ["jax", "pallas"])
    @pytest.mark.parametrize("margin", [0.0, 0.5])
    def test_full_mask_matches_exhaustive_admission_oracle(backend, margin):
        """Every (path, start, format) is stepped from the same state.

        snr_margin applies to mask selection only (admission checks the bare
        thresholds), so with zero margin the mask must allow exactly the starts
        some format admits and select the highest-SE admitted format; with a
        margin every allowed start must still be admitted.
        """
        _, env, p, s = setup("rmsa_gn_model", "lifetime_snr")
        p = p.replace(gn_mask_mode="all_slots", gn_kernel_backend=backend, snr_margin=margin)
        # Partially load the network so existing lightpaths constrain admission.
        for key in (21, 22):
            s = with_mask(env, p, s)
            action = np.flatnonzero(np.asarray(s.link_slot_mask))[0]
            s = env.step(jax.random.PRNGKey(key), s, jnp.int32(action), p)[1]
        s = with_mask(env, p, s)
        num_mods = p.modulations_array.val.shape[0]
        actions = jnp.arange(p.k_paths * p.link_resources)

        def admitted(action, mod):
            trial = s.replace(
                mod_format_mask=s.mod_format_mask.at[action].set(
                    mod.astype(s.mod_format_mask.dtype)
                )
            )
            after = env.step(jax.random.PRNGKey(0), trial, action, p)[1]
            return after.accepted_services > s.accepted_services

        table = jax.jit(
            jax.vmap(lambda a: jax.vmap(lambda m: admitted(a, m))(jnp.arange(num_mods)))
        )(actions)
        table = np.asarray(table)
        mask = np.asarray(s.link_slot_mask[: len(actions)]) > 0
        assert mask.sum() > 1 and (~mask).sum() > 1
        chosen = np.asarray(s.mod_format_mask[: len(actions)]).astype(int)
        assert np.all(table[np.flatnonzero(mask), chosen[mask]])
        if margin > 0:
            return
        np.testing.assert_array_equal(mask, table.any(axis=1))
        se = np.asarray(p.modulations_array.val[:, 1])
        for action in np.flatnonzero(mask):
            best = max(np.flatnonzero(table[action]), key=lambda m: (se[m], -m))
            assert int(s.mod_format_mask[action]) == best

    @staticmethod
    @pytest.mark.parametrize("screen_batch", [1, 8, 512])
    def test_pallas_endpoint_screen_preserves_full_mask(screen_batch):
        _, _, p, s = setup("rmsa_gn_model", "lifetime_snr")
        p = p.replace(gn_mask_mode="all_slots", gn_mask_batch_size=3)
        reference = reference_mask(s, p)
        kernel = p.replace(gn_kernel_backend="pallas", gn_mask_screen_batch_size=screen_batch)
        actual = mask_slots_rmsa_gn_model(s, kernel, s.request_array)
        np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
        np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)

    @staticmethod
    @pytest.mark.parametrize("pruning", [True, False])
    def test_screened_power_reuse_preserves_the_exact_power_limit(pruning):
        _, _, p, state = fixture("all_slots")
        p = p.replace(gn_mask_batch_size=3)
        power = p.slot_launch_power_array.val[0]
        # Include both sides of the exact one-channel limit and the boundary itself.
        for limit in (jnp.nextafter(power, jnp.float32(0)), power, jnp.nextafter(power, jnp.inf)):
            q = p.replace(max_power_per_fibre=float(limit))
            reference = reference_mask(state, q)
            kernel = q.replace(gn_kernel_backend="pallas")
            actual = jax.jit(
                partial(ef._mask_slots_rmsa_gn_model, prune_formats=pruning), static_argnums=1
            )(state, kernel, state.request_array)
            np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
            np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)

    @staticmethod
    def test_screened_power_reuse_checks_unaffected_overloaded_links():
        _, _, p, state = fixture("all_slots")
        p = p.replace(gn_mask_batch_size=3)
        state = state.replace(
            channel_power_array=state.channel_power_array.at[0, 0].set(1.0),
            channel_centre_bw_array=state.channel_centre_bw_array.at[0, 0].set(p.slot_size),
            path_index_array=state.path_index_array.at[0, 0].set(0),
            link_slot_array=state.link_slot_array.at[0, 0].set(1),
        )
        p = p.replace(max_power_per_fibre=0.1)
        reference = reference_mask(state, p)
        actual = mask_slots_rmsa_gn_model(
            state, p.replace(gn_kernel_backend="pallas"), state.request_array
        )
        assert not actual.link_slot_mask.any()
        np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
        np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)

    @staticmethod
    def test_pallas_mask_matches_reference_across_admissions():
        _, env, p, state = setup("rmsa_gn_model", "lifetime_snr")
        p = p.replace(gn_mask_mode="all_slots", gn_mask_batch_size=3)
        kernel = p.replace(gn_kernel_backend="pallas")
        for seed in (28, 29):
            reference = mask_slots_rmsa_gn_model(state, p, state.request_array)
            actual = mask_slots_rmsa_gn_model(state, kernel, state.request_array)
            np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
            np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)
            _, state, *_ = env.step(
                jax.random.PRNGKey(seed), actual, jnp.argmax(actual.link_slot_mask), kernel
            )

    @staticmethod
    @pytest.mark.parametrize("chunk", [1, 2])
    def test_mask_environment_chunks_preserve_outer_vmap(chunk):
        _, _, p, s = fixture("all_slots")
        p = p.replace(gn_mask_batch_size=3)
        # Different candidate counts, plus a non-divisible final environment chunk.
        states = [s.replace(link_slot_array=s.link_slot_array.at[:, :i].set(-1)) for i in (0, 3, 7)]
        states = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
        reference = jax.vmap(lambda st: mask_slots_rmsa_gn_model(st, p, st.request_array))(states)
        q = p.replace(gn_mask_env_batch_size=chunk)
        actual = jax.vmap(lambda st: mask_slots_rmsa_gn_model(st, q, st.request_array))(states)
        np.testing.assert_array_equal(actual.link_slot_mask, reference.link_slot_mask)
        np.testing.assert_array_equal(actual.mod_format_mask, reference.mod_format_mask)

    @staticmethod
    def test_selected_slot_falls_back_to_physically_valid_modulation():
        _, env, p, s = fixture("selected_slot")
        mods = p.modulations_array.val.at[1:, 2].set(1000.0)
        p = p.replace(modulations_array=HashableArrayWrapper(mods))
        s = with_mask(env, p, s)
        assert s.link_slot_mask[3]
        _, next_state, reward, *_ = env.step(jax.random.PRNGKey(8), s, jnp.array(3), p)
        assert reward >= 0 and next_state.accepted_services == 1
        occupied = np.asarray(next_state.channel_power_array) > 0
        assert occupied.any()
        assert np.all(np.asarray(next_state.modulation_format_index_array)[occupied] == 0)

    @staticmethod
    def test_selected_slot_resolution_under_env_vmap():
        """Environments needing different numbers of fallback formats batch correctly."""
        _, env, p, s = fixture("selected_slot")
        mods = p.modulations_array.val.at[1:, 2].set(1000.0)
        p = p.replace(modulations_array=HashableArrayWrapper(mods))
        s = with_mask(env, p, s)
        actions = jnp.array([3, 5, 3])
        states = jax.tree.map(lambda x: jnp.stack([x, x, x]), s)
        step = jax.vmap(lambda st, a: env.step(jax.random.PRNGKey(8), st, a, p)[1])
        batched = step(states, actions)
        for i, action in enumerate(np.asarray(actions)):
            single = env.step(jax.random.PRNGKey(8), s, jnp.array(action), p)[1]
            np.testing.assert_array_equal(
                batched.modulation_format_index_array[i], single.modulation_format_index_array
            )
            assert batched.accepted_services[i] == single.accepted_services

    @staticmethod
    def test_selected_slot_physical_rejection_rolls_back_and_counts_blocking():
        _, env, p, s = fixture("selected_slot")
        mods = p.modulations_array.val.at[:, 2].set(1000.0)
        p = p.replace(modulations_array=HashableArrayWrapper(mods))
        s = with_mask(env, p, s)
        assert s.link_slot_mask[3]  # Physical feasibility is intentionally deferred.
        _, next_state, reward, *_ = env.step(jax.random.PRNGKey(9), s, jnp.array(3), p)
        assert reward < 0 and next_state.accepted_services == 0
        assert next_state.blocked_snr == 1 and next_state.blocked_spectrum == 0
        np.testing.assert_array_equal(next_state.link_slot_array, s.link_slot_array)
        np.testing.assert_array_equal(next_state.channel_power_array, s.channel_power_array)

    @staticmethod
    def test_selected_slot_rejects_forced_band_gap_action():
        _, env, p, s = fixture("selected_slot")
        s = s.replace(link_slot_array=s.link_slot_array.at[:, 3].set(-1))
        s = with_mask(env, p, s)
        assert not s.link_slot_mask[3]
        _, ns, _, *_ = env.step(jax.random.PRNGKey(10), s, jnp.array(3), p)
        assert ns.blocked_spectrum == 1 and ns.accepted_services == 0
        np.testing.assert_array_equal(ns.link_slot_array, s.link_slot_array)


def departure_fixture(relative):
    """Two lightpaths on one path; switching off the high-frequency one lowers the
    low-frequency survivor's SNR (ISRS tilt / amplifier gain), as in issue #67."""
    env, p = make(
        dict(
            env_type="rmsa_gn_model",
            topology_name="5node_directed",
            k=2,
            link_resources=16,
            slot_size=50,
            guardband=0,
            values_bw=[100],
            max_requests=50,
            include_no_op=False,
            power_per_channel=0,
            gn_mask_mode="all_slots",
            incremental_loading=True,
            relative_arrival_times=relative,
        ),
        log_wrapper=False,
    )
    # A stronger Raman slope makes the departure-induced dip ~0.1 dB.
    p = cast(RMSAGNModelEnvParams, p)
    p = p.replace(raman_gain_slope=p.raman_gain_slope * 100)
    _, s = env.reset(jax.random.PRNGKey(0), p)
    for action in (0, 14):
        s = s.replace(request_array=jnp.array([0, 100, 1]))
        s = with_mask(env, p, s)
        s = env.step(jax.random.PRNGKey(action), s, jnp.int32(action), p)[1]
    assert int(s.accepted_services) == 2
    survivor = (s.channel_centre_bw_array != 0) & (jnp.arange(16) < 8)[None, :]
    departing = (s.channel_centre_bw_array != 0) & ~survivor
    # The departing lightpath has its own format so its threshold can differ.
    num_mods = p.modulations_array.val.shape[0]
    s = s.replace(
        modulation_format_index_array=jnp.where(departing, num_mods - 1, 0).astype(
            s.modulation_format_index_array.dtype
        )
    )

    def fresh_snr(state, params):
        state = state.replace(link_snr_array=get_snr_link_array(state, params))
        return np.asarray(get_lightpath_snr(state, params))

    before = fresh_snr(s, p)[np.asarray(survivor)].min()
    link, slot = np.argwhere(np.asarray(lightpath_representatives(s, p) & departing))[0]
    after = fresh_snr(ef._switch_off_lightpath(s, link, slot), p)[np.asarray(survivor)].min()
    assert after < before - 0.05
    now = 0.0 if relative else float(jnp.ravel(s.current_time)[0])
    s = s.replace(
        link_slot_departure_array=jnp.where(
            departing, now + 1e-6, jnp.where(survivor, now + 1e6, s.link_slot_departure_array)
        ).astype(s.link_slot_departure_array.dtype)
    )
    # Survivor threshold between its SNR before and after the departure; the
    # departing lightpath's own threshold is unattainable (it no longer matters).
    mods = p.modulations_array.val.at[0, 2].set((before + after) / 2).at[-1, 2].set(1000.0)
    p = p.replace(incremental_loading=False, modulations_array=HashableArrayWrapper(mods))
    return env, p, s, survivor, departing, fresh_snr, (before, after)


class TestSafeReleaseDepartures:
    @staticmethod
    @pytest.mark.parametrize("relative", [True, False])
    def test_immediate_departure_pushes_survivor_below_threshold(relative):
        _, p, s, survivor, _, fresh_snr, _ = departure_fixture(relative)
        p = p.replace(departure_policy="immediate")
        nxt = generate_request_rsa(jax.random.PRNGKey(1), s, p)
        threshold = float(p.modulations_array.val[0, 2])
        assert int(jnp.sum(nxt.channel_centre_bw_array != 0)) == int(jnp.sum(survivor))
        assert fresh_snr(nxt, p)[np.asarray(survivor)].min() < threshold
        # Outage: with fresh physics every later admission now fails the global check.
        fresh = nxt.replace(link_snr_array=get_snr_link_array(nxt, p))
        assert bool(check_snr_sufficient(fresh, p))

    @staticmethod
    @pytest.mark.parametrize("relative", [True, False])
    def test_safe_release_holds_until_switching_off_is_safe(relative):
        env, p, s, survivor, departing, fresh_snr, (before, after) = departure_fixture(relative)
        safe = p.replace(departure_policy="safe_release")
        held = generate_request_rsa(jax.random.PRNGKey(1), s, safe)
        threshold = float(safe.modulations_array.val[0, 2])
        # The departing lightpath stays lit, so the survivor keeps its SNR.
        np.testing.assert_array_equal(held.channel_centre_bw_array != 0, survivor | departing)
        assert int(jnp.sum(held_lightpath_mask(held, safe))) == int(jnp.sum(departing))
        assert fresh_snr(held, safe)[np.asarray(survivor)].min() >= threshold
        # Held lightpaths impose no SNR requirement (their own is unattainable here)
        # but still occupy spectrum.
        fresh = held.replace(link_snr_array=get_snr_link_array(held, safe))
        assert not bool(check_snr_sufficient(fresh, safe))
        mask, _, _ = env.action_mask(held.replace(request_array=jnp.array([0, 100, 1])), safe)
        held_slots = np.flatnonzero(np.asarray(departing).any(axis=0))  # Path 0 actions.
        assert not np.asarray(mask)[held_slots].any() and np.asarray(mask)[8]
        # Once switching off is safe, the next arrival releases it.
        relaxed_mods = safe.modulations_array.val.at[0, 2].set(after - 0.05)
        relaxed = safe.replace(modulations_array=HashableArrayWrapper(relaxed_mods))
        released = generate_request_rsa(jax.random.PRNGKey(2), held, relaxed)
        np.testing.assert_array_equal(released.channel_centre_bw_array != 0, survivor)
        assert not bool(held_lightpath_mask(released, relaxed).any())

    @staticmethod
    def test_safe_release_step_reports_held_lightpaths_and_admits():
        env, p, s, survivor, departing, _, _ = departure_fixture(True)
        safe = p.replace(departure_policy="safe_release")
        s = generate_request_rsa(jax.random.PRNGKey(1), s, safe).replace(
            request_array=jnp.array([0, 100, 1])
        )
        s = with_mask(env, safe, s)
        _, nxt, _, _, _, info = env.step(
            jax.random.PRNGKey(3), s, jnp.argmax(s.link_slot_mask), safe
        )
        assert int(info["held_lightpaths"]) == 1
        assert int(nxt.accepted_services) == int(s.accepted_services) + 1

    @staticmethod
    @pytest.mark.parametrize("margin", [0.0, 0.5])
    def test_safe_release_keeps_the_link_snr_cache_fresh(margin):
        """Release checks re-solve only the departing path's links, which is exact
        only if the cached link SNR is fresh after every step; pin that invariant
        through admissions, blocks and departures."""
        env, p = make(
            dict(
                env_type="rmsa_gn_model",
                topology_name="5node_directed",
                k=2,
                link_resources=16,
                slot_size=50,
                guardband=0,
                values_bw=[100, 200],
                max_requests=200,
                include_no_op=False,
                power_per_channel=0,
                load=8,
                mean_service_holding_time=4,
                existing_snr_margin=margin,
                snr_margin=max(margin, 0.5),
            ),
            log_wrapper=False,
        )
        p = cast(RMSAGNModelEnvParams, p)
        assert p.departure_policy == "safe_release"
        _, s = env.reset(jax.random.PRNGKey(5), p)
        step = jax.jit(lambda key, st, a: env.step(key, st, a, p)[1])
        accepted = blocked = 0
        for i in range(40):
            s = with_mask(env, p, s)
            before = int(s.accepted_services)
            s = step(jax.random.PRNGKey(i), s, jnp.argmax(s.link_slot_mask))
            accepted += int(s.accepted_services) - before
            blocked += 1 - (int(s.accepted_services) - before)
            np.testing.assert_allclose(
                s.link_snr_array, get_snr_link_array(s, p), rtol=1e-5, atol=1e-6
            )
        # Most admitted lightpaths have departed through the release path.
        present = int(jnp.sum(lightpath_representatives(s, p)))
        assert accepted > 20 and present < accepted // 2

    @staticmethod
    def test_safe_release_is_the_default_for_hard_rmsa_gn():
        base = dict(env_type="rmsa_gn_model", topology_name="5node_directed", link_resources=8)
        _, params = make(base, log_wrapper=False)
        assert cast(RMSAGNModelEnvParams, params).departure_policy == "safe_release"
        _, params = make(base | {"departure_policy": "immediate"}, log_wrapper=False)
        assert cast(RMSAGNModelEnvParams, params).departure_policy == "immediate"
        # The differentiable relaxation keeps immediate departures.
        _, params = make(base | {"differentiable": True}, log_wrapper=False)
        assert cast(RMSAGNModelEnvParams, params).departure_policy == "immediate"
        # Other environment types have no departure policy.
        make(dict(env_type="rsa_gn_model", topology_name="5node_directed"), log_wrapper=False)


class TestGetSnrLinkArray:
    @staticmethod
    def test_pallas_backend_matches_jax_with_mixed_channels():
        _, _, p, s = setup("rmsa_gn_model", "lifetime_snr")
        f = jnp.linspace(-1000.0, 1000.0, p.link_resources)
        power = jnp.array([1e-4, 2e-4, 0, 1e-4, 3e-4, 0, 1e-4, 2e-4])
        s = s.replace(
            channel_power_array=jnp.broadcast_to(power, s.channel_power_array.shape),
            channel_centre_bw_array=jnp.full_like(s.channel_centre_bw_array, 25.0),
            channel_centre_freq_array=jnp.broadcast_to(f, s.channel_centre_freq_array.shape),
        )
        kernel = p.replace(gn_kernel_backend="pallas")
        expected = get_snr_link_array(s, p)
        np.testing.assert_allclose(get_snr_link_array(s, kernel), expected, rtol=1e-5, atol=1e-5)
        links = jnp.array([3, 1, 3])
        np.testing.assert_allclose(
            get_snr_link_array(s, kernel, links), expected[links], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            get_snr_link_array_fused(s, kernel), get_snr_link_array_fused(s, p), rtol=1e-5
        )


class TestGetObsTransformer:
    @staticmethod
    @pytest.mark.parametrize("env_type", ["rsa_gn_model", "rmsa_gn_model"])
    @pytest.mark.parametrize("mode", ["snr", "occupancy_snr", "departure_snr"])
    def test_snr_feature_values_shapes_and_policy_gradients(env_type, mode):
        config, _, params, state = setup(env_type, mode)
        snr = state.link_snr_array.at[0, :5].set(jnp.array([10.0, 100.0, jnp.nan, jnp.inf, -1.0]))
        power = state.channel_power_array.at[0, :5].set(1e-4)
        # Empty slots must stay zero even when their cache contains a positive value.
        snr = snr.at[0, 5].set(100.0)
        state = state.replace(link_snr_array=snr, channel_power_array=power)
        tokens = get_obs_transformer(state, params)
        start = config.num_wire_features + (params.link_resources if mode != "snr" else 0)
        np.testing.assert_allclose(
            tokens[0, start : start + 6], [1 / 3, 2 / 3, 0, 1, 0, 0], atol=1e-6
        )
        request_cols = 6 if mode == "departure_snr" else 5
        expected_width = (
            config.num_wire_features
            + params.link_resources * (1 if mode == "snr" else 2)
            + 2
            + request_cols
        )
        assert tokens.shape == (params.num_links, expected_width)
        assert np.all(np.isfinite(tokens))
        model = init_network(config, jax.random.PRNGKey(4))

        @eqx.filter_value_and_grad
        def loss(network):
            pi, value = network(state, params)
            assert pi.logits.shape == (config.ACTION_DIM,)
            return jnp.mean(pi.logits**2) + jnp.mean(value**2)

        value, gradient = loss(model)
        assert np.isfinite(value)
        assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(gradient))


class TestGenerateRequestRsaGn:
    @staticmethod
    @pytest.mark.parametrize("env_type", ["rsa_gn_model", "rmsa_gn_model"])
    def test_snr_cache_is_refreshed_after_departures(env_type):
        _, env, params, state = setup(env_type, "occupancy_snr")
        mask_output = env.action_mask(state, params)
        state = state.replace(link_slot_mask=mask_output[0], full_link_slot_mask=mask_output[1])
        if env_type == "rmsa_gn_model":
            state = state.replace(mod_format_mask=mask_output[2])
            action = jnp.array([jnp.argmax(mask_output[0]), 0])
        else:
            action = jnp.argmax(mask_output[0])
        _, state, *_ = env.step(jax.random.PRNGKey(5), state, action, params)
        assert np.any(np.asarray(state.channel_power_array) > 0)
        # Force expiration and poison the old cache: the next request must rebuild it.
        state = state.replace(
            link_slot_departure_array=jnp.zeros_like(state.link_slot_departure_array),
            link_snr_array=jnp.full_like(state.link_snr_array, 123),
        )
        params = params.replace(incremental_loading=False)
        next_state = generate_request_rsa(jax.random.PRNGKey(6), state, params)
        assert not np.any(np.asarray(next_state.channel_power_array) > 0)
        np.testing.assert_allclose(
            next_state.link_snr_array, get_snr_link_array(next_state, params)
        )


class TestInitTransceiverAmplifierNoiseArrays:
    @staticmethod
    def test_noise_lookup_preserves_boundaries_gaps_and_first_overlapping_band(tmp_path):
        csv = tmp_path / "noise.csv"
        csv.write_text(
            "band,lmin,lmax,fmin,fmax,nf,trx,express,ad,rnf\n"
            "a,0,0,10,20,1,2,3,4,5\n"
            "b,0,0,20,30,6,7,8,9,10\n"
        )
        frequencies = np.array([9, 10, 19.999, 20, 20.001, 30, 31], np.float64)
        actual = ef.init_transceiver_amplifier_noise_arrays(
            len(frequencies), 1550e-9, 12.5, str(csv), frequencies
        )
        expected = np.array(
            [
                [0] * 5,
                [2, 1, 3, 4, 5],
                [2, 1, 3, 4, 5],
                [2, 1, 3, 4, 5],
                [7, 6, 8, 9, 10],
                [7, 6, 8, 9, 10],
                [0] * 5,
            ],
            np.float32,
        )
        np.testing.assert_array_equal(np.stack(actual, axis=1), expected)

    @staticmethod
    @pytest.mark.parametrize("source", ["default", "jax", "numpy"])
    def test_noise_lookup_retains_legacy_scalar_boundary_rounding(tmp_path, source):
        csv = tmp_path / "noise.csv"
        csv.write_text(
            "band,lmin,lmax,fmin,fmax,nf,trx,express,ad,rnf\na,0,0,193414.501,193415,1,2,3,4,5\n"
        )
        frequencies = np.array([193414.5], np.float32)
        supplied = {"default": None, "jax": jnp.asarray(frequencies), "numpy": frequencies}[source]
        actual = ef.init_transceiver_amplifier_noise_arrays(
            1, 299792458 / (193414.5 * 1e9), 12.5, str(csv), supplied
        )
        # Legacy default/JAX scalars round the CSV bound to float32 before comparing.
        expected = [0] * 5 if source == "numpy" else [2, 1, 3, 4, 5]
        np.testing.assert_array_equal(np.stack(actual, axis=1), [expected])


if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
