"""
Unit tests for `env_funcs.py`.
See `chex` github and docs for info on test framework.

Key points:
There is a class for each function under test.
chex.all_variants() decorator runs the test once for each variant (e.g. jitted, non-jitted, pmapped, etc.) of the function under test.
parameterized.named_parameters() decorator runs the test once for each set of parameters passed to the function under test.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import pathlib

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from xlron.environments.dataclasses import *
from xlron.environments.env_funcs import *
from xlron.environments.make_env import make
from xlron.environments.rsa.rsa import *
from xlron.environments.wrappers import *


def keys_test_setup():
    rng = jax.random.PRNGKey(0)  # N.B. all test rely on 0 seed for reproducibility
    rng, key_init, key_reset, key_policy, key_step = jax.random.split(rng, 5)
    return rng, key_init, key_reset, key_policy, key_step


def settings_rwa_4node():
    return dict(
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
    )


# Module-level caches for expensive make() + reset() calls.
# Each cache stores (env, params) for a specific config - the expensive part.
# obs and state are cheap to recompute via env.reset() and must be fresh per
# test to avoid buffer donation/deletion issues with chex device variants.
_cache = {}


def _cached_setup(cache_key, settings, seed=0):
    """Return (key, env, obs, state, params) using cached env/params."""
    key = jax.random.PRNGKey(seed)
    if cache_key not in _cache:
        env, params = make(settings, log_wrapper=False)
        _cache[cache_key] = (env, params)
    env, params = _cache[cache_key]
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_4node_test_setup(**kwargs):
    if not kwargs:
        return _cached_setup("rwa_4node", settings_rwa_4node())
    key = jax.random.PRNGKey(0)
    settings = settings_rwa_4node()
    settings.update(kwargs)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_4node_agg_slots_test_setup(**kwargs):
    if not kwargs:
        settings = settings_rwa_4node()
        settings["aggregate_slots"] = 2
        settings["link_resources"] = 6
        return _cached_setup("rwa_4node_agg_slots", settings)
    key = jax.random.PRNGKey(0)
    settings = settings_rwa_4node()
    settings["aggregate_slots"] = 2
    settings["link_resources"] = 6
    settings.update(kwargs)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_4node_3_slot_request_test_setup(**kwargs):
    if not kwargs:
        settings = settings_rwa_4node()
        settings["env_type"] = "rsa"
        settings["values_bw"] = [3]
        settings["link_resources"] = 5
        return _cached_setup("rsa_4node_3_slot", settings)
    key = jax.random.PRNGKey(0)
    settings = settings_rwa_4node()
    settings["env_type"] = "rsa"
    settings["values_bw"] = [3]
    settings["link_resources"] = 5
    settings.update(kwargs)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_16_test_setup(**kwargs):
    base_settings = dict(
        load=100,
        k=5,
        topology_name="nsfnet_deeprmsa_undirected",
        link_resources=16,
        max_requests=10,
        values_bw=[1, 2, 3],
        slot_size=1,
        mean_service_holding_time=10,
        env_type="rsa",
    )
    if not kwargs:
        return _cached_setup("rsa_nsfnet_16", base_settings)
    key = jax.random.PRNGKey(0)
    base_settings.update(kwargs)
    env, params = make(base_settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_16_mod_test_setup(**kwargs):
    base_settings = dict(
        load=100,
        k=5,
        topology_name="nsfnet_deeprmsa_undirected",
        link_resources=16,
        max_requests=10,
        consider_modulation_format=True,
        slot_size=12.5,
        mean_service_holding_time=10,
        env_type="rsa",
    )
    if not kwargs:
        return _cached_setup("rsa_nsfnet_16_mod", base_settings)
    key = jax.random.PRNGKey(0)
    base_settings.update(kwargs)
    env, params = make(base_settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_4_test_setup(**kwargs):
    base_settings = dict(
        load=1000,
        k=5,
        topology_name="nsfnet_deeprmsa_undirected",
        link_resources=4,
        max_requests=10,
        values_bw=[1, 2, 3],
        slot_size=1,
        mean_service_holding_time=10,
        env_type="rsa",
    )
    if not kwargs:
        return _cached_setup("rsa_nsfnet_4", base_settings)
    key = jax.random.PRNGKey(0)
    base_settings.update(kwargs)
    env, params = make(base_settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


class InitPathLinkArrayTest(parameterized.TestCase):
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


class GetPathIndicesTest(parameterized.TestCase):
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


class GetPathsTest(parameterized.TestCase):
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
                    [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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


class GenerateArrivalHoldingTimesTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base", (jnp.array([0.1089808]), jnp.array([0.06516063]))),
    )
    def test_generate_arrival_times(self, expected):
        min_arr, min_hold = 0, 0
        arrival_time, holding_time = 0, 0
        rng = jax.random.PRNGKey(0)
        for i in range(1, 10000):
            rng, key = jax.random.split(rng)
            arrival_time, holding_time = self.variant(generate_arrival_holding_times)(
                key, self.params, jnp.array(self.params.arrival_rate), jnp.array(self.params.mean_service_holding_time)
            )
            min_arr = jnp.minimum(min_arr, arrival_time)
            min_hold = jnp.minimum(min_hold, holding_time)
            print(i, arrival_time, holding_time)
        print(min_arr, min_hold)
        chex.assert_trees_all_close(arrival_time, expected[0])
        chex.assert_trees_all_close(holding_time, expected[1])


class SetPathLinksTest(parameterized.TestCase):
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


class UpdatePathLinksTest(parameterized.TestCase):
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


class RemoveExpiredSlotRequestsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()
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


class UndoLinkSlotActionTest(parameterized.TestCase):
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


class ImplementPathActionTest(parameterized.TestCase):
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


class CheckNoSpectrumReuseTest(parameterized.TestCase):
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


class InitPathLengthArrayTest(parameterized.TestCase):
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


class InitModulationsArrayTest(parameterized.TestCase):
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


class InitPathSEArrayTest(parameterized.TestCase):
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


class RequiredSlotsTest(parameterized.TestCase):
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


class MaskSlotsBitRateModFormat(parameterized.TestCase):
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


class AggregateSlotsTest(parameterized.TestCase):
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


class ProcessPathActionTest(parameterized.TestCase):
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


class FindBlockStartsTest(parameterized.TestCase):
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


class FindBlockEndsTest(parameterized.TestCase):
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


class FindBlockSizesTest(parameterized.TestCase):
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


if __name__ == "__main__":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
