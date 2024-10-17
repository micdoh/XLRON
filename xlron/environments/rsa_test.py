"""
Unit tests for `env_funcs.py`.
See `chex` github and docs for info on test framework.

Key points:
There is a class for each function under test.
chex.all_variants() decorator runs the test once for each variant (e.g. jitted, non-jitted, pmapped, etc.) of the function under test.
parameterized.named_parameters() decorator runs the test once for each set of parameters passed to the function under test.
"""
import os
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4"
import distrax
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from xlron.environments.env_funcs import *
from xlron.environments.rsa import *
from xlron.environments.wrappers import *
from xlron.environments.dataclasses import *


def keys_test_setup():
    rng = jax.random.PRNGKey(0)  # N.B. all test rely on 0 seed for reproducibility
    rng, key_init, key_reset, key_policy, key_step = jax.random.split(rng, 5)
    return rng, key_init, key_reset, key_policy, key_step


def settings_rsa_4node():
    return dict(load=100, k=2, topology_name="4node", link_resources=4, max_requests=10, mean_service_holding_time=10,
                env_type="rwa", values_bw=[0], slot_size=1)


def rsa_4node_test_setup():
    key = jax.random.PRNGKey(0)
    env, params = make_rsa_env(settings_rsa_4node())
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_4node_agg_slots_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rsa_4node_agg_slots = settings_rsa_4node()
    settings_rsa_4node_agg_slots["aggregate_slots"] = 2
    settings_rsa_4node_agg_slots["link_resources"] = 5
    env, params = make_rsa_env(settings_rsa_4node_agg_slots)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_4node_3_slot_request_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rsa_4node_3_slots = settings_rsa_4node()
    settings_rsa_4node_3_slots["values_bw"] = [3]
    settings_rsa_4node_3_slots["link_resources"] = 5
    env, params = make_rsa_env(settings_rsa_4node_3_slots)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_16_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rsa_nsfnet_16 = dict(load=100, k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=16, max_requests=10,
                                  values_bw=[1, 2, 3], slot_size=1, mean_service_holding_time=10)
    env, params = make_rsa_env(settings_rsa_nsfnet_16)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_16_mod_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rsa_nsfnet_16_mod = dict(load=100, k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=16, max_requests=10,
                                  consider_modulation_format=True, slot_size=12.5, mean_service_holding_time=10)
    env, params = make_rsa_env(settings_rsa_nsfnet_16_mod)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_4_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rsa_nsfnet_4 = dict(load=1000, k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=4, max_requests=10,
                                 values_bw=[1, 2, 3], slot_size=1, mean_service_holding_time=10)
    env, params = make_rsa_env(settings_rsa_nsfnet_4)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_lightpath_reuse_4_nsfnet_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rwa_lr_nsfnet_4 = dict(k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=4, max_requests=1000,
                                 values_bw=[100], incremental_loading=True, env_type="rwa_lightpath_reuse", scale_factor=0.2)
    env, params = make_rsa_env(settings_rwa_lr_nsfnet_4)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_lightpath_reuse_4node_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rwa_lr_4 = dict(k=2, topology_name="4node", link_resources=4, max_requests=1000,
                                 values_bw=[100], incremental_loading=True, env_type="rwa_lightpath_reuse",)
    env, params = make_rsa_env(settings_rwa_lr_4)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_gn_model_4_nsfnet_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rwa_lr_nsfnet_4 = dict(
        k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=4, max_requests=1000,
        values_bw=[100], incremental_loading=True, env_type="rsa_gn_model",
        interband_gap=0, slot_size=25, mod_format_correction=False, launch_power=0.0
    )
    env, params = make_rsa_env(settings_rwa_lr_nsfnet_4)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


class GenerateRSARequestTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0, 0, 2])),
    )
    def test_generate_rsa_request(self, expected):
        key = np.array([1, 2], dtype=np.uint32)
        state = self.variant(generate_request_rsa)(key, self.state, self.params)
        request = state.request_array
        chex.assert_trees_all_close(request, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', (jnp.array([0, 1, 1]), jnp.array([1,1,2]))),
    )
    def test_generate_rsa_request_from_list(self, expected):
        key = np.array([1, 2], dtype=np.uint32)
        self.params = self.params.replace(deterministic_requests=True, list_of_requests=HashableArrayWrapper(jnp.array([[0,1,1], [1,1,2]])))
        self.state = self.variant(generate_request_rsa)(key, self.state, self.params)
        request1 = self.state.request_array
        self.state = self.variant(generate_request_rsa)(key, self.state, self.params)
        request2 = self.state.request_array
        chex.assert_trees_all_close((request1, request2), expected)


class InitPathLinkArrayTest(parameterized.TestCase):

    # N.B. jit not required here as this function is
    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ('case_base', "4node", 2, jnp.array([
            [1,0,0,0],
            [0,1,1,1],
            [1,0,1,0],
            [0,1,0,1],
            [0,1,0,0],
            [1,0,1,1],
            [0,0,1,0],
            [1,1,0,1],
            [1,1,0,0],
            [0,0,1,1],
            [0,0,0,1],
            [1,1,1,0],
        ])),
    )
    def test_init_path_link_array(self, topology_name, k, expected):
        graph = make_graph(topology_name)
        path_link_array = self.variant(init_path_link_array)(graph, k)
        chex.assert_trees_all_close(path_link_array, expected)


class GetPathIndicesTest(parameterized.TestCase):

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 3, 2, 4, 4),
        ('case_conus', 73, 74, 2, 75, 5548),
        ("case_nsfnet", 1, 2, 5, 14, 65),
    )
    def test_get_path_indices(self, s, d, k, N, expected):
        paths_start_index = self.variant(get_path_indices, static_argnums=(2, 3))(s, d, k, N)
        chex.assert_trees_all_close(paths_start_index, expected)


class GetPathsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0, 3]), jnp.array([[0,1,0,0],[1,0,1,1]])),
        ('case_switched', jnp.array([3, 0]), jnp.array([[0, 1, 0, 0], [1, 0, 1, 1]])),
    )
    def test_get_paths(self, nodes, expected):
        paths = self.variant(get_paths, static_argnums=(0,))(self.params, nodes)
        chex.assert_trees_all_close(paths, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_01', jnp.array([0, 1]),
         jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], ])),
        ('case_12', jnp.array([1, 2]),
         jnp.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],])),
    )
    def test_get_paths_nsfnet(self, nodes, expected):
        self.key, self.env, self.obs, self.state, self.params = rsa_nsfnet_16_test_setup()
        paths = self.variant(get_paths, static_argnums=(0,))(self.params, nodes)
        chex.assert_trees_all_close(paths, expected)


class GenerateArrivalHoldingTimesTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', (jnp.array([0.08356591]), jnp.array([0.29246438]))),
    )
    def test_generate_arrival_times(self, expected):
        min_arr, min_hold = 0, 0
        rng = jax.random.PRNGKey(0)
        for i in range(1, 10000):
            rng, key = jax.random.split(rng)
            arrival_time, holding_time = self.variant(generate_arrival_holding_times)(key, self.params)
            min_arr = jnp.minimum(min_arr, arrival_time)
            min_hold = jnp.minimum(min_hold, holding_time)
            print(i, arrival_time, holding_time)
        print(min_arr, min_hold)
        chex.assert_trees_all_close(arrival_time, expected[0])
        chex.assert_trees_all_close(holding_time, expected[1])


class UpdateActionHistoryTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([1, 2, 3]), jnp.array([-1, -1, -1, -1, 3, 2, 1])),
    )
    def test_update_action_history_once(self, action, expected):
        updated_action_history = self.variant(update_action_history)(self.state.action_history, self.state.action_counter, action)
        chex.assert_trees_all_close(updated_action_history, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', (jnp.array([1, 2, 3]), jnp.array([3, 4, 5]), jnp.array([5, 6, 7])), jnp.array([7, 6, 5, 4, 3, 2, 1])),
        ('case_not_masked', (jnp.array([1, 2, 3]), jnp.array([4, 5, 6]), jnp.array([7, 8, 9])), jnp.array([9, 8, 7, 5, 4, 2, 1])),
    )
    def test_update_action_history_multiple(self, actions, expected):
        for action in actions:
            self.state = self.state.replace(action_history=self.variant(update_action_history)(self.state.action_history, self.state.action_counter, action))
            self.state = self.state.replace(action_counter=decrease_last_element(self.state.action_counter))
        chex.assert_trees_all_close(self.state.action_history, expected)

    @chex.all_variants()
    def test_decrease_last_element(self):
        action_counter = jnp.array([1, 1, 1])
        expected = jnp.array([1, 1, 0])
        updated_action_counter = self.variant(decrease_last_element)(action_counter)
        chex.assert_trees_all_close(updated_action_counter, expected)


class UpdateLinkTest(parameterized.TestCase):

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0, 0, 0, 0, 0]), 3, 2, -1,
         jnp.array([0, 0, 0, 1, 1])),
        ('case_value', jnp.array([0, 0, 0, 0, 0]), 0, 5, -99,
         jnp.array([99, 99, 99, 99, 99])),
    )
    def test_update_link(self, link, initial_slot, num_slots, value, expected):
        updated_link = self.variant(update_link)(link, initial_slot, num_slots, value)
        chex.assert_trees_all_close(updated_link, expected)


class UpdatePathTest(parameterized.TestCase):

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0, 0, 0, 0, 0]), 1, 3, 2, -1,
         jnp.array([0, 0, 0, 1, 1])),
        ('case_value', jnp.array([0, 0, 0, 0, 0]), 1, 0, 2, -99,
         jnp.array([99, 99, 0, 0, 0])),
        ('case_not_in_path', jnp.array([0, 0, 0, 0, 0]), 0, 0, 5, -99,
         jnp.array([0, 0, 0, 0, 0])),
    )
    def test_update_path(self, link, link_in_path, initial_slot, num_slots, value, expected):
        updated_link = self.variant(update_path)(link, link_in_path, initial_slot, num_slots, value)
        chex.assert_trees_all_close(updated_link, expected)


class VmapUpdatePathLinksTest(parameterized.TestCase):

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base',
         jnp.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]), jnp.array([1,0,0,0]), 3, 2, -1,
         jnp.array([[0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])),
         ('case_value',
          jnp.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]), jnp.array([1, 0, 1, 1]), 0, 4, -99,
          jnp.array([[99, 99, 99, 99, 0],
                    [0, 0, 0, 0, 0],
                    [99, 99, 99, 99, 0],
                    [99, 99, 99, 99, 0]])),
        ('case_no_path',
         jnp.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]), jnp.array([0,0,0,0]), 3, 2, -1,
         jnp.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])),
    )
    def test_vmap_update_path_links(self, link_array, path, initial_slot, num_slots, value, expected):
        updated_link_array = self.variant(vmap_update_path_links)(link_array, path, initial_slot, num_slots, value)
        chex.assert_trees_all_close(updated_link_array, expected)


class UpdateNodeDepartureTest(parameterized.TestCase):

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base',
         jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf]), 3, 0.1,
         jnp.array([jnp.inf, jnp.inf, jnp.inf, 0.1, jnp.inf])),
    )
    def test_update_node_departure(self, node_row, inf_index, value, expected):
        updated_node_row = self.variant(update_node_departure)(node_row, inf_index, value)
        chex.assert_trees_all_close(updated_node_row, expected)


class UpdateSelectedNodeDepartureTest(parameterized.TestCase):

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base',
         jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf]), 1, 3, 0.1,
         jnp.array([jnp.inf, jnp.inf, jnp.inf, 0.1, jnp.inf])),
        ('case_not_selected',
          jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf]), 0, 3, 0.1,
          jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf])),
    )
    def test_update_selected_node_departure(self, node_row, node_selected, inf_index, value, expected):
        updated_node_row = self.variant(update_selected_node_departure)(node_row, node_selected, inf_index, value)
        chex.assert_trees_all_close(updated_node_row, expected)


class VmapUpdateNodeDepartureTest(parameterized.TestCase):

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base',
         jnp.array([[jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                   [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                   [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                   [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf]]), jnp.array([1,0,0,0]), 0.1,
         jnp.array([[0.1, jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                    [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                    [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                    [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf]])),
        ('case_occupied',
         jnp.array([[jnp.inf, 1, 1, jnp.inf, jnp.inf],
                    [1, jnp.inf, 1, jnp.inf, jnp.inf],
                    [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                    [1, 1, 1, jnp.inf, jnp.inf]]), jnp.array([1, 1, 1, 1]), 0.99,
         jnp.array([[0.99, 1, 1, jnp.inf, jnp.inf],
                    [1, 0.99, 1, jnp.inf, jnp.inf],
                    [0.99, jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                    [1, 1, 1, 0.99, jnp.inf]])),
    )
    def test_vmap_update_node_departure(self, node_departure_array, selected_nodes, value, expected):
        updated_link_array = self.variant(vmap_update_node_departure)(node_departure_array, selected_nodes, value)
        chex.assert_trees_all_close(updated_link_array, expected)


class UpdateNodeArrayTest(parameterized.TestCase):
    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.arange(10), jnp.full(10, 10), 0, 4, jnp.array([6, 10, 10, 10, 10, 10, 10, 10, 10, 10])),
    )
    def test_update_node_array(self, node_indices, node_array, node, request, expected):
        updated_node_array = self.variant(update_node_array)(node_indices, node_array, node, request)
        chex.assert_trees_all_close(updated_node_array, expected)


class UpdateNodeResourcesTest(parameterized.TestCase):

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.zeros(10), 4, 6, jnp.array([0, 0, 0, 0, 6, 0, 0, 0, 0, 0])),
    )
    def test_update_node_resources(self, node_row, zero_index, value, expected):
        updated_node_resources = self.variant(update_node_resources)(node_row, zero_index, value)
        chex.assert_trees_all_close(updated_node_resources, expected)


class UpdateSelectedNodeResourcesTest(parameterized.TestCase):

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.zeros(10), 6, 4, jnp.array([0, 0, 0, 0, 6, 0, 0, 0, 0, 0])),
    )
    def test_update_selected_node_resources(self, node_row, request, zero_index, expected):
        updated_node_resources = self.variant(update_selected_node_resources)(node_row, request, zero_index)
        chex.assert_trees_all_close(updated_node_resources, expected)


class VmapUpdateNodeResourcesTest(parameterized.TestCase):

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base',
         jnp.full((4, 10), 0), jnp.array([4,5,6,7]),
         jnp.array([[4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [7, 0, 0, 0, 0, 0, 0, 0, 0, 0]])),
        ('case_occupied',
         jnp.array([[4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [7, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
         jnp.array([4, 0, 0, 7]),
         jnp.array([[4, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [7, 7, 0, 0, 0, 0, 0, 0, 0, 0]])),
        ('case_negatives',
         jnp.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
         jnp.array([4, 0, 0, 7]),
         jnp.array([[4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [7, 0, 0, 0, 0, 0, 0, 0, 0, 0]])),
    )
    def test_vmap_update_node_resources(self, node_resource_array, selected_nodes, expected):
        updated_node_resource_array = self.variant(vmap_update_node_resources)(node_resource_array, selected_nodes)
        chex.assert_trees_all_close(updated_node_resource_array, expected)


class RemoveExpiredSlotRequestsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()
        path = jnp.array([1, 0, 0, 0])
        initial_slot_index = 0
        num_slots = 2
        self.state = self.state.replace(
            link_slot_array=vmap_update_path_links(self.state.link_slot_array, path, initial_slot_index, num_slots, 1),
            link_slot_departure_array=vmap_update_path_links(self.state.link_slot_departure_array, path, initial_slot_index,
                                                             num_slots, -self.state.current_time - self.state.holding_time)
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.full((4, 4), 0))
    )
    def test_remove_expired_slot_requests(self, expected):
        state = self.state.replace(current_time=10e4)
        jax.debug.print('departure {}', state.link_slot_departure_array, ordered=True)
        jax.debug.print('state.link_slot_array {}', state.link_slot_array, ordered=True)
        updated_state = self.variant(remove_expired_services_rsa)(state)
        jax.debug.print('updated_state.link_slot_departure_array {}', updated_state.link_slot_departure_array, ordered=True)
        jax.debug.print('updated_state.link_slot_array {}', updated_state.link_slot_array, ordered=True)
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.full((4, 4), 0))
    )
    def test_remove_expired_slot_requests_departure(self, expected):
        state = self.state.replace(current_time=10e4)
        updated_state = self.variant(remove_expired_services_rsa)(state)
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class UndoNodeActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.full(4, 4))
    )
    def test_undo_node_action_capacity(self, s, d, sr, dr, n, expected):
        state = implement_node_action(self.state, s, d, sr, dr, n=n)
        state = state.replace(current_time=10e4)
        updated_state = self.variant(undo_node_action)(state)
        chex.assert_trees_all_close(updated_state.node_capacity_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.full((4, 4), 0))
    )
    def test_undo_node_action_resource(self, s, d, sr, dr, n, expected):
        state = implement_node_action(self.state, s, d, sr, dr, n=n)
        state = state.replace(current_time=10e4)
        updated_state = self.variant(undo_node_action)(state)
        chex.assert_trees_all_close(updated_state.node_resource_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.full((4, 4), jnp.inf))
    )
    def test_undo_node_action_departure(self, s, d, sr, dr, n, expected):
        state = implement_node_action(self.state, s, d, sr, dr, n=n)
        state = state.replace(current_time=10e4)
        updated_state = self.variant(undo_node_action)(state)
        chex.assert_trees_all_close(updated_state.node_departure_array, expected)


class UndoLinkSlotActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()
        path = jnp.array([1, 0, 0, 0])
        initial_slot_index = 0
        num_slots = 2
        self.state = self.state.replace(
            link_slot_array=vmap_update_path_links(self.state.link_slot_array, path, initial_slot_index, num_slots, 1),
            link_slot_departure_array=vmap_update_path_links(self.state.link_slot_departure_array, path, initial_slot_index,
                                                             num_slots, self.state.current_time+self.state.holding_time)
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.full((4, 4), 0))
    )
    def test_undo_link_slot_action(self, expected):
        updated_state = self.variant(undo_action_rsa)(self.state)
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.full((4, 4), 0))
    )
    def test_undo_link_slot_action_departure(self, expected):
        updated_state = self.variant(undo_action_rsa)(self.state)
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class ImplementNodeActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.array([3, 3, 4, 4])),
        ('case_base_single', 0, 3, 1, 2, 1, jnp.array([4, 4, 4, 2]))
    )
    def test_implement_node_action_capacity(self, s, d, sr, dr, n, expected):
        updated_state = self.variant(implement_node_action)(self.state, s, d, sr, dr, n=n)
        chex.assert_trees_all_close(updated_state.node_capacity_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.array([[1,0,0,0], [1,0,0,0], [0,0,0,0], [0,0,0,0]])),
        ('case_base_single', 0, 3, 1, 2, 1, jnp.array([[0,0,0,0], [0,0,0,0], [0,0,0,0], [2,0,0,0]]))
    )
    def test_implement_node_action_resource(self, s, d, sr, dr, n, expected):
        updated_state = self.variant(implement_node_action)(self.state, s, d, sr, dr, n=n)
        chex.assert_trees_all_close(updated_state.node_resource_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.array([[-2,jnp.inf,jnp.inf,jnp.inf], [-2,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf]])),
        ('case_base_single', 0, 3, 1, 2, 1, jnp.array([[jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [-2,jnp.inf,jnp.inf,jnp.inf]]))
    )
    def test_implement_node_action_departure(self, s, d, sr, dr, n, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_node_action)(state, s, d, sr, dr, n=n)
        chex.assert_trees_all_close(updated_state.node_departure_array, expected)


class ImplementPathActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([1,0,0,0]), 0, 2, jnp.array([[-1,-1,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])),
        ('case_extra', jnp.array([1, 0, 1, 0]), 2, 2,
         jnp.array([[0, 0, -1, -1], [0, 0, 0, 0], [0, 0, -1, -1], [0, 0, 0, 0]])),
    )
    def test_implement_path_action(self, path, initial_slot_index, num_slots, expected):
        updated_state = self.variant(implement_path_action)(self.state, path, initial_slot_index, num_slots)
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([1,0,0,0]), 0, 2, jnp.array([[-2,-2,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])),
    )
    def test_implement_path_action_departure(self, path, initial_slot_index, num_slots, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_path_action)(state, path, initial_slot_index, num_slots)
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class ImplementRsaActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array(0), jnp.array([[-1,0,0,0], [0,0,0,0], [-1,0,0,0], [0,0,0,0]])),
        ('case_base_long_path', jnp.array(5), jnp.array([[0,0,0,0], [0,-1,0,0], [0,0,0,0], [0,-1,0,0]])),
    )
    def test_implement_action_rsa_slots(self, action, expected):
        updated_state = self.variant(implement_action_rsa, static_argnums=(2,))(self.state, action, self.params)
        jax.debug.print("params.path_link_array {}", self.params.path_link_array.val, ordered=True)
        jax.debug.print("updated_state.request_array {}", updated_state.request_array, ordered=True)
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array(19),
         jnp.array([
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., -1.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., -1.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., -1.],
             [0., 0., 0., -1.],
             [0., 0., 0., 0.],
             [0., 0., 0., -1.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
         ])),
    )
    def test_implement_action_rsa_slots_nsfnet4(self, action, expected):
        key, env, obs, state, params = rsa_nsfnet_4_test_setup()
        updated_state = self.variant(implement_action_rsa, static_argnums=(2,))(state, action, params)
        jax.debug.print("updated_state.link_slot_array {}", updated_state.link_slot_array, ordered=True)
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array(0), jnp.array(
            [[-2, 0, 0, 0], [0, 0, 0, 0],
             [-2, 0, 0, 0], [0, 0, 0, 0]])),
        ('case_base_long_path', jnp.array(5), jnp.array(
            [[0, 0, 0, 0], [0, -2, 0, 0],
             [0, 0, 0, 0], [0, -2, 0, 0]])),

    )
    def test_implement_action_rsa_slots_departure(self, action, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_action_rsa, static_argnums=(2,))(state, action, self.params)
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class CheckAllNodesAssignedTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, 2, jnp.array(False)),
        ('case_base_single', 0, 3, 1, 2, 1, 1, jnp.array(False)),
        ('case_base_fail', 0, 1, 1, 1, 2, 3, jnp.array(True)),
        ('case_base_single_fail', 0, 3, 1, 2, 1, 3, jnp.array(True)),
    )
    def test_check_all_nodes_assigned(self, s, d, sr, dr, n, total_requested_nodes, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_node_action)(state, s, d, sr, dr, n=n)
        actual = self.variant(check_all_nodes_assigned)(updated_state.node_departure_array, total_requested_nodes)
        chex.assert_trees_all_close(actual, expected)


class CheckNoSpectrumReuseTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([1,0,0,0]), 0, 2, jnp.array(False)),
        ('case_extra', jnp.array([1, 0, 1, 0]), 2, 2,
         jnp.array(False)),
    )
    def test_check_no_spectrum_reuse(self, path, initial_slot_index, num_slots, expected):
        updated_state = self.variant(implement_path_action)(self.state, path, initial_slot_index, num_slots)
        actual = self.variant(check_no_spectrum_reuse)(updated_state.link_slot_array)
        chex.assert_trees_all_close(actual, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([1,0,0,0]), 0, 2, jnp.array(True)),
        ('case_extra', jnp.array([1, 0, 1, 0]), 2, 2,
         jnp.array(True)),
    )
    def test_check_no_spectrum_reuse_fail(self, path, initial_slot_index, num_slots, expected):
        updated_state = self.variant(implement_path_action)(self.state, path, initial_slot_index, num_slots)
        updated_state = self.variant(implement_path_action)(updated_state, path, initial_slot_index, num_slots)
        actual = self.variant(check_no_spectrum_reuse)(updated_state.link_slot_array)
        chex.assert_trees_all_close(actual, expected)


class CheckRsaActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_pass', (jnp.array(0), jnp.array(2)), jnp.array(False)),
        ('case_fail', (jnp.array(0), jnp.array(0)), jnp.array(True)),
    )
    def test_check_action_rsa(self, actions, expected):
        for action in actions:
            self.state = implement_action_rsa(self.state, action, self.params)
        actual = self.variant(check_action_rsa)(self.state)
        chex.assert_trees_all_close(actual, expected)


class FinaliseRsaActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_pass', jnp.array([[0, 0, 0, 0],
                                 [4, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [4, 0, 0, 0]]),
         jnp.array([[0, 0, 0, 0],
                   [-2, 0, 0, 0],
                   [0, 0, 0, 0],
                   [-2, 0, 0, 0]])),
    )
    def test_finalise_rsa_action(self, expected_dept, expected_link_slot):
        self.state = self.state.replace(current_time=1, holding_time=1)
        self.state = implement_action_rsa(self.state, jnp.array(0), self.params)
        self.state = implement_action_rsa(self.state, jnp.array(0), self.params)
        actual_dept = self.variant(finalise_action_rsa)(self.state).link_slot_departure_array
        actual_link_slot = self.variant(finalise_action_rsa)(self.state).link_slot_array
        chex.assert_trees_all_close(actual_dept, expected_dept)
        chex.assert_trees_all_close(actual_link_slot, expected_link_slot)


class RsaStepTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_success", (jnp.array(0),),
         jnp.array([1.,  0.,  3., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.])),
        ("case_failure", (jnp.array(0), jnp.array(0)),
         jnp.array([1., 0., 3., -1., 0., 0., 0., 0., 0., 0., 0., -1., 0.,
                    0., 0., 0., 0., 0., 0.]))
    )
    def test_rsa_step_obs(self, actions, expected):
        for action in actions:
            obs, self.state, reward, done, info = self.variant(self.env.step, static_argnums=(3,))(
                self.key, self.state, action, self.params
            )
            jax.debug.print("dept {}", self.state.link_slot_departure_array, ordered=True)
            jax.debug.print("obs {}", obs, ordered=True)
        chex.assert_trees_all_close(obs, expected)


class RsaResetTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base",
         jnp.array([0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0.])),
    )
    def test_rsa_reset_obs(self, expected):
        obs, self.state = self.variant(self.env.reset, static_argnums=(1,))(self.key, self.params)
        chex.assert_trees_all_close(obs, expected)


class RsaActionMaskTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_empty", jnp.array([0, 0, 1]),
         jnp.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0], ]),
         jnp.array([1, 1, 1, 1, 1, 1, 1, 1])),
        ("case_full", jnp.array([0, 0, 1]),
         jnp.array([[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1], ]),
         jnp.array([0., 0., 0., 0., 0., 0., 0., 0.])),
        ("case_start_edge", jnp.array([0, 0, 1]),
         jnp.array([[0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1], ]),
         jnp.array([1., 0., 0., 0., 1., 0., 0., 0.])),
        ("case_end_edge", jnp.array([0, 0, 1]),
         jnp.array([[1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0], ]),
         jnp.array([0., 0., 0., 1., 0., 0., 0., 1.])),
    )
    def test_rsa_action_mask(self, request_array, link_slot_array, expected):
        self.state = self.state.replace(request_array=request_array, link_slot_array=link_slot_array)
        state = self.variant(self.env.action_mask, static_argnums=(1,))(self.state, self.params)
        jax.debug.print("actual {}", state.link_slot_mask, ordered=True)
        jax.debug.print("expected {}", expected, ordered=True)
        chex.assert_trees_all_close(state.link_slot_mask, expected)

    # N.B. that requested bandwidth (middle number of request array) will lead to bw+1 slots allocated
    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_start_edge_3", jnp.array([0, 2, 1]),
         jnp.array([[0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1], ]),
         jnp.array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0.])),
        ("case_end_edge_3", jnp.array([0, 2, 1]),
         jnp.array([[1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0], ]),
         jnp.array([0., 0., 1., 0., 0., 0., 0., 1., 0., 0.])),
        ("case_start_edge_2", jnp.array([0, 1, 1]),
         jnp.array([[0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1], ]),
         jnp.array([1., 1., 0., 0., 0., 1., 1., 0., 0., 0.])),
        ("case_end_edge_2", jnp.array([0, 1, 1]),
         jnp.array([[1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0], ]),
         jnp.array([0., 0., 1., 1., 0., 0., 0., 1., 1., 0.])),
        ("case_middle_2", jnp.array([0, 1, 1]),
         jnp.array([[1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1], ]),
         jnp.array([0., 0., 1., 0., 0., 0., 0., 1., 0., 0.])),
        ("case_middle_3", jnp.array([0, 2, 1]),
         jnp.array([[1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1], ]),
         jnp.array([0., 1., 1., 0., 0., 0., 1., 0., 0., 0.])),
        ("case_middle_1", jnp.array([0, 0, 1]),
         jnp.array([[1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 1], ]),
         jnp.array([0., 0., 1., 1., 1., 0., 0., 1., 0., 0.])),
    )
    def test_rsa_action_mask_3_slot_request(self, request_array, link_slot_array, expected):
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_3_slot_request_test_setup()
        self.state = self.state.replace(request_array=request_array, link_slot_array=link_slot_array)
        self.params = self.params.replace(max_slots=3)
        state = self.variant(self.env.action_mask, static_argnums=(1,))(self.state, self.params)
        chex.assert_trees_all_close(state.link_slot_mask, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_start_edge", jnp.array([0, 2, 1]), True,
         jnp.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],]),
         # N.B. that modulation format consideration means double spectral efficiency on the first path, hence two 1's
         jnp.array(
             [
                 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             ]
         )
         ),
        ("case_rwa", jnp.array([0, 0, 1]), False,
         jnp.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],]),
         jnp.array(
             [
                 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             ]
         )
         ),
    )
    def test_rsa_action_mask_nsfnet_16(self, request_array, consider_mod, link_slot_array, expected):
        self.key, self.env, self.obs, self.state, self.params = rsa_nsfnet_16_test_setup()
        self.state = self.state.replace(request_array=request_array, link_slot_array=link_slot_array)
        jax.debug.print("request_array {}", request_array, ordered=True)
        self.params = self.params.replace(max_slots=3, consider_modulation_format=consider_mod)
        state = self.variant(self.env.action_mask, static_argnums=(1,))(self.state, self.params)
        jax.debug.print("request_array {}", request_array, ordered=True)
        jax.debug.print("actual {}", state.link_slot_mask, ordered=True)
        jax.debug.print("expected {}", expected, ordered=True)
        chex.assert_trees_all_close(state.link_slot_mask, expected)


class InitPathLengthArrayTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("case_nsfnet", "nsfnet", 1, jnp.array([2100, 1200, 3600, 4800])),
        ("case_4node", "4node", 1, jnp.array([4, 7, 1, 3]))
    )
    def test_init_path_length_array(self, topology_name, k, expected):
        graph = make_graph(topology_name)
        path_link_array = init_path_link_array(graph, k)
        path_length_array = self.variant(init_path_length_array)(path_link_array, graph)
        chex.assert_trees_all_close(path_length_array[:4], expected)


class InitModulationsArrayTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("case_base", "modulations.csv", jnp.array([[100000.0, 1.0, 12.6, -14.0],
                                                    [2000.0, 2.0, 12.6, -17.0],
                                                    [1000.0, 3.0, 18.6, -20.0],
                                                    [500.0, 4.0, 22.4, -23.0],
                                                    [250.0, 5.0, 26.4, -26.0],
                                                    [125.0, 6.0, 30.4, -29.0]])),
    )
    def test_init_modulations_array(self, modulations_file, expected):
        modulations_array = self.variant(init_modulations_array)(modulations_file)
        chex.assert_trees_all_close(modulations_array, expected)


class InitPathSEArrayTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("case_nsfnet", "nsfnet", 1, jnp.array([1.0, 2.0, 1.0, 1.0])),
        ("case_4node", "4node", 1, jnp.array([6., 6., 6., 6.]))
    )
    def test_init_path_se_array(self, topology_name, k, expected):
        graph = make_graph(topology_name)
        path_link_array = init_path_link_array(graph, k)
        path_length_array = init_path_length_array(path_link_array, graph)
        modulations_array = init_modulations_array("modulations.csv")
        path_se_array = self.variant(init_path_se_array)(path_length_array, modulations_array)
        chex.assert_trees_all_close(path_se_array[:4], expected)


class RequiredSlotsTest(parameterized.TestCase):
    # TODO - why does with_device fail with this? 12.5 should be an acceptable static arg
    #  (probably will disappear if channel_width is moved to params)

    def setUp(self):
        super().setUp()

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("case_empty", 100, 2, 12.5, jnp.array([5])),
    )
    def test_required_slots(self, bit_rate, se, channel_width, expected):
        slots = self.variant(required_slots)(bit_rate, se, channel_width)
        chex.assert_trees_all_close(slots, expected)


class MaskSlotsBitRateModFormat(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_nsfnet_16_mod_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_empty", jnp.array([0, 100, 1]),
         jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]),
            jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,])),
        ("case_full", jnp.array([0, 49, 1]),
         jnp.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ]),
         jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("case_middle", jnp.array([0, 100, 1]),
         jnp.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
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
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], ]),
         jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    )
    def test_mask_slots_bit_rate_mod_format(self, request, link_slot_array, expected):
        self.state = self.state.replace(link_slot_array=link_slot_array, request_array=request)
        jax.debug.print("state.link_slot_mask {}", self.state.link_slot_mask, ordered=True)
        updated_state = self.variant(self.env.action_mask, static_argnums=(1,))(self.state, self.params)
        jax.debug.print("state.link_slot_mask {}", updated_state.link_slot_mask, ordered=True)
        jax.debug.print("expected {}", expected, ordered=True)
        chex.assert_trees_all_close(updated_state.link_slot_mask, expected)


class CalculatePathStatsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_nsfnet_16_mod_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_middle", jnp.array([0, 100, 1]),
         jnp.array([
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
             [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], ]),
         jnp.array([[0.625, 0.5625, 0.25, 1.8, 0.9,],
                    [0.625, 0.5625, 0.25, 1.8, 0.9,],
                    [1.125, 0.5625, 0.25, 1., 0.5],
                    [1.125, 0.5625, 0.25, 1., 0.5],
                    [1.125, 0.5625, 0.25, 1., 0.5],]),),
        ("case_edges", jnp.array([0, 100, 1]),
         jnp.array([
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         ]),
         jnp.array(
             [[0.625, 0.5, 0., 0.8, 0.533333336],
              [0.625, 0.5, 0., 0.8, 0.533333336],
              [1.125, 0.5, 0., 0.44444445, 0.2962963],
              [1.125, 0.5, 0., 0.44444445, 0.2962963],
              [1.125, 0.5, 0., 0.44444445, 0.2962963],]),)
    )
    def test_calculate_path_stats(self, request, link_slot_array, expected):
        self.state = self.state.replace(link_slot_array=link_slot_array, request_array=request)
        stats = self.variant(calculate_path_stats, static_argnums=(1,))(self.state, self.params, request)
        jax.debug.print("stats {}", stats, ordered=True)
        jax.debug.print("expected {}", expected, ordered=True)
        chex.assert_trees_all_close(stats, expected)


class AggregateSlotsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_agg_slots_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_all_invalid", jnp.array([[0.,0.,0.,0.,0.], [0.,0.,0.,0.,0.]]), jnp.array([[0.,0.,0.], [0.,0.,0.]]),),
        ("case_all_valid", jnp.array([[1.,1.,1.,1.,1.], [1.,1.,1.,1.,1.]]), jnp.array([[1.,1.,1.], [1.,1.,1.]]),),
        ("case_first_edge_valid", jnp.array([[1.,0.,0.,0.,0.], [0.,0.,0.,0.,0.]]), jnp.array([[1.,0.,0.], [0.,0.,0.]]),),
        ("case_last_edge_valid", jnp.array([[0.,0.,0.,0.,0.], [0.,0.,0.,0.,1.]]), jnp.array([[0.,0.,0.], [0.,0.,1.]]),),
        ("case_middle_valid", jnp.array([[0.,0.,1.,0.,0.], [0.,0.,1.,0.,0.]]), jnp.array([[0.,1.,0.], [0.,1.,0.]]),),
    )
    def test_aggregate_slots(self, mask, expected):
        result, mask = self.variant(aggregate_slots, static_argnums=(1,))(mask, self.params)
        jax.debug.print("result {}", result, ordered=True)
        chex.assert_trees_all_close(result, expected)


class ProcessPathActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_agg_slots_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_last_fit",
         jnp.array([0,0,0,0,0,0,0,0,1,0]),
         jnp.array([4]),
         jnp.array([3]),
         ),
        ("case_first_fit",
         jnp.array([0,0,1,0,0,0,0,0,0,0]),
         jnp.array([1]),
         jnp.array([2]),
         ),
    )
    def test_process_path_action(self, full_link_slot_mask, path_action, expected):
        state = self.state.replace(full_link_slot_mask=full_link_slot_mask)
        result_path, result_slot = self.variant(process_path_action, static_argnums=(1,))(state, self.params, path_action)
        jax.debug.print("result path {}", result_path, ordered=True)
        jax.debug.print("result slot {}", result_slot, ordered=True)
        chex.assert_trees_all_close(result_slot, expected)
        
        
class CheckLightpathAvailableAndExistingTest(parameterized.TestCase):
    
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_lightpath_reuse_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_available",
         jnp.array([[-1, -1, -1, -1, ],
                    [-1, -1, -1, -1, ],
                    [-1, -1, -1, -1, ],
                    [-1, -1, -1, -1, ]]),
         jnp.array([3]),
         jnp.array(True),
         jnp.array(False),
        ),
        ("case_not_available",
         jnp.array([[-1, -1, -1, 1, ],
                    [-1, -1, -1, -1, ],
                    [-1, -1, -1, -1, ],
                    [-1, -1, -1, -1, ]]),
         jnp.array([3]),
         jnp.array(False),
         jnp.array(False),
         ),
        ("case_available_existing",
         jnp.array([[1, 2, 3, 0,],
                    [4, 5, 6, 7,],
                    [8, 9, 10, 11,],
                    [12, 13, 14, 15,]]),
         jnp.array([3]),
         jnp.array(True),
         jnp.array(True),
         ),
        ("case_not_available_existing",
         jnp.array([[-1, -1, -1, 0, ],
                    [-1, -1, -1, -1, ],
                    [-1, -1, -1, -1, ],
                    [-1, -1, -1, -1, ]]),
         jnp.array([3]),
         jnp.array(True), # Always available if exists
         jnp.array(True),
         ),
    )
    def test_lightpath_available_existing(self,path_index_array, action, expected_available, expected_existing, request=jnp.array([0, 100, 1])):
        state = self.state.replace(path_index_array=path_index_array, request_array=request)
        result_available, result_existing, curr_lightpath_capacity, lightpath_index = self.variant(
            check_lightpath_available_and_existing, static_argnums=(1,)
        )(state, self.params, action)
        jax.debug.print("request_array {}", state.request_array, ordered=True)
        jax.debug.print("state.path_capacity_array {}", state.path_capacity_array, ordered=True)
        jax.debug.print("state.path_index_array {}", state.path_index_array, ordered=True)
        jax.debug.print("curr_lightpath_capacity {}", curr_lightpath_capacity, ordered=True)
        jax.debug.print("lightpath_index {}", lightpath_index, ordered=True)
        jax.debug.print("result available {}", result_available, ordered=True)
        jax.debug.print("result existing {}", result_existing, ordered=True)
        chex.assert_trees_all_close(result_available, expected_available)
        chex.assert_trees_all_close(result_existing, expected_existing)


class MaskSlotsRWALightpathReuseTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_lightpath_reuse_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_no_capacity",
         jnp.array([[0., 0., 0., 0., ],
                    [0., 0., 0., 0., ],
                    [0., 0., 0., 0., ],
                    [0., 0., 0., 0., ]]),
         jnp.array([[-1, -1, -1, -1, ],
                    [-1, -1, -1, -1, ],
                    [-1, -1, -1, -1, ],
                    [-1, -1, -1, -1, ]]),
         jnp.array([0, 100, 1]),
         jnp.array([0., 0., 0., 0.,
                     0., 0., 0., 0., ])
         ),
         ("case_capacity_no_path",
         jnp.array([[100., 100., 100., 100., ],
                    [100., 100., 100., 100., ],
                    [100., 100., 100., 100., ],
                    [100., 100., 100., 100., ],]),
         jnp.array([[99, 99, 99, 99, ],
                    [99, 99, 99, 99, ],
                    [99, 99, 99, 99, ],
                    [99, 99, 99, 99, ],]),
         jnp.array([0, 100, 1]),
         jnp.array([0., 0., 0., 0.,
                     0., 0., 0., 0., ])
         ),
        ("case_capacity_path",
         jnp.array([[100., 100., 100., 100., ],
                    [100., 100., 100., 100., ],
                    [100., 100., 100., 100., ],
                    [100., 100., 100., 100., ], ]),
         jnp.array([[99, 99, 99, -1, ],
                    [99, 99, 99, 99, ],
                    [99, 99, 99, 99, ],
                    [99, 99, 99, 99, ], ]),
         jnp.array([0, 100, 1]),
         jnp.array([0., 0., 0., 1.,
                     0., 0., 0., 0., ])
         ),
        ("case_no_capacity_path",
         jnp.array([[100., 100., 100., 0., ],
                    [100., 100., 100., 100., ],
                    [100., 100., 100., 100., ],
                    [100., 100., 100., 100., ], ]),
         jnp.array([[99, 99, 99, -1, ],
                    [99, 99, 99, 99, ],
                    [99, 99, 99, 99, ],
                    [99, 99, 99, 99, ], ]),
         jnp.array([0, 100, 1]),
         jnp.array([0., 0., 0., 0.,
                     0., 0., 0., 0., ])
         ),
        ("case_2_path",
         jnp.array([[100., 100., 100., 100., ],
                    [100., 100., 100., 100., ],
                    [100., 100., 100., 100., ],
                    [100., 100., 100., 100., ], ]),
         jnp.array([[99, 99, 99, 0, ],
                    [99, 1, 99, 99, ],
                    [99, 1, 99, 99, ],
                    [99, 1, 99, 99, ], ]),
         jnp.array([0, 100, 1]),
         jnp.array([0., 0., 0., 1.,
                     0., 1., 0., 0., ])
         ),
    )
    def test_mask_slots_rwa_lightpath_reuse(self, link_capacity_array, path_index_array, request, expected):
        state = self.state.replace(
            link_capacity_array=link_capacity_array, path_index_array=path_index_array
        )
        state = self.variant(mask_slots_rwalr, static_argnums=(1,))(state, self.params, request)
        jax.debug.print("state.link_slot_mask {}", state.link_slot_mask, ordered=True)
        jax.debug.print("expected {}", expected, ordered=True)
        chex.assert_trees_all_close(state.link_slot_mask, expected)


class RWALightpathReuseTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_lightpath_reuse_4_nsfnet_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_end_episode",
         jnp.array([[ 200.,  800.,  500.,    0.,],
                    [1100.,  700.,  700.,  700.,],
                    [   0.,    0.,  300.,    0.,],
                    [ 800.,  500.,  600.,  500.,],
                    [ 400.,  100.,  200.,    0.,],
                    [ 300.,    0.,    0.,    0.,],
                    [ 400.,  300.,  300.,  600.,],
                    [   0.,    0.,    0.,    0.,],
                    [ 400.,    0.,  200.,    0.,],
                    [ 600.,  500.,  900.,  900.,],
                    [ 400.,  300.,  200.,  400.,],
                    [   0.,    0.,  400.,  700.,],
                    [ 700.,  500.,  600.,  300.,],
                    [ 600.,  800.,  500.,  700.,],
                    [ 500.,  600.,  700.,  300.,],
                    [ 600.,  200.,  200.,  400.,],
                    [1100.,  900.,  700.,  900.,],
                    [1000.,  800.,  800.,  900.,],
                    [ 400.,  600.,  400.,  600.,],
                    [ 700.,  200.,  400.,  300.,],
                    [ 600.,  900.,  800., 1300.,],
                    [1100., 1200., 1300., 1500.,]])
         ),
    )
    def test_end_episode(self, expected):
        rng, reset_rng = jax.random.split(self.key)
        obsv, env_state = self.env.reset(reset_rng, self.params)
        reward = jnp.array([1.])
        i = 0
        while reward > 0:
            i += 1
            rng, rng_sample, rng_step = jax.random.split(rng, 3)
            # get mask
            env_state = self.env.action_mask(env_state, self.params)
            mask = env_state.link_slot_mask
            # make distribution
            action_dist = distrax.Categorical(logits=jnp.where(mask, mask, -1e8))
            jax.debug.print("action dist {}", action_dist.logits, ordered=True)
            # sample distribution
            action = action_dist.sample(seed=rng_sample)
            # step env
            remaining_capacity = env_state.link_capacity_array  # capture to avoid being reset to initial state
            obsv, env_state, reward, done, info = self.variant(self.env.step, static_argnums=(3))(
                rng_step, env_state, action, self.params
            )
            jax.debug.print("action mask {}", env_state.link_slot_mask, ordered=True)
            jax.debug.print("action {}", action, ordered=True)
            jax.debug.print("reward {}", reward, ordered=True)
            jax.debug.print("-----END-----")
            if i == 1000:
                break
        chex.assert_trees_all_close(remaining_capacity, expected)


class FindBlockStartsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_full', jnp.array([1, 1, 1, 1, 1, 1, 1]), jnp.array([0, 0, 0, 0, 0, 0, 0])),
        ('case_empty', jnp.array([0, 0, 0, 0, 0, 0, 0]), jnp.array([1, 0, 0, 0, 0, 0, 0])),
        ('case_few', jnp.array([0, 1, 0, 0, 1, 0, 0]), jnp.array([1, 0, 1, 0, 0, 1, 0])),
        ('case_many', jnp.array([0, 1, 1, 0, 1, 1, 1]), jnp.array([1, 0, 0, 1, 0, 0, 0])),
    )
    def test_find_block_starts(self, slots, expected):
        actual = self.variant(find_block_starts)(slots)
        chex.assert_trees_all_close(actual, expected)


class FindBlockEndsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_full', jnp.array([1, 1, 1, 1, 1, 1, 1]), jnp.array([0, 0, 0, 0, 0, 0, 0])),
        ('case_empty', jnp.array([0, 0, 0, 0, 0, 0, 0]), jnp.array([0, 0, 0, 0, 0, 0, 1])),
        ('case_few', jnp.array([0, 1, 0, 0, 1, 0, 0]), jnp.array([1, 0, 0, 1, 0, 0, 1])),
        ('case_many', jnp.array([0, 1, 1, 0, 1, 1, 1]), jnp.array([1, 0, 0, 1, 0, 0, 0])),
    )
    def test_find_block_ends(self, slots, expected):
        actual = self.variant(find_block_ends)(slots)
        chex.assert_trees_all_close(actual, expected)


class FindBlockSizesTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_full', jnp.array([1, 1, 1, 1, 1, 1, 1]), jnp.array([0, 0, 0, 0, 0, 0, 0])),
        ('case_empty', jnp.array([0, 0, 0, 0, 0, 0, 0]), jnp.array([7, 0, 0, 0, 0, 0, 0])),
        ('case_few', jnp.array([0, 1, 0, 0, 1, 0, 0]), jnp.array([1, 0, 2, 0, 0, 2, 0])),
        ('case_many', jnp.array([0, 1, 1, 0, 1, 1, 1]), jnp.array([1, 0, 0, 1, 0, 0, 0])),
        ('case_gap', jnp.array([1, 0, 0, 0, 0, 0, 1]), jnp.array([0, 5, 0, 0, 0, 0, 0])),
    )
    def test_find_block_sizes(self, slots, expected):
        actual = self.variant(find_block_sizes)(slots)
        chex.assert_trees_all_close(actual, expected)

    # Don't test on device as it causes static_argnames to be traced...
    @chex.variants(without_device=True)
    @parameterized.named_parameters(
        ('case_full', jnp.array([1, 1, 1, 1, 1, 1, 1]), jnp.array([0, 0, 0, 0, 0, 0, 0])),
        ('case_empty', jnp.array([0, 0, 0, 0, 0, 0, 0]), jnp.array([1, 2, 3, 4, 5, 6, 7])),
        ('case_few', jnp.array([0, 1, 0, 0, 1, 0, 0]), jnp.array([1, 0, 1, 2, 0, 1, 2])),
        ('case_many', jnp.array([0, 1, 1, 0, 1, 1, 1]), jnp.array([1, 0, 0, 1, 0, 0, 0])),
        ('case_gap', jnp.array([1, 0, 0, 0, 0, 0, 1]), jnp.array([0, 1, 2, 3, 4, 5, 0])),
        ('case_tricky', jnp.array([0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,1]),
         jnp.array([1,2,3,4,5,6,7,0,0,0,0,1,2,3,4,5,0,0,1,0]))
    )
    def test_find_block_sizes_reverse(self, slots, expected):
        actual = self.variant(find_block_sizes, static_argnames=("starts_only", "reverse"))(slots, starts_only=False, reverse=True)
        chex.assert_trees_all_close(actual, expected)

    @chex.variants(without_device=True)
    @parameterized.named_parameters(
        ('case_full', jnp.array([1, 1, 1, 1, 1, 1, 1]), jnp.array([0, 0, 0, 0, 0, 0, 0])),
        ('case_empty', jnp.array([0, 0, 0, 0, 0, 0, 0]), jnp.array([7, 6, 5, 4, 3, 2, 1])),
        ('case_few', jnp.array([0, 1, 0, 0, 1, 0, 0]), jnp.array([1, 0, 2, 1, 0, 2, 1])),
        ('case_many', jnp.array([0, 1, 1, 0, 1, 1, 1]), jnp.array([1, 0, 0, 1, 0, 0, 0])),
        ('case_gap', jnp.array([1, 0, 0, 0, 0, 0, 1]), jnp.array([0, 5, 4, 3, 2, 1, 0])),
    )
    def test_find_block_sizes_not_starts_only(self, slots, expected):
        actual = self.variant(find_block_sizes, static_argnames=("starts_only", "reverse"))(slots, starts_only=False)
        chex.assert_trees_all_close(actual, expected)


class RSAGNModelTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_gn_model_4_nsfnet_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_end_episode",
         jnp.array([[ 200.,  800.,  500.,    0.,],
                    [1100.,  700.,  700.,  700.,],
                    [   0.,    0.,  300.,    0.,],
                    [ 800.,  500.,  600.,  500.,],
                    [ 400.,  100.,  200.,    0.,],
                    [ 300.,    0.,    0.,    0.,],
                    [ 400.,  300.,  300.,  600.,],
                    [   0.,    0.,    0.,    0.,],
                    [ 400.,    0.,  200.,    0.,],
                    [ 600.,  500.,  900.,  900.,],
                    [ 400.,  300.,  200.,  400.,],
                    [   0.,    0.,  400.,  700.,],
                    [ 700.,  500.,  600.,  300.,],
                    [ 600.,  800.,  500.,  700.,],
                    [ 500.,  600.,  700.,  300.,],
                    [ 600.,  200.,  200.,  400.,],
                    [1100.,  900.,  700.,  900.,],
                    [1000.,  800.,  800.,  900.,],
                    [ 400.,  600.,  400.,  600.,],
                    [ 700.,  200.,  400.,  300.,],
                    [ 600.,  900.,  800., 1300.,],
                    [1100., 1200., 1300., 1500.,]])
         ),
    )
    def test_end_episode(self, expected):
        rng, reset_rng = jax.random.split(self.key)
        obsv, env_state = self.env.reset(reset_rng, self.params)
        reward = jnp.array([1.])
        i = 0
        while reward > 0:
            i += 1
            rng, rng_sample, rng_step = jax.random.split(rng, 3)
            # get mask
            env_state = self.env.action_mask(env_state, self.params)
            mask = env_state.link_slot_mask
            # make distribution
            action_dist = distrax.Categorical(logits=jnp.where(mask, mask, -1e8))
            #jax.debug.print("action dist {}", action_dist.logits, ordered=True)
            # sample distribution
            action = action_dist.sample(seed=rng_sample)
            path_index, slot_index = process_path_action(env_state, self.params, action)
            path = get_paths(self.params, read_rsa_request(env_state.request_array)[0])[path_index]
            jax.debug.print("path {}", path, ordered=True)
            jax.debug.print("slot {}", slot_index, ordered=True)
            # step env
            obsv, env_state, reward, done, info = self.variant(self.env.step, static_argnums=(3))(
                rng_step, env_state, action, self.params
            )
            jax.debug.print("action mask {}", env_state.link_slot_mask, ordered=True)
            jax.debug.print("action {}", action, ordered=True)
            jax.debug.print("reward {}", reward, ordered=True)
            jax.debug.print("link snr array {}", env_state.link_snr_array, ordered=True)
            jax.debug.print("path_snr {}", get_snr_for_path(path, env_state.link_snr_array, self.params), ordered=True)
            jax.debug.print("link_slot_array {}", env_state.link_slot_array, ordered=True)
            jax.debug.print("path_index_array {}", env_state.path_index_array, ordered=True)
            jax.debug.print("channel_centre_bw_array {}", env_state.channel_centre_bw_array, ordered=True)
            jax.debug.print("channel_power_array {}", env_state.channel_power_array, ordered=True)
            jax.debug.print("modulation_format_index_array {}", env_state.modulation_format_index_array, ordered=True)
            jax.debug.print("-----END-----")
            jax.debug.print("request_array {}", env_state.request_array, ordered=True)
            if i == self.params.max_requests:
                break
        chex.assert_trees_all_close(env_state.link_snr_array, expected)


if __name__ == '__main__':
    jax.config.update('jax_numpy_rank_promotion', 'raise')
    absltest.main()
