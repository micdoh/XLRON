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
from xlron.environments.rsa.rsa import *
from xlron.environments.wrappers import *
from xlron.environments.dataclasses import *
from xlron.environments.make_env import make


def keys_test_setup():
    rng = jax.random.PRNGKey(0)  # N.B. all test rely on 0 seed for reproducibility
    rng, key_init, key_reset, key_policy, key_step = jax.random.split(rng, 5)
    return rng, key_init, key_reset, key_policy, key_step


def settings_rwa_4node():
    return dict(load=100, k=2, topology_name="4node", link_resources=4, max_requests=10, mean_service_holding_time=10,
                env_type="rwa", values_bw=[1], slot_size=1, guardband=0)


def rwa_4node_test_setup(**kwargs):
    key = jax.random.PRNGKey(0)
    settings_rwa_4node().update(kwargs)
    env, params = make(settings_rwa_4node(), log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_4node_agg_slots_test_setup(**kwargs):
    key = jax.random.PRNGKey(0)
    settings_rwa_4node_agg_slots = settings_rwa_4node()
    settings_rwa_4node_agg_slots["aggregate_slots"] = 2
    settings_rwa_4node_agg_slots["link_resources"] = 5
    settings_rwa_4node_agg_slots.update(kwargs)
    env, params = make(settings_rwa_4node_agg_slots, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_4node_3_slot_request_test_setup(**kwargs):
    key = jax.random.PRNGKey(0)
    settings_rsa_4node_3_slots = settings_rwa_4node()
    settings_rsa_4node_3_slots["env_type"] = "rsa"
    settings_rsa_4node_3_slots["values_bw"] = [3]
    settings_rsa_4node_3_slots["link_resources"] = 5
    settings_rsa_4node_3_slots.update(kwargs)
    env, params = make(settings_rsa_4node_3_slots, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_16_test_setup(**kwargs):
    key = jax.random.PRNGKey(0)
    settings_rsa_nsfnet_16 = dict(load=100, k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=16, max_requests=10,
                                  values_bw=[1, 2, 3], slot_size=1, mean_service_holding_time=10, env_type='rsa')
    settings_rsa_nsfnet_16.update(kwargs)
    env, params = make(settings_rsa_nsfnet_16, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_16_mod_test_setup(**kwargs):
    key = jax.random.PRNGKey(0)
    settings_rsa_nsfnet_16_mod = dict(load=100, k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=16, max_requests=10,
                                  consider_modulation_format=True, slot_size=12.5, mean_service_holding_time=10, env_type='rsa')
    settings_rsa_nsfnet_16_mod.update(kwargs)
    env, params = make(settings_rsa_nsfnet_16_mod, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_4_test_setup(**kwargs):
    key = jax.random.PRNGKey(0)
    settings_rsa_nsfnet_4 = dict(load=1000, k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=4, max_requests=10,
                                 values_bw=[1, 2, 3], slot_size=1, mean_service_holding_time=10, env_type='rsa')
    settings_rsa_nsfnet_4.update(kwargs)
    env, params = make(settings_rsa_nsfnet_4, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


class GenerateRSARequestTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0, 1, 2])),
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
        self.params = self.params.replace(deterministic_requests=True)
        self.state = self.state.replace(list_of_requests=jnp.array([[0,1,1], [1,1,2]]))
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
            [0, 1, 0, 1],
            [1,0,1,0],
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 0, 1, 0],
            [1, 1, 0, 1],
            [1,1,0,0],
            [0,0,1,1],
            [0,0,0,1],
            [1,1,1,0],
        ])),
    )
    def test_init_path_link_array(self, topology_name, k, expected):
        graph = make_graph(topology_name)
        path_link_array = self.variant(init_path_link_array)(graph, k)
        print(path_link_array)
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
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

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
                    [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        print(paths)
        chex.assert_trees_all_close(paths, expected)


class GenerateArrivalHoldingTimesTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', (jnp.array([0.07049651]), jnp.array([8.057691]))),
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
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()
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


class UndoLinkSlotActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()
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
        updated_state = self.variant(undo_action_rsa)(self.state, self.params)
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.full((4, 4), 0))
    )
    def test_undo_link_slot_action_departure(self, expected):
        updated_state = self.variant(undo_action_rsa)(self.state, self.params)
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class ImplementPathActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

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
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array(0), jnp.array([[-1,0,0,0], [0,0,0,0], [-1,0,0,0], [0,0,0,0]])),
        ('case_base_long_path', jnp.array(5), jnp.array([[0,0,0,0], [0,-1,0,0], [0,0,0,0], [0,-1,0,0]])),
    )
    def test_implement_action_rsa_slots(self, action, expected):
        updated_state = self.variant(implement_action_rsa, static_argnums=(2,))(self.state, action, self.params)
        jax.debug.print("params.path_link_array {}", self.params.path_link_array.val, ordered=True)
        jax.debug.print("updated_state.request_array {}", updated_state.request_array, ordered=True)
        jax.debug.print("updated_state.link_slot_array {}", updated_state.link_slot_array, ordered=True)
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


class CheckNoSpectrumReuseTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

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
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

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
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_pass', jnp.array([[4, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [4, 0, 0, 0],
                                 [0, 0, 0, 0]]),
         jnp.array([[-2, 0, 0, 0],
                   [0, 0, 0, 0],
                   [-2, 0, 0, 0],
                   [0, 0, 0, 0]])),
    )
    def test_finalise_rsa_action(self, expected_dept, expected_link_slot):
        self.state = self.state.replace(current_time=1, holding_time=1)
        self.state = implement_action_rsa(self.state, jnp.array(0), self.params)
        self.state = implement_action_rsa(self.state, jnp.array(0), self.params)
        final_state = self.variant(finalise_action_rsa)(self.state, self.params)
        chex.assert_trees_all_close(final_state.link_slot_departure_array, expected_dept)
        chex.assert_trees_all_close(final_state.link_slot_array, expected_link_slot)


class RsaStepTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_success", (jnp.array(0),),
         jnp.array([1.,  1.,  3., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.])),
        ("case_failure", (jnp.array(0), jnp.array(0)),
         jnp.array([1., 1., 3., -1., 0., 0., 0., 0., 0., 0., 0., -1., 0.,
                    0., 0., 0., 0., 0., 0.]))
    )
    def test_rsa_step_obs(self, actions, expected):
        for action in actions:
            print(action)
            obs, self.state, reward, done, info = self.variant(self.env.step, static_argnums=(3,))(
                self.key, self.state, action, self.params
            )
            jax.debug.print("dept {}", self.state.link_slot_departure_array, ordered=True)
            jax.debug.print("obs {}", obs, ordered=True)
        chex.assert_trees_all_close(obs, expected)


class RsaResetTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base",
         jnp.array([0., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0.])),
    )
    def test_rsa_reset_obs(self, expected):
        obs, self.state = self.variant(self.env.reset, static_argnums=(1,))(self.key, self.params)
        chex.assert_trees_all_close(obs, expected)


class RsaActionMaskTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_empty", jnp.array([0, 1, 1]),
         jnp.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0], ]),
         jnp.array([1, 1, 1, 1, 1, 1, 1, 1])),
        ("case_full", jnp.array([0, 1, 1]),
         jnp.array([[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1], ]),
         jnp.array([0., 0., 0., 0., 0., 0., 0., 0.])),
        ("case_start_edge", jnp.array([0, 1, 1]),
         jnp.array([[0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1], ]),
         jnp.array([1., 0., 0., 0., 1., 0., 0., 0.])),
        ("case_end_edge", jnp.array([0, 1, 1]),
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
        ("case_start_edge_3", jnp.array([0, 3, 1]),
         jnp.array([[0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1], ]),
         jnp.array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0.])),
        ("case_end_edge_3", jnp.array([0, 3, 1]),
         jnp.array([[1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0], ]),
         jnp.array([0., 0., 1., 0., 0., 0., 0., 1., 0., 0.])),
        ("case_start_edge_2", jnp.array([0, 2, 1]),
         jnp.array([[0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1], ]),
         jnp.array([1., 1., 0., 0., 0., 1., 1., 0., 0., 0.])),
        ("case_end_edge_2", jnp.array([0, 2, 1]),
         jnp.array([[1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0], ]),
         jnp.array([0., 0., 1., 1., 0., 0., 0., 1., 1., 0.])),
        ("case_middle_2", jnp.array([0, 2, 1]),
         jnp.array([[1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1], ]),
         jnp.array([0., 0., 1., 0., 0., 0., 0., 1., 0., 0.])),
        ("case_middle_3", jnp.array([0, 3, 1]),
         jnp.array([[1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1], ]),
         jnp.array([0., 1., 1., 0., 0., 0., 1., 0., 0., 0.])),
        ("case_middle_1", jnp.array([0, 1, 1]),
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
                 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             ]
         ).astype(jnp.float32)
         ),
        ("case_rwa", jnp.array([0, 1, 1]), False,
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
         ).astype(jnp.float32)
         ),
    )
    def test_rsa_action_mask_nsfnet_16(self, request_array, consider_mod, link_slot_array, expected):
        self.key, self.env, self.obs, self.state, self.params = rsa_nsfnet_16_test_setup(guardband=0, env_type="rmsa")
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
        ("case_nsfnet", "nsfnet_deeprmsa_undirected", 1, jnp.array([1050., 1500., 1800., 2400.])),
        ("case_4node", "4node", 1, jnp.array([4, 3, 1, 3]))
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
        ("case_base", jnp.array([[100000.0, 1.0, 12.6, -14.0],
                                [2000.0, 2.0, 12.6, -17.0],
                                [1000.0, 3.0, 18.6, -20.0],
                                [500.0, 4.0, 22.4, -23.0],
                                [250.0, 5.0, 26.4, -26.0],
                                [125.0, 6.0, 30.4, -29.0]])),
    )
    def test_init_modulations_array(self, expected):
        modulations_array = self.variant(init_modulations_array)()
        chex.assert_trees_all_close(modulations_array, expected)


class InitPathSEArrayTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        ("case_nsfnet", "nsfnet_deeprmsa_undirected", 1, jnp.array([2.0, 2.0, 2.0, 1.0])),
        ("case_4node", "4node", 1, jnp.array([6., 6., 6., 6.]))
    )
    def test_init_path_se_array(self, topology_name, k, expected):
        graph = make_graph(topology_name)
        path_link_array = init_path_link_array(graph, k)
        path_length_array = init_path_length_array(path_link_array, graph)
        modulations_array = init_modulations_array()
        path_se_array = self.variant(init_path_se_array)(path_length_array, modulations_array)
        chex.assert_trees_all_close(path_se_array[:4], expected)


class RequiredSlotsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_3_slot_request_test_setup(slot_size=12.5)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("case_empty", 100, 2, jnp.array([5])),
    )
    def test_required_slots(self, bit_rate, se, expected):
        slots = self.variant(required_slots)(bit_rate, se, self.params.slot_size)
        chex.assert_trees_all_close(slots, expected)


class MaskSlotsBitRateModFormat(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_nsfnet_16_mod_test_setup(env_type="rmsa")

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
                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,]).astype(jnp.float32)),
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
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(jnp.float32)),
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
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(jnp.float32)),
    )
    def test_mask_slots_bit_rate_mod_format(self, request, link_slot_array, expected):
        self.state = self.state.replace(link_slot_array=link_slot_array, request_array=request)
        jax.debug.print("state.link_slot_mask {}", self.state.link_slot_mask, ordered=True)
        updated_state = self.variant(self.env.action_mask, static_argnums=(1,))(self.state, self.params)
        jax.debug.print("state.link_slot_mask {}", updated_state.link_slot_mask, ordered=True)
        jax.debug.print("expected {}", expected, ordered=True)
        chex.assert_trees_all_close(updated_state.link_slot_mask, expected)


class AggregateSlotsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_agg_slots_test_setup()

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
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_agg_slots_test_setup()

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


if __name__ == '__main__':
    jax.config.update('jax_numpy_rank_promotion', 'raise')
    absltest.main()
