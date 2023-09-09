"""
Unit tests for `env_funcs.py`.
See `chex` github and docs for info on test framework.

Key points:
There is a class for each function under test.
chex.all_variants() decorator runs the test once for each variant (e.g. jitted, non-jitted, pmapped, etc.) of the function under test.
parameterized.named_parameters() decorator runs the test once for each set of parameters passed to the function under test.
"""
import os
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
from xlron.environments.env_funcs import *
from xlron.environments.vone import *
from xlron.environments.rsa import *


# Set the number of (emulated) host devices
num_devices = 4
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={num_devices}"


def keys_test_setup():
    rng = jax.random.PRNGKey(0)  # N.B. all test rely on 0 seed for reproducibility
    rng, key_init, key_reset, key_policy, key_step = jax.random.split(rng, 5)
    return rng, key_init, key_reset, key_policy, key_step


def settings_rsa_4node():
    return dict(load=100, k=2, topology_name="4node", link_resources=4, max_requests=10, min_slots=1,
                              max_slots=1, mean_service_holding_time=10)

def settings_vone_4node():
    return dict(**settings_rsa_4node(), node_resources=4, virtual_topologies=["3_ring"], min_node_resources=1, max_node_resources=1)


def vone_4node_test_setup():
    key = jax.random.PRNGKey(0)
    env, params = make_vone_env(**settings_vone_4node())
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def vone_nsfnet_16_test_setup():
    key = jax.random.PRNGKey(0)
    settings_vone_nsfnet_16 = dict(load=100, k=5, topology_name="nsfnet", link_resources=16, max_requests=10, min_slots=2,
                              max_slots=4, mean_service_holding_time=10, node_resources=4,
                                  virtual_topologies=["3_ring"], min_node_resources=1, max_node_resources=2)
    env, params = make_vone_env(**settings_vone_nsfnet_16)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_4node_test_setup():
    key = jax.random.PRNGKey(0)
    env, params = make_rsa_env(**settings_rsa_4node())
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_4node_3_slot_request_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rsa_4node_3_slots = settings_rsa_4node()
    settings_rsa_4node_3_slots["min_slots"] = 3
    settings_rsa_4node_3_slots["max_slots"] = 3
    settings_rsa_4node_3_slots["link_resources"] = 5
    env, params = make_rsa_env(**settings_rsa_4node_3_slots)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_16_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rsa_nsfnet_16 = dict(load=100, k=5, topology_name="nsfnet", link_resources=16, max_requests=10, min_slots=2,
                              max_slots=4, mean_service_holding_time=10)
    env, params = make_rsa_env(**settings_rsa_nsfnet_16)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params



class GenerateVoneRequestTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([[1,1,1,1,1,1,1], [2,1,3,1,4,1,2]])),
    )
    def test_generate_vone_request(self, expected):
        key = np.array([1, 2], dtype=np.uint32)
        state = self.variant(generate_vone_request)(key, self.state, self.params)
        request = state.request_array
        chex.assert_trees_all_close(request, expected)


class GenerateRSARequestTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0, 1, 1])),
    )
    def test_generate_rsa_request(self, expected):
        key = np.array([1, 2], dtype=np.uint32)
        state = self.variant(generate_rsa_request)(key, self.state, self.params)
        request = state.request_array
        chex.assert_trees_all_close(request, expected)


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
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

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
         jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                    [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                    [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ],
                    [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ], ])),
        ('case_12', jnp.array([1, 2]),
         jnp.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                    [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ],
                    [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
                    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, ],])),
    )
    def test_get_paths_nsfnet(self, nodes, expected):
        self.key, self.env, self.obs, self.state, self.params = vone_nsfnet_16_test_setup()
        paths = self.variant(get_paths, static_argnums=(0,))(self.params, nodes)
        chex.assert_trees_all_close(paths, expected)


class GenerateArrivalHoldingTimesTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', (jnp.array([0.08356591]), jnp.array([0.29246438]))),
    )
    def test_generate_arrival_times(self, expected):
        key = np.array([1, 2], dtype=np.uint32)
        arrival_time, holding_time = self.variant(generate_arrival_holding_times)(key, self.params)
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
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()
        path = jnp.array([1, 0, 0, 0])
        initial_slot_index = 0
        num_slots = 2
        self.state = self.state.replace(
            link_slot_array=vmap_update_path_links(self.state.link_slot_array, path, initial_slot_index, num_slots, 1),
            link_slot_departure_array=vmap_update_path_links_departure(self.state.link_slot_departure_array, path, initial_slot_index,
                                                             num_slots, self.state.current_time + self.state.holding_time)
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.full((4, 4), 0))
    )
    def test_remove_expired_slot_requests(self, expected):
        state = self.state.replace(current_time=10e4)
        updated_state = self.variant(remove_expired_slot_requests)(state)
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.full((4, 4), jnp.inf))
    )
    def test_remove_expired_slot_requests_departure(self, expected):
        state = self.state.replace(current_time=10e4)
        updated_state = self.variant(remove_expired_slot_requests)(state)
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class RemoveExpiredNodeRequestsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.full(4, 4))
    )
    def test_remove_expired_node_requests_capacity(self, s, d, sr, dr, n, expected):
        state = implement_node_action(self.state, s, d, sr, dr, n=n)
        state = finalise_vone_action(state)
        state = state.replace(current_time=10e4)
        updated_state = self.variant(remove_expired_node_requests)(state)
        chex.assert_trees_all_close(updated_state.node_capacity_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.full((4, 4), 0))
    )
    def test_remove_expired_node_requests_resource(self, s, d, sr, dr, n, expected):
        state = implement_node_action(self.state, s, d, sr, dr, n=n)
        state = finalise_vone_action(state)
        state = state.replace(current_time=10e4)
        updated_state = self.variant(remove_expired_node_requests)(state)
        chex.assert_trees_all_close(updated_state.node_resource_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.full((4, 4), jnp.inf))
    )
    def test_remove_expired_node_requests_departure(self, s, d, sr, dr, n, expected):
        state = implement_node_action(self.state, s, d, sr, dr, n=n)
        state = state.replace(current_time=10e4)
        updated_state = self.variant(remove_expired_node_requests)(state)
        chex.assert_trees_all_close(updated_state.node_departure_array, expected)


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


# TODO - test undo_link_slot_action
class UndoLinkSlotActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()
        path = jnp.array([1, 0, 0, 0])
        initial_slot_index = 0
        num_slots = 2
        self.state = self.state.replace(
            link_slot_array=vmap_update_path_links(self.state.link_slot_array, path, initial_slot_index, num_slots, 1),
            link_slot_departure_array=vmap_update_path_links_departure(self.state.link_slot_departure_array, path, initial_slot_index,
                                                             num_slots, -self.state.current_time-self.state.holding_time)
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.full((4, 4), 0))
    )
    def test_undo_link_slot_action(self, expected):
        state = self.state.replace(current_time=10e4)
        updated_state = self.variant(undo_link_slot_action)(state)
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.full((4, 4), jnp.inf))
    )
    def test_undo_link_slot_action_departure(self, expected):
        state = self.state.replace(current_time=10e4)
        updated_state = self.variant(undo_link_slot_action)(state)
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
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

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
        ('case_base', jnp.array([1,0,0,0]), 0, 2, jnp.array([[-2,-2,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf]])),
    )
    def test_implement_path_action_departure(self, path, initial_slot_index, num_slots, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_path_action)(state, path, initial_slot_index, num_slots)
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class ImplementVoneActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0,0,1]), 3, 3, jnp.array([[-1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])),
        ('case_base_long_path', jnp.array([0,5,1]), 3, 3, jnp.array([[0,0,0,0], [0,-1,0,0], [0,-1,0,0], [0,-1,0,0]])),
        ('case_base_single', jnp.array([0,0,1]), 3, 2, jnp.array([[-1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])),
        ('case_base_no_nodes', jnp.array([0,0,1]), 3, 1,
         jnp.array([[-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
    )
    def test_implement_vone_action_slots(self, action, total, remaining, expected):
        updated_state = self.variant(implement_vone_action, static_argnums=(4,))(self.state, action, total, remaining, self.params)
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)


    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0,0,1]), 3, 3, jnp.array([[-2,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf]])),
        ('case_base_long_path', jnp.array([0,5,1]), 3, 3, jnp.array([[jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,-2,jnp.inf,jnp.inf], [jnp.inf,-2,jnp.inf,jnp.inf], [jnp.inf,-2,jnp.inf,jnp.inf]])),
        ('case_base_single', jnp.array([0,0,1]), 3, 2, jnp.array([[-2,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf]])),
        ('case_base_no_nodes', jnp.array([0,0,1]), 3, 1,
         jnp.array([[-2, jnp.inf, jnp.inf, jnp.inf], [jnp.inf, jnp.inf, jnp.inf, jnp.inf], [jnp.inf, jnp.inf, jnp.inf, jnp.inf], [jnp.inf, jnp.inf, jnp.inf, jnp.inf]])),
    )
    def test_implement_vone_action_slots_departure(self, action, total, remaining, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_vone_action, static_argnums=(4,))(state, action, total, remaining, self.params)
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0,0,1]), 3, 3, jnp.array([3,3,4,4])),
        ('case_base_long_path', jnp.array([0,5,1]), 3, 3, jnp.array([3,3,4,4])),
        ('case_base_single', jnp.array([0,0,1]), 3, 2, jnp.array([4,3,4,4])),
        ('case_base_no_nodes', jnp.array([0,0,1]), 3, 1, jnp.array([4,4,4,4])),
    )
    def test_implement_vone_action_nodes_capacity(self, action, total, remaining, expected):
        updated_state = self.variant(implement_vone_action, static_argnums=(4,))(self.state, action, total, remaining, self.params)
        chex.assert_trees_all_close(updated_state.node_capacity_array, expected)


    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0,0,1]), 3, 3, jnp.array([[1,0,0,0], [1,0,0,0], [0,0,0,0], [0,0,0,0]])),
        ('case_base_long_path', jnp.array([0,5,1]), 3, 3, jnp.array([[1,0,0,0], [1,0,0,0], [0,0,0,0], [0,0,0,0]])),
        ('case_base_single', jnp.array([0,0,1]), 3, 2, jnp.array([[0,0,0,0], [1,0,0,0], [0,0,0,0], [0,0,0,0]])),
        ('case_base_no_nodes', jnp.array([0,0,1]), 3, 1,
         jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
    )
    def test_implement_vone_action_nodes_resource(self, action, total, remaining, expected):
        updated_state = self.variant(implement_vone_action, static_argnums=(4,))(self.state, action, total, remaining, self.params)
        chex.assert_trees_all_close(updated_state.node_resource_array, expected)


    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([0,0,1]), 3, 3, jnp.array([[-2,jnp.inf,jnp.inf,jnp.inf], [-2,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf]])),
        ('case_base_long_path', jnp.array([0,5,1]), 3, 3, jnp.array([[-2,jnp.inf,jnp.inf,jnp.inf], [-2,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf]])),
        ('case_base_single', jnp.array([0,0,1]), 3, 2, jnp.array([[jnp.inf,jnp.inf,jnp.inf,jnp.inf], [-2,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf], [jnp.inf,jnp.inf,jnp.inf,jnp.inf]])),
        ('case_base_no_nodes', jnp.array([0,0,1]), 3, 1,
         jnp.array([[jnp.inf, jnp.inf, jnp.inf, jnp.inf], [jnp.inf, jnp.inf, jnp.inf, jnp.inf], [jnp.inf, jnp.inf, jnp.inf, jnp.inf], [jnp.inf, jnp.inf, jnp.inf, jnp.inf]])),
    )
    def test_implement_vone_action_nodes_departure(self, action, total, remaining, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_vone_action, static_argnums=(4,))(state, action, total, remaining, self.params)
        chex.assert_trees_all_close(updated_state.node_departure_array, expected)


class ImplementRsaActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()


    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array(0), jnp.array([[0,0,0,0], [-1,0,0,0], [0,0,0,0], [0,0,0,0]])),
        ('case_base_long_path', jnp.array(5), jnp.array([[0,-1,0,0], [0,0,0,0], [0,-1,0,0], [0,-1,0,0]])),
    )
    def test_implement_rsa_action_slots(self, action, expected):
        updated_state = self.variant(implement_rsa_action, static_argnums=(2,))(self.state, action, self.params)
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)


    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array(0), jnp.array(
            [[jnp.inf, jnp.inf, jnp.inf, jnp.inf], [-2, jnp.inf, jnp.inf, jnp.inf],
             [jnp.inf, jnp.inf, jnp.inf, jnp.inf], [jnp.inf, jnp.inf, jnp.inf, jnp.inf]])),
        ('case_base_long_path', jnp.array(5), jnp.array(
            [[jnp.inf, -2, jnp.inf, jnp.inf], [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
             [jnp.inf, -2, jnp.inf, jnp.inf], [jnp.inf, -2, jnp.inf, jnp.inf]])),

    )
    def test_implement_rsa_action_slots_departure(self, action, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_rsa_action, static_argnums=(2,))(state, action, self.params)
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class CheckUniqueNodesTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.array(False)),
        ('case_base_single', 0, 3, 1, 2, 1, jnp.array(False))
    )
    def test_check_unique_nodes(self, s, d, sr, dr, n, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_node_action)(state, s, d, sr, dr, n=n)
        actual = self.variant(check_unique_nodes)(updated_state.node_departure_array)
        chex.assert_trees_all_close(actual, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.array(True)),
        ('case_base_single', 0, 3, 1, 2, 1, jnp.array(True))
    )
    def test_check_unique_nodes_fail(self, s, d, sr, dr, n, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_node_action)(state, s, d, sr, dr, n=n)
        updated_state = self.variant(implement_node_action)(updated_state, s, d, sr, dr, n=n)
        actual = self.variant(check_unique_nodes)(updated_state.node_departure_array)
        chex.assert_trees_all_close(actual, expected)


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


class CheckMinTwoNodesAssignedTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_pass', 0, 1, 1, 1, 2, jnp.array(False)),
        ('case_fail', 0, 1, 1, 1, 1, jnp.array(True)),
    )
    def test_check_min_two_nodes_assigned(self, s, d, sr, dr, n, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_node_action)(state, s, d, sr, dr, n=n)
        actual = self.variant(check_min_two_nodes_assigned)(updated_state.node_departure_array)
        chex.assert_trees_all_close(actual, expected)


class CheckNodeCapacitiesTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.array(False)),
    )
    def test_check_node_capacities(self, s, d, sr, dr, n, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_node_action)(state, s, d, sr, dr, n=n)
        actual = self.variant(check_node_capacities)(updated_state.node_capacity_array)
        chex.assert_trees_all_close(actual, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', 0, 1, 1, 1, 2, jnp.array(True)),
    )
    def test_check_node_capacities_fail(self, s, d, sr, dr, n, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        updated_state = self.variant(implement_node_action)(state, s, d, sr, dr, n=n)
        updated_state = self.variant(implement_node_action)(updated_state, s, d, sr, dr, n=n)
        updated_state = self.variant(implement_node_action)(updated_state, s, d, sr, dr, n=n)
        updated_state = self.variant(implement_node_action)(updated_state, s, d, sr, dr, n=n)
        updated_state = self.variant(implement_node_action)(updated_state, s, d, sr, dr, n=n)
        actual = self.variant(check_node_capacities)(updated_state.node_capacity_array)
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


class CheckTopologyTest(parameterized.TestCase):
    """check_topology() checks that each virtual node is assigned to a unique and consistent physical node.
    It compares the actions in action_history with the virtual topology pattern. Each virtual node in the pattern
     should line up with the same physical node in the action history, and vice versa."""

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_bus', jnp.array([3, 1, 2, 1, 1]), "3_bus", jnp.array(False)),
        ('case_bus_mixed', jnp.array([3, 1, 2, 2, 1]), "3_bus", jnp.array(False)),
        ('case_bus_zeroes', jnp.array([3, 1, 2, 2, 0]), "3_bus", jnp.array(False)),
        ('case_bus_fail', jnp.array([1, 1, 2, 1, 1]), "3_bus", jnp.array(True)),
        ('case_bus_zeroes_fail', jnp.array([0, 1, 2, 1, 0]), "3_bus", jnp.array(True)),
        ('case_ring', jnp.array([1, 1, 3, 1, 2, 1, 1]), "3_ring", jnp.array(False)),
        ('case_ring_mixed', jnp.array([1, 3, 3, 1, 2, 2, 1]), "3_ring", jnp.array(False)),
        ('case_ring_zeroes', jnp.array([0, 3, 3, 1, 2, 2, 0]), "3_ring", jnp.array(False)),
        ('case_ring_fail', jnp.array([0, 1, 3, 1, 2, 1, 1]), "3_ring", jnp.array(True)),
        ('case_ring_zeroes_fail', jnp.array([0, 1, 3, 1, 2, 1, 1]), "3_ring", jnp.array(True)),
    )
    def test_check_topology(self, action_history, pattern_name, expected):
        pattern = init_virtual_topology_patterns(pattern_name)[0]
        topology_pattern = jax.lax.dynamic_slice(pattern, (3,), (pattern.shape[0] - 3,))
        actual = self.variant(check_topology)(action_history, topology_pattern)
        chex.assert_trees_all_close(actual, expected)


class CheckVoneActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()


    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_pass', (jnp.array([0, 1, 1]),), 2,
         jnp.array(False)),
        ('case_fail', (jnp.array([1, 1, 1]),), 2,
         jnp.array(True)),
        ('case_no_remaining_action_pass', (jnp.array([0, 0, 1]), jnp.array([1, 1, 2]), jnp.array([2, 2, 0])), 3,
            jnp.array(False)),
        ('case_no_remaining_action_fail_all_nodes', (jnp.array([0, 0, 1]), jnp.array([1, 1, 2]), jnp.array([2, 2, 0])), 4,
            jnp.array(True)),
        ('case_no_remaining_action_fail_topology', (jnp.array([0, 0, 1]), jnp.array([1, 1, 2]), jnp.array([2, 2, 3])), 3,
            jnp.array(True)),
    )
    def test_check_vone_action(self, actions, total_requested_nodes, expected):
        for action in actions:
            total_actions = jnp.squeeze(jax.lax.dynamic_slice(self.state.action_counter, (1,), (1,)))
            remaining_actions = jnp.squeeze(jax.lax.dynamic_slice(self.state.action_counter, (2,), (1,)))
            self.state = self.variant(implement_vone_action, static_argnums=(4,))\
                (self.state, action, total_actions, remaining_actions, self.params)
            self.state = self.state.replace(action_history=update_action_history(self.state.action_history, self.state.action_counter, action))
            self.state = self.state.replace(action_counter=decrease_last_element(self.state.action_counter))
        actual = self.variant(check_vone_action)(self.state, remaining_actions, total_requested_nodes)
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
    def test_check_rsa_action(self, actions, expected):
        for action in actions:
            self.state = implement_rsa_action(self.state, action, self.params)
        actual = self.variant(check_rsa_action)(self.state)
        chex.assert_trees_all_close(actual, expected)


class FinaliseVoneActiontest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()
        self.state = self.state.replace(current_time=1, holding_time=1)
        self.state = implement_vone_action(self.state, jnp.array([0, 1, 1]), 3, 3, self.params)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([[2, jnp.inf, jnp.inf, jnp.inf],
                                 [2, jnp.inf, jnp.inf, jnp.inf],
                                 [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                                 [jnp.inf, jnp.inf, jnp.inf, jnp.inf]])),
    )
    def test_finalise_vone_action_node_departure(self, expected):
        actual = self.variant(finalise_vone_action)(self.state).node_departure_array
        chex.assert_trees_all_close(actual, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_base', jnp.array([[jnp.inf, 2, jnp.inf, jnp.inf],
                                 [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                                 [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                                 [jnp.inf, jnp.inf, jnp.inf, jnp.inf]])),
    )
    def test_finalise_vone_action_link_slot_departure(self, expected):
        actual = self.variant(finalise_vone_action)(self.state).link_slot_departure_array
        chex.assert_trees_all_close(actual, expected)


class FinaliseRsaActionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()
        self.state = self.state.replace(current_time=1, holding_time=1)
        self.state = implement_rsa_action(self.state, jnp.array(0), self.params)

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_pass', jnp.array([[jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                                 [2, jnp.inf, jnp.inf, jnp.inf],
                                 [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                                 [jnp.inf, jnp.inf, jnp.inf, jnp.inf]]))
    )
    def test_finalise_rsa_action(self, expected):
        actual = self.variant(finalise_rsa_action)(self.state).link_slot_departure_array
        chex.assert_trees_all_close(actual, expected)


class PathActionOnlyTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_not_already_assigned', jnp.array([2, 1, 3, 1, 4]), jnp.array([3, 2, 1]), 1, jnp.array(False)),
        ('case_already_assigned', jnp.array([2, 1, 3, 1, 4, 1, 2]), jnp.array([3, 2, 1]), 1, jnp.array(True)),
        ('case_not_already_assigned_longer', jnp.array([2, 1, 3, 1, 4, 1, 2, 1, 5, 1, 4]), jnp.array([4, 5, 2]), 2, jnp.array(False)),
        ('case_already_assigned_longer', jnp.array([2, 1, 3, 1, 4, 1, 2, 1, 5, 1, 4]), jnp.array([4, 5, 2]), 1, jnp.array(True)),
        ('case_first_action', jnp.array([2, 1, 3, 1, 4, 1, 2, 1, 5, 1, 4]), jnp.array([4, 5, 5]), 5,
         jnp.array(False)),
    )
    def test_path_action_only(self, topology_pattern, action_counter, remaining_actions, expected):
        actual = self.variant(path_action_only)(topology_pattern, action_counter, remaining_actions)
        chex.assert_trees_all_close(actual, expected)


# TODO - could potentially add more test cases here
class VoneStepTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_success", (jnp.array([0, 0, 1]), jnp.array([1, 1, 2]), jnp.array([2, 2, 0])),
         jnp.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 4, 1, 2, 3, 3, 3, 4, -1, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0])),
        ("case_failure", (jnp.array([0, 0, 0]),),
        jnp.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 4, 1, 2, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("case_neutral", (jnp.array([0, 0, 1]),),
        jnp.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 4, 1, 2, 3, 3, 4, 4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    )
    def test_vone_step_obs(self, actions, expected):
        for action in actions:
            obs, self.state, reward, done, info = self.variant(self.env.step, static_argnums=(3,))(
                self.key, self.state, action, self.params
            )
        chex.assert_trees_all_close(obs, expected)


class VoneResetTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base",
         jnp.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 4, 1, 2, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    )
    def test_vone_reset_obs(self, expected):
        obs, self.state = self.variant(self.env.reset, static_argnums=(1,))(self.key, self.params)
        chex.assert_trees_all_close(obs, expected)


class RsaStepTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_success", (jnp.array(0),),
         jnp.array([0.,  1.,  3., 0.,  0.,  0.,  0.,  -1.,  0.,  0.,  0., 0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.])),
        ("case_failure", (jnp.array(0), jnp.array(0)),
         jnp.array([0., 1., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0.]))
    )
    def test_rsa_step_obs(self, actions, expected):
        for action in actions:
            obs, self.state, reward, done, info = self.variant(self.env.step, static_argnums=(3,))(
                self.key, self.state, action, self.params
            )
        chex.assert_trees_all_close(obs, expected)


class RsaResetTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base",
         jnp.array([0., 1., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0.])),
    )
    def test_vone_reset_obs(self, expected):
        obs, self.state = self.variant(self.env.reset, static_argnums=(1,))(self.key, self.params)
        chex.assert_trees_all_close(obs, expected)


class RsaActionMaskTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

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
        chex.assert_trees_all_close(state.link_slot_mask, expected)

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
        state = self.variant(self.env.action_mask, static_argnums=(1,))(self.state, self.params)
        chex.assert_trees_all_close(state.link_slot_mask, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_start_edge", jnp.array([0, 3, 1]),
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
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],]),
         jnp.array(
             [
                 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             ]
         )
         ),
    )
    def test_rsa_action_mask_nsfnet_16(self, request_array, link_slot_array, expected):
        self.key, self.env, self.obs, self.state, self.params = rsa_nsfnet_16_test_setup()
        self.state = self.state.replace(request_array=request_array, link_slot_array=link_slot_array)
        state = self.variant(self.env.action_mask, static_argnums=(1,))(self.state, self.params)
        chex.assert_trees_all_close(state.link_slot_mask, expected)


class MaskNodesTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_empty",
         jnp.array([4, 4, 4, 4]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, -1, -1, -1]),  # action_history
         jnp.array([3, 3, 3]),  # action_counter
         jnp.array([[1, 1, 1, 1, 1, 1, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([1, 1, 1, 1, 1, 1, 1, 1])),  # expected
        ("case_full",
         jnp.array([0, 0, 0, 0]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, -1, -1, -1]),  # action_history
         jnp.array([3, 3, 3]),  # action_counter
         jnp.array([[1, 1, 1, 1, 1, 1, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([0, 0, 0, 0, 0, 0, 0, 0])),  # expected
        ("case_second_move",
         jnp.array([4, 4, 3, 3]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, 2, 1, 3]),  # action_history
         jnp.array([3, 3, 2]),  # action_counter
         jnp.array([[1, 1, 1, 1, 1, 1, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([0, 0, 1, 0, 1, 1, 0, 0])),  # expected
        ("case_third_move",
         jnp.array([3, 4, 3, 3]),  # node_capacity_array
         jnp.array([-1, -1, 0, 1, 2, 1, 3]),  # action_history
         jnp.array([3, 3, 1]),  # action_counter
         jnp.array([[1, 1, 1, 1, 1, 1, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([1, 0, 0, 0, 0, 0, 0, 1])),  # expected
        ("case_third_move_big_requests",
         jnp.array([3, 4, 3, 3]),  # node_capacity_array
         jnp.array([-1, -1, 0, 1, 2, 1, 3]),  # action_history
         jnp.array([3, 3, 1]),  # action_counter
         jnp.array([[3, 1, 3, 1, 3, 1, 3], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([1, 0, 0, 0, 0, 0, 0, 1])),  # expected
    )
    def test_mask_nodes(self, node_capacity_array, action_history, action_counter, request_array, expected):
        self.state = self.state.replace(
            node_capacity_array=node_capacity_array,
            action_history=action_history,
            action_counter=action_counter,
            request_array=request_array
        )
        state = self.variant(mask_nodes, static_argnums=(1,))(self.state, self.params.num_nodes)
        node_mask = jnp.concatenate([state.node_mask_s, state.node_mask_d], axis=0)
        chex.assert_trees_all_close(node_mask, expected)


class VoneActionMaskTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_empty",
         jnp.array([0, 1, 1]),  # action
         jnp.array([4, 4, 4, 4]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, -1, -1, -1]),  # action_history
         jnp.array([3, 3, 3]),  # action_counter
         jnp.array([[1, 1, 1, 1, 1, 1, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),  # link_slot_array
         jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),  # expected
        ("case_full_nodes",
         jnp.array([0, 1, 1]),  # action
         jnp.array([0, 0, 0, 0]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, -1, -1, -1]),  # action_history
         jnp.array([3, 3, 3]),  # action_counter
         jnp.array([[1, 1, 1, 1, 1, 1, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),  # link_slot_array
         jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])),  # expected
        ("case_full_slots",
         jnp.array([0, 1, 1]),  # action
         jnp.array([1, 1, 1, 1]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, -1, -1, -1]),  # action_history
         jnp.array([3, 3, 3]),  # action_counter
         jnp.array([[1, 1, 1, 1, 1, 1, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),  # link_slot_array
         jnp.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])),  # expected
        ("case_second_move",
         jnp.array([2, 1, 1]),  # action
         jnp.array([4, 4, 3, 3]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, 2, 0, 3]),  # action_history
         jnp.array([3, 3, 2]),  # action_counter
         jnp.array([[1, 1, 1, 1, 1, 1, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]),  # link_slot_array
         jnp.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0])),  # expected
        ("case_third_move_blocked_link",
         jnp.array([0, 1, 3]),  # action
         jnp.array([3, 4, 3, 3]),  # node_capacity_array
         jnp.array([-1, -1, 0, 1, 2, 1, 3]),  # action_history
         jnp.array([3, 3, 1]),  # action_counter
         jnp.array([[1, 1, 1, 1, 1, 1, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]]),  # link_slot_array
         jnp.array([1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])),  # expected
        ("case_third_move_free_link",
         jnp.array([0, 1, 3]),  # action
         jnp.array([3, 4, 3, 3]),  # node_capacity_array
         jnp.array([-1, -1, 0, 1, 2, 1, 3]),  # action_history
         jnp.array([3, 3, 1]),  # action_counter
         jnp.array([[1, 1, 1, 1, 1, 1, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),  # link_slot_array
         jnp.array([1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1])),  # expected
    )
    def test_vone_action_mask(self, action, node_capacity_array, action_history, action_counter, request_array, link_slot_array, expected):
        self.state = self.state.replace(
            node_capacity_array=node_capacity_array,
            action_history=action_history,
            action_counter=action_counter,
            request_array=request_array,
            link_slot_array=link_slot_array,
        )
        state = self.variant(self.env.action_mask_nodes, static_argnums=(1,))(self.state, self.params)
        state = self.variant(self.env.action_mask_slots, static_argnums=(1,))(state, self.params, action)
        mask = jnp.concatenate([state.node_mask_s, state.link_slot_mask, state.node_mask_d], axis=0)
        chex.assert_trees_all_close(mask, expected)


if __name__ == '__main__':
    jax.config.update('jax_numpy_rank_promotion', 'raise')
    absltest.main()
