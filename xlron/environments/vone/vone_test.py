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
from xlron.environments.vone import *
from xlron.environments.rsa import *
from xlron.environments.wrappers import *
from xlron.environments.dataclasses import *
from xlron.environments.make_env import make


def keys_test_setup():
    rng = jax.random.PRNGKey(0)  # N.B. all test rely on 0 seed for reproducibility
    rng, key_init, key_reset, key_policy, key_step = jax.random.split(rng, 5)
    return rng, key_init, key_reset, key_policy, key_step


def settings_vone_4node():
    return dict(load=100, k=2, topology_name="4node", link_resources=4, max_requests=10, mean_service_holding_time=10,
                node_resources=4, virtual_topologies=["3_ring"], min_node_resources=1, max_node_resources=1,
                values_bw=[1], slot_size=1, env_type="vone", guardband=0)


def settings_vone_4node_mod():
    return dict(load=100, k=2, topology_name="4node", link_resources=4, max_requests=10, mean_service_holding_time=10,
                node_resources=4, virtual_topologies=["3_ring"], min_node_resources=1, max_node_resources=1, consider_modulation_format=True,
                env_type="vone", guardband=0)


def vone_4node_test_setup():
    key = jax.random.PRNGKey(0)
    env, params = make(settings_vone_4node(), log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def vone_4node_mod_test_setup():
    key = jax.random.PRNGKey(0)
    env, params = make(settings_vone_4node_mod(), log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def vone_nsfnet_16_test_setup():
    key = jax.random.PRNGKey(0)
    settings_vone_nsfnet_16 = dict(load=100, k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=16, max_requests=10,
                                   values_bw=[1, 2, 3], slot_size=1,  consider_modulation_format=False,
                                   mean_service_holding_time=10, node_resources=4, virtual_topologies=["3_ring"],
                                   min_node_resources=1, max_node_resources=2, env_type="vone")
    env, params = make(settings_vone_nsfnet_16, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def vone_nsfnet_16_mod_test_setup():
    key = jax.random.PRNGKey(0)
    settings_vone_nsfnet_16_mod = dict(load=100, k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=16, max_requests=10,
                                   mean_service_holding_time=10, node_resources=4, virtual_topologies=["3_ring"],
                                   min_node_resources=1, max_node_resources=2, consider_modulation_format=True,
                                       env_type="vone")
    env, params = make(settings_vone_nsfnet_16_mod, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


class GenerateVoneRequestTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = vone_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_1', np.array([1, 2], dtype=np.uint32), jnp.array([[1, 0, 1, 0, 1, 0, 1], [2,1,3,1,4,1,2]])),
        ('case_2', np.array([2, 1], dtype=np.uint32), jnp.array([[1, 0, 1, 0, 1, 0, 1], [2, 1, 3, 1, 4, 1, 2]])),
        ('case_3', np.array([5, 5], dtype=np.uint32), jnp.array([[1, 0, 1, 0, 1, 0, 1], [2, 1, 3, 1, 4, 1, 2]])),
    )
    def test_generate_vone_request(self, key, expected):
        state = self.variant(generate_vone_request)(key, self.state, self.params)
        request = state.request_array
        chex.assert_trees_all_close(request, expected)


    @chex.all_variants()
    @parameterized.named_parameters(
        ('case_1', np.array([1, 2], dtype=np.uint32), jnp.array([[2, 2, 1, 3, 2, 3, 2], [2,1,3,1,4,1,2]])),
        ('case_2', np.array([2, 1], dtype=np.uint32), jnp.array([[2, 1, 2, 1, 1, 2, 2], [2, 1, 3, 1, 4, 1, 2]])),
        ('case_3', np.array([5, 5], dtype=np.uint32), jnp.array([[1, 3, 1, 1, 2, 3, 1], [2, 1, 3, 1, 4, 1, 2]])),
    )
    def test_generate_vone_nsfnet(self, key, expected):
        self.key, self.env, self.obs, self.state, self.params = vone_nsfnet_16_test_setup()
        state = self.variant(generate_vone_request)(key, self.state, self.params)
        request = state.request_array
        chex.assert_trees_all_close(request, expected)


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
        state = finalise_vone_action(state)
        state = state.replace(current_time=10e4)
        updated_state = self.variant(remove_expired_node_requests)(state)
        chex.assert_trees_all_close(updated_state.node_departure_array, expected)


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
        ('case_base', jnp.array([0,0,1]), 3, 3, jnp.array([[-2,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])),
        ('case_base_long_path', jnp.array([0,5,1]), 3, 3, jnp.array([[0,0,0,0], [0,-2,0,0], [0,-2,0,0], [0,-2,0,0]])),
        ('case_base_single', jnp.array([0,0,1]), 3, 2, jnp.array([[-2,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])),
        ('case_base_no_nodes', jnp.array([0,0,1]), 3, 1,
         jnp.array([[-2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
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
        ('case_base', jnp.array([[0, 2, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]])),
    )
    def test_finalise_vone_action_link_slot_departure(self, expected):
        actual = self.variant(finalise_vone_action)(self.state).link_slot_departure_array
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
         jnp.array([1, 0, 1, 0, 1, 0, 1, 2, 1, 3, 1, 4, 1, 2, 3, 3, 3, 4, -1, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0])),
        ("case_failure", (jnp.array([0, 0, 0]),),
        jnp.array([1, 0, 1, 0, 1, 0, 1, 2, 1, 3, 1, 4, 1, 2, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ("case_neutral", (jnp.array([0, 0, 1]),),
        jnp.array([1, 0, 1, 0, 1, 0, 1, 2, 1, 3, 1, 4, 1, 2, 3, 3, 4, 4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
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
         jnp.array([1, 0, 1, 0, 1, 0, 1, 2, 1, 3, 1, 4, 1, 2, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    )
    def test_vone_reset_obs(self, expected):
        obs, self.state = self.variant(self.env.reset, static_argnums=(1,))(self.key, self.params)
        chex.assert_trees_all_close(obs, expected)


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

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_third_move_big_requests",
         jnp.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, 7, 15, 1]),  # action_history
         jnp.array([3, 3, 2]),  # action_counter
         jnp.array([[2, 2, 1, 4, 2, 4, 2], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])),  # expected
    )
    def test_mask_nodes_nsfnet(self, node_capacity_array, action_history, action_counter, request_array, expected):
        self.key, self.env, self.obs, self.state, self.params = vone_nsfnet_16_test_setup()
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
         jnp.array([[1, 0, 1, 0, 1, 0, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),  # link_slot_array
         jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),  # expected
        ("case_full_nodes",
         jnp.array([0, 1, 1]),  # action
         jnp.array([0, 0, 0, 0]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, -1, -1, -1]),  # action_history
         jnp.array([3, 3, 3]),  # action_counter
         jnp.array([[1, 0, 1, 0, 1, 0, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),  # link_slot_array
         jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])),  # expected
        ("case_full_slots",
         jnp.array([0, 1, 1]),  # action
         jnp.array([1, 1, 1, 1]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, -1, -1, -1]),  # action_history
         jnp.array([3, 3, 3]),  # action_counter
         jnp.array([[1, 0, 1, 0, 1, 0, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),  # link_slot_array
         jnp.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])),  # expected
        ("case_second_move",
         jnp.array([2, 1, 1]),  # action
         jnp.array([4, 4, 3, 3]),  # node_capacity_array
         jnp.array([-1, -1, -1, -1, 2, 0, 3]),  # action_history
         jnp.array([3, 3, 2]),  # action_counter
         jnp.array([[1, 0, 1, 0, 1, 0, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]),  # link_slot_array
         jnp.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0])),  # expected
        ("case_third_move_blocked_link",
         jnp.array([0, 1, 3]),  # action
         jnp.array([3, 4, 3, 3]),  # node_capacity_array
         jnp.array([-1, -1, 0, 1, 2, 1, 3]),  # action_history
         jnp.array([3, 3, 1]),  # action_counter
         jnp.array([[1, 0, 1, 0, 1, 0, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
         jnp.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]]),  # link_slot_array
         jnp.array([1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])),  # expected
        ("case_third_move_free_link",
         jnp.array([0, 1, 3]),  # action
         jnp.array([3, 4, 3, 3]),  # node_capacity_array
         jnp.array([-1, -1, 0, 1, 2, 1, 3]),  # action_history
         jnp.array([3, 3, 1]),  # action_counter
         jnp.array([[1, 0, 1, 0, 1, 0, 1], [2, 1, 3, 1, 4, 1, 2]]),  # request_array
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
