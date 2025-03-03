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
    os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4"
    jax.config.update('jax_numpy_rank_promotion', 'raise')
    absltest.main()
