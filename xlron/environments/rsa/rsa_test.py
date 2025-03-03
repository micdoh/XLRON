"""
Unit tests for `rsa.py`.
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
from xlron.environments.env_funcs_test import *
from xlron.environments.rsa.rsa import *
from xlron.environments.wrappers import *
from xlron.environments.dataclasses import *
from xlron.environments.make_env import make


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


if __name__ == '__main__':
    os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4"
    jax.config.update('jax_numpy_rank_promotion', 'raise')
    absltest.main()
