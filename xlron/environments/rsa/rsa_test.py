"""
Unit tests for `rsa.py`.
See `chex` github and docs for info on test framework.

Key points:
There is a class for each function under test.
chex.all_variants() decorator runs the test once for each variant (e.g. jitted, non-jitted, pmapped, etc.) of the function under test.
parameterized.named_parameters() decorator runs the test once for each set of parameters passed to the function under test.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from xlron.environments.dataclasses import *
from xlron.environments.env_funcs import *
from xlron.environments.env_funcs_test import *
from xlron.environments.rsa.rsa import *
from xlron.environments.wrappers import *


class GenerateRSARequestTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base", jnp.array([1.0, 1.0, 3.0])),
    )
    def test_generate_rsa_request(self, expected):
        key = np.array([1, 2], dtype=np.uint32)
        state = self.variant(generate_request_rsa)(key, self.state, self.params)
        request = state.request_array
        chex.assert_trees_all_close(request, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_base", (jnp.array([0, 1, 1]), jnp.array([1, 1, 2]))),
    )
    def test_generate_rsa_request_from_list(self, expected):
        key = np.array([1, 2], dtype=np.uint32)
        self.params = self.params.replace(deterministic_requests=True)
        # Fresh-episode counter (-1): the next generated request is number 0 = list row 0
        self.state = self.state.replace(
            list_of_requests=jnp.array([[0, 1, 1], [1, 1, 2]]),
            total_requests=jnp.array(-1, dtype=self.state.total_requests.dtype),
        )
        self.state = self.variant(generate_request_rsa)(key, self.state, self.params)
        request1 = self.state.request_array
        self.state = self.variant(generate_request_rsa)(key, self.state, self.params)
        request2 = self.state.request_array
        chex.assert_trees_all_close((request1, request2), expected)


class ImplementRsaActionTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    def _action_info(self, state, action, params, env=None):
        env = env or self.env
        return env.process_action(state, action, params)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_base",
            jnp.array(0),
            jnp.array([[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]),
        ),
        (
            "case_base_long_path",
            jnp.array(5),
            jnp.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]),
        ),
    )
    def test_implement_action_rsa_slots(self, action, expected):
        action_info = self._action_info(self.state, action, self.params)
        updated_state = self.variant(implement_action_rsa, static_argnums=(2,))(
            self.state, action_info, self.params
        )
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_base",
            jnp.array(19),
            jnp.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    )
    def test_implement_action_rsa_slots_nsfnet4(self, action, expected):
        key, env, obs, state, params = rsa_nsfnet_4_test_setup()
        action_info = self._action_info(state, action, params, env=env)
        updated_state = self.variant(implement_action_rsa, static_argnums=(2,))(
            state, action_info, params
        )
        chex.assert_trees_all_close(updated_state.link_slot_array, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_base",
            jnp.array(0),
            jnp.array([[2, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]]),
        ),
        (
            "case_base_long_path",
            jnp.array(5),
            jnp.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]]),
        ),
    )
    def test_implement_action_rsa_slots_departure(self, action, expected):
        state = self.state.replace(current_time=1, holding_time=1)
        action_info = self._action_info(state, action, self.params)
        updated_state = self.variant(implement_action_rsa, static_argnums=(2,))(
            state, action_info, self.params
        )
        chex.assert_trees_all_close(updated_state.link_slot_departure_array, expected)


class CheckRsaActionTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_pass", (jnp.array(0), jnp.array(2)), jnp.array(False)),
        ("case_fail", (jnp.array(0), jnp.array(0)), jnp.array(True)),
    )
    def test_check_action_rsa(self, actions, expected):
        action_info = None
        for action in actions:
            action_info = self.env.process_action(self.state, action, self.params)
            self.state = implement_action_rsa(self.state, action_info, self.params)
        actual = self.variant(check_action_rsa)(self.state, action_info, self.params)
        chex.assert_trees_all_close(actual, expected)


class RsaStepTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_success",
            (jnp.array(0),),
            jnp.array(
                [
                    2.0,
                    1.0,
                    3.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
        (
            "case_failure",
            (jnp.array(0), jnp.array(0)),
            jnp.array(
                [
                    2.0,
                    1.0,
                    3.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
    )
    def test_rsa_step_obs(self, actions, expected):
        obs: Array = jnp.array(0)
        for action in actions:
            print(action)
            obs, self.state, reward, done, truncated, info = self.variant(
                self.env.step, static_argnums=(3,)
            )(self.key, self.state, action, self.params)
        chex.assert_trees_all_close(obs, expected)


class RsaResetTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_base",
            jnp.array(
                [
                    0.0,
                    1.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
    )
    def test_rsa_reset_obs(self, expected):
        obs, self.state = self.variant(self.env.reset, static_argnums=(1,))(self.key, self.params)
        chex.assert_trees_all_close(obs, expected)


class RsaActionMaskTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Expected masks below include the trailing no-op action, so pin
        # include_no_op=True (the flag default, now applied to dict configs too, is False).
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup(
            include_no_op=True
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_empty",
            jnp.array([0, 1, 1]),
            jnp.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ),
            jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ),
        (
            "case_full",
            jnp.array([0, 1, 1]),
            jnp.array(
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ),
            jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            "case_start_edge",
            jnp.array([0, 1, 1]),
            jnp.array(
                [
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                ]
            ),
            jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            "case_end_edge",
            jnp.array([0, 1, 1]),
            jnp.array(
                [
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                ]
            ),
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
        ),
    )
    def test_rsa_action_mask(self, request_array, link_slot_array, expected):
        self.state = self.state.replace(
            request_array=request_array, link_slot_array=link_slot_array
        )
        link_slot_mask, full_link_slot_mask = self.variant(
            self.env.action_mask, static_argnums=(1,)
        )(self.state, self.params)
        chex.assert_trees_all_close(link_slot_mask, expected)

    # N.B. that requested bandwidth (middle number of request array) will lead to bw+1 slots allocated
    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_start_edge_3",
            jnp.array([0, 3, 1]),
            jnp.array(
                [
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                ]
            ),
            jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            "case_end_edge_3",
            jnp.array([0, 3, 1]),
            jnp.array(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                ]
            ),
            jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
        ),
        (
            "case_start_edge_2",
            jnp.array([0, 2, 1]),
            jnp.array(
                [
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                ]
            ),
            jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            "case_end_edge_2",
            jnp.array([0, 2, 1]),
            jnp.array(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                ]
            ),
            jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        ),
        (
            "case_middle_2",
            jnp.array([0, 2, 1]),
            jnp.array(
                [
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                ]
            ),
            jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
        ),
        (
            "case_middle_3",
            jnp.array([0, 3, 1]),
            jnp.array(
                [
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                ]
            ),
            jnp.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        (
            "case_middle_1",
            jnp.array([0, 1, 1]),
            jnp.array(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 1],
                ]
            ),
            jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
        ),
    )
    def test_rsa_action_mask_3_slot_request(self, request_array, link_slot_array, expected):
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_3_slot_request_test_setup(
            include_no_op=True
        )
        self.state = self.state.replace(
            request_array=request_array, link_slot_array=link_slot_array
        )
        self.params = self.params.replace(max_slots=3)
        link_slot_mask, full_link_slot_mask = self.variant(
            self.env.action_mask, static_argnums=(1,)
        )(self.state, self.params)
        chex.assert_trees_all_close(link_slot_mask, expected)

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_start_edge",
            jnp.array([0, 2, 1]),
            True,
            jnp.array(
                [
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
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
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            # N.B. that modulation format consideration means double spectral efficiency on the first path, hence two 1's
            jnp.array(
                [
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
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
                    0,
                    0,
                    0,
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
                    0,
                    0,
                    0,
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
                    0,
                    0,
                    0,
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
                    0,
                    0,
                    0,
                    1,
                ]
            ).astype(jnp.float32),
        ),
        (
            "case_rwa",
            jnp.array([0, 1, 1]),
            False,
            jnp.array(
                [
                    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
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
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
            jnp.array(
                [
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
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
                    0,
                    0,
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
                    0,
                    0,
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
                    0,
                    0,
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
                    0,
                    0,
                    1,
                ]
            ).astype(jnp.float32),
        ),
    )
    def test_rsa_action_mask_nsfnet_16(
        self, request_array, consider_mod, link_slot_array, expected
    ):
        self.key, self.env, self.obs, self.state, self.params = rsa_nsfnet_16_test_setup(
            guardband=0,
            env_type="rmsa",
            include_no_op=True,
            # Expectations were computed with this modulations table (the
            # pre-unification implicit default); the flag default is now
            # modulations_deeprmsa.csv, so pin it explicitly.
            modulations_csv_filepath="./xlron/data/modulations/modulations.csv",
        )
        self.state = self.state.replace(
            request_array=request_array, link_slot_array=link_slot_array
        )
        jax.debug.print("request_array {}", request_array, ordered=True)
        self.params = self.params.replace(max_slots=3, consider_modulation_format=consider_mod)
        link_slot_mask, full_link_slot_mask = self.variant(
            self.env.action_mask, static_argnums=(1,)
        )(self.state, self.params)
        chex.assert_trees_all_close(link_slot_mask, expected)


class RewardTypeBitrateTest(chex.TestCase):
    """Regression tests for reward_type='bitrate'.

    The success reward was previously zeroed by multiplying the bitrate by the
    zero-initialised reward, so a successful step returned 0 while a failure
    returned -bitrate/max(values_bw) (asymmetric, no positive reinforcement).
    """

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_3_slot_request_test_setup(
            reward_type="bitrate"
        )

    @chex.all_variants()
    def test_success_reward_is_normalised_bitrate(self):
        # values_bw=[3], so a successful placement gives 3 / max([3]) = 1.0
        obs, state, reward, done, truncated, info = self.variant(
            self.env.step, static_argnums=(3,)
        )(self.key, self.state, jnp.array(0), self.params)
        chex.assert_trees_all_close(reward, jnp.array(1.0, dtype=reward.dtype))

    @chex.all_variants()
    def test_failure_reward_is_negative_normalised_bitrate(self):
        # Force a blocked action by filling the network
        state = self.state.replace(link_slot_array=jnp.ones_like(self.state.link_slot_array))
        obs, state, reward, done, truncated, info = self.variant(
            self.env.step, static_argnums=(3,)
        )(self.key, state, jnp.array(0), self.params)
        chex.assert_trees_all_close(reward, jnp.array(-1.0, dtype=reward.dtype))


class EndFirstBlockingBitrateTest(chex.TestCase):
    """Regression test: with end_first_blocking + reward_type='bitrate', is_terminal
    used to compare the reward against the failure reward of the NEXT request (the
    request array is regenerated before is_terminal runs), so blocked steps were not
    terminal whenever consecutive request bitrates differed.
    """

    def setUp(self):
        super().setUp()
        settings = settings_rwa_4node()
        settings.update(
            env_type="rsa",
            values_bw=[1, 3],
            link_resources=5,
            incremental_loading=True,
            end_first_blocking=True,
            reward_type="bitrate",
        )
        self.key = jax.random.PRNGKey(0)
        self.env, self.params = make(settings, log_wrapper=False)
        self.obs, self.state = self.env.reset(self.key, self.params)

    def test_blocked_step_is_terminal(self):
        step = jax.jit(self.env.step, static_argnums=(3,))
        state = self.state
        rng = self.key
        bitrates = []
        for _ in range(10):
            rng, key_step = jax.random.split(rng)
            # Force every action to be blocked by filling the network
            state = state.replace(link_slot_array=jnp.ones_like(state.link_slot_array))
            # Read the current bitrate before stepping (step donates state buffers)
            bitrates.append(float(state.request_array[1]))
            obs, state, reward, done, truncated, info = step(
                key_step, state, jnp.array(0), self.params
            )
            self.assertLess(float(reward), 0.0)
            self.assertTrue(bool(done), "Blocked step must terminate with end_first_blocking")
        # Sanity: the traffic contains different bitrates, so the pre-fix behavior
        # (terminal only when consecutive bitrates match) would have failed above
        self.assertGreater(len(set(bitrates)), 1)


def rsa_multiband_4node_test_setup(**kwargs):
    settings = dict(
        load=100,
        k=2,
        topology_name="4node",
        link_resources=20,
        max_requests=10,
        mean_service_holding_time=10,
        env_type="rsa_multiband",
        values_bw=[25],
        slot_size=12.5,
        guardband=0,
        # 25 GHz gap starting at 100 GHz -> 2-slot gap at slots 8-9
        interband_gap_width=[25],
        interband_gap_start=[100],
        # These tests exercise the custom interband_gap_* branch, which is only
        # reached when enforce_band_gaps is off (the flag default, now applied to
        # dict configs too, is True = CSV-derived band gaps).
        enforce_band_gaps=False,
    )
    settings.update(kwargs)
    key = jax.random.PRNGKey(0)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


class RsaMultibandBandGapTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        (
            self.key,
            self.env,
            self.obs,
            self.state,
            self.params,
        ) = rsa_multiband_4node_test_setup()

    def test_custom_gap_flags_respected(self):
        """--interband_gap_width/--interband_gap_start must produce the requested gaps.

        Regression: an inverted conditional in make() discarded user-supplied values
        (yielding no gaps at all) and only ever applied the hardcoded defaults."""
        chex.assert_trees_all_equal(self.params.gap_starts.val, jnp.array([8]))
        chex.assert_trees_all_equal(self.params.gap_widths.val, jnp.array([2]))
        self.assertTrue(jnp.all(self.state.link_slot_array[:, 8:10] == -1))
        self.assertTrue(jnp.all(self.state.link_slot_array[:, :8] == 0))
        self.assertTrue(jnp.all(self.state.link_slot_array[:, 10:] == 0))

    def test_default_gaps_when_flags_unset(self):
        """Without gap flags, the [200, 200] GHz @ [4425, 8425] GHz defaults apply."""
        _, _, _, _, params = rsa_multiband_4node_test_setup(
            interband_gap_width=None, interband_gap_start=None
        )
        chex.assert_trees_all_equal(params.gap_starts.val, jnp.array([354, 674]))
        chex.assert_trees_all_equal(params.gap_widths.val, jnp.array([16, 16]))

    def test_expiry_preserves_gap_sentinels(self):
        """remove_expired_services_rsa must clear expired services but keep -1 gaps.

        Regression: link_slot_array was multiplied by keep=(dep > t), and gap slots
        carry dep == 0, so the first expiry pass erased the sentinels and opened the
        inter-band gaps to placement."""
        lsa = self.state.link_slot_array
        dep = self.state.link_slot_departure_array
        # Occupy slot 0 on link 0 with a service departing at t=5, then expire at t=10
        lsa = lsa.at[0, 0].set(jnp.asarray(1, dtype=lsa.dtype))
        dep = dep.at[0, 0].set(jnp.asarray(5, dtype=dep.dtype))
        t = jnp.asarray(10)
        state = self.state.replace(
            link_slot_array=lsa,
            link_slot_departure_array=dep,
            current_time=t.astype(self.state.current_time.dtype),
            arrival_time=t.astype(self.state.arrival_time.dtype),
        )
        new_state = remove_expired_services_rsa(state, self.params)
        self.assertEqual(float(new_state.link_slot_array[0, 0]), 0.0)  # ty: ignore[unresolved-attribute]
        self.assertTrue(jnp.all(new_state.link_slot_departure_array == 0))  # ty: ignore[unresolved-attribute]
        self.assertTrue(
            jnp.all(new_state.link_slot_array[:, 8:10] == -1),  # ty: ignore[unresolved-attribute]
            f"Gap sentinels erased by expiry: {new_state.link_slot_array}",  # ty: ignore[unresolved-attribute]
        )

    def test_utilisation_excludes_gap_slots(self):
        """Utilisation must count only positively-occupied slots over usable slots."""
        mask, _ = self.env.action_mask(self.state, self.params)
        self.assertTrue(bool(jnp.any(mask > 0)))
        action = jnp.argmax(mask)
        _, new_state, _, _, _, info = self.env.step(self.key, self.state, action, self.params)
        lsa = np.asarray(new_state.link_slot_array)
        occupied = np.count_nonzero(lsa > 0)
        usable = np.count_nonzero(lsa >= 0)
        # Gap slots (2 per link) are excluded from the usable spectrum
        self.assertEqual(usable, lsa.size - 2 * lsa.shape[0])
        self.assertGreater(occupied, 0)
        chex.assert_trees_all_close(
            info["_utilisation"], jnp.asarray(occupied / usable, dtype=info["_utilisation"].dtype)
        )


class RsaResetPreservesTrafficParamsTest(chex.TestCase):
    """Regression tests for the episodic load-sweep bug: runtime-patched
    arrival_rate/mean_service_holding_time (as set by the load sweep in
    train.py) must survive episode auto-resets instead of reverting to the
    construction-time values baked into initial_state."""

    def setUp(self):
        super().setUp()
        # rwa_4node settings are episodic (max_requests=10, no continuous_operation)
        self.key, self.env, self.obs, self.state, self.params = rwa_4node_test_setup()

    def test_auto_reset_preserves_swept_traffic_params(self):
        # Patch the live state the way _update_experiment_input_load does
        patched = self.state.replace(
            arrival_rate=jnp.full_like(self.state.arrival_rate, 99.0),
            mean_service_holding_time=jnp.full_like(self.state.mean_service_holding_time, 7.0),
            # Next step's request generation reaches max_requests -> truncation
            total_requests=jnp.array(
                self.params.max_requests - 1, dtype=self.state.total_requests.dtype
            ),
        )
        _, new_state, _, terminal, truncated, _ = self.env.step(
            self.key, patched, jnp.array(0), self.params
        )
        self.assertTrue(bool(truncated | terminal))
        chex.assert_trees_all_close(
            new_state.arrival_rate, jnp.full_like(new_state.arrival_rate, 99.0)
        )
        chex.assert_trees_all_close(
            new_state.mean_service_holding_time,
            jnp.full_like(new_state.mean_service_holding_time, 7.0),
        )

    def test_fresh_reset_uses_params_values(self):
        _, state = self.env.reset(self.key, self.params)
        chex.assert_trees_all_close(
            state.arrival_rate,
            jnp.full_like(state.arrival_rate, self.params.arrival_rate),
        )
        chex.assert_trees_all_close(
            state.mean_service_holding_time,
            jnp.full_like(state.mean_service_holding_time, self.params.mean_service_holding_time),
        )

    def test_reset_env_with_state_preserves_traffic_params(self):
        patched = self.state.replace(
            arrival_rate=jnp.full_like(self.state.arrival_rate, 42.5),
            mean_service_holding_time=jnp.full_like(self.state.mean_service_holding_time, 3.0),
        )
        _, state = self.env.reset_env(self.key, self.params, patched)
        chex.assert_trees_all_close(state.arrival_rate, jnp.full_like(state.arrival_rate, 42.5))
        chex.assert_trees_all_close(
            state.mean_service_holding_time,
            jnp.full_like(state.mean_service_holding_time, 3.0),
        )
        # Everything else resets: spectrum empty, counters back to start
        chex.assert_trees_all_close(state.link_slot_array, jnp.zeros_like(state.link_slot_array))


if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
