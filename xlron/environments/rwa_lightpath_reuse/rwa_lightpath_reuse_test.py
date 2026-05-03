import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import chex
import distrax
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from xlron.environments.dataclasses import *
from xlron.environments.env_funcs import *
from xlron.environments.make_env import make
from xlron.environments.rsa.rsa import *
from xlron.environments.wrappers import *


# Module-level caches for expensive make() calls.
# Only env and params are cached; obs/state are recomputed per call via
# env.reset() to avoid buffer donation/deletion issues with chex device variants.
_rwa_lr_cache = {}


def _rwa_lr_cached_setup(cache_key, settings):
    """Return (key, env, obs, state, params) using cached env/params."""
    key = jax.random.PRNGKey(0)
    if cache_key not in _rwa_lr_cache:
        env, params = make(settings, log_wrapper=False)
        _rwa_lr_cache[cache_key] = (env, params)
    env, params = _rwa_lr_cache[cache_key]
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_lightpath_reuse_4_nsfnet_test_setup():
    return _rwa_lr_cached_setup(
        "rwa_lr_nsfnet_4",
        dict(
            k=5,
            topology_name="nsfnet_deeprmsa_undirected",
            link_resources=4,
            max_requests=1000,
            values_bw=[100],
            incremental_loading=True,
            env_type="rwa_lightpath_reuse",
            scale_factor=1.0,
        ),
    )


def rwa_lightpath_reuse_4node_test_setup():
    return _rwa_lr_cached_setup(
        "rwa_lr_4node",
        dict(
            k=2,
            topology_name="4node",
            link_resources=4,
            max_requests=1000,
            values_bw=[100],
            incremental_loading=True,
            env_type="rwa_lightpath_reuse",
        ),
    )


class CheckLightpathAvailableAndExistingTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = (
            rwa_lightpath_reuse_4node_test_setup()
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_available",
            jnp.array(
                [
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                ]
            ),
            jnp.int32(3),
            jnp.array(True),
            jnp.array(False),
        ),
        (
            "case_not_available",
            jnp.array(
                [
                    [
                        -1,
                        -1,
                        -1,
                        1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                ]
            ),
            jnp.int32(3),
            jnp.array(False),
            jnp.array(False),
        ),
        (
            "case_available_existing",
            jnp.array(
                [
                    [
                        1,
                        2,
                        3,
                        0,
                    ],
                    [
                        4,
                        5,
                        6,
                        7,
                    ],
                    [
                        8,
                        9,
                        10,
                        11,
                    ],
                    [
                        12,
                        13,
                        14,
                        15,
                    ],
                ]
            ),
            jnp.int32(3),
            jnp.array(True),
            jnp.array(True),
        ),
        (
            "case_not_available_existing",
            jnp.array(
                [
                    [
                        -1,
                        -1,
                        -1,
                        0,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                ]
            ),
            jnp.int32(3),
            jnp.array(True),  # Always available if exists
            jnp.array(True),
        ),
    )
    def test_lightpath_available_existing(
        self,
        path_index_array,
        action,
        expected_available,
        expected_existing,
        request=jnp.array([0, 100, 1]),
    ):
        state = self.state.replace(path_index_array=path_index_array, request_array=request)
        action_info = self.env.process_action(state, action, self.params)
        result_available, result_existing, curr_lightpath_capacity, lightpath_index = self.variant(
            check_lightpath_available_and_existing, static_argnums=(2,)
        )(state, action_info, self.params)
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
        self.key, self.env, self.obs, self.state, self.params = (
            rwa_lightpath_reuse_4node_test_setup()
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_no_capacity",
            jnp.array(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                ]
            ),
            jnp.array([0, 100, 1]),
            jnp.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ),
        ),
        (
            "case_capacity_no_path",
            jnp.array(
                [
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        99,
                        99,
                        99,
                        99,
                    ],
                    [
                        99,
                        99,
                        99,
                        99,
                    ],
                    [
                        99,
                        99,
                        99,
                        99,
                    ],
                    [
                        99,
                        99,
                        99,
                        99,
                    ],
                ]
            ),
            jnp.array([0, 100, 1]),
            jnp.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ),
        ),
        (
            "case_capacity_path",
            jnp.array(
                [
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        99,
                        99,
                        99,
                        -1,
                    ],
                    [
                        99,
                        99,
                        99,
                        99,
                    ],
                    [
                        99,
                        99,
                        99,
                        99,
                    ],
                    [
                        99,
                        99,
                        99,
                        99,
                    ],
                ]
            ),
            jnp.array([0, 100, 1]),
            jnp.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ),
        ),
        (
            "case_no_capacity_path",
            jnp.array(
                [
                    [
                        100.0,
                        100.0,
                        100.0,
                        0.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        99,
                        99,
                        99,
                        -1,
                    ],
                    [
                        99,
                        99,
                        99,
                        99,
                    ],
                    [
                        99,
                        99,
                        99,
                        99,
                    ],
                    [
                        99,
                        99,
                        99,
                        99,
                    ],
                ]
            ),
            jnp.array([0, 100, 1]),
            jnp.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ),
        ),
        (
            "case_2_path",
            jnp.array(
                [
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                    [
                        100.0,
                        100.0,
                        100.0,
                        100.0,
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        99,
                        99,
                        99,
                        0,
                    ],
                    [
                        99,
                        1,
                        99,
                        99,
                    ],
                    [
                        99,
                        1,
                        99,
                        99,
                    ],
                    [
                        99,
                        1,
                        99,
                        99,
                    ],
                ]
            ),
            jnp.array([0, 100, 1]),
            jnp.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ),
        ),
    )
    def test_mask_slots_rwa_lightpath_reuse(
        self, link_capacity_array, path_index_array, request, expected
    ):
        state = self.state.replace(
            link_capacity_array=link_capacity_array, path_index_array=path_index_array
        )
        link_slot_mask, _ = self.variant(mask_slots_rwalr, static_argnums=(1,))(
            state, self.params, request
        )
        jax.debug.print("link_slot_mask {}", link_slot_mask, ordered=True)
        jax.debug.print("expected {}", expected, ordered=True)
        chex.assert_trees_all_close(link_slot_mask, expected)


class RWALightpathReuseTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = (
            rwa_lightpath_reuse_4_nsfnet_test_setup()
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_end_episode",
            jnp.array(
                [
                    [1.0e06, 1.0e06, 6.0e02, 1.0e06],
                    [1.0e06, 1.0e06, 1.0e06, 5.0e02],
                    [1.0e06, 1.0e06, 6.0e02, 5.0e02],
                    [1.0e06, 1.0e06, 1.0e06, 7.0e02],
                    [1.0e06, 7.0e02, 6.0e02, 7.0e02],
                    [1.0e06, 6.0e02, 1.0e06, 5.0e02],
                    [7.0e02, 7.0e02, 6.0e02, 1.0e03],
                    [7.0e02, 8.0e02, 6.0e02, 7.0e02],
                    [7.0e02, 7.0e02, 6.0e02, 1.0e06],
                    [1.0e06, 6.0e02, 1.0e06, 1.0e03],
                    [7.0e02, 6.0e02, 1.0e06, 5.0e02],
                    [7.0e02, 7.0e02, 6.0e02, 9.0e02],
                    [1.0e06, 1.0e06, 1.0e06, 5.0e02],
                    [1.0e06, 6.0e02, 1.0e06, 1.0e06],
                    [1.0e06, 1.0e06, 6.0e02, 1.0e06],
                    [7.0e02, 1.0e06, 1.0e03, 1.0e03],
                    [7.0e02, 1.0e06, 6.0e02, 1.0e06],
                    [1.0e06, 1.0e06, 1.0e03, 1.0e03],
                    [7.0e02, 8.0e02, 1.2e03, 1.0e06],
                    [7.0e02, 1.0e06, 6.0e02, 1.0e06],
                    [1.0e06, 7.0e02, 1.0e03, 1.4e03],
                    [7.0e02, 1.0e06, 1.0e03, 1.0e03],
                ]
            ),
        ),
    )
    def test_end_episode(self, expected):
        remaining_capacity: Array = jnp.array(0)
        rng, reset_rng = jax.random.split(self.key)
        obsv, env_state = self.env.reset(reset_rng, self.params)
        reward = jnp.array([0.0])
        i = 0
        while reward == 0:
            i += 1
            rng, rng_sample, rng_step = jax.random.split(rng, 3)
            # get mask
            mask, _ = self.env.action_mask(env_state, self.params)
            env_state = env_state.replace(link_slot_mask=mask)
            # make distribution
            action_dist = distrax.Categorical(logits=jnp.where(mask, mask, -1e8))
            jax.debug.print("action dist {}", action_dist.logits, ordered=True)
            # sample distribution
            action = action_dist.sample(seed=rng_sample)
            jax.debug.print("action {}", action, ordered=True)
            # step env
            remaining_capacity = (
                env_state.link_capacity_array
            )  # capture to avoid being reset to initial state
            obsv, env_state, reward, terminal, truncated, info = self.variant(
                self.env.step, static_argnums=(3)
            )(rng_step, env_state, action, self.params)
            jax.debug.print("action mask {}", env_state.link_slot_mask, ordered=True)
            jax.debug.print("action {}", action, ordered=True)
            jax.debug.print("reward {}", reward, ordered=True)
            jax.debug.print("remaining_capacity {}", remaining_capacity, ordered=True)
            jax.debug.print("-----END-----")
            if i == 1000:
                break
        chex.assert_trees_all_close(remaining_capacity, expected)


if __name__ == "__main__":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
