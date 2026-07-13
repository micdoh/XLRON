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


def rwa_lightpath_reuse_4_nsfnet_test_setup(**kwargs):
    settings = dict(
        k=5,
        topology_name="nsfnet_deeprmsa_undirected",
        link_resources=4,
        max_requests=1000,
        values_bw=[100],
        incremental_loading=True,
        env_type="rwa_lightpath_reuse",
        scale_factor=1.0,
    )
    if not kwargs:
        return _rwa_lr_cached_setup("rwa_lr_nsfnet_4", settings)
    settings.update(kwargs)
    key = jax.random.PRNGKey(0)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_lightpath_reuse_4node_test_setup(**kwargs):
    settings = dict(
        k=2,
        topology_name="4node",
        link_resources=4,
        max_requests=1000,
        values_bw=[100],
        incremental_loading=True,
        env_type="rwa_lightpath_reuse",
    )
    if not kwargs:
        return _rwa_lr_cached_setup("rwa_lr_4node", settings)
    settings.update(kwargs)
    key = jax.random.PRNGKey(0)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


class CheckLightpathAvailableAndExistingTest(chex.TestCase):
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


class MaskSlotsRWALightpathReuseTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Expected masks below include the trailing no-op action, so pin
        # include_no_op=True (the flag default, now applied to dict configs too, is False).
        self.key, self.env, self.obs, self.state, self.params = (
            rwa_lightpath_reuse_4node_test_setup(include_no_op=True)
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


class RWALightpathReuseTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        # test_end_episode samples actions from the mask, so the expected capacities
        # depend on the action-space size and path ordering: pin include_no_op=True
        # and path_sort_criteria="hops" (the flag defaults, now applied to dict
        # configs too, are False and "spectral_resources").
        self.key, self.env, self.obs, self.state, self.params = (
            rwa_lightpath_reuse_4_nsfnet_test_setup(include_no_op=True, path_sort_criteria="hops")
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_end_episode",
            jnp.array(
                [
                    [1.0e06, 1.0e06, 6.0e02, 7.0e02],
                    [1.0e06, 1.0e06, 1.0e06, 1.0e06],
                    [1.0e06, 1.0e06, 6.0e02, 7.0e02],
                    [1.0e06, 1.0e06, 1.0e06, 1.0e06],
                    [1.0e06, 1.0e06, 6.0e02, 1.0e06],
                    [1.0e06, 1.0e06, 1.0e06, 1.0e06],
                    [7.0e02, 1.0e06, 1.0e06, 1.0e06],
                    [7.0e02, 1.0e06, 1.0e06, 1.0e06],
                    [7.0e02, 1.0e06, 1.0e06, 1.0e06],
                    [1.0e06, 1.0e06, 1.0e06, 1.0e06],
                    [1.0e06, 7.0e02, 1.0e06, 1.0e06],
                    [1.0e06, 7.0e02, 1.0e06, 9.0e02],
                    [1.0e06, 1.0e06, 1.0e06, 1.0e06],
                    [1.0e06, 1.0e06, 1.0e06, 1.0e06],
                    [1.0e06, 1.0e06, 6.0e02, 7.0e02],
                    [1.0e06, 7.0e02, 1.0e06, 1.0e03],
                    [1.0e06, 7.0e02, 6.0e02, 7.0e02],
                    [1.0e06, 1.0e06, 1.0e06, 1.0e03],
                    [1.0e06, 7.0e02, 1.2e03, 1.0e06],
                    [7.0e02, 1.0e06, 1.0e06, 1.0e06],
                    [1.0e06, 1.0e06, 1.0e06, 1.0e03],
                    [1.0e06, 7.0e02, 1.0e06, 1.0e03],
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


class AggregateSlotsMaskTest(chex.TestCase):
    """aggregate_slots > 1 must not crash RWA-LR masking (regression: mask_slots_rwalr
    unpacked aggregate_slots' single-array return into two names). Uses non-divisible
    link_resources to exercise the ceil/padding path end to end."""

    def test_masked_step_with_aggregation(self):
        settings = dict(
            k=5,
            topology_name="nsfnet_deeprmsa_undirected",
            link_resources=5,
            max_requests=10,
            values_bw=[100],
            incremental_loading=True,
            env_type="rwa_lightpath_reuse",
            scale_factor=1.0,
            aggregate_slots=2,
            include_no_op=False,
        )
        env, params = make(settings, log_wrapper=False)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        mask, full_mask = env.action_mask(state, params)
        # k * ceil(5 / 2) = 5 * 3 aggregated actions; full mask stays k * 5
        self.assertEqual(mask.shape[0], 15)
        self.assertEqual(full_mask.shape[0], 25)
        self.assertTrue(bool(jnp.any(mask > 0)))
        state = state.replace(link_slot_mask=mask, full_link_slot_mask=full_mask)
        action = jnp.argmax(mask)  # first valid aggregated action
        _, state, reward, terminal, _, _ = jax.jit(env.step, static_argnums=(3,))(
            key, state, action, params
        )
        # The step must implement a real (in-range) slot: exactly one lightpath placed
        self.assertTrue(bool(jnp.any(state.path_index_array >= 0)))


class OverCapacityRejectionTest(chex.TestCase):
    """Over-capacity placements must be rejected via the total_mask == 2 sentinel.

    Regression: the exhausted/over masks were computed after restoring the over-drawn
    capacity, so a mask-bypassing action on a full lightpath was counted as accepted
    while no capacity was deducted."""

    def test_repeated_action_stops_being_accepted_at_zero_capacity(self):
        key, env, obs, state, params = rwa_lightpath_reuse_4_nsfnet_test_setup()
        step = jax.jit(env.step, static_argnums=(3,))
        # Fix the request so every step asks for the same 100 on the same source-dest
        request = state.request_array
        action = jnp.array(0)  # first path, first slot
        accepted_history = []
        capacity = None
        slots = None
        for i in range(14):
            state = state.replace(request_array=request)
            if i == 0:
                # Locate the affected slots from the pre-step state with the fixed request
                action_info = env.process_action(state, action, params)
                slots = jnp.where(action_info.affected_slots_mask > 0)
            _, state, reward, *_ = step(key, state, action, params)
            accepted_history.append(int(state.accepted_services))
            if i == 0:
                # Lightpath established: remaining = initial - 100
                capacity = float(state.link_capacity_array[slots][0])
        initial_capacity = capacity + 100.0
        expected_accepts = int(initial_capacity // 100.0)
        # Accepts must stop exactly when capacity is exhausted, then stay frozen
        self.assertEqual(accepted_history[-1], expected_accepts)
        self.assertEqual(accepted_history[expected_accepts - 1], expected_accepts)
        # The full lightpath sits at exactly 0 and no capacity ever goes negative
        self.assertEqual(float(state.link_capacity_array[slots][0]), 0.0)
        self.assertGreaterEqual(float(jnp.min(state.link_capacity_array)), 0.0)


class MixedPrecisionCarryTest(chex.TestCase):
    """RWA-LR under --mixed_precision must keep link_slot_array at the SMALL_FLOAT tier
    (regression: implement_action_rwalr wrote it back as LARGE_FLOAT, a lax.scan carry
    dtype mismatch)."""

    def test_step_preserves_small_float_link_slot_array(self):
        from xlron import dtype_config

        settings = dict(
            k=5,
            topology_name="nsfnet_deeprmsa_undirected",
            link_resources=4,
            max_requests=10,
            values_bw=[100],
            incremental_loading=True,
            env_type="rwa_lightpath_reuse",
            scale_factor=1.0,
            mixed_precision=True,
        )
        try:
            env, params = make(settings, log_wrapper=False)
            key = jax.random.PRNGKey(0)
            obs, state = env.reset(key, params)
            init_dtype = state.link_slot_array.dtype
            self.assertEqual(init_dtype, dtype_config.SMALL_FLOAT_DTYPE)
            mask, full_mask = env.action_mask(state, params)
            # Store masks at SMALL_FLOAT exactly as select_action does
            state = state.replace(
                link_slot_mask=mask.astype(dtype_config.SMALL_FLOAT_DTYPE),
                full_link_slot_mask=full_mask.astype(dtype_config.SMALL_FLOAT_DTYPE),
            )
            _, state, *_ = jax.jit(env.step, static_argnums=(3,))(
                key, state, jnp.argmax(mask), params
            )
            self.assertEqual(state.link_slot_array.dtype, init_dtype)
        finally:
            # initialize_dtypes mutates module globals; restore defaults for other tests
            settings["mixed_precision"] = False
            make(settings, log_wrapper=False)


class DynamicExpiryCapacityRestoreTest(chex.TestCase):
    """Expired RWA-LR services must return their slots to the 1e6 empty-capacity sentinel
    (regression: capacity was never restored, so freed slots stayed masked at their stale
    reduced capacity)."""

    def test_expiry_restores_capacity(self):
        settings = dict(
            k=5,
            topology_name="nsfnet_deeprmsa_undirected",
            link_resources=4,
            max_requests=100,
            values_bw=[100],
            incremental_loading=False,
            env_type="rwa_lightpath_reuse",
            scale_factor=1.0,
            load=100,
            mean_service_holding_time=25,
            # This test drives expiry with an absolute far-future current_time, so pin
            # the absolute-time mode (the flag default, now applied to dict configs
            # too, is relative).
            relative_arrival_times=False,
        )
        env, params = make(settings, log_wrapper=False)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        mask, full_mask = env.action_mask(state, params)
        self.assertTrue(bool(jnp.any(mask > 0)))
        state = state.replace(link_slot_mask=mask, full_link_slot_mask=full_mask)
        _, state, *_ = jax.jit(env.step, static_argnums=(3,))(key, state, jnp.argmax(mask), params)
        occupied = state.path_index_array != -1
        self.assertTrue(bool(jnp.any(occupied)))
        self.assertLess(float(state.link_capacity_array[occupied].min()), 1e6)

        far_future = jnp.full_like(state.current_time, 1e7)
        expired = remove_expired_services_rwalr(state.replace(current_time=far_future), params)
        self.assertTrue(bool(jnp.all(expired.path_index_array == -1)))  # ty: ignore[unresolved-attribute]
        self.assertTrue(bool(jnp.all(expired.link_capacity_array == 1e6)))  # ty: ignore[unresolved-attribute]
        self.assertTrue(bool(jnp.all(expired.link_slot_array == 0)))  # ty: ignore[unresolved-attribute]


if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
