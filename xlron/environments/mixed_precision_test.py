"""Tests for mixed-precision support (see xlron/dtype_config.py and --mixed_precision).

Design contract under test:
  * Default mode keeps every env array at 32-bit (reclassification is a no-op).
  * --mixed_precision shrinks the dominant carried env arrays (link_slot_array, departure,
    times) to 16-bit while keeping NN compute/params, accumulators and physical floats at f32.
  * Time arrays auto-fall back to float32 when the clock is unbounded (absolute arrival times or
    incremental_loading), and a forced 16-bit time_dtype in those cases raises.
  * Blocking-probability parity holds between default and mixed precision.

Note: dtype constants are process-global; make()/process_config() re-resolves them from each
config, so every test re-initialises them through make() and restores defaults on tearDown.
"""

import jax
import jax.numpy as jnp
from absl.testing import absltest
from box import Box

import xlron.dtype_config as dtype_config
from xlron.environments.make_env import make


def _cfg(env_type="rmsa", mixed=False, **extra):
    cfg = dict(
        env_type=env_type,
        topology_name="nsfnet_deeprmsa_directed",
        link_resources=40,
        k=5,
        load=200,
        continuous_operation=True,
        mean_service_holding_time=25,
        truncate_holding_time=True,
        relative_arrival_times=True,
        ENV_WARMUP_STEPS=0,
        TOTAL_TIMESTEPS=4000,
        NUM_ENVS=1,
        ROLLOUT_LENGTH=40,
        STEPS_PER_INCREMENT=40,
        NUM_MINIBATCHES=1,
        mixed_precision=mixed,
        SEED=42,
    )
    cfg.update(extra)
    return Box(cfg)


def _pytree_bytes(tree):
    return sum(int(x.nbytes) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "nbytes"))


def _unwrap(state):
    return state.env_state if hasattr(state, "env_state") else state


def _rollout(env, params, n_steps=80, seed=0):
    """Greedy first-valid-action rollout; returns final unwrapped env state."""
    raw = env._env
    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key, params)
    step = jax.jit(env.step)
    for _ in range(n_steps):
        key, skey = jax.random.split(key)
        es = _unwrap(state)
        mask = raw.action_mask(es, params)
        mask = mask[0] if isinstance(mask, tuple) else mask
        action = jnp.argmax(mask)
        # LogWrapper.step returns (obs, log_state, reward, terminal, truncated, info).
        out = step(skey, state, action, params)
        state = out[1]
    return _unwrap(state)


class DtypeResolutionTest(absltest.TestCase):
    def tearDown(self):
        dtype_config.initialize_dtypes(Box({}))  # restore 32-bit defaults
        super().tearDown()

    def test_default_is_all_32bit(self):
        dtype_config.initialize_dtypes(Box({}))
        for c in [
            dtype_config.LARGE_FLOAT_DTYPE,
            dtype_config.SMALL_FLOAT_DTYPE,
            dtype_config.TIME_DTYPE,
        ]:
            self.assertEqual(jnp.dtype(c), jnp.dtype(jnp.float32))
        for c in [
            dtype_config.LARGE_INT_DTYPE,
            dtype_config.SMALL_INT_DTYPE,
            dtype_config.BINARY_DTYPE,
        ]:
            self.assertEqual(jnp.dtype(c), jnp.dtype(jnp.int32))

    def test_mixed_relative_shrinks_tiers(self):
        dtype_config.initialize_dtypes(
            Box({"mixed_precision": True, "relative_arrival_times": True})
        )
        self.assertEqual(jnp.dtype(dtype_config.SMALL_FLOAT_DTYPE), jnp.dtype(jnp.float16))
        self.assertEqual(jnp.dtype(dtype_config.TIME_DTYPE), jnp.dtype(jnp.float16))
        self.assertEqual(jnp.dtype(dtype_config.SMALL_INT_DTYPE), jnp.dtype(jnp.int16))
        self.assertEqual(jnp.dtype(dtype_config.BINARY_DTYPE), jnp.dtype(jnp.int8))
        # Precision tiers stay 32-bit.
        self.assertEqual(jnp.dtype(dtype_config.LARGE_FLOAT_DTYPE), jnp.dtype(jnp.float32))
        self.assertEqual(jnp.dtype(dtype_config.LARGE_INT_DTYPE), jnp.dtype(jnp.int32))
        self.assertEqual(jnp.dtype(dtype_config.COMPUTE_DTYPE), jnp.dtype(jnp.float32))
        self.assertEqual(jnp.dtype(dtype_config.PARAMS_DTYPE), jnp.dtype(jnp.float32))

    def test_mixed_absolute_time_stays_f32(self):
        dtype_config.initialize_dtypes(
            Box({"mixed_precision": True, "relative_arrival_times": False})
        )
        self.assertEqual(jnp.dtype(dtype_config.TIME_DTYPE), jnp.dtype(jnp.float32))
        # Bulk float tier still shrinks.
        self.assertEqual(jnp.dtype(dtype_config.SMALL_FLOAT_DTYPE), jnp.dtype(jnp.float16))

    def test_mixed_incremental_loading_time_stays_f32(self):
        dtype_config.initialize_dtypes(
            Box(
                {
                    "mixed_precision": True,
                    "relative_arrival_times": True,
                    "incremental_loading": True,
                }
            )
        )
        self.assertEqual(jnp.dtype(dtype_config.TIME_DTYPE), jnp.dtype(jnp.float32))

    def test_differentiable_overrides_to_f32(self):
        dtype_config.initialize_dtypes(Box({"mixed_precision": True, "differentiable": True}))
        for c in [
            dtype_config.SMALL_FLOAT_DTYPE,
            dtype_config.TIME_DTYPE,
            dtype_config.BINARY_DTYPE,
        ]:
            self.assertEqual(jnp.dtype(c), jnp.dtype(jnp.float32))

    def test_granular_flag_overrides_mixed_default(self):
        dtype_config.initialize_dtypes(
            Box(
                {
                    "mixed_precision": True,
                    "relative_arrival_times": True,
                    "small_float_dtype": "bfloat16",
                }
            )
        )
        self.assertEqual(jnp.dtype(dtype_config.SMALL_FLOAT_DTYPE), jnp.dtype(jnp.bfloat16))


class EnvStateDtypeTest(absltest.TestCase):
    def tearDown(self):
        dtype_config.initialize_dtypes(Box({}))
        super().tearDown()

    def test_rmsa_carried_arrays_are_f16_in_mixed(self):
        env, params = make(_cfg("rmsa", mixed=True))
        es = _rollout(env, params, n_steps=40)
        self.assertEqual(jnp.dtype(es.link_slot_array.dtype), jnp.dtype(jnp.float16))
        self.assertEqual(jnp.dtype(es.link_slot_departure_array.dtype), jnp.dtype(jnp.float16))
        self.assertEqual(jnp.dtype(es.current_time.dtype), jnp.dtype(jnp.float16))
        # Counters / accumulators stay 32-bit.
        self.assertEqual(jnp.dtype(es.total_requests.dtype), jnp.dtype(jnp.int32))
        self.assertEqual(jnp.dtype(es.accepted_bitrate.dtype), jnp.dtype(jnp.float32))

    def test_rmsa_default_arrays_are_f32(self):
        env, params = make(_cfg("rmsa", mixed=False))
        es = _rollout(env, params, n_steps=40)
        self.assertEqual(jnp.dtype(es.link_slot_array.dtype), jnp.dtype(jnp.float32))
        self.assertEqual(jnp.dtype(es.link_slot_departure_array.dtype), jnp.dtype(jnp.float32))

    def test_mixed_reduces_env_state_memory(self):
        env_d, params_d = make(_cfg("rmsa", mixed=False))
        _, state_d = env_d.reset(jax.random.PRNGKey(0), params_d)
        bytes_default = _pytree_bytes(_unwrap(state_d))
        env_m, params_m = make(_cfg("rmsa", mixed=True))
        _, state_m = env_m.reset(jax.random.PRNGKey(0), params_m)
        bytes_mixed = _pytree_bytes(_unwrap(state_m))
        self.assertLess(bytes_mixed, bytes_default)


def _ensemble_blocking_prob(env, params, n_envs=128, n_steps=80, seed=0):
    """Aggregate blocking probability over an ensemble of parallel envs (vmapped over keys).

    A single greedy trajectory is chaotic: one float16 rounding at a tie-break forks the path and
    divergence cascades. The aggregate over many independent envs is the meaningful, self-averaging
    quantity (this mirrors how XLRON actually runs -- thousands of parallel envs).
    """
    raw = env._env
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    _, state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    step_v = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    mask_v = jax.vmap(raw.action_mask, in_axes=(0, None))

    def body(carry, _):
        state, key = carry
        key, sub = jax.random.split(key)
        es = _unwrap(state)
        mask = mask_v(es, params)
        mask = mask[0] if isinstance(mask, tuple) else mask
        actions = jnp.argmax(mask, axis=-1)
        skeys = jax.random.split(sub, actions.shape[0])
        out = step_v(skeys, state, actions, params)
        return (out[1], key), None

    (state, _), _ = jax.lax.scan(body, (state, jax.random.PRNGKey(seed + 1)), None, length=n_steps)
    es = _unwrap(state)
    accepted = float(jnp.sum(es.accepted_services))
    total = float(jnp.sum(es.total_timesteps))
    return 1.0 - accepted / total


class BlockingParityTest(absltest.TestCase):
    def tearDown(self):
        dtype_config.initialize_dtypes(Box({}))
        super().tearDown()

    def test_blocking_probability_parity_rmsa(self):
        env_d, params_d = make(_cfg("rmsa", mixed=False))
        bp_default = _ensemble_blocking_prob(env_d, params_d)
        env_m, params_m = make(_cfg("rmsa", mixed=True))
        bp_mixed = _ensemble_blocking_prob(env_m, params_m)
        # Aggregate blocking probability must agree within statistical noise; float16 occupancy/
        # time arrays do not change the steady-state rate (validated at larger scale in the PR).
        self.assertLessEqual(abs(bp_default - bp_mixed), 0.01)


class TimePrecisionGuardTest(absltest.TestCase):
    def tearDown(self):
        dtype_config.initialize_dtypes(Box({}))
        super().tearDown()

    def test_forced_f16_time_with_absolute_clock_raises(self):
        # Forcing a 16-bit time_dtype with an unbounded absolute clock must be rejected.
        with self.assertRaises(ValueError):
            make(_cfg("rmsa", mixed=False, time_dtype="float16", relative_arrival_times=False))

    def test_forced_f16_time_with_incremental_loading_raises(self):
        with self.assertRaises(ValueError):
            make(
                _cfg(
                    "rmsa",
                    mixed=False,
                    time_dtype="float16",
                    relative_arrival_times=True,
                    incremental_loading=True,
                    end_first_blocking=True,
                )
            )


if __name__ == "__main__":
    absltest.main()
