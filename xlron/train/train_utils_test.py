"""Unit tests for train_utils.py helpers."""

import jax
import jax.numpy as jnp
from absl.testing import absltest

from xlron.environments.make_env import make
from xlron.train import ppo
from xlron.train.train_utils import (
    diagnostics_metrics,
    loss_metrics,
    prioritization_metrics,
    reset_warmup_metric_counters,
    reward_centering_metrics,
)


class ResetWarmupMetricCountersTest(absltest.TestCase):
    """Regression test: with continuous_operation the cumulative metric counters were
    never reset after ENV_WARMUP_STEPS, so every logged blocking probability included
    the near-zero-blocking network-fill transient.
    """

    def _stepped_log_state(self):
        settings = dict(
            env_type="rsa",
            topology_name="4node",
            k=2,
            link_resources=5,
            values_bw=[1],
            slot_size=1,
            guardband=0,
            load=100,
            mean_service_holding_time=10,
            max_requests=10,
            continuous_operation=True,
        )
        env, params = make(settings)  # LogWrapper-wrapped
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        step = jax.jit(env.step, static_argnums=(3,))
        for _ in range(5):
            key, step_key = jax.random.split(key)
            obs, state, reward, terminal, truncated, info = step(
                step_key, state, jnp.array(0), params
            )
        return state

    def test_metric_counters_zeroed_and_total_requests_kept(self):
        state = self._stepped_log_state()
        # Counters have accumulated during the (simulated) warmup
        self.assertGreater(int(state.lengths), 0)
        self.assertGreater(int(state.env_state.total_bitrate), 0)
        total_requests_before = int(state.env_state.total_requests)
        self.assertGreater(total_requests_before, 0)

        new_state = reset_warmup_metric_counters(state)

        for field in (
            "lengths",
            "cum_returns",
            "accepted_services",
            "accepted_bitrate",
            "total_bitrate",
        ):
            self.assertEqual(float(getattr(new_state, field)), 0.0, f"LogEnvState.{field}")
        for field in ("accepted_services", "accepted_bitrate", "total_bitrate"):
            self.assertEqual(float(getattr(new_state.env_state, field)), 0.0, f"env.{field}")
        # total_requests drives episode truncation and must be preserved
        self.assertEqual(int(new_state.env_state.total_requests), total_requests_before)

    def test_shapes_and_dtypes_preserved(self):
        state = self._stepped_log_state()
        new_state = reset_warmup_metric_counters(state)
        old_leaves = jax.tree_util.tree_leaves(state)
        new_leaves = jax.tree_util.tree_leaves(new_state)
        self.assertEqual(len(old_leaves), len(new_leaves))
        for old, new in zip(old_leaves, new_leaves):
            self.assertEqual(jnp.shape(old), jnp.shape(new))
            self.assertEqual(jnp.asarray(old).dtype, jnp.asarray(new).dtype)


class LossInfoMetricRegistrationTest(absltest.TestCase):
    """Every key that _update_step puts in loss_info must be in one of the registered
    wandb metric lists, otherwise it is computed but silently dropped from logging
    (see CLAUDE.md 'Adding New Loss/Diagnostic Metrics').
    """

    def test_loss_info_keys_are_registered(self):
        import inspect

        source = inspect.getsource(ppo)
        registered = set(
            loss_metrics + prioritization_metrics + reward_centering_metrics + diagnostics_metrics
        )
        for prefix in ("loss/", "prioritization/", "reward_centering/"):
            for line in source.splitlines():
                line = line.strip()
                if line.startswith(f'"{prefix}'):
                    key = line.split('"')[1]
                    self.assertIn(key, registered, f"loss_info key {key} not registered")


if __name__ == "__main__":
    absltest.main()
