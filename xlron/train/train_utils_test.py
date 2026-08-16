"""Unit tests for train_utils heuristic plumbing, schedules, and checkpointing."""

import io

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from box import Box

from xlron.environments.make_env import make
from xlron.heuristics.heuristics import ksp_ff
from xlron.models.mlp import ActorCriticMLP
from xlron.train import ppo
from xlron.train.train_utils import (
    diagnostics_metrics,
    get_sweep_rewarm_fn,
    get_warmup_fn,
    heuristic_eval_obs_placeholder,
    loss_metrics,
    make_ent_schedule,
    make_vml_schedule,
    prioritization_metrics,
    reset_warmup_metric_counters,
    reward_centering_metrics,
    select_action_eval,
    steps_per_train_state_unit,
)

# Mirrors PATH_HEURISTICS in xlron/gui/widgets.py: every name selectable in the
# GUI (and documented in docs/heuristic_evaluation.md) must be dispatchable.
GUI_PATH_HEURISTICS = [
    "ksp_ff",
    "ksp_lf",
    "ksp_bf",
    "ksp_mu",
    "ff_ksp",
    "lf_ksp",
    "bf_ksp",
    "mu_ksp",
    "kmc_ff",
    "kmf_ff",
    "kme_ff",
    "kca_ff",
    "arbr_ff",
]


def _rwa_4node_settings():
    return dict(
        load=100,
        k=2,
        topology_name="4node",
        link_resources=4,
        max_requests=10,
        mean_service_holding_time=10,
        env_type="rwa",
        values_bw=[1],
        slot_size=1,
        guardband=0,
    )


def _make_log_wrapped_env():
    key = jax.random.PRNGKey(0)
    env, params = make(_rwa_4node_settings())
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


class SelectActionEvalDispatchTest(chex.TestCase):
    @parameterized.parameters(*GUI_PATH_HEURISTICS)
    def test_dispatches_all_gui_heuristics(self, heuristic_name):
        key, env, obs, state, params = _make_log_wrapped_env()
        config = Box(
            dict(
                EVAL_HEURISTIC=True,
                env_type="rwa",
                path_heuristic=heuristic_name,
            )
        )
        select_action_state = (key, state, obs)
        _, action, _, _ = select_action_eval(select_action_state, env, params, None, config)
        num_actions = params.k_paths * params.link_resources
        self.assertTrue(0 <= int(action) < num_actions)


class WarmupHeuristicActionTest(chex.TestCase):
    def test_warmup_action_type_heuristic_uses_heuristic_during_rl_training(self):
        """During RL training (EVAL_HEURISTIC=False), warmup_action_type='heuristic'
        must select heuristic actions. train_state=None makes any fallback to the
        policy path (select_action) fail loudly."""
        key, env, obs, state, params = _make_log_wrapped_env()
        config = Box(
            dict(
                EVAL_HEURISTIC=False,
                env_type="rwa",
                path_heuristic="ksp_ff",
                warmup_action_type="heuristic",
                launch_power_type="fixed",
                ENV_WARMUP_STEPS=1,
                USE_GNN=False,
                USE_TRANSFORMER=False,
            )
        )
        warmup_state = (key, state, tuple([obs]))
        warmup_fn = get_warmup_fn(warmup_state, env, params, None, config)
        warmed_state, _ = warmup_fn(warmup_state)

        # Replicate the single warmup step manually with the ksp_ff action
        _, _, step_key = jax.random.split(key, 3)
        expected_action = ksp_ff(state.env_state, params)
        _, expected_state, *_ = env.step(step_key, state, expected_action, params)

        chex.assert_trees_all_close(
            warmed_state.env_state.link_slot_array,
            expected_state.env_state.link_slot_array,
        )

    def test_warmup_action_type_heuristic_rejects_aggregate_slots(self):
        key, env, obs, state, params = _make_log_wrapped_env()
        config = Box(
            dict(
                EVAL_HEURISTIC=False,
                env_type="rwa",
                path_heuristic="ksp_ff",
                warmup_action_type="heuristic",
                launch_power_type="fixed",
                aggregate_slots=2,
                ENV_WARMUP_STEPS=1,
                USE_GNN=False,
                USE_TRANSFORMER=False,
            )
        )
        warmup_state = (key, state, tuple([obs]))
        with self.assertRaisesRegex(ValueError, "aggregate_slots"):
            get_warmup_fn(warmup_state, env, params, None, config)


def _schedule_config(step_on_gradient):
    return Box(
        dict(
            NUM_UPDATES=10,
            NUM_INCREMENTS=2,
            UPDATE_EPOCHS=4,
            NUM_MINIBATCHES=4,
            STEP_ON_GRADIENT=step_on_gradient,
            ENT_COEF=0.01,
            ENT_END_FRACTION=0.1,
            ENT_SCHEDULE="linear",
            ENT_SCHEDULE_MULTIPLIER=1.0,
            VALID_MASS_LOSS_COEF=0.5,
            VML_END_FRACTION=0.2,
            VML_SCHEDULE="linear",
            VML_SCHEDULE_MULTIPLIER=1.0,
        )
    )


class ScheduleStepUnitTest(chex.TestCase):
    @parameterized.named_parameters(
        ("per_update_loop", False),
        ("per_gradient_step", True),
    )
    def test_ent_schedule_completes_at_end_of_training(self, step_on_gradient):
        config = _schedule_config(step_on_gradient)
        final_step = config.NUM_UPDATES * config.NUM_INCREMENTS * steps_per_train_state_unit(config)
        ent_schedule = make_ent_schedule(config)
        chex.assert_trees_all_close(
            ent_schedule(final_step),
            jnp.array(config.ENT_COEF * config.ENT_END_FRACTION),
            atol=1e-8,
        )
        chex.assert_trees_all_close(ent_schedule(0), jnp.array(config.ENT_COEF), atol=1e-8)

    @parameterized.named_parameters(
        ("per_update_loop", False),
        ("per_gradient_step", True),
    )
    def test_vml_schedule_completes_at_end_of_training(self, step_on_gradient):
        config = _schedule_config(step_on_gradient)
        final_step = config.NUM_UPDATES * config.NUM_INCREMENTS * steps_per_train_state_unit(config)
        vml_schedule = make_vml_schedule(config)
        chex.assert_trees_all_close(
            vml_schedule(final_step),
            jnp.array(config.VALID_MASS_LOSS_COEF * config.VML_END_FRACTION),
            atol=1e-8,
        )


class LearnerStackedCheckpointTest(chex.TestCase):
    def _model(self, key):
        return ActorCriticMLP(4, 8, num_layers=1, num_units=8, key=key)

    def test_unreplicated_params_roundtrip(self):
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        params_static = [eqx.partition(self._model(k), eqx.is_inexact_array) for k in keys]
        static = params_static[0][1]
        stacked = jax.tree.map(
            lambda a, b: jnp.stack([a, b]), params_static[0][0], params_static[1][0]
        )
        template = self._model(jax.random.PRNGKey(1))

        buf = io.BytesIO()
        eqx.tree_serialise_leaves(buf, eqx.combine(stacked, static))
        buf.seek(0)
        with self.assertRaises(Exception):
            eqx.tree_deserialise_leaves(buf, template)

        buf = io.BytesIO()
        unreplicated = jax.tree.map(lambda x: x[0], stacked)
        eqx.tree_serialise_leaves(buf, eqx.combine(unreplicated, static))
        buf.seek(0)
        restored = eqx.tree_deserialise_leaves(buf, template)
        chex.assert_trees_all_close(
            eqx.partition(restored, eqx.is_inexact_array)[0], params_static[0][0]
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

    def test_gn_model_blocking_cause_counters_zeroed(self):
        """The GN-model blocking-cause counters accumulate during warmup too; if they are
        not zeroed alongside lengths, spectrum/snr/power_blocking_probability divide
        warmup-contaminated counts by post-warmup lengths and no longer sum to
        service_blocking_probability (they can even exceed 1 early in a run).
        """
        settings = dict(
            env_type="rmsa_gn_model",
            topology_name="5node_directed",
            k=2,
            link_resources=5,
            values_bw=[100],
            slot_size=25,
            guardband=0,
            mod_format_correction=False,
            load=100,
            mean_service_holding_time=10,
            max_requests=10,
            continuous_operation=True,
        )
        env, params = make(settings)  # LogWrapper-wrapped
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        # Simulate warmup-accumulated blocking-cause counts at both levels
        count = jnp.array(2, dtype=state.env_state.blocked_spectrum.dtype)
        state = state.replace(
            blocked_spectrum=count,
            blocked_snr=count,
            blocked_power=count,
            env_state=state.env_state.replace(
                blocked_spectrum=count,
                blocked_snr=count,
                blocked_power=count,
            ),
        )

        new_state = reset_warmup_metric_counters(state)

        for field in ("blocked_spectrum", "blocked_snr", "blocked_power"):
            self.assertEqual(int(getattr(new_state, field)), 0, f"LogEnvState.{field}")
            self.assertEqual(int(getattr(new_state.env_state, field)), 0, f"env.{field}")

    def test_shapes_and_dtypes_preserved(self):
        state = self._stepped_log_state()
        new_state = reset_warmup_metric_counters(state)
        old_leaves = jax.tree_util.tree_leaves(state)
        new_leaves = jax.tree_util.tree_leaves(new_state)
        self.assertEqual(len(old_leaves), len(new_leaves))
        for old, new in zip(old_leaves, new_leaves):
            self.assertEqual(jnp.shape(old), jnp.shape(new))
            self.assertEqual(jnp.asarray(old).dtype, jnp.asarray(new).dtype)


class SweepRewarmTest(absltest.TestCase):
    """Regression tests for the load-sweep warmup bias: each swept load must
    re-equilibrate the network at its own arrival rate (re-running the
    ENV_WARMUP_STEPS warmup) and zero the metric counters, instead of measuring
    from a state equilibrated at the original --load.
    """

    WARMUP_STEPS = 40

    def _setup(self, num_envs=1):
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
            continuous_operation=True,
        )
        env, params = make(settings)  # LogWrapper-wrapped (max_requests defaults to 1e4)
        config = Box(
            dict(
                EVAL_HEURISTIC=True,
                env_type="rsa",
                path_heuristic="ksp_ff",
                launch_power_type="fixed",
                ENV_WARMUP_STEPS=self.WARMUP_STEPS,
                NUM_ENVS=num_envs,
                NUM_LEARNERS=1,
                USE_GNN=False,
                USE_TRANSFORMER=False,
            )
        )
        reset_key = jax.random.PRNGKey(0)
        if num_envs > 1:
            reset_key = jax.random.split(reset_key, num_envs)
            _, state = jax.vmap(env.reset, in_axes=(0, None))(reset_key, params)
        else:
            _, state = env.reset(reset_key, params)
        # Mirror experiment_data_setup: under EVAL_HEURISTIC the pipeline carries the
        # placeholder obs (get_warmup_fn's loop body returns the same placeholder, so a
        # full-shape obs here would mismatch the fori_loop carry)
        return env, params, config, state, heuristic_eval_obs_placeholder(num_envs)

    @staticmethod
    def _experiment_input(state, obsv):
        rng = jax.random.PRNGKey(1)
        # runner_state=None: heuristic eval has no train state (as in select_action_eval)
        return (None, state, obsv, rng, rng)

    def test_rewarm_advances_requests_and_zeroes_counters(self):
        env, params, config, state, obsv = self._setup()
        rewarm = get_sweep_rewarm_fn(env, params, config)
        total_before = int(state.env_state.total_requests)
        out = rewarm(self._experiment_input(state, obsv), jax.random.PRNGKey(2))
        _, new_state, _, _, _ = out
        # Warmup actually ran at the new load...
        self.assertEqual(int(new_state.env_state.total_requests), total_before + self.WARMUP_STEPS)
        # ...and its transient is excluded from the per-load metrics
        for field in ("lengths", "cum_returns", "accepted_services"):
            self.assertEqual(float(getattr(new_state, field)), 0.0, f"LogEnvState.{field}")
        for field in ("accepted_services", "accepted_bitrate", "total_bitrate"):
            self.assertEqual(float(getattr(new_state.env_state, field)), 0.0, f"env.{field}")

    def test_rewarm_reequilibrates_occupancy_to_new_load(self):
        """Sweeping from a high to a much lower load must not inherit the high-load
        occupancy: after the re-warmup the network reflects the new steady state."""
        env, params, config, state, obsv = self._setup()
        rewarm = get_sweep_rewarm_fn(env, params, config)
        # Equilibrate at the original (high) load: arrival_rate = 100/10 = 10
        _, high_state, high_obsv, rng_step, rng_epoch = rewarm(
            self._experiment_input(state, obsv), jax.random.PRNGKey(2)
        )
        occupied_high = int(jnp.count_nonzero(high_state.env_state.link_slot_array))
        self.assertGreater(occupied_high, 0)

        # Sweep to a near-zero load: only arrival_rate changes, exactly as
        # train.py:_update_experiment_input_load does
        inner = high_state.env_state.replace(
            arrival_rate=jnp.full_like(high_state.env_state.arrival_rate, 0.01)
        )
        low_input = (None, high_state.replace(env_state=inner), high_obsv, rng_step, rng_epoch)
        _, low_state, _, _, _ = rewarm(low_input, jax.random.PRNGKey(3))
        occupied_low = int(jnp.count_nonzero(low_state.env_state.link_slot_array))
        # Inter-arrival time (100) >> holding time (10): the high-load occupancy
        # must have drained during the re-warmup
        self.assertLess(occupied_low, occupied_high)

    def test_rewarm_preserves_structure_for_compiled_experiment(self):
        """The rewarmed experiment_input feeds the already-compiled experiment fn,
        so every leaf must keep its shape and dtype."""
        env, params, config, state, obsv = self._setup()
        rewarm = get_sweep_rewarm_fn(env, params, config)
        experiment_input = self._experiment_input(state, obsv)
        out = rewarm(experiment_input, jax.random.PRNGKey(2))
        old_leaves = jax.tree_util.tree_leaves(experiment_input)
        new_leaves = jax.tree_util.tree_leaves(out)
        self.assertEqual(len(old_leaves), len(new_leaves))
        for old, new in zip(old_leaves, new_leaves):
            self.assertEqual(jnp.shape(old), jnp.shape(new))
            self.assertEqual(jnp.asarray(old).dtype, jnp.asarray(new).dtype)

    def test_rewarm_vmapped_envs(self):
        """NUM_ENVS > 1: the warmup is vmapped and each env gets its own key."""
        num_envs = 2
        env, params, config, state, obsv = self._setup(num_envs=num_envs)
        rewarm = get_sweep_rewarm_fn(env, params, config)
        total_before = jnp.asarray(state.env_state.total_requests)
        out = rewarm(self._experiment_input(state, obsv), jax.random.PRNGKey(2))
        _, new_state, _, _, _ = out
        self.assertTrue(
            bool(jnp.all(new_state.env_state.total_requests == total_before + self.WARMUP_STEPS))
        )
        self.assertTrue(bool(jnp.all(new_state.lengths == 0)))


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
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
