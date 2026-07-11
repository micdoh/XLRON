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
from xlron.train.train_utils import (
    get_warmup_fn,
    make_ent_schedule,
    make_vml_schedule,
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


if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
