"""Unit tests for heuristic action selection plumbing in train_utils.

Covers:
* select_action_eval dispatches every heuristic name exposed in the GUI
  (xlron/gui/widgets.py PATH_HEURISTICS) without raising.
* get_warmup_fn with --warmup_action_type=heuristic uses the heuristic during
  RL training (EVAL_HEURISTIC=False) rather than silently falling back to the
  (untrained) policy.
"""

import chex
import jax
from absl.testing import absltest, parameterized
from box import Box

from xlron.environments.make_env import make
from xlron.heuristics.heuristics import ksp_ff
from xlron.train.train_utils import get_warmup_fn, select_action_eval

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


if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
