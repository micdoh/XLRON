"""Unit tests for training utilities: schedule step-unit consistency and checkpointing.

The entropy/VML schedules are evaluated at ``train_state.step``, which increments once per
update loop by default and once per gradient (minibatch) step when STEP_ON_GRADIENT is set.
Their horizons must be expressed in the same unit or the schedules never complete (default)
or complete UPDATE_EPOCHS*NUM_MINIBATCHES times too early (STEP_ON_GRADIENT). The LR
schedules are exempt: optax drives them with its internal per-tx.update count.
"""

import chex
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from box import Box

from xlron.train.train_utils import (
    make_ent_schedule,
    make_vml_schedule,
    steps_per_train_state_unit,
)


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
        # And the start of training is the initial coefficient
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


if __name__ == "__main__":
    absltest.main()
