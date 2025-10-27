"""Multi-device training modules for XLRON.

This package provides PPO training implementations for multi-device setups.
"""

from xlron.train.multidevice.ppo_multidevice import (
    get_learner_fn as get_learner_fn_multidevice,
    learner_data_setup,
)

from xlron.train.multidevice.ppo_stoix import (
    get_learner_fn as get_learner_fn_stoix,
    get_warmup_fn,
    learner_setup,
    setup_experiment,
)

__all__ = [
    # ppo_multidevice exports
    "get_learner_fn_multidevice",
    "learner_data_setup",
    # ppo_stoix exports
    "get_learner_fn_stoix",
    "get_warmup_fn",
    "learner_setup",
    "setup_experiment",
]
