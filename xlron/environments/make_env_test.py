"""Tests for lightweight run-configuration validation (validate_config)."""

import pytest
from box import Box

from xlron.environments.make_env import validate_config


def _cfg(**overrides):
    base = dict(
        NUM_ENVS=4,
        ROLLOUT_LENGTH=32,
        NUM_MINIBATCHES=1,
        TOTAL_TIMESTEPS=4096,
        NUM_UPDATES=4,
        continuous_operation=True,
        end_first_blocking=False,
        max_requests=4,
    )
    base.update(overrides)
    return Box(base)


def test_structural_error_on_zero_envs():
    with pytest.raises(ValueError):
        validate_config(_cfg(NUM_ENVS=0), is_eval=False)


def test_sane_continuous_config_is_quiet(capsys):
    validate_config(_cfg(), is_eval=False)
    assert "WARNING" not in capsys.readouterr().out


def test_warns_when_window_shorter_than_episode(capsys):
    # Episodic (not continuous, not end_first_blocking): window 8 < episode 1000.
    validate_config(
        _cfg(continuous_operation=False, NUM_UPDATES=1, ROLLOUT_LENGTH=8, max_requests=1000),
        is_eval=False,
    )
    assert "episode-end metrics will be empty" in capsys.readouterr().out


def test_eval_skips_episode_window_check(capsys):
    validate_config(
        _cfg(continuous_operation=False, NUM_UPDATES=1, ROLLOUT_LENGTH=8, max_requests=1000),
        is_eval=True,
    )
    assert "WARNING" not in capsys.readouterr().out


def test_warns_on_indivisible_minibatches(capsys):
    validate_config(_cfg(NUM_MINIBATCHES=3), is_eval=False)
    assert "not divisible" in capsys.readouterr().out
