"""Standalone evaluation shares the periodic-training eval metric namespace."""

from types import SimpleNamespace

import numpy as np
import pytest
from box import Box

from xlron.train import train_utils as tu


@pytest.mark.parametrize("mode", ["training", "heuristic", "model"])
@pytest.mark.parametrize("continuous", [True, False])
def test_evaluation_metric_registration_and_logged_values(monkeypatch, mode, continuous):
    definitions, logs = {}, []
    fake = SimpleNamespace(
        setup=lambda *_: None,
        Settings=lambda **kw: kw,
        init=lambda **_: SimpleNamespace(name=None),
        config=SimpleNamespace(update=lambda _: None),
        define_metric=lambda name, **kw: definitions.update({name: kw}),
        log=logs.append,
    )
    monkeypatch.setattr(tu, "wandb", fake)
    c = Box(
        dict(
            EVAL_HEURISTIC=mode == "heuristic",
            EVAL_MODEL=mode == "model",
            continuous_operation=continuous,
            LOG_ALL_INFO=False,
            WANDB=True,
            LOG_LOSS_INFO=False,
            DOWNSAMPLE_FACTOR=2,
            STEPS_PER_INCREMENT=8,
            NUM_ENVS=2,
            NUM_LEARNERS=1,
        )
    )
    tu.setup_wandb(c, "project", "name")
    stats = {key: np.arange(1, 5, dtype=float) for key in ("mean", "std", "iqr_upper", "iqr_lower")}
    stats.update(
        {
            "episode_end_" + key: np.array([0.25, 0.5])
            for key in ("mean", "std", "iqr_upper", "iqr_lower")
        }
    )
    processed = {
        metric: {k: v.copy() for k, v in stats.items()}
        for metric in (
            "service_blocking_probability",
            "bitrate_blocking_probability",
            "accepted_services",
            "accepted_bitrate",
        )
    }
    monkeypatch.setattr(tu, "process_metrics", lambda *_: ({}, None, processed, np.ones(2, bool)))
    tu.log_metrics(
        c, {"runner_state": (None, None)}, 0, 1, lambda x: x, step_count=100, episode_count=7
    )
    name = "eval/service_blocking_probability_mean"
    if mode == "training":
        expected = (
            "train/service_blocking_probability_mean"
            if continuous
            else "train/service_blocking_probability_episode_end_mean"
        )
        assert expected in logs[0] and name not in logs[0]
        assert "performance/time_s" in logs[0]
    else:
        assert name in logs[0]
        assert "service_blocking_probability_mean" not in definitions
        assert "eval/time_s" in logs[0] and "training_time" not in logs[0]
        assert not any(k.startswith("service_blocking_probability_") for k in logs[0])
        assert definitions[name]["step_metric"] == ("env_step" if continuous else "episode_count")
        assert float(logs[0][name]) == (1.5 if continuous else 0.25)
    assert logs[-1]["performance/environment_transitions_per_s"] == 8
    for row in logs:
        for key in row:
            assert "/" in key or definitions[key]["hidden"] is True
    assert all("/" in key or definition.get("hidden") for key, definition in definitions.items())
