"""Preserve per-replica outliers and the physical state needed to diagnose them."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from box import Box

from xlron.train.gn_campaign_diagnostics import (
    install_diagnostics,
    per_env_record,
    summarize_replicas,
)


def metrics():
    return {
        key: np.asarray(values)[None, None, :]
        for key, values in {
            "accepted_services": [7, 4],
            "accepted_bitrate": [700, 400],
            "total_bitrate": [1000, 1000],
            "blocked_spectrum": [3, 1],
            "blocked_snr": [0, 5],
            "blocked_power": [0, 0],
        }.items()
    }


def test_replica_counters_preserve_high_snr_outlier():
    record = per_env_record(metrics(), 2, 10)
    np.testing.assert_allclose(record["service_blocking_probability"], [0.3, 0.6])
    np.testing.assert_allclose(record["snr_blocking_probability"], [0, 0.5])
    np.testing.assert_allclose(record["bitrate_blocking_probability"], [0.3, 0.6])


def test_summary_keeps_failure_replicas_in_mean_and_exposes_the_tail():
    result = summarize_replicas(
        {
            "service_blocking_probability": [0.25, 0.26, 0.9],
            "snr_blocking_probability": [0.001, 0.002, 0.65],
        }
    )
    assert result["replica_count"] == 3
    np.testing.assert_allclose(result["service_blocking_mean"], 0.47)
    assert result["service_blocking_median"] == 0.26
    assert result["service_blocking_min"] == 0.25
    assert result["service_blocking_max"] == 0.9
    assert result["high_snr_blocking_count"] == 1
    np.testing.assert_allclose(result["high_snr_share_of_snr_blocks"], 0.65 / 0.653)


def test_summary_handles_no_snr_blocks():
    result = summarize_replicas({"service_blocking_probability": [0.25, 0.26]})
    assert result["high_snr_share_of_snr_blocks"] == 0
    assert result["high_snr_blocking_count"] == 0


def test_snapshot_selects_outlier_and_logs_table(tmp_path):
    @dataclass
    class State:
        link_slot_array: np.ndarray

    saved = []
    run = SimpleNamespace(summary={})
    wandb = SimpleNamespace(run=run, save=saved.append, Table=lambda **kwargs: kwargs)
    train_utils = SimpleNamespace(log_metrics=lambda *_a, **_k: "logged", wandb=wandb)
    path = tmp_path / "replicas.jsonl"
    install_diagnostics(train_utils, path)
    config = Box(
        {
            "NUM_LEARNERS": 1,
            "continuous_operation": True,
            "NUM_ENVS": 2,
            "STEPS_PER_INCREMENT": 20,
            "NUM_INCREMENTS": 1,
            "WANDB": True,
        }
    )
    state = State(np.arange(12).reshape(2, 2, 3))
    out = {"metrics": metrics(), "runner_state": (None, SimpleNamespace(env_state=state))}
    assert train_utils.log_metrics(config, out) == "logged"
    record = json.loads(path.read_text())
    assert "anomaly_snapshot" in record
    snapshot = np.load(record["anomaly_snapshot"])
    np.testing.assert_array_equal(snapshot["link_slot_array"], state.link_slot_array[1])
    assert run.summary["replicas/high_snr_blocking_count"] == 1
    np.testing.assert_allclose(run.summary["replicas/service_blocking_median"], 0.45)
    assert run.summary["replicas/final_blocking"]["data"][1][-1] == 0.5
    assert len(saved) == 3
