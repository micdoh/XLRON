"""Wrapper arguments must not consume arguments belonging to the workload."""

import sys

from experimental.benchmarks import gn_scaling
from xlron.kernels import gn_environment_benchmark


def test_reference_snapshot_is_forwarded_to_workload(monkeypatch):
    captured = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--xpm-backend", "jax", "--reference", "baseline.npz", "--slots", "100"],
    )
    monkeypatch.setattr(gn_scaling, "main", lambda: captured.extend(sys.argv[1:]))
    gn_environment_benchmark.main()
    assert captured == ["--reference", "baseline.npz", "--slots", "100"]
