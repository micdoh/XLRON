"""Runtime configuration and compiled executables persist across fresh processes."""

import os
import subprocess
import sys

import pytest

from xlron.train.compilation import (
    DETERMINISTIC_XLA_FLAGS,
    DEVICE_LOOP_XLA_FLAGS,
    PERFORMANCE_XLA_FLAGS,
    deterministic_ops_requested,
    merge_xla_flags,
)


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], False),
        (["--DETERMINISTIC_OPS"], True),
        (["--DETERMINISTIC_OPS=True"], True),
        (["--DETERMINISTIC_OPS=1"], True),
        (["--DETERMINISTIC_OPS=false"], False),
        (["--DETERMINISTIC_OPS", "--noDETERMINISTIC_OPS"], False),
        (["--DETERMINISTIC_OPS_EXTRA"], False),
    ],
)
def test_deterministic_flag_parsing_matches_absl_booleans(argv, expected):
    assert deterministic_ops_requested(argv) is expected


def test_user_xla_flags_are_kept_and_determinism_overrides_conflicts():
    user = "--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=true"
    merged = merge_xla_flags(user, DETERMINISTIC_XLA_FLAGS, override=True).split()
    assert "--xla_gpu_autotune_level=0" in merged
    assert "--xla_gpu_enable_triton_gemm=false" in merged
    assert "--xla_gpu_deterministic_ops=true" in merged
    assert "--xla_gpu_enable_triton_gemm=true" not in merged


def test_performance_defaults_never_override_user_values():
    user = "--xla_gpu_triton_gemm_any=False"
    merged = merge_xla_flags(user, PERFORMANCE_XLA_FLAGS, override=False).split()
    assert "--xla_gpu_triton_gemm_any=False" in merged
    assert "--xla_gpu_enable_latency_hiding_scheduler=true" in merged


def _run(program, tmp_path, **env):
    environment = dict(os.environ, JAX_PLATFORMS="cpu", **env)
    for name in ("JAX_COMPILATION_CACHE_DIR", "XLRON_COMPILATION_CACHE"):
        if name not in env:
            environment.pop(name, None)
    return subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_compilation_cache_is_opt_in(tmp_path):
    program = """
from xlron.train.compilation import configure_jax_runtime
configure_jax_runtime([])
import jax
print(jax.config.jax_compilation_cache_dir)
"""
    run = _run(program, tmp_path, XDG_CACHE_HOME=str(tmp_path))
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "None"
    assert not (tmp_path / "xlron").exists()


def test_opt_in_persistent_cache_reuses_compilation(tmp_path):
    program = """
from xlron.train.compilation import configure_jax_runtime
configure_jax_runtime([])
import jax
import jax.numpy as jnp
f = jax.jit(lambda x: jnp.sum(jnp.sin(x)))
print(float(f(jnp.arange(128, dtype=jnp.float32)).block_until_ready()))
"""
    env = dict(
        JAX_LOG_COMPILES="1",
        XLRON_COMPILATION_CACHE=str(tmp_path / "cache"),
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
    )
    runs = [_run(program, tmp_path, **env) for _ in range(2)]
    for run in runs:
        assert run.returncode == 0, run.stderr
    assert runs[0].stdout == runs[1].stdout
    assert "Persistent compilation cache hit" in runs[1].stderr


@pytest.mark.parametrize("argv", [[], ["--DETERMINISTIC_OPS"]])
def test_device_side_loops_are_enabled_unless_the_user_sets_command_buffers(argv, monkeypatch):
    from xlron.train.compilation import configure_jax_runtime

    monkeypatch.delenv("XLRON_COMPILATION_CACHE", raising=False)
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    configure_jax_runtime(argv)
    flags = os.environ["XLA_FLAGS"].split()
    assert DEVICE_LOOP_XLA_FLAGS[0] in flags and "--xla_gpu_autotune_level=0" in flags
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    configure_jax_runtime(argv)
    assert "--xla_gpu_enable_command_buffer=" in os.environ["XLA_FLAGS"].split()
    assert DEVICE_LOOP_XLA_FLAGS[0] not in os.environ["XLA_FLAGS"].split()
