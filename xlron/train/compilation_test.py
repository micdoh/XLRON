"""Runtime configuration and compiled executables persist across fresh processes."""

import os
import subprocess
import sys


def test_persistent_cache_reuses_compilation_and_preserves_explicit_flags(tmp_path):
    program = """
import os
from xlron.train.compilation import configure_jax_runtime
configure_jax_runtime([])
assert os.environ['XLA_FLAGS'] == '--xla_gpu_autotune_level=0'
import jax
import jax.numpy as jnp
f = jax.jit(lambda x: jnp.sum(jnp.sin(x)))
print(float(f(jnp.arange(128, dtype=jnp.float32)).block_until_ready()))
"""
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_LOG_COMPILES="1",
        XLA_FLAGS="--xla_gpu_autotune_level=0",
        JAX_ENABLE_COMPILATION_CACHE="true",
        JAX_COMPILATION_CACHE_DIR=str(tmp_path / "cache"),
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", program],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        for _ in range(2)
    ]
    for run in runs:
        assert run.returncode == 0, run.stderr
    assert runs[0].stdout == runs[1].stdout
    assert "Persistent compilation cache hit" in runs[1].stderr
