"""Configure compilation before any model/environment module initializes JAX."""

import os
from pathlib import Path


def configure_jax_runtime(argv):
    """Respect explicit XLA/cache settings and reuse compiled programs across runs."""
    deterministic = any(arg in {"--DETERMINISTIC_OPS", "--DETERMINISTIC_OPS=true"} for arg in argv)
    if deterministic:
        os.environ.setdefault(
            "XLA_FLAGS", "--xla_gpu_deterministic_ops=true --xla_gpu_enable_triton_gemm=false"
        )
        os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    else:
        os.environ.setdefault(
            "XLA_FLAGS",
            "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_highest_priority_async_stream=true",
        )

    import jax

    if not jax.config.jax_enable_compilation_cache:
        return
    if jax.config.jax_compilation_cache_dir is None:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache = root / "xlron" / "jax"
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        jax.config.update("jax_compilation_cache_dir", str(cache))
    if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)
    if "JAX_COMPILATION_CACHE_MAX_SIZE" not in os.environ:
        jax.config.update("jax_compilation_cache_max_size", 2 * 1024**3)
