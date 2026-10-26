"""Configure XLA before any model/environment module initializes JAX."""

import os
from pathlib import Path

DETERMINISTIC_XLA_FLAGS = (
    "--xla_gpu_deterministic_ops=true",
    "--xla_gpu_enable_triton_gemm=false",
)
# Run while/cond loops as CUDA-graph conditional nodes instead of copying each
# loop predicate to the host. The GN masks and departure releases use
# data-dependent loops; on a time-sliced (shared) GPU every host round trip costs
# a scheduling slice, and this made full-mask rollouts ~7.5x faster. Applied in
# both modes; it does not change results.
DEVICE_LOOP_XLA_FLAGS = ("--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL",)
PERFORMANCE_XLA_FLAGS = (
    "--xla_gpu_triton_gemm_any=True",
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_highest_priority_async_stream=true",
)


def deterministic_ops_requested(argv) -> bool:
    """Parse --DETERMINISTIC_OPS with absl boolean syntax before flags are parsed."""
    enabled = False
    for arg in argv:
        if arg == "--DETERMINISTIC_OPS":
            enabled = True
        elif arg == "--noDETERMINISTIC_OPS":
            enabled = False
        elif arg.startswith("--DETERMINISTIC_OPS="):
            enabled = arg.split("=", 1)[1].strip().lower() in {"true", "t", "1", "yes", "y"}
    return enabled


def merge_xla_flags(existing: str, flags, *, override: bool) -> str:
    """Add flags to an XLA_FLAGS string without discarding the user's other flags.

    With override=False a flag the user already set keeps the user's value; with
    override=True (determinism) our value replaces it.
    """
    merged = {}
    for flag in existing.split():
        merged[flag.split("=", 1)[0]] = flag
    for flag in flags:
        name = flag.split("=", 1)[0]
        if override or name not in merged:
            merged[name] = flag
    return " ".join(merged.values())


def configure_jax_runtime(argv):
    """Set XLA flags and, when requested, a persistent compilation cache.

    The cache is opt-in: set XLRON_COMPILATION_CACHE=1 for the per-user default
    directory ($XDG_CACHE_HOME/xlron/jax), or to a directory path. JAX's own
    JAX_COMPILATION_CACHE_DIR / JAX_ENABLE_COMPILATION_CACHE settings are respected.
    """
    deterministic = deterministic_ops_requested(argv)
    os.environ["XLA_FLAGS"] = merge_xla_flags(
        merge_xla_flags(
            os.environ.get("XLA_FLAGS", ""),
            DETERMINISTIC_XLA_FLAGS if deterministic else PERFORMANCE_XLA_FLAGS,
            override=deterministic,
        ),
        DEVICE_LOOP_XLA_FLAGS,
        override=False,
    )
    if deterministic:
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    requested = os.environ.get("XLRON_COMPILATION_CACHE", "").strip()
    if requested.lower() in {"", "0", "false", "no"}:
        return

    import jax

    if jax.config.jax_compilation_cache_dir is None:
        if requested.lower() in {"1", "true", "yes"}:
            root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            cache = root / "xlron" / "jax"
        else:
            cache = Path(requested).expanduser()
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        jax.config.update("jax_compilation_cache_dir", str(cache))
    if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)
    if "JAX_COMPILATION_CACHE_MAX_SIZE" not in os.environ:
        jax.config.update("jax_compilation_cache_max_size", 2 * 1024**3)
