import time
import timeit
from collections import defaultdict
from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import Array, tree_util

from xlron import dtype_config
from xlron.environments.dataclasses import (
    LogEnvState,
    RSAEnvParams,
    RSAEnvState,
    RSAGNModelEnvParams,
)
from xlron.environments.env_funcs import (
    get_path_indices,
    get_snr_for_path,
    process_path_action,
    read_rsa_request,
)


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths.
    Modified from: https://github.com/RobertTLange/gymnax/blob/master/gymnax/wrappers/purerl.py
    """

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        params: Optional[RSAEnvParams] = None,
        state: Optional[RSAEnvState] = None,
    ) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params, state)
        log_state = LogEnvState(
            env_state=env_state,
            lengths=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            returns=jnp.array(0, dtype=dtype_config.REWARD_DTYPE),
            cum_returns=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            accepted_services=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            accepted_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            total_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            utilisation=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            terminal=jnp.array(False),
            truncated=jnp.array(False),
        )
        return obs, log_state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        log_state: LogEnvState,
        action: Union[int, float] | Tuple[Union[int, float], Union[int, float]],
        params: RSAEnvParams,
    ) -> Tuple[Array, LogEnvState, float, bool, bool, dict]:
        obs, env_state, reward, terminal, truncated, info = self._env.step(
            key, log_state.env_state, action, params
        )
        done = jnp.logical_or(terminal, truncated)
        log_state = LogEnvState(
            env_state=env_state,
            lengths=log_state.lengths * (1 - done) + 1,
            returns=jnp.asarray(reward, dtype=dtype_config.REWARD_DTYPE),
            cum_returns=log_state.cum_returns * (1 - done) + reward,
            accepted_services=env_state.accepted_services,
            accepted_bitrate=env_state.accepted_bitrate,
            total_bitrate=env_state.total_bitrate,
            utilisation=jnp.count_nonzero(env_state.link_slot_array)
            / env_state.link_slot_array.size,
            terminal=terminal,
            truncated=truncated,
        )
        info["lengths"] = log_state.lengths
        info["returns"] = log_state.returns
        info["cum_returns"] = log_state.cum_returns
        info["accepted_services"] = log_state.accepted_services
        info["accepted_bitrate"] = log_state.accepted_bitrate
        info["total_bitrate"] = log_state.total_bitrate
        info["utilisation"] = log_state.utilisation
        info["terminal"] = terminal
        info["truncated"] = truncated
        # First check if we're dealing with RSAGNModelEnvParams
        is_gn_params = isinstance(params, RSAGNModelEnvParams)

        # For RSA params, unpack the action
        if is_gn_params:
            action, power_action = action
            info["launch_power"] = power_action

        # Now, if we need to log actions OR we have RSA params, compute the common fields
        if is_gn_params or params.log_actions:
            # Compute common fields
            nodes_sd, dr_request = read_rsa_request(log_state.env_state.request_array)
            source, dest = nodes_sd
            i = get_path_indices(
                params, source, dest, params.k_paths, params.num_nodes, directed=params.directed_graph
            ).astype(jnp.int32)
            path_index, slot_index = process_path_action(log_state.env_state, params, action)

            # Set common info
            info["path_index"] = i + path_index
            info["slot_index"] = slot_index
            info["source"] = source
            info["dest"] = dest
            info["data_rate"] = dr_request[0]

            # RSA-specific throughput info
            if is_gn_params:
                info["throughput"] = env_state.throughput

            # Logging-specific info
            if params.log_actions:
                # RSA-specific logging
                if is_gn_params:
                    path = params.path_link_array.val[path_index.astype(jnp.int32)]
                    info["path_snr"] = get_snr_for_path(path, env_state.link_snr_array, params)[
                        slot_index.astype(jnp.int32)
                    ]
                # Common logging fields
                info["arrival_time"] = env_state.current_time[0]
                info["departure_time"] = env_state.current_time[0] + env_state.holding_time[0]
        return obs, log_state, reward, terminal, truncated, info

    def _tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        children = ()  # arrays / dynamic values
        aux_data = (self._env,)  # static values, e.g. env params
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data: Tuple[Any, ...], children: Tuple[Any, ...]) -> "LogWrapper":
        return cls(*children, *aux_data)


class TimeIt:
    """Context manager for timing execution of code blocks."""

    def __init__(self, tag, frames=None):
        self.tag = tag
        self.frames = frames

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.frames:
            msg += ", FPS=%.2e" % (self.frames / self.elapsed_secs)
        print(msg)


class Profiler:
    """Simple wall-clock profiler that tracks named sections.

    Usage:
        profiler = Profiler()

        with profiler.section("compilation"):
            ...

        for i in range(10):
            with profiler.section("training_step", frames=1000):
                ...

        profiler.summary()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        # Each key maps to a list of (elapsed_secs, frames) tuples
        self._records: dict[str, list[tuple[float, int | None]]] = {}
        self._order: list[str] = []  # Insertion order of section names

    def section(self, tag: str, frames: int | None = None) -> "_ProfileSection":
        """Return a context manager that times the enclosed block.

        Args:
            tag: Name for this section. Repeated uses accumulate.
            frames: Optional work-unit count (e.g. timesteps) for throughput.
        """
        return _ProfileSection(self, tag, frames)

    def _record(self, tag: str, elapsed: float, frames: int | None):
        if not self.enabled:
            return
        if tag not in self._records:
            self._records[tag] = []
            self._order.append(tag)
        self._records[tag].append((elapsed, frames))

    def summary(self):
        """Print a table summarising all recorded sections."""
        if not self._records:
            return
        total_wall = sum(e for entries in self._records.values() for e, _ in entries)
        header = (
            f"{'Section':<30} {'Calls':>6} {'Total (s)':>10} {'Mean (s)':>10} {'%':>6} {'FPS':>12}"
        )
        print("\n" + "=" * len(header))
        print("PROFILER SUMMARY")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for tag in self._order:
            entries = self._records[tag]
            n = len(entries)
            total_t = sum(e for e, _ in entries)
            mean_t = total_t / n
            pct = 100.0 * total_t / total_wall if total_wall > 0 else 0.0
            total_frames = sum(f for _, f in entries if f is not None)
            fps_str = f"{total_frames / total_t:.2e}" if total_frames and total_t > 0 else ""
            print(f"{tag:<30} {n:>6} {total_t:>10.2f} {mean_t:>10.4f} {pct:>5.1f}% {fps_str:>12}")
        print("-" * len(header))
        print(f"{'TOTAL':<30} {'':>6} {total_wall:>10.2f}")
        print("=" * len(header) + "\n")


class _ProfileSection:
    """Context manager returned by Profiler.section()."""

    def __init__(self, profiler: Profiler, tag: str, frames: int | None):
        self._profiler = profiler
        self._tag = tag
        self._frames = frames
        self.elapsed_secs = 0.0

    def __enter__(self):
        self._start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self._start
        self._profiler._record(self._tag, self.elapsed_secs, self._frames)
        msg = self._tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self._frames:
            msg += ", FPS=%.2e" % (self._frames / self.elapsed_secs)
        print(msg)


class JitProfiler:
    """Wall-clock profiler for JAX JIT-compiled code.

    On CPU:
        • Uses host callbacks for fine-grained section timing.

    On GPU:
        • Automatically switches to *first-call-only* timing.
        • Measures compilation + first execution latency.
        • Fine-grained per-call timings are intentionally disabled
          to avoid misleading synchronization artifacts.
    
    This profiler records host-side timestamps using `jax.debug.callback`,
    allowing coarse wall-clock profiling of sections inside JIT-compiled code.
    Profiling is designed to be gated by a *static* Python boolean (e.g.
    `params.profile`) so that all profiling logic is resolved at trace time and
    introduces zero runtime overhead when disabled.

    Features
    --------
    • Manual markers via `mark()`, `start()`, and `end()`
    • Function-level profiling via `call()`
    • Automatic `jax.named_scope` integration for clearer JAX traces
    • Safety checks to ensure the profiling flag is static at trace time
    • Aggregation across repeated calls with a readable summary table

    Basic usage inside JIT (manual markers):

        if params.profile:
            jit_profiler.mark("process_action:start")
        with jax.named_scope("process_action"):
            ...
        if params.profile:
            jit_profiler.mark("process_action:end")

    Function-wrapping usage inside JIT (recommended):

        action_mask = jit_profiler.call(
            params.profile,
            mask_slots,
            state,
            params,
            name="mask_actions",  # optional, defaults to fn.__name__
        )

    Block-style usage inside JIT:

        jit_profiler.start(params.profile, "action_logic")
        ...
        jit_profiler.end(params.profile, "action_logic")

    Notes
    -----
    • The `enabled` flag *must* be a Python `bool` known at trace time
        (e.g. `params.profile` with `pytree_node=False`).
    • Passing a traced or JAX boolean will raise a `TypeError`.
    • All timing is wall-clock time measured on the host, not device time.

    After execution (outside JIT):

        jit_profiler.summary()
    """

    def __init__(self):
        self._timestamps: list[tuple[str, float]] = []
        self._backend = jax.default_backend()
        self._is_gpu = self._backend == "gpu"
        self._warned_gpu = False
        self._seen_first_call: set[str] = set()

        if self._is_gpu and not self._warned_gpu:
            print(
                "JitProfiler warning:\n"
                "  GPU backend detected. Fine-grained host-side timings inside JIT\n"
                "  are unreliable due to asynchronous execution.\n"
                "  `call()` will record ONLY first-call (compile + first execution)\n"
                "  latency per section.\n"
                "  Use jax.profiler / TensorBoard / Nsight for steady-state GPU timing."
            )
            self._warned_gpu = True

    def _record(self, label):
        self._timestamps.append((str(label), time.time()))

    @staticmethod
    def _assert_static_bool(x, name="enabled"):
        if not isinstance(x, bool):
            raise TypeError(
                f"JitProfiler.call(): `{name}` must be a Python bool "
                "(static at trace time), e.g. params.profile"
            )

    def mark(self, label: str):
        """Insert a timing marker. Safe to call inside JIT."""
        jax.debug.callback(self._record, label)

    def reset(self):
        """Clear all recorded timestamps and first-call tracking."""
        self._timestamps.clear()
        self._seen_first_call.clear()

    def start(self, enabled: bool, name: str):
        """Insert a start marker for a block."""
        self._assert_static_bool(enabled)
        if enabled:
            self.mark(f"{name}:start")
    
    def end(self, enabled: bool, name: str):
        """Insert an end marker for a block."""
        self._assert_static_bool(enabled)
        if enabled:
            self.mark(f"{name}:end")

    # -----------------------
    # GPU-only first-call path
    # -----------------------

    def _call_first_only(self, enabled: bool, fn, *args, name=None, **kwargs):
        """Record compile + first execution latency (GPU only)."""
        self._assert_static_bool(enabled)
    
        section = name or fn.__name__
    
        if section in self._seen_first_call or not enabled:
            return fn(*args, **kwargs)
    
        self._seen_first_call.add(section)
    
        start_time = time.time()
        out = fn(*args, **kwargs)
        out = jax.block_until_ready(out)
        elapsed = time.time() - start_time
    
        # Append synthetic start/end entries so summary sees them
        self._timestamps.append((f"{section}:start", start_time))
        self._timestamps.append((f"{section}:end", start_time + elapsed))
    
        # Also keep a :first entry for clarity
        self._timestamps.append((f"{section}:first", elapsed))
    
        return out

    # -----------------------
    # Public API
    # -----------------------

    def call(self, enabled: bool, fn, *args, name: str | None = None, **kwargs):
        """Profile a function call inside JIT-compiled code.

        CPU:
            Fine-grained section timing via callbacks.

        GPU:
            Records only first-call (compile + first execution) latency.
        """
        self._assert_static_bool(enabled)

        if self._is_gpu:
            return self._call_first_only(enabled, fn, *args, name=name, **kwargs)

        # CPU path
        section = name or fn.__name__
        if enabled:
            self.mark(f"{section}:start")
        with jax.named_scope(section):
            out = fn(*args, **kwargs)
        if enabled:
            self.mark(f"{section}:end")
        return out

    # -----------------------
    # Summary
    # -----------------------

    def summary(self):
        """Print timing breakdown from collected start/end marker pairs.

        Expects markers in the format "name:start" and "name:end".
        Aggregates across repeated calls (e.g. many step_env invocations).

        Also reports GPU `:first` entries if present.
        """
        if len(self._timestamps) < 2:
            print("JitProfiler: not enough markers recorded.")
            return

        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        order: list[str] = []
        pending: dict[str, float] = {}

        # Aggregate :start / :end
        for label, ts in self._timestamps:
            if label.endswith(":start"):
                name = label[: -len(":start")]
                pending[name] = ts
            elif label.endswith(":end"):
                name = label[: -len(":end")]
                if name in pending:
                    elapsed = ts - pending.pop(name)
                    totals[name] += elapsed
                    counts[name] += 1
                    if name not in order:
                        order.append(name)

        # Include any :first entries for GPU
        for label, val in self._timestamps:
            if label.endswith(":first"):
                name = label[: -len(":first")]
                totals[name] += val
                counts[name] += 1
                if name not in order:
                    order.append(name)

        if not totals:
            print("JitProfiler: no matched start/end pairs found.")
            return

        total_wall = sum(totals.values())
        header = f"{'Section':<30} {'Calls':>8} {'Total (s)':>10} {'Mean (us)':>10} {'%':>6}"
        print("\n" + "=" * len(header))
        print("JIT PROFILER SUMMARY")
        if self._backend.upper() == "GPU":
            print("  (GPU backend --> profile only indicates first-call latencies)")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for name in order:
            n = counts[name]
            total_t = totals[name]
            mean_us = 1e6 * total_t / n
            pct = 100.0 * total_t / total_wall if total_wall > 0 else 0.0
            print(f"{name:<30} {n:>8} {total_t:>10.4f} {mean_us:>10.1f} {pct:>5.1f}%")
        print("-" * len(header))
        print(f"{'TOTAL':<30} {'':>8} {total_wall:>10.4f}")
        print("=" * len(header) + "\n")



# Module-level singleton so rsa.py and train.py can share the same instance
jit_profiler = JitProfiler()

tree_util.register_pytree_node(LogWrapper, LogWrapper._tree_flatten, LogWrapper._tree_unflatten)
