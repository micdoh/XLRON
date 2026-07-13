# Usage: import all of these at the top of every file that creates arrays
#
# IMPORTANT: import the *module* (``from xlron import dtype_config``) and read the
# constants as ``dtype_config.LARGE_FLOAT_DTYPE`` etc. ``initialize_dtypes`` rebinds
# these as module globals at runtime, so a by-value import
# (``from xlron.dtype_config import LARGE_FLOAT_DTYPE``) freezes the import-time value
# and will NOT pick up mixed-precision reconfiguration.

import os
from typing import Any, Dict

import jax.numpy as jnp
from absl import flags
from box import Box

FLAGS = flags.FLAGS

DTYPE_MAP = {
    "int32": jnp.int32,
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float8": jnp.float8_e4m3fn,
    "int16": jnp.int16,
    "uint8": jnp.uint8,
    "int8": jnp.int8,
    "int4": jnp.int4,
}

# 16-bit (and smaller) float dtypes, used to decide when extra precision guards are needed.
HALF_FLOAT_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn)

global \
    COMPUTE_DTYPE, \
    PARAMS_DTYPE, \
    LARGE_FLOAT_DTYPE, \
    SMALL_FLOAT_DTYPE, \
    TIME_DTYPE, \
    LARGE_INT_DTYPE, \
    SMALL_INT_DTYPE, \
    BINARY_DTYPE, \
    REWARD_DTYPE, \
    ACTION_DTYPE

# Neural-network compute/params: always keep at full precision for training stability.
COMPUTE_DTYPE = jnp.float32
PARAMS_DTYPE = jnp.float32
# Env-state floats. LARGE_FLOAT is the "precision" tier (accumulators, physical SNR/power,
# importance weights); SMALL_FLOAT is the "bulk" tier (occupancy, masks, normalised features).
LARGE_FLOAT_DTYPE = jnp.float32
SMALL_FLOAT_DTYPE = jnp.float32
# Time/departure arrays. Separated out because they are precision-sensitive: in absolute-time
# mode the clock accumulates unbounded and must stay float32; in relative-arrival-time mode
# (the default) values stay bounded by the holding time, so a 16-bit float is safe.
TIME_DTYPE = jnp.float32
# Env-state ints. LARGE_INT is the "counter" tier (total_requests/timesteps, large indices);
# SMALL_INT is the "bounded index" tier (required_slots, node/datarate fields).
LARGE_INT_DTYPE = jnp.int32
SMALL_INT_DTYPE = jnp.int32
BINARY_DTYPE = jnp.int32  # Default for binary arrays.
REWARD_DTYPE = jnp.float32  # Default for rewards, can be float or int.
ACTION_DTYPE = jnp.int32  # Default for actions, must be int for indexing.
INDEX_DTYPE = jnp.int32  # Default for indexing arrays. Always int32.


def initialize_dtypes(flags: flags.FlagValues | Box | Dict) -> None:
    """Resolve the global dtype constants from flags / env vars / config.

    Resolution order for each constant: explicit per-constant flag -> the mode default.
    Three modes (mutually exclusive, highest priority first):
      * ``differentiable``: float32 everywhere (incl. ints) so straight-through gradients flow.
      * ``mixed_precision``: shrink the bulk/bounded env-state arrays (see tier defaults below)
        while keeping NN compute/params and precision-sensitive accumulators at float32.
      * otherwise (default): 32-bit everywhere (previous behaviour, fully backwards compatible).

    Safe to call repeatedly (idempotent); it is invoked from ``process_config`` so every entry
    point (train, eval, bounds, GUI, tests) picks up the configured dtypes before any env array
    is built.
    """

    def get_flag_value_or_none(flag_name: str, default: Any) -> Any:
        """Get the value of a flag / config entry / env var, returning ``default`` if not set.

        Handles all three accepted input shapes: a Mapping (dict / Box, via .get), an
        absl FlagValues object (via getattr, guarding the unparsed-flag access error), and
        falls back to an environment variable of the same name.
        """
        flag_value = None
        # Box subclasses dict, so Mapping access covers both dict and Box.
        if isinstance(flags, dict):
            flag_value = flags.get(flag_name, None)
        else:
            try:
                flag_value = getattr(flags, flag_name, None)
            except Exception:
                flag_value = None
        if flag_value is not None:
            return flag_value

        env_var = os.environ.get(flag_name)
        if env_var is not None:
            return env_var
        return default

    differentiable = bool(get_flag_value_or_none("differentiable", False))
    mixed_precision = bool(get_flag_value_or_none("mixed_precision", False))
    # Time arrays are kept at float32 by default in every mode (see the mixed-precision
    # branch below for why float16 time corrupts service lifetimes). These regime checks
    # remain to warn if a user explicitly opts into float16 time where it cannot work:
    # absolute time accumulates without bound, and incremental_loading sets
    # mean_service_holding_time ~1e6, both of which overflow float16.
    relative_arrival_times = bool(get_flag_value_or_none("relative_arrival_times", False))
    incremental_loading = bool(get_flag_value_or_none("incremental_loading", False))
    times_can_be_half = relative_arrival_times and not incremental_loading

    if differentiable:
        # Differentiable mode uses float32 for everything (including ints) for gradient flow.
        print("Differentiable flag is set - using float 32-bit data types")
        compute_default = "float32"
        params_default = "float32"
        large_float_default = "float32"
        small_float_default = "float32"
        time_default = "float32"
        large_int_default = "float32"
        small_int_default = "float32"
        binary_default = "float32"
        reward_default = "float32"
        action_default = "float32"
        index_dtype = "int32"
    elif mixed_precision:
        # Mixed precision: keep NN/precision-sensitive arrays at 32-bit, shrink bulk env state.
        compute_default = "float32"
        params_default = "float32"
        large_float_default = "float32"  # accumulators, physical SNR/power, importance weights
        small_float_default = "float16"  # occupancy, masks, normalised observation features
        # Time stays float32 even though relative arrival times bound its magnitude:
        # the per-step departure decrement (dep - t) rounds to float16 ulp bins, and
        # because inter-arrival times are exponentially distributed (density decreasing
        # across each rounding bin) round-to-nearest systematically under-decrements.
        # Services then overstay, inflating occupancy and blocking probability — up to
        # 2x measured blocking on large topologies where ulp(remaining_time) is
        # comparable to the mean inter-arrival time. float16 time remains available
        # as an explicit opt-in via --time_dtype=float16 (bounded regimes only).
        time_default = "float32"
        large_int_default = "int32"  # global counters / large indices
        small_int_default = "int16"  # bounded indices (required_slots, node/datarate fields)
        binary_default = "int8"  # binary arrays (path-link incidence)
        reward_default = "float32"
        action_default = "int32"
        index_dtype = "int32"
    else:
        compute_default = "float32"
        params_default = "float32"
        large_float_default = "float32"
        small_float_default = "float32"
        time_default = "float32"
        large_int_default = "int32"
        small_int_default = "int32"
        binary_default = "int32"
        reward_default = "float32"
        action_default = "int32"
        index_dtype = "int32"

    # A global ``float_dtype``/``int_dtype`` flag, if set, overrides BOTH tiers uniformly.
    float_dtype_flag = get_flag_value_or_none("float_dtype", None)
    if float_dtype_flag is not None:
        large_float_default = small_float_default = float_dtype_flag
    int_dtype_flag = get_flag_value_or_none("int_dtype", None)
    if int_dtype_flag is not None:
        large_int_default = small_int_default = int_dtype_flag

    compute_dtype_flag = get_flag_value_or_none("compute_dtype", compute_default)
    params_dtype_flag = get_flag_value_or_none("params_dtype", params_default)
    large_float_dtype_flag = get_flag_value_or_none("large_float_dtype", large_float_default)
    small_float_dtype_flag = get_flag_value_or_none("small_float_dtype", small_float_default)
    time_dtype_flag = get_flag_value_or_none("time_dtype", time_default)
    # Unbounded regimes (absolute time / incremental_loading) are rejected later by
    # make_env's validation; here we only warn about the bounded-but-biased opt-in.
    if time_dtype_flag == "float16" and times_can_be_half:
        print(
            "WARNING: --time_dtype=float16 biases service lifetimes (departure-time "
            "decrements round to float16 ulp bins), inflating occupancy and blocking. "
            "Use only where memory outweighs simulation fidelity."
        )
    large_int_dtype_flag = get_flag_value_or_none("large_int_dtype", large_int_default)
    small_int_dtype_flag = get_flag_value_or_none("small_int_dtype", small_int_default)
    binary_dtype_flag = get_flag_value_or_none("binary_dtype", binary_default)
    reward_dtype_flag = get_flag_value_or_none("reward_dtype", reward_default)
    action_dtype_flag = get_flag_value_or_none("action_dtype", action_default)

    globals()["COMPUTE_DTYPE"] = DTYPE_MAP[compute_dtype_flag]
    globals()["PARAMS_DTYPE"] = DTYPE_MAP[params_dtype_flag]
    globals()["LARGE_FLOAT_DTYPE"] = DTYPE_MAP[large_float_dtype_flag]
    globals()["SMALL_FLOAT_DTYPE"] = DTYPE_MAP[small_float_dtype_flag]
    globals()["TIME_DTYPE"] = DTYPE_MAP[time_dtype_flag]
    globals()["LARGE_INT_DTYPE"] = DTYPE_MAP[large_int_dtype_flag]
    globals()["SMALL_INT_DTYPE"] = DTYPE_MAP[small_int_dtype_flag]
    globals()["BINARY_DTYPE"] = DTYPE_MAP[binary_dtype_flag]
    globals()["REWARD_DTYPE"] = DTYPE_MAP[reward_dtype_flag]
    globals()["ACTION_DTYPE"] = DTYPE_MAP[action_dtype_flag]
    globals()["INDEX_DTYPE"] = DTYPE_MAP[index_dtype]


initialize_dtypes(FLAGS)
