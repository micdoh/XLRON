# Usage: import all of these at the top of every file that creates arrays

import inspect
import os
from typing import Any, Dict

import jax
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

global \
    COMPUTE_DTYPE, \
    PARAMS_DTYPE, \
    LARGE_FLOAT_DTYPE, \
    SMALL_FLOAT_DTYPE, \
    LARGE_INT_DTYPE, \
    SMALL_INT_DTYPE, \
    BINARY_DTYPE, \
    REWARD_DTYPE, \
    ACTION_DTYPE

COMPUTE_DTYPE = jnp.bfloat16
PARAMS_DTYPE = jnp.bfloat16
LARGE_FLOAT_DTYPE = jnp.bfloat16
SMALL_FLOAT_DTYPE = jnp.float16
LARGE_INT_DTYPE = jnp.int32
SMALL_INT_DTYPE = jnp.int8
BINARY_DTYPE = jnp.int8  # Default for binary arrays.
REWARD_DTYPE = jnp.int16  # Default for rewards, can be float or int.
ACTION_DTYPE = jnp.int16  # Default for actions, must be int for indexing.
INDEX_DTYPE = jnp.int16  # Default for indexing arrays.


def _is_called_from_train_py() -> bool:
    """Check if this module is being imported/called from train.py"""
    stack = inspect.stack()
    for frame_info in stack:
        filename = os.path.basename(frame_info.filename)
        if filename == "train.py" or filename == "train_transformer.py":
            return True
    return False


def initialize_dtypes(flags: flags.FlagValues | Box | Dict) -> None:
    # TODO - work out best dtypes to use based on environment params and hardware
    def get_flag_value_or_none(flag_name: str, default: Any) -> Any:
        """Get the value of a flag or environment variable, returning None if not set."""
        try:
            flag_value = getattr(flags, flag_name, default)
            if flag_value is not None:
                return flag_value
            else:
                return default
        except Exception:
            pass
        import os

        env_var = os.environ.get(flag_name)
        if env_var is not None:
            return env_var
        return None

    # Check if called from train.py
    called_from_train = _is_called_from_train_py()

    platform = jax.local_devices()[0].platform

    # Force 32-bit types if called from train.py, otherwise use platform-specific defaults
    if get_flag_value_or_none("differentiable", False):
        print("Differentiable flag is set - using float 32-bit data types")
        compute_default = "float32"
        float_default = "float32"
        int_default = "float32"
        binary_default = "float32"
        reward_default = "float32"
        action_default = "float32"
        index_dtype = "int32"
    elif called_from_train:
        print("Detected import from train.py - using 32-bit data types")
        compute_default = "float32"
        float_default = "float32"
        int_default = "int32"
        binary_default = "int32"
        reward_default = "float32"
        action_default = "int32"
        index_dtype = "int32"
    else:
        compute_default = "float32" if platform == "cpu" else "bfloat16"
        float_default = "float32" if platform == "cpu" else "float16"
        int_default = "int32" if platform == "cpu" else "int16"
        binary_default = "int32" if platform == "cpu" else "int8"
        reward_default = "float32" if platform == "cpu" else "int16"
        action_default = "int32" if platform == "cpu" else "int16"
        index_dtype = "int32" if platform == "cpu" else "int16"

    # Allow global int_dtype and float_dtype flags
    compute_dtype_flag = get_flag_value_or_none("compute_dtype", compute_default)
    params_dtype_flag = get_flag_value_or_none("params_dtype", compute_default)
    float_dtype_flag = get_flag_value_or_none("float_dtype", float_default)
    large_float_dtype_flag = get_flag_value_or_none("large_float_dtype", float_dtype_flag)
    small_float_dtype_flag = get_flag_value_or_none("small_float_dtype", float_dtype_flag)
    int_dtype_flag = get_flag_value_or_none("int_dtype", int_default)
    large_int_dtype_flag =get_flag_value_or_none("large_int_dtype", int_dtype_flag)
    small_int_dtype_flag = get_flag_value_or_none("small_int_dtype", int_dtype_flag)
    binary_dtype_flag = get_flag_value_or_none("binary_dtype", binary_default)
    # Ensure reward is float by default if edge difference is normalised (fractional)
    reward_dtype_flag = get_flag_value_or_none(
        "reward_dtype",
        reward_default if not getattr(flags, "normalise_reward", True) else "float32",
    )
    action_dtype_flag = get_flag_value_or_none("action_dtype", action_default)

    globals()["COMPUTE_DTYPE"] = DTYPE_MAP[compute_dtype_flag]
    globals()["PARAMS_DTYPE"] = DTYPE_MAP[params_dtype_flag]
    globals()["LARGE_FLOAT_DTYPE"] = DTYPE_MAP[large_float_dtype_flag]
    globals()["SMALL_FLOAT_DTYPE"] = DTYPE_MAP[small_float_dtype_flag]
    globals()["LARGE_INT_DTYPE"] = DTYPE_MAP[large_int_dtype_flag]
    globals()["SMALL_INT_DTYPE"] = DTYPE_MAP[small_int_dtype_flag]
    globals()["BINARY_DTYPE"] = DTYPE_MAP[binary_dtype_flag]
    globals()["REWARD_DTYPE"] = DTYPE_MAP[reward_dtype_flag]
    globals()["ACTION_DTYPE"] = DTYPE_MAP[action_dtype_flag]
    globals()["INDEX_DTYPE"] = DTYPE_MAP[index_dtype]

    print("Data types initialized:")
    print(f"COMPUTE_DTYPE: {globals()['COMPUTE_DTYPE']}")
    print(f"PARAMS_DTYPE: {globals()['PARAMS_DTYPE']}")
    print(f"LARGE_FLOAT_DTYPE: {globals()['LARGE_FLOAT_DTYPE']}")
    print(f"SMALL_FLOAT_DTYPE: {globals()['SMALL_FLOAT_DTYPE']}")
    print(f"LARGE_INT_DTYPE: {globals()['LARGE_INT_DTYPE']}")
    print(f"SMALL_INT_DTYPE: {globals()['SMALL_INT_DTYPE']}")
    print(f"BINARY_DTYPE: {globals()['BINARY_DTYPE']}")
    print(f"REWARD_DTYPE: {globals()['REWARD_DTYPE']}")
    print(f"ACTION_DTYPE: {globals()['ACTION_DTYPE']}")
    print(f"INDEX_DTYPE: {globals()['INDEX_DTYPE']}")


initialize_dtypes(FLAGS)

