# dtype_config.py
import jax.numpy as jnp
import os
from absl import flags

# Get the FLAGS object
FLAGS = flags.FLAGS

# Try to ensure flag is defined if it isn't already
try:
    flags.DEFINE_string('DEFAULT_DTYPE_BITS', '32', 'Default bit width for dtypes (16 or 32)')
except flags.DuplicateFlagError:
    # Flag is already defined, which is fine
    pass

# Mapping from string flag values to actual dtypes
INT_DTYPE_MAP = {
    "16": jnp.int16,
    "32": jnp.int32
}

FLOAT_DTYPE_MAP = {
    "16": jnp.bfloat16,
    "32": jnp.float32
}

TIME_FLOAT_DTYPE_MAP = {
    "16": jnp.float16,
    "32": jnp.float32
}


# Define a function to get the current dtype bits
def _get_dtype_bits():
    # Try to get from FLAGS
    try:
        # See if flags have been parsed
        _ = FLAGS.DEFAULT_DTYPE_BITS
        return FLAGS.DEFAULT_DTYPE_BITS
    except (AttributeError, flags.UnparsedFlagAccessError):
        # Flags not parsed yet, try environment variable
        return os.environ.get('DEFAULT_DTYPE_BITS', '32')


# Initialize the global variables
INT_DTYPE = INT_DTYPE_MAP[_get_dtype_bits()]
FLOAT_DTYPE = FLOAT_DTYPE_MAP[_get_dtype_bits()]
TIME_DTYPE = TIME_FLOAT_DTYPE_MAP[_get_dtype_bits()]

# Print a message when the module is imported
print(f"dtype_config: Using int dtype: {INT_DTYPE} and float dtype: {FLOAT_DTYPE}")


# Keep the initialize_dtypes function for explicit updates
def initialize_dtypes(default_bits=None):
    """
    Explicitly initialize dtypes. This can be called after flags are parsed
    to ensure the correct values are used.
    """
    global INT_DTYPE, FLOAT_DTYPE

    if default_bits is None:
        default_bits = _get_dtype_bits()

    INT_DTYPE = INT_DTYPE_MAP[default_bits]
    FLOAT_DTYPE = FLOAT_DTYPE_MAP[default_bits]

    print(f"dtype_config: Explicitly updated to int dtype: {INT_DTYPE} and float dtype: {FLOAT_DTYPE}")