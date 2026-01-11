# dtype_config.py
import jax
import jax.numpy as jnp
import os
from absl import flags
from typing import Dict, Any, Optional, Tuple

# Get the FLAGS object
FLAGS = flags.FLAGS

# Mapping from string flag values to actual dtypes
FLOAT_DTYPE_MAP = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}

INT_DTYPE_MAP = {
    "int32": jnp.int32,
    "int16": jnp.int16,
    "int8": jnp.int8,
}

# Default dtype configurations by device type
DEVICE_CONFIGS = {
    "CPU": {
        "COMPUTE_DTYPE": jnp.float32,
        "PARAMS_DTYPE": jnp.float32,
        "LARGE_FLOAT_DTYPE": jnp.float32,
        "SMALL_FLOAT_DTYPE": jnp.float32,
        "LARGE_INT_DTYPE": jnp.int32,
        "MED_INT_DTYPE": jnp.int32,
        "SMALL_INT_DTYPE": jnp.int32,
    },
    "GPU_OLDER": {
        "COMPUTE_DTYPE": jnp.float32,
        "PARAMS_DTYPE": jnp.float32,
        "LARGE_FLOAT_DTYPE": jnp.float32,
        "SMALL_FLOAT_DTYPE": jnp.float32,
        "LARGE_INT_DTYPE": jnp.int32,
        "MED_INT_DTYPE": jnp.int32,
        "SMALL_INT_DTYPE": jnp.int32,
    },
    "GPU_A100": {
        "COMPUTE_DTYPE": jnp.float32,
        "PARAMS_DTYPE": jnp.float32,
        "LARGE_FLOAT_DTYPE": jnp.float32,
        "SMALL_FLOAT_DTYPE": jnp.float32,
        "LARGE_INT_DTYPE": jnp.int32,
        "MED_INT_DTYPE": jnp.int32,
        "SMALL_INT_DTYPE": jnp.int32,
    },
    "TPU": {
        "COMPUTE_DTYPE": jnp.bfloat16,
        "PARAMS_DTYPE": jnp.bfloat16,
        "LARGE_FLOAT_DTYPE": jnp.bfloat16,
        "SMALL_FLOAT_DTYPE": jnp.float16,
        "LARGE_INT_DTYPE": jnp.int32,
        "MED_INT_DTYPE": jnp.int16,
        "SMALL_INT_DTYPE": jnp.int8,
    }
}

# Global variables for dtypes, will be initialized properly
COMPUTE_DTYPE = None
PARAMS_DTYPE = None
LARGE_FLOAT_DTYPE = None
SMALL_FLOAT_DTYPE = None
LARGE_INT_DTYPE = None
MED_INT_DTYPE = None
SMALL_INT_DTYPE = None


def detect_device_type() -> str:
    """
    Detect the device type (CPU, GPU_A100, or TPU) currently being used.

    Returns:
        str: The detected device type as a string
    """
    # Get the default backend
    platform = jax.devices()[0].platform

    if platform == "cpu":
        return "CPU"
    elif platform == "gpu":
        # Check the GPU type to determine if it's A100 or newer
        try:
            import subprocess
            nvidia_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode()

            # List of NVIDIA GPUs that support efficient bfloat16 and mixed precision
            # A100, H100, H200 and newer support efficient bfloat16 operations
            # Ampere (A100, A10, etc.), Hopper (H100), etc.
            advanced_gpus = ["A100", "A10", "A30", "A40", "H100", "H200", "L40", "L4", "L40S"]

            # Check if we're running on one of the advanced GPUs
            if any(gpu_type in nvidia_output for gpu_type in advanced_gpus):
                return "GPU_A100"
            else:
                # For older GPUs (Volta, Turing, Pascal, etc.), use the CPU config (full precision)
                print(f"Detected older GPU: {nvidia_output.strip()}. Using 32-bit precision defaults.")
                return "CPU"  # Using CPU config for older GPUs that work better with float32

        except (ImportError, FileNotFoundError, subprocess.SubprocessError):
            # Alternative method using JAX to get compute capability
            try:
                # Try to get compute capability from JAX
                device = jax.devices()[0]
                if hasattr(device, "device_kind"):
                    device_kind = device.device_kind

                    # NVIDIA compute capability 8.0+ (Ampere, e.g., A100) or newer supports efficient bfloat16
                    # Extract compute capability from "sm_XX" or similar format
                    if "sm_" in device_kind:
                        compute_cap = int(device_kind.split("sm_")[1][:2])
                        if compute_cap >= 80:  # Ampere or newer
                            return "GPU_A100"

                print(f"Could not determine precise GPU capabilities. Using full precision (CPU config).")
                return "CPU"  # Default to full precision for unknown GPUs
            except Exception as e:
                print(f"Error detecting GPU capabilities: {e}. Using full precision (CPU config).")
                return "CPU"
    elif platform == "tpu":
        return "TPU"
    else:
        # Default to CPU if unknown
        print(f"Warning: Unknown platform '{platform}', defaulting to CPU configuration.")
        return "CPU"


def get_flag_value_or_none(flag_name: str) -> Optional[str]:
    """
    Get the value of a flag if it's set and the flags have been parsed.

    Args:
        flag_name: The name of the flag to get the value for

    Returns:
        The flag value if set and parsed, None otherwise
    """
    try:
        flag_value = getattr(FLAGS, flag_name)
        # Check if flag is explicitly set (not None)
        if flag_value is not None:
            return flag_value
    except (AttributeError, flags.UnparsedFlagAccessError):
        # Flags not parsed yet
        pass

    # Check for environment variable override
    env_var = os.environ.get(flag_name)
    if env_var is not None:
        return env_var

    return None


def initialize_dtypes(device_type: Optional[str] = None, force_precision: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize all dtype variables based on detected device and flag overrides.

    Args:
        device_type: Optional device type to use instead of detecting
        force_precision: Force "full" (32-bit) or "mixed" (16-bit) precision regardless of device

    Returns:
        Dictionary containing all the dtype variables
    """
    global COMPUTE_DTYPE, PARAMS_DTYPE, LARGE_FLOAT_DTYPE, SMALL_FLOAT_DTYPE, LARGE_INT_DTYPE, MED_INT_DTYPE, SMALL_INT_DTYPE

    # Detect device type if not provided
    if device_type is None:
        device_type = detect_device_type()

    # Handle force_precision override
    if force_precision is not None:
        if force_precision.lower() == "full":
            device_type = "CPU"  # Use CPU config for full precision
        elif force_precision.lower() == "mixed":
            device_type = "GPU_A100"  # Use A100 config for mixed precision

    # Check for a global precision override flag
    precision_flag = get_flag_value_or_none("PRECISION")
    if precision_flag is not None:
        if precision_flag.lower() in ["full", "32bit", "float32"]:
            device_type = "CPU"  # Use CPU config for full precision
        elif precision_flag.lower() in ["mixed", "16bit", "bfloat16"]:
            device_type = "GPU_A100"  # Use A100 config for mixed precision

    # Get the base configuration for this device
    config = DEVICE_CONFIGS[device_type].copy()

    # Override with any explicitly set flags
    for var_name in config.keys():
        flag_value = get_flag_value_or_none(var_name)
        if flag_value is not None:
            if "FLOAT" in var_name:
                config[var_name] = FLOAT_DTYPE_MAP[flag_value]
            elif "INT" in var_name:
                config[var_name] = INT_DTYPE_MAP[flag_value]

    # Set the global variables
    COMPUTE_DTYPE = config["COMPUTE_DTYPE"]
    PARAMS_DTYPE = config["PARAMS_DTYPE"]
    LARGE_FLOAT_DTYPE = config["LARGE_FLOAT_DTYPE"]
    SMALL_FLOAT_DTYPE = config["SMALL_FLOAT_DTYPE"]
    LARGE_INT_DTYPE = config["LARGE_INT_DTYPE"]
    MED_INT_DTYPE = config["MED_INT_DTYPE"]
    SMALL_INT_DTYPE = config["SMALL_INT_DTYPE"]

    # Get actual hardware for display
    try:
        if jax.devices()[0].platform == "gpu":
            import subprocess
            hardware = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip()
        else:
            hardware = jax.devices()[0].device_kind
    except:
        hardware = "unknown"

    print(f"dtype_config: Initialized for {device_type} (Hardware: {hardware})")
    print(f"  COMPUTE_DTYPE: {COMPUTE_DTYPE}")
    print(f"  PARAMS_DTYPE: {PARAMS_DTYPE}")
    print(f"  LARGE_FLOAT_DTYPE: {LARGE_FLOAT_DTYPE}")
    print(f"  SMALL_FLOAT_DTYPE: {SMALL_FLOAT_DTYPE}")
    print(f"  LARGE_INT_DTYPE: {LARGE_INT_DTYPE}")
    print(f"  MED_INT_DTYPE: {MED_INT_DTYPE}")
    print(f"  SMALL_INT_DTYPE: {SMALL_INT_DTYPE}")

    return {
        "COMPUTE_DTYPE": COMPUTE_DTYPE,
        "PARAMS_DTYPE": PARAMS_DTYPE,
        "LARGE_FLOAT_DTYPE": LARGE_FLOAT_DTYPE,
        "SMALL_FLOAT_DTYPE": SMALL_FLOAT_DTYPE,
        "LARGE_INT_DTYPE": LARGE_INT_DTYPE,
        "MED_INT_DTYPE": MED_INT_DTYPE,
        "SMALL_INT_DTYPE": SMALL_INT_DTYPE,
    }


# Initialize the dtype variables on module import
initialize_dtypes()


def get_dtype_config() -> Dict[str, Any]:
    """
    Get the current dtype configuration.

    Returns:
        Dictionary containing all dtype variables
    """
    return {
        "COMPUTE_DTYPE": COMPUTE_DTYPE,
        "PARAMS_DTYPE": PARAMS_DTYPE,
        "LARGE_FLOAT_DTYPE": LARGE_FLOAT_DTYPE,
        "SMALL_FLOAT_DTYPE": SMALL_FLOAT_DTYPE,
        "LARGE_INT_DTYPE": LARGE_INT_DTYPE,
        "MED_INT_DTYPE": MED_INT_DTYPE,
        "SMALL_INT_DTYPE": SMALL_INT_DTYPE,
    }
