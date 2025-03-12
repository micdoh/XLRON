import jax
import jax.numpy as jnp
import chex
from functools import partial


def straight_through(hard_output, soft_output):
    """
    General straight-through gradient trick.

    Args:
        hard_output: The non-differentiable (but correct) output for forward pass
        soft_output: The differentiable approximation for backward pass

    Returns:
        An output that uses hard_output in the forward pass but has
        gradients of soft_output in the backward pass
    """
    return soft_output + jax.lax.stop_gradient(hard_output - soft_output)


def differentiable_where(condition, true_val, false_val, threshold, temperature=1.0):
    """
    A differentiable version of jnp.where.

    Args:
        condition: The condition to check (typically a boolean or 0/1 array)
        true_val: Value to use where condition is True
        false_val: Value to use where condition is False
        temperature: Controls the sharpness of the sigmoid approximation

    Returns:
        Result that behaves like jnp.where in forward pass but is differentiable
    """
    # Hard version for forward pass
    hard_result = jnp.where(condition, true_val, false_val)
    # Soft version for backward pass
    # Convert boolean condition to 0/1 if needed
    condition_float = jnp.asarray(condition, dtype=jnp.float32)
    # Sigmoid approximation of the condition
    soft_mask = jax.nn.sigmoid(temperature * (condition_float - threshold))
    soft_result = soft_mask * true_val + (1 - soft_mask) * false_val
    # Apply straight-through gradient trick
    return straight_through(hard_result, soft_result)


def differentiable_equals(x, y, temperature=1.0):
    """
    A differentiable version of x == y.

    Args:
        x: First tensor
        y: Second tensor or scalar
        temperature: Controls the sharpness of the sigmoid approximation

    Returns:
        Result that behaves like x == y in forward pass but is differentiable
    """
    # Hard version
    hard_result = (x == y)
    # Soft version using sigmoid centered at y
    soft_result = jax.nn.sigmoid(temperature * (1.0 - jnp.abs(x - y)))
    # Apply straight-through gradient trick
    return straight_through(hard_result, soft_result)


def differentiable_argmax(x, temperature=1.0):
    """
    A differentiable version of jnp.argmax.

    Args:
        x: Input tensor to find argmax
        temperature: Controls the sharpness of the softmax approximation

    Returns:
        Result that behaves like argmax in forward pass but is differentiable
    """
    # Hard version: Standard argmax (non-differentiable)
    hard_argmax = jnp.argmax(x)

    # Soft version: Weighted expectation using softmax
    logits = x * temperature  # Scale for sharper distribution
    probs = jax.nn.softmax(logits)
    indices = jnp.arange(x.shape[0], dtype=jnp.float32)
    soft_argmax = jnp.sum(probs * indices)

    # Apply straight-through gradient trick
    return straight_through(hard_argmax, soft_argmax)


def masked_select(values, mask, replacement, threshold, temperature=1.0):
    """
    A differentiable version of selecting values based on a mask.
    Equivalent to jnp.where(mask, values, replacement).

    Args:
        values: The values to select from
        mask: Boolean or 0/1 mask indicating which values to keep
        replacement: Value to use where mask is False
        temperature: Controls the sharpness of the sigmoid approximation

    Returns:
        Masked values that are differentiable
    """
    return differentiable_where(mask, values, replacement, threshold, temperature)


def differentiable_round_simple(x, temperature=1.0):
    """
    A simpler differentiable approximation of rounding.

    Args:
        x: Input tensor to round
        temperature: Controls the steepness of the sigmoid at rounding boundaries

    Returns:
        Tensor with rounded values in forward pass but differentiable gradients
    """
    # Hard version: Standard round (non-differentiable)
    hard_round = jnp.round(x)
    # Soft version: Use sigmoid for each fractional part
    fractional = x - jnp.floor(x)
    # For values < 0.5, we round down (output = floor(x))
    # For values >=.5, we round up (output = floor(x) + 1)
    # This sigmoid approaches 1.0 when fractional >= 0.5
    sigmoid_result = jax.nn.sigmoid(temperature * (fractional - 0.5))
    # Combine floor and ceiling with sigmoid weighting
    soft_round = jnp.floor(x) + sigmoid_result
    # Apply straight-through gradient trick
    return straight_through(hard_round, soft_round)


def differentiable_round(x, decimals=0, temperature=1.0):
    """
    A differentiable approximation of rounding to specified decimals.

    Args:
        x: Input tensor to round
        decimals: Number of decimal places to round to
        temperature: Controls the steepness of the sigmoid at rounding boundaries

    Returns:
        Tensor with rounded values in forward pass but differentiable gradients
    """
    # Scale to make decimal rounding equivalent to integer rounding
    scale = 10.0 ** decimals
    x_scaled = x * scale
    # Hard version: Standard round (non-differentiable)
    hard_round = jnp.round(x_scaled) / scale
    # Soft version: Use sigmoid for each fractional part
    floor_scaled = jnp.floor(x_scaled)
    fractional = x_scaled - floor_scaled
    # This sigmoid approaches 1.0 when fractional >= 0.5
    sigmoid_result = jax.nn.sigmoid(temperature * (fractional - 0.5))
    # Combine floor and ceiling with sigmoid weighting
    soft_round = (floor_scaled + sigmoid_result) / scale
    # Apply straight-through gradient trick
    return straight_through(hard_round, soft_round)


def differentiable_ceil(x, temperature=1.0):
    """
    A differentiable approximation of ceiling function.

    Args:
        x: Input tensor
        temperature: Controls the steepness of the sigmoid at integer boundaries

    Returns:
        Tensor with ceiling values in forward pass but differentiable gradients
    """
    # Hard version: Standard ceiling (non-differentiable)
    hard_ceil = jnp.ceil(x)
    # Soft version:
    # The fractional part determines how close we are to the next integer
    floor_x = jnp.floor(x)
    fractional = x - floor_x
    # When fractional is 0, we're exactly at an integer and ceil = floor
    # When fractional is > 0, ceiling is floor + 1
    # We use a sigmoid that approaches 1 for any fractional > 0
    # Higher temperature makes this transition sharper
    ceil_offset = jax.nn.sigmoid(temperature * fractional)
    soft_ceil = floor_x + ceil_offset
    # Apply straight-through gradient trick
    return straight_through(hard_ceil, soft_ceil)


def differentiable_one_hot_index_update(array, indices, values):
    """
    A differentiable version of array.at[indices].set(values).

    Args:
        array: Base array to update
        indices: Indices where to set values
        values: Values to set at indices

    Returns:
        Updated array that maintains gradient flow
    """
    # Hard version: Standard index update (non-differentiable for indices)
    hard_result = array.at[indices].set(values)
    # Create a soft version using a mask
    # This isn't strictly necessary for most cases since typically
    # the values and indices are fixed, but it helps if indices depend on learned parameters
    mask = jnp.zeros_like(array)
    mask = mask.at[indices].set(1.0)
    # For each position, either keep original value or use new value based on mask
    soft_result = (1 - mask) * array + mask * values
    # Apply straight-through gradient trick
    return straight_through(hard_result, soft_result)


def differentiable_index(array, index, temperature=1.0):
    """
    Differentiable indexing along the 0-axis (first dimension).
    e.g. can replace array[index] with differentiable_index(array, index)

    Args:
        array: Input array/tensor to index into
        index: Index to select (can be fractional)
        temperature: Controls sharpness of selection (larger = sharper)

    Returns:
        The indexed value with gradient flow through the index
    """
    hard_result = array[index.astype(jnp.int32)]
    # Ensure index is a float for gradient flow
    index = jnp.asarray(index, dtype=jnp.float32)
    # Get the length of the first dimension
    length = array.shape[0]
    # Create positions
    positions = jnp.arange(length, dtype=jnp.float32)
    # Create weights based on proximity to the target index
    weights = jnp.exp(-((positions - index) ** 2) * temperature)
    # Normalize weights to sum to 1
    weights = weights / jnp.sum(weights)
    # Reshape weights for broadcasting along the first dimension only
    weights_shape = [length] + [1] * (array.ndim - 1)
    weights = weights.reshape(weights_shape)
    # Apply weights and sum along the first dimension
    soft_result = jnp.sum(array * weights, axis=0)
    # Apply straight-through gradient trick
    return straight_through(hard_result, soft_result)


def differentiable_window_mask(positions, index, half_window, temperature=1.0):
    """
    Create a differentiable mask for positions within a window of the index.
    Args:
        positions: Array of position indices
        index: Target index (can be fractional)
        half_window: Half-width of the window
        temperature: Controls sharpness of the mask boundaries
    Returns:
        Soft mask with values close to 1 inside window, close to 0 outside
    """
    # Calculate distance from window boundaries
    # Positive inside window, negative outside
    jax.debug.print("positions: {}", positions)
    jax.debug.print("index: {}", index)
    jax.debug.print("half_window: {}", half_window)
    dist_from_lower = positions - (index - half_window)
    dist_from_upper = (index + half_window) - positions
    # Apply sigmoid to create soft boundaries
    # Values close to 1 inside window, close to 0 outside
    lower_mask = jax.nn.sigmoid(temperature * dist_from_lower)
    upper_mask = jax.nn.sigmoid(temperature * dist_from_upper)
    # Combine masks (both need to be close to 1 to be in window)
    return lower_mask * upper_mask


def differentiable_index_windowed(array, index, window_size=5, temperature=1.0):
    """
    Numerically stable differentiable indexing along the 0-axis.
    """
    # Get hard result for forward pass
    index_int = jnp.floor(index).astype(jnp.int32)
    # Clip index to valid range to prevent out-of-bounds
    index_int = jnp.clip(index_int, 0, array.shape[0] - 1)
    hard_result = array[index_int]

    # Ensure index is a float for gradient flow
    index = jnp.asarray(index, dtype=jnp.float32)

    # Get window bounds (clip to array bounds)
    half_window = window_size // 2
    window_start = jnp.maximum(0, jnp.floor(index - half_window).astype(jnp.int32))
    window_end = jnp.minimum(array.shape[0], jnp.ceil(index + half_window).astype(jnp.int32) + 1)

    # Use dynamic_slice to get window content
    window_size_actual = window_end - window_start
    window_slice = jax.lax.dynamic_slice(
        array, (window_start,), (window_size_actual,))

    # Create window positions
    positions = jnp.arange(window_start, window_end, dtype=jnp.float32)

    # Calculate weights with numerical stability safeguards
    # Use squared distance but scale it properly with temperature
    distances = positions - index
    # Scaling factor should make temperature DECREASE the gradient magnitude
    weights_logits = -(distances ** 2) * temperature
    # Subtract max for numerical stability before exp
    weights_logits = weights_logits - jnp.max(weights_logits)
    weights = jnp.exp(weights_logits)
    weights_sum = jnp.sum(weights) + 1e-10  # Avoid division by zero
    weights = weights / weights_sum

    # Compute weighted sum
    soft_result = jnp.sum(window_slice * weights)

    # Apply straight-through estimator
    return straight_through(hard_result, soft_result)


# def differentiable_index_windowed(array, index, window_size=5, temperature=1.0):
#     """
#     Differentiable indexing along the 0-axis (first dimension) with windowed weight calculation.
#     Only calculates weights for indices within a window around the target index.
#
#     Args:
#         array: Input array/tensor to index into
#         index: Index to select (can be fractional)
#         window_size: Size of the window around the target index to consider
#         temperature: Controls sharpness of selection (larger = sharper)
#
#     Returns:
#         The indexed value with gradient flow through the index
#     """
#     # Get hard result for forward pass
#     hard_result = array[index.astype(jnp.int32)]
#     # Ensure index is a float for gradient flow
#     index = jnp.asarray(index, dtype=jnp.float32)
#     # Get the length of the first dimension
#     length = array.shape[0]
#     # Calculate window bounds
#     half_window = window_size // 2
#     positions = jnp.arange(length, dtype=jnp.float32)
#     in_window = differentiable_window_mask(positions, index, half_window)
#     # Calculate weights only for positions in the window
#     # Use where to avoid computing exponentials for all positions
#     distance_squared = differentiable_where(in_window, (positions - index) ** 2, 1e8, 1.0)
#     weights = differentiable_where(in_window, jnp.exp(-distance_squared * temperature), 0.0, 1.0)
#     # Normalize weights to sum to 1
#     # Add small epsilon to prevent division by zero
#     weights = weights / (jnp.sum(weights) + 1e-10)
#     # Reshape weights for broadcasting along the first dimension only
#     weights_shape = [length] + [1] * (array.ndim - 1)
#     weights = weights.reshape(weights_shape)
#     # Apply weights and sum along the first dimension
#     soft_result = jnp.sum(array * weights, axis=0)
#     # Apply straight-through gradient trick
#     return straight_through(hard_result, soft_result)


def differentiable_slice(array, start_idx, slice_size=None, window_size=5, temperature=1.0):
    """
    Differentiable version of array[start_idx:start_idx+slice_size] that supports
    both single-index operations and contiguous slices efficiently.

    Args:
        array: Input array to slice from
        start_idx: Starting index (can be fractional)
        slice_size: Number of elements to slice (None for single-index operation)
        window_size: Size of the window around each target index
        temperature: Controls sharpness of selection (larger = sharper)

    Returns:
        Sliced values with gradient flow through indices
    """
    # Handle single-index case
    if slice_size is None:
        return differentiable_index_windowed(array, start_idx, window_size, temperature)

    # Handle contiguous slice case
    # Ensure start_idx is float for gradient flow
    start_idx = jnp.asarray(start_idx, dtype=jnp.float32)

    # Create slice indices
    indices = start_idx + jnp.arange(slice_size, dtype=jnp.float32)

    # Get the hard result for forward pass (standard dynamic slice)
    int_start = jnp.floor(start_idx).astype(jnp.int32)
    hard_result = jax.lax.dynamic_slice(
        array,
        (int_start,) + (0,) * (array.ndim - 1),
        (slice_size,) + array.shape[1:]
    )

    # Define a function to compute soft indexing for a single position
    def soft_index(idx):
        # Calculate window positions centered around idx
        half_window = window_size // 2

        # Get window start and end while keeping within array bounds
        window_start = jnp.maximum(0, jnp.floor(idx - half_window).astype(jnp.int32))
        window_end = jnp.minimum(array.shape[0], jnp.ceil(idx + half_window).astype(jnp.int32) + 1)

        # Create positions array for this window
        positions = jnp.arange(window_start, window_end, dtype=jnp.float32)

        # Calculate weights based on proximity to the target index
        distances = jnp.abs(positions - idx)
        weights = jnp.exp(-(distances ** 2) * temperature)
        weights = weights / (jnp.sum(weights) + 1e-10)

        # Get window values from array
        window_size_actual = window_end - window_start
        window_slice = jax.lax.dynamic_slice(
            array,
            (window_start,) + (0,) * (array.ndim - 1),
            (window_size_actual,) + array.shape[1:]
        )

        # Apply weights (reshape for broadcasting)
        weights_shaped = weights.reshape((window_size_actual,) + (1,) * (array.ndim - 1))
        weighted_sum = jnp.sum(window_slice * weights_shaped, axis=0)

        return weighted_sum

    # Apply soft indexing function to all indices in parallel
    soft_result = jax.vmap(soft_index)(indices)

    # Apply straight-through gradient trick
    return straight_through(hard_result, soft_result)


def differentiable_indexing(array, indices, window_size=5, temperature=1.0):
    """
    Unified differentiable indexing function that supports:
    1. Single indices: array[index]
    2. Contiguous slices: array[start:end]
    3. Multiple arbitrary indices: array[indices]

    Args:
        array: Input array to index into
        indices: Either a single index, a tuple of (start, size) for slicing,
                 or an array of indices
        window_size: Size of the window around each target index
        temperature: Controls sharpness of selection (larger = sharper)

    Returns:
        Indexed values with gradient flow through indices
    """
    # Check what type of indexing we're doing
    if isinstance(indices, tuple) and len(indices) == 2:
        # Contiguous slice with (start, size)
        start_idx, slice_size = indices
        return differentiable_slice(array, start_idx, slice_size, window_size, temperature)
    elif hasattr(indices, 'shape') and indices.shape:  # JAX array with multiple indices
        # Multiple arbitrary indices
        return jax.vmap(lambda idx: differentiable_index_windowed(
            array, idx, window_size, temperature))(indices)
    elif isinstance(indices, (list, tuple)) and len(indices) > 1:
        # Multiple arbitrary indices as list/tuple
        indices_array = jnp.array(indices, dtype=jnp.float32)
        return jax.vmap(lambda idx: differentiable_index_windowed(
            array, idx, window_size, temperature))(indices_array)
    else:
        # Single index
        return differentiable_index_windowed(array, indices, window_size, temperature)


# def differentiable_conditional(condition, true_fn, false_fn, *args):
#     """
#     A differentiable version of if-else using weighted combination.
#
#     Args:
#         condition: Boolean condition
#         true_fn: Function to call if condition is True
#         false_fn: Function to call if condition is False
#         *args: Arguments to pass to both functions
#
#     Returns:
#         Result with gradient flow maintained
#     """
#     # Hard version: Standard conditional
#     hard_result = jax.lax.cond(condition, true_fn, false_fn, *args)
#     # For a differentiable version, we could compute both branches and blend
#     true_result = true_fn(*args)
#     false_result = false_fn(*args)
#     # Convert condition to float for weighting
#     weight = jnp.asarray(condition, dtype=jnp.float32)
#     # Weighted combination
#     soft_result = weight * true_result + (1 - weight) * false_result
#     # Apply straight-through gradient trick
#     return straight_through(hard_result, soft_result)