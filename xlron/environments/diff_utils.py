import jax
import jax.numpy as jnp
import chex
from functools import partial

# TODO - for each of these functions. investigate what is the optimal value of temperature.
#  N.B. 1.0 seems to be good from initial investigations of indexing but some funcs might differ


def straight_through(hard_output, soft_output):
    """
    General straight-through gradient trick.

    Args:
    Args
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


@partial(jax.jit, static_argnums=(2, 3))
def differentiable_compare(x, y, op_type='==', temperature=1.0):
    """
    A unified differentiable comparison function that supports multiple operators.

    Args:
        x: First tensor
        y: Second tensor or scalar
        op_type: String specifying the comparison operation:
                '==', '!=', '>=', '<=', '>', '<', '!='
        temperature: Controls the sharpness of the sigmoid approximation

    Returns:
        Result that behaves like the specified comparison in forward pass
        but is differentiable in backward pass
    """
    # Define hard results (for forward pass)
    if op_type == '==':
        hard_result = (x == y)
        #soft_result = jax.nn.sigmoid(temperature * (1.0 - jnp.abs(x - y)))
        #soft_result = jax.nn.sigmoid(-temperature * jnp.abs(x - y))
        soft_result = jnp.exp(-temperature * (x - y) ** 2)
    elif op_type == '>=':
        hard_result = (x >= y)
        soft_result = jax.nn.sigmoid(temperature * (x - y))
    elif op_type == '<=':
        hard_result = (x <= y)
        soft_result = jax.nn.sigmoid(temperature * (y - x))
    elif op_type == '>':
        hard_result = (x > y)
        # Slightly sharper version for strict inequality
        soft_result = jax.nn.sigmoid(temperature * (x - y - 1e-5))
    elif op_type == '<':
        hard_result = (x < y)
        # Slightly sharper version for strict inequality
        soft_result = jax.nn.sigmoid(temperature * (y - x - 1e-5))
    elif op_type == '!=':
        hard_result = (x != y)
        # Invert the equality result
        #soft_result = 1.0 - jax.nn.sigmoid(temperature * (1.0 - jnp.abs(x - y)))
        #soft_result = 1.0 - jax.nn.sigmoid(-temperature * jnp.abs(x - y))
        soft_result = 1.0 - jnp.exp(-temperature * (x - y) ** 2)
    else:
        raise ValueError(f"Unknown operation type: {op_type}")

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
    temperature = 0.4
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
    return straight_through(hard_round, x)


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
    temperature = 0.4
    # Hard version: Standard round (non-differentiable)
    hard_round = jnp.round(x, decimals)
    # Scale to make decimal rounding equivalent to integer rounding
    scale = 10.0 ** decimals
    x_scaled = x * scale
    # Soft version: Use sigmoid for each fractional part
    fractional = x_scaled - jnp.floor(x_scaled)
    # This sigmoid approaches 1.0 when fractional >= 0.5
    sigmoid_result = jax.nn.sigmoid(temperature * (fractional - 0.5))
    # Combine floor and ceiling with sigmoid weighting
    soft_round = (jnp.floor(x_scaled) + sigmoid_result) / scale
    # Apply straight-through gradient trick
    return straight_through(hard_round, x)


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
    fractional = x - jnp.floor(x)
    # When fractional is 0, we're exactly at an integer and ceil = floor
    # When fractional is > 0, ceiling is floor + 1
    # We use a sigmoid that approaches 1 for any fractional > 0
    # Higher temperature makes this transition sharper
    ceil_offset = jax.nn.sigmoid(temperature * fractional)
    soft_ceil = jnp.floor(x) + ceil_offset
    # Apply straight-through gradient trick
    return straight_through(hard_ceil, x)


def differentiable_floor(x, temperature=1.0):
    """
    A differentiable approximation of floor function.

    Args:
        x: Input tensor
        temperature: Controls the steepness of the sigmoid at integer boundaries

    Returns:
        Tensor with floor values in forward pass but differentiable gradients
    """
    #temperature = 0.1
    # Hard version: Standard floor (non-differentiable)
    hard_floor = jnp.floor(x)
    # Soft version:
    # The fractional part determines how close we are to the previous integer
    fractional = jnp.ceil(x) - x
    # When fractional is 0, we're exactly at an integer and floor = ceil
    # When fractional is > 0, floor is ceil - 1
    # We use a sigmoid that approaches 1 for any fractional > 0
    # Higher temperature makes this transition sharper
    floor_offset = jax.nn.sigmoid(temperature * fractional)
    soft_floor = jnp.ceil(x) - floor_offset
    # Apply straight-through gradient trick
    return straight_through(hard_floor, x)


# TODO - this is broken
def differentiable_one_hot_index_update(array, indices, values, temperature):
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
    dist_from_lower = positions - (index - half_window)
    dist_from_upper = (index + half_window) - positions
    # Apply sigmoid to create soft boundaries
    # Values close to 1 inside window, close to 0 outside
    lower_mask = jax.nn.sigmoid(temperature * dist_from_lower)
    upper_mask = jax.nn.sigmoid(temperature * dist_from_upper)
    # Combine masks (both need to be close to 1 to be in window)
    return lower_mask * upper_mask


def differentiable_index(array, index, temperature=1.0):
    """
    Differentiable indexing along the 0-axis (first dimension) with windowed weight calculation.
    Only calculates weights for indices within a window around the target index.

    Args:
        array: Input array/tensor to index into
        index: Index to select (can be fractional)
        temperature: Controls sharpness of selection (larger = sharper)

    Returns:
        The indexed value with gradient flow through the index
    """
    # Get hard result for forward pass
    hard_result = array[jnp.asarray(index).astype(jnp.int32)]
    # Ensure index is a float for gradient flow
    index = jnp.asarray(index, dtype=jnp.float32)
    # Get the length of the first dimension
    length = array.shape[0]
    positions = jnp.arange(length, dtype=jnp.float32)
    # Use squared distance but scale it properly with temperature
    distances = positions - index
    # Scaling factor should make temperature DECREASE the gradient magnitude
    weights_logits = -(distances ** 2) * temperature
    # Subtract max for numerical stability before exp
    weights_logits = weights_logits - jnp.max(weights_logits)
    weights = jnp.exp(weights_logits)
    weights_sum = jnp.sum(weights) + 1e-10  # Avoid division by zero
    weights = weights / weights_sum
    # Reshape weights for broadcasting along the first dimension only
    weights_shape = [length] + [1] * (array.ndim - 1)
    weights = weights.reshape(weights_shape)
    # Apply weights and sum along the first dimension
    soft_result = jnp.sum(array * weights, axis=0)
    # Apply straight-through gradient trick
    return straight_through(hard_result, soft_result)


def differentiable_indexing(array, indices, temperature=1.0):
    """
    Unified differentiable indexing function that supports:
    1. Single indices: array[index]
    2. Multiple arbitrary indices: array[indices]

    Args:
        array: Input array to index into
        indices: Either a single index, a tuple of (start, size) for slicing,
                 or an array of indices
        temperature: Controls sharpness of selection (larger = sharper)

    Returns:
        Indexed values with gradient flow through indices
    """
    if hasattr(indices, 'shape') and indices.shape:
        # Multiple arbitrary indices as list/tuple or JAX array with multiple indices
        # Multiple arbitrary indices
        return jax.vmap(lambda idx: differentiable_index(
            array, idx, temperature
        ))(indices)
    else:
        # Single index
        return differentiable_index(array, indices, temperature)


def differentiable_cond(condition, true_fn, false_fn, operand, threshold=0.0, temperature=1.0):
    """
    A differentiable version of jax.lax.cond that is fully jittable.

    Args:
        condition: The condition to check (boolean or numeric)
        true_fn: Function to execute if condition is True
        false_fn: Function to execute if condition is False
        operand: Operand to pass to both functions
        threshold: Condition threshold value
        temperature: Controls the sharpness of the sigmoid approximation

    Returns:
        Result that behaves like jax.lax.cond in forward pass but is differentiable
    """
    # Hard result for forward pass
    hard_result = jax.lax.cond(condition, true_fn, false_fn, operand)

    # Get results from both branches using JAX control flow
    true_result = true_fn(operand)
    false_result = false_fn(operand)
    # Create a soft weight using sigmoid
    soft_weight = jax.nn.sigmoid(temperature * (condition - threshold))

    # Interpolate between results for gradient computation
    # Use tree_map to handle nested structures
    soft_result = jax.tree.map(
        lambda t, f: soft_weight * t + (1 - soft_weight) * f,
        true_result, false_result
    )

    # Apply straight-through trick with jax.lax.stop_gradient
    return jax.tree.map(
        straight_through,
        hard_result, soft_result
    )


# def differentiable_cond(condition, true_fn, false_fn, operand, threshold=0.0, temperature=1.0):
#     """
#     A differentiable version of jax.lax.cond with identity gradients.
#     """
#     # Get the actual result from the correct branch
#     hard_result = jax.lax.cond(condition, true_fn, false_fn, operand)
#
#     # For the soft result, we compute which branch we actually took
#     # and create a dummy path for gradients to flow through
#     took_true_branch = differentiable_where(condition, 1.0, 0.0, threshold, temperature)
#
#     # Compute both branches for gradient computation
#     true_result = true_fn(operand)
#     false_result = false_fn(operand)
#
#     # Direct gradient to whichever branch was taken
#     # Use tree_map to handle nested structures
#     soft_result = jax.tree.map(
#         lambda t, f: took_true_branch * t + (1 - took_true_branch) * f,
#         true_result, false_result
#     )
#
#     # Apply straight-through
#     return jax.tree.map(
#         straight_through,
#         hard_result, soft_result
#     )
