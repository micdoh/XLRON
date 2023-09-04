import jax
import jax.numpy as jnp
from xlron.environments.env_funcs import *


@partial(jax.jit, static_argnums=(1,))
def ksp_ff(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the first available slot from all k-shortest paths
    Method: Go through action mask and find the first available slot, starting from shortest path

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Action
    """
    state = mask_slots(state, params, state.request_array)
    mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
    mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
    # Get index of first available slots for each path
    first_slots = jax.vmap(jnp.argmax, in_axes=(0))(mask)
    # Chosen path is the first one with an available slot
    path_index = jnp.argmax(first_slots < params.link_resources)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * state.link_slot_array.shape[0] + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def ff_ksp(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the first available slot from the first k-shortest paths
    Method: Go through action mask and find the first available slot on all paths

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Action
    """
    state = mask_slots(state, params, state.request_array)
    mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
    mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
    # Get index of first available slots for each path
    first_slots = jax.vmap(jnp.argmax, in_axes=(0))(mask)
    # Chosen path is the one with the lowest index of first available slot
    path_index = jnp.argmin(first_slots)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * state.link_slot_array.shape[0] + slot_index
    return action
