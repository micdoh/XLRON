import jax
import jax.numpy as jnp
from xlron.environments.env_funcs import *

# TODO - define lower/highest GSNR heuristics. Will require returning an alternative mask
#  e.g. of available SNR on path or of required slots


@partial(jax.jit, static_argnums=(1,))
def ksp_ff(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the first available slot from the shortest available path
    Method: Go through action mask and find the first available slot, starting from shortest path

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Action
    """
    first_slots = first_fit(state, params)
    # Chosen path is the first one with an available slot
    path_index = jnp.argmax(first_slots < params.link_resources)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


def ksp_ff_multiband(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the first available slot from all k-shortest paths in multiband scenario
    Method: Go through action mask and find the first available slot, starting from shortest path

    Args:
        state (MultiBandRSAEnvState): Environment state specific to multiband operations
        params (MultiBandRSAEnvParams): Environment parameters including multiband details
    Returns:
        chex.Array: Action
    """
    pass


@partial(jax.jit, static_argnums=(1,))
def ksp_lf(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the last available slot on the shortest available path
    Method: Go through action mask and find the last available slot, starting from shortest path

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Action
    """
    last_slots = last_fit(state, params)
    # Chosen path is the first one with an available slot
    path_index = jnp.argmax(last_slots < params.link_resources)
    slot_index = last_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def ff_ksp(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the first available slot from all paths
    Method: Go through action mask and find the first available slot on all paths

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Action
    """
    first_slots = first_fit(state, params)
    # Chosen path is the one with the lowest index of first available slot
    path_index = jnp.argmin(first_slots)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def lf_ksp(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the last available slot from all paths
    Method: Go through action mask and find the last available slot on all paths

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Action
    """
    last_slots = last_fit(state, params)
    # Chosen path is the one with the highest index of last available slot
    path_index = jnp.argmax(last_slots)
    slot_index = last_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def ksp_bf(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the first available slot from all k-shortest paths
    Method: Go through action mask and find the first available slot, starting from shortest path

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Action
    """
    best_slots, fitness = best_fit(state, params)
    # Chosen path is the first one with an available slot
    path_index = jnp.argmin(jnp.where(fitness < jnp.inf, 0, 1))
    slot_index = best_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def bf_ksp(state: EnvState, params: EnvParams) -> chex.Array:
    """Get the first available slot from the first k-shortest paths
    Method: Go through action mask and find the first available slot on all paths

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Action
    """
    best_slots, fitness = best_fit(state, params)
    # Chosen path is the one with the best fit
    path_index = jnp.argmin(fitness)
    slot_index = best_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1, 2, 3))
def ksp_mu(state: EnvState, params: EnvParams, unique_lightpaths: bool, relative: bool) -> chex.Array:
    """Get the most-used slot on the shortest available path.
    Method: Go through action mask and find the utilisation of available slots on each path.
    Find the shortest available path and choose the most utilised slot on that path.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
        unique_lightpaths (bool): Whether to consider unique lightpaths
        relative (bool): Whether to return relative utilisation

    Returns:
        chex.Array: Action
    """
    mask = get_action_mask(state, params)
    most_used_slots = most_used(state, params, unique_lightpaths, relative)
    # Get usage of available slots
    most_used_mask = most_used_slots * mask
    # Get index of most-used available slot for each path
    most_used_slots = jnp.argmax(most_used_mask, axis=1).astype(jnp.int32)
    # Chosen path is the first one with an available slot
    available_paths = jnp.max(mask, axis=1)
    path_index = jnp.argmax(available_paths)
    slot_index = most_used_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1, 2, 3))
def mu_ksp(state: EnvState, params: EnvParams, unique_lightpaths: bool, relative: bool) -> chex.Array:
    """Use the most-used available slot on any path.
    The most-used slot is that which has the most unique lightpaths (if unique_lightpaths=True) or active lightpaths.
    Method: Go through action mask and find the usage of available slots, choose available slot that is most utilised.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
        unique_lightpaths (bool): Whether to consider unique lightpaths
        relative (bool): Whether to return relative utilisation

    Returns:
        chex.Array: Action
    """
    mask = get_action_mask(state, params)
    # Get most used slots by summing the link_slot_array along the links
    most_used_slots = most_used(state, params, unique_lightpaths, relative)
    # Get usage of available slots
    most_used_mask = most_used_slots * mask
    # Chosen slot is the most used globally
    action = jnp.argmax(most_used_mask)
    return action


@partial(jax.jit, static_argnums=(1,))
def kmc_ff(state: EnvState, params: EnvParams) -> chex.Array:
    """K-Minimum Cut. Only suitable for RSA/RMSA.
    Method:
    1. Go through action mask and find the first available slot on all paths.
    2. For each path, allocate the first available slot.
    3. Sum number of new consecutive zero regions (cuts) created by assignment (on each link)
    4. Choose path that creates the fewest cuts.
    """
    mask = get_action_mask(state, params)
    first_slots = first_fit(state, params)
    link_slot_array = jnp.where(state.link_slot_array < 0, 1., state.link_slot_array)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)
    block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(link_slot_array)
    block_sizes_mask = jnp.where(block_sizes > 0, 1, 0.)  # Binary array showing initial block starts
    block_count = jnp.sum(block_sizes_mask, axis=1)

    def get_cuts_on_path(i, result):
        initial_slot_index = first_slots[i] % params.link_resources
        path = get_paths(params, nodes_sd)[i]
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        num_slots = required_slots(requested_bw, se, params.slot_size, guardband=params.guardband)
        # Make link-slot_array positive
        updated_slots = vmap_set_path_links(link_slot_array, path, initial_slot_index, num_slots, 1.)
        updated_block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(updated_slots)
        updated_block_sizes_mask = jnp.where(updated_block_sizes > 0, 1, 0)  # Binary array showing updated block starts
        updated_block_count = jnp.sum(updated_block_sizes_mask, axis=1)
        num_cuts = jax.lax.cond(
            mask[i][initial_slot_index] == 0.,  # If true, no valid action for path
            lambda x: jnp.full((1,), params.link_resources*params.num_links).astype(jnp.float32),  # Return max no. of cuts
            lambda x: jnp.sum(jnp.maximum(updated_block_count - block_count, 0.)).reshape((1,)),  # Else, return number of cuts
            1.
        )
        result = jax.lax.dynamic_update_slice(result, num_cuts, (i,))
        return result

    # Initialise array to hold number of cuts on each path
    path_cuts_array = jnp.full((mask.shape[0],), 0.)
    path_cuts_array = jax.lax.fori_loop(0, mask.shape[0], get_cuts_on_path, path_cuts_array)
    path_index = jnp.argmin(path_cuts_array)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def kmf_ff(state: RSAEnvState, params: RSAEnvParams) -> chex.Array:
    """K-Minimum Frag-size. Only suitable for RSA/RMSA.
    Method:
    1. Go through action mask and find the first available slot on all paths.
    2. For each path, allocate the first available slot.
    3. Sum number of new consecutive zero regions (cuts) created by assignment (on each link)
    4. Choose path that creates the fewest cuts.
    """
    mask = get_action_mask(state, params)
    first_slots = first_fit(state, params)
    link_slot_array = jnp.where(state.link_slot_array < 0, 1., state.link_slot_array)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)
    blocks = jax.vmap(find_block_sizes, in_axes=(0,))(link_slot_array)

    def get_frags_on_path(i, result):
        initial_slot_index = first_slots[i] % params.link_resources
        path = get_paths(params, nodes_sd)[i]
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        num_slots = required_slots(requested_bw, se, params.slot_size, guardband=params.guardband)
        # Mask on path links
        block_sizes = jax.vmap(lambda x, y: jnp.where(x > 0, y, 0.), in_axes=(0, 0))(path, blocks)
        updated_slots = vmap_set_path_links(state.link_slot_array, path, initial_slot_index, num_slots, -1)
        updated_block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(updated_slots)
        # Mask on path links
        updated_block_sizes = jax.vmap(lambda x, y: jnp.where(x > 0, y, 0.), in_axes=(0, 0))(path, updated_block_sizes)
        difference = updated_block_sizes - block_sizes
        new_frags = jnp.where(difference != 0, block_sizes + difference, 0.)
        # Slice new frags up to initial slot index (so as to only consider frags to the left)
        new_frags = jnp.where(jnp.arange(params.link_resources) < initial_slot_index, new_frags, 0.)
        new_frag_size = jnp.sum(new_frags)
        num_frags = jax.lax.cond(
            mask[i][initial_slot_index] == 0.,  # If true, no valid action for path
            lambda x: jnp.full((1,), float(params.link_resources * params.num_links)),  # Return max frag size
            lambda x: new_frag_size.reshape((1,)),
            # Else, return number of cuts
            1.
        )
        result = jax.lax.dynamic_update_slice(result, num_frags, (i,))
        return result

    # Initialise array to hold number of cuts on each path
    path_frags_array = jnp.full((mask.shape[0],), 0.)
    path_frags_array = jax.lax.fori_loop(0, mask.shape[0], get_frags_on_path, path_frags_array)
    path_index = jnp.argmin(path_frags_array)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def kme_ff(state: EnvState, params: EnvParams) -> chex.Array:
    """K-Minimum Entropy. Only suitable for RSA/RMSA.
    Method:
    1. Go through action mask and find the first available slot on all paths.
    2. For each path, allocate the first available slot.
    3. Sum number of new consecutive zero regions (cuts) created by assignment (on each link)
    4. Choose path that creates the fewest cuts.
    """
    mask = get_action_mask(state, params)
    first_slots = first_fit(state, params)
    link_slot_array = jnp.where(state.link_slot_array < 0, 1., state.link_slot_array)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)
    max_entropy = jnp.sum(jnp.log(params.link_resources)) * params.num_links

    def get_link_entropy(blocks):
        ent = jax.vmap(lambda x: jnp.sum(x/params.link_resources * jnp.log(params.link_resources/x)), in_axes=0)(blocks)
        return jnp.sum(jnp.where(blocks > 0, ent, 0))

    def get_entropy_on_path(i, result):
        initial_slot_index = first_slots[i] % params.link_resources
        path = get_paths(params, nodes_sd)[i]
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        num_slots = required_slots(requested_bw, se, params.slot_size, guardband=params.guardband)
        # Make link-slot_array positive
        updated_slots = vmap_set_path_links(link_slot_array, path, initial_slot_index, num_slots, 1.)
        updated_block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(updated_slots)
        updated_entropy = jax.vmap(get_link_entropy, in_axes=(0,))(updated_block_sizes)
        new_path_entropy = jnp.sum(jnp.dot(path, updated_entropy)).reshape((1,))
        new_path_entropy = jax.lax.cond(
            mask[i][initial_slot_index] == 0.,  # If true, no valid action for path
            lambda x: max_entropy.astype(jnp.float32).reshape((1,)),  # Return maximum entropy
            lambda x: new_path_entropy,  # Else, return number of cuts
            1.
        )
        result = jax.lax.dynamic_update_slice(result, new_path_entropy, (i,))
        return result

    path_entropy_array = jnp.full((mask.shape[0],), 0.)
    path_entropy_array = jax.lax.fori_loop(0, mask.shape[0], get_entropy_on_path, path_entropy_array)
    path_index = jnp.argmin(path_entropy_array)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def kca_ff(state: EnvState, params: EnvParams) -> chex.Array:
    """Congestion-aware First Fit. Only suitable for RSA/RMSA.
    Method:

    """
    mask = get_action_mask(state, params)
    # Get index of first available slots for each path
    first_slots = first_fit(state, params)
    # Get nodes
    nodes_sd, _ = read_rsa_request(state.request_array)
    # Initialise array to hold congestion on each path
    path_congestion_array = jnp.full((mask.shape[0],), 0.)
    link_weights = get_link_weights(state, params)

    def get_path_congestion(i, val):
        # Get links on path
        path = get_paths(params, nodes_sd)[i]
        # Get congestion
        path_link_congestion = jnp.multiply(link_weights, path)
        path_congestion = jnp.sum(path_link_congestion).reshape((1,))
        return jax.lax.dynamic_update_slice(val, path_congestion, (i,))

    path_congestion_array = jax.lax.fori_loop(0, mask.shape[0], get_path_congestion, path_congestion_array)
    path_index = jnp.argmin(path_congestion_array)
    slot_index = first_slots[path_index] % params.link_resources
    action = path_index * params.link_resources + slot_index
    return action


def get_link_weights(state: EnvState, params: EnvParams):
    """Get link weights based on occupancy for use in congestion-aware routing heuristics.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        chex.Array: Link weights
    """
    if params.__class__.__name__ != "RWALightpathReuseEnvParams":
        link_occupancy = jnp.count_nonzero(state.link_slot_array, axis=1)
    else:
        initial_path_capacity = init_path_capacity_array(
            params.link_length_array.val, params.path_link_array.val, scale_factor=1.0
        )
        initial_path_capacity = jnp.squeeze(jax.vmap(lambda x: initial_path_capacity[x])(state.path_index_array))
        utilisation = jnp.where(initial_path_capacity - state.link_capacity_array < 0, 0,
                                initial_path_capacity - state.link_capacity_array) / initial_path_capacity
        link_occupancy = jnp.sum(utilisation, axis=1)
    link_weights = jnp.multiply(params.link_length_array.val.T, (1 / (1 - link_occupancy / (params.link_resources + 1))))[0]
    return link_weights


def get_action_mask(state: EnvState, params: EnvParams) -> chex.Array:
    """N.B. The mask must already be present in the state!"""
    mask = jnp.reshape(state.link_slot_mask, (params.k_paths, -1))
    return mask


def best_fit(state: EnvState, params: EnvParams) -> chex.Array:
    """Best-Fit Spectrum Allocation. Returns the best fit slot for each path."""
    mask = get_action_mask(state, params)
    link_slot_array = jnp.where(state.link_slot_array < 0, 1., state.link_slot_array)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)

    # We need to define a wrapper function in order to vmap with named arguments
    def _find_block_sizes(arr, starts_only=False, reverse=True):
        return jax.vmap(find_block_sizes, in_axes=(0, None, None))(arr, starts_only, reverse)

    block_sizes_right = _find_block_sizes(link_slot_array, starts_only=False, reverse=False)
    block_sizes_left = _find_block_sizes(link_slot_array, starts_only=False, reverse=True)
    block_sizes = jnp.maximum((block_sizes_left + block_sizes_right) - 1, 0)
    paths = get_paths(params, nodes_sd)
    se = get_paths_se(params, nodes_sd) if params.consider_modulation_format else jnp.ones((params.k_paths,))
    num_slots = jax.vmap(required_slots, in_axes=(None, 0, None, None))(requested_bw, se, params.slot_size, params.guardband)

    # Quantify how well the request fits within a free spectral block
    def get_bf_on_path(path, blocks, req_slots):
        fits = jax.vmap(lambda x: x - req_slots, in_axes=0)(blocks)
        fits = jnp.where(fits >= 0, fits, params.link_resources)
        path_fit = jnp.dot(path, fits) / jnp.sum(path)
        return path_fit
    fits_block = jax.vmap(lambda x, y, z: get_bf_on_path(x, y, z), in_axes=(0, None, 0))(paths, block_sizes, num_slots)

    # Quantity much of a gap there is between the assigned slots and the next occupied slots on the left
    def get_bf_on_path_left(path, blocks, req_slots):
        fits = jax.vmap(lambda x: x - req_slots, in_axes=0)(blocks)
        fits = jnp.where(fits >= 0, fits, params.link_resources)
        fits_shift = jax.lax.dynamic_slice(fits, (0, 1), (fits.shape[0], fits.shape[1]-1))
        fits_shift = jnp.concatenate((jnp.full((fits.shape[0], 1), params.link_resources), fits_shift), axis=1)
        fits = fits + 1/jnp.maximum(fits_shift, 1)
        path_fit = jnp.dot(path, fits) / jnp.sum(path)
        return path_fit
    fits_left = jax.vmap(lambda x, y, z: get_bf_on_path_left(x, y, z), in_axes=(0, None, 0))(paths, block_sizes_left, num_slots)

    # Quantity much of a gap there is between the assigned slots and the next occupied slots on the right
    def get_bf_on_path_right(path, blocks, req_slots):
        fits = jax.vmap(lambda x: x - req_slots, in_axes=0)(blocks)
        fits = jnp.where(fits >= 0, fits, params.link_resources)
        fits_shift = jax.lax.dynamic_slice(fits, (0, 0), (fits.shape[0], fits.shape[1] - 1))
        fits_shift = jnp.concatenate((fits_shift, jnp.full((fits.shape[0], 1), params.link_resources)), axis=1)
        fits = fits + 1/jnp.maximum(fits_shift, 1)
        path_fit = jnp.dot(path, fits) / jnp.sum(path)
        return path_fit
    fits_right = jax.vmap(lambda x, y, z: get_bf_on_path_right(x, y, z), in_axes=(0, None, 0))(paths, block_sizes_right, num_slots)

    # Sum the contribution to the overall quality of fit, and scale down the left/right contributions
    fits = jnp.sum(jnp.stack((fits_block, fits_left/params.link_resources, fits_right/params.link_resources), axis=0), axis=0)
    # Mask out occupied lightpaths (in case the quality of fit on some links is good enough to be considered, even if the overall path is invalid)
    fits = jnp.where(mask == 0, jnp.inf, fits)
    best_slots = jnp.argmin(fits, axis=1)
    best_fits = jnp.min(fits, axis=1)
    return best_slots, best_fits


def first_fit(state: EnvState, params: EnvParams) -> chex.Array:
    """First-Fit Spectrum Allocation. Returns the first fit slot for each path."""
    mask = get_action_mask(state, params)
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "first_slots"
    mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
    # Get index of first available slots for each path
    first_slots = jnp.argmax(mask, axis=1)
    return first_slots


def last_fit(state: EnvState, params: EnvParams) -> chex.Array:
    """Last-Fit Spectrum Allocation. Returns the last fit slot for each path."""
    mask = get_action_mask(state, params)
    # Add a column of ones to the mask to make sure that occupied paths have non-zero index in "last_slots"
    mask = jnp.concatenate((jnp.full((mask.shape[0], 1), 1), mask), axis=1)
    # Get index of last available slots for each path
    last_slots = jnp.argmax(mask[:, ::-1], axis=1)
    # Convert to index from the left
    last_slots = params.link_resources - last_slots - 1
    return last_slots


@partial(jax.jit, static_argnums=(1, 2, 3))
def most_used(state: EnvState, params: EnvParams, unique_lightpaths, relative) -> chex.Array:
    """Get the amount of utilised bandwidth on each lightpath.
    If RWA-LR environment, the utilisation of a slot is defined by either the count of unique active lightpahts on the
    slot (if unique_lightpaths is True) or the count of active lightpaths on the slot (if unique_lightpaths is False).
    If RSA-type environment, utilisation is the count of active lightpaths on that slot.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
        unique_lightpaths (bool): Whether to consider unique lightpaths
        relative (bool): Whether to return relative utilisation

    Returns:
        chex.Array: Most used slots (array length = link_resources)
    """
    if params.__class__.__name__ != "RWALightpathReuseEnvParams":
        most_used_slots = jnp.count_nonzero(state.link_slot_array, axis=0) + 1
    elif params.__class__.__name__ == "RWALightpathReuseEnvParams" and not unique_lightpaths:
        # Get initial path capacity
        initial_path_capacity = init_path_capacity_array(
            params.link_length_array.val, params.path_link_array.val, scale_factor=1.0
        )
        initial_path_capacity = jnp.squeeze(jax.vmap(lambda x: initial_path_capacity[x])(state.path_index_array))
        utilisation = jnp.where(initial_path_capacity - state.link_capacity_array < 0, 0,
                                initial_path_capacity - state.link_capacity_array)
        if relative:
            utilisation = utilisation / initial_path_capacity
        # Get most used slots by summing the utilisation along the slots
        most_used_slots = jnp.sum(utilisation, axis=0) + 1
    else:
        most_used_slots = jnp.count_nonzero(state.path_index_array + 1, axis=0) + 1
    return most_used_slots
