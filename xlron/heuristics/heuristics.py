from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from typing import cast

from xlron.environments.dataclasses import (
    EnvState,
    GNModelEnvParams,
    RMSAGNModelEnvParams,
    RSAEnvParams,
    RSAEnvState,
    RWALightpathReuseEnvParams,
    RWALightpathReuseEnvState,
)
from xlron.environments.env_funcs import (
    find_block_sizes,
    get_affected_slots_mask,
    get_paths,
    get_paths_se,
    init_path_capacity_array,
    mask_slots,
    mask_slots_rmsa_gn_model,
    mask_slots_rwalr,
    read_rsa_request,
    required_slots,
    set_path_links,
)
from xlron.environments.wrappers import jit_profiler

# TODO - define lower/highest GSNR heuristics. Will require returning an alternative mask
#  e.g. of available SNR on path or of required slots


@partial(jax.jit, static_argnums=(1,))
def ksp_ff(state: RSAEnvState, params: RSAEnvParams) -> Array:
    """Get the first available slot from the shortest available path
    Method: Go through action mask and find the first available slot, starting from shortest path

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    first_slots = first_fit(state, params)
    # Chosen path is the first one with an available slot
    path_index = jnp.argmax(first_slots < params.link_resources)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


def ksp_ff_multiband(state: EnvState, params: RSAEnvParams) -> None:
    """Get the first available slot from all k-shortest paths in multiband scenario
    Method: Go through action mask and find the first available slot, starting from shortest path

    Args:
        state (MultiBandRSAEnvState): Environment state specific to multiband operations
        params (MultiBandRSAEnvParams): Environment parameters including multiband details
    Returns:
        Array: Action
    """
    pass


@partial(jax.jit, static_argnums=(1,))
def ksp_lf(state: RSAEnvState, params: RSAEnvParams) -> Array:
    """Get the last available slot on the shortest available path
    Method: Go through action mask and find the last available slot, starting from shortest path

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    last_slots = last_fit(state, params)
    # Chosen path is the first one with an available slot.
    # last_fit signals "no slot" with -1 (standard branch) or
    # params.link_resources (GN band-order branch); exclude both.
    available = (last_slots >= 0) & (last_slots < params.link_resources)
    path_index = jnp.argmax(available)
    slot_index = last_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def ksp_ef(state: RSAEnvState, params: RSAEnvParams) -> Array:
    """K-Shortest Path, Exact-Fit. Only suitable for RSA/RMSA.
    On the shortest available path, allocate the first free block whose size exactly
    matches the required slots; if no exact-fit block exists, fall back to first-fit.
    Exact-fit consumes blocks whole, avoiding the creation of small unusable fragments.
    Reference: Chatterjee, Sarma & Oki, "Routing and spectrum allocation in elastic
    optical networks: A tutorial", IEEE Comms. Surveys & Tutorials 17(3), 2015.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    first_exact, _, _ = exact_fit(state, params)
    first_slots = first_fit(state, params)
    # Chosen path is the first one with an available slot (as in ksp_ff)
    path_index = jnp.argmax(first_slots < params.link_resources)
    slot_index = jnp.where(
        first_exact[path_index] < params.link_resources,
        first_exact[path_index],
        first_slots[path_index] % params.link_resources,
    )
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def ksp_flf(state: RSAEnvState, params: RSAEnvParams) -> Array:
    """K-Shortest Path, First-Last-Fit.
    On the shortest available path, allocate small requests first-fit (from the low
    end of the spectrum) and large requests last-fit (from the high end). Requests
    are "large" if their datarate exceeds the midpoint of the requestable range.
    Segregating request sizes to opposite spectrum ends reduces the size-mismatch
    fragmentation that a single first-fit pointer creates.
    Reference: Fadini & Oki, "A subcarrier-slot partition scheme for wavelength
    assignment in elastic optical networks", IEEE ICC 2014; Chatterjee, Sarma & Oki,
    IEEE Comms. Surveys & Tutorials 17(3), 2015.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    is_large = is_large_request(state, params)
    ff_action = ksp_ff(state, params)
    lf_action = ksp_lf(state, params)
    return jnp.where(is_large, lf_action, ff_action)


@partial(jax.jit, static_argnums=(1,))
def flf_ksp(state: RSAEnvState, params: RSAEnvParams) -> Array:
    """First-Last-Fit across K-Shortest Paths.
    Small requests take the globally first available slot across all paths (ff_ksp);
    large requests take the globally last available slot across all paths (lf_ksp).
    See ksp_flf for the partitioning rationale and references.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    is_large = is_large_request(state, params)
    ff_action = ff_ksp(state, params)
    lf_action = lf_ksp(state, params)
    return jnp.where(is_large, lf_action, ff_action)


@partial(jax.jit, static_argnums=(1,))
def ksp_flef(state: RSAEnvState, params: RSAEnvParams) -> Array:
    """K-Shortest Path, First-Last-Exact-Fit. Only suitable for RSA/RMSA.
    On the shortest available path: small requests take the lowest exactly-fitting
    free block, falling back to first-fit; large requests take the highest
    exactly-fitting free block, falling back to last-fit. Combines the size-class
    segregation of first-last-fit with exact-fit's fragment avoidance.
    Reference: Chatterjee, Fadini & Oki, "A spectrum allocation scheme based on
    first-last-exact fit policy for elastic optical networks", JNCA 68, 2016.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    is_large = is_large_request(state, params)
    first_exact, last_exact, _ = exact_fit(state, params)
    first_slots = first_fit(state, params)
    last_slots = last_fit(state, params)
    # Chosen path is the first one with an available slot (as in ksp_ff)
    path_index = jnp.argmax(first_slots < params.link_resources)
    small_slot = jnp.where(
        first_exact[path_index] < params.link_resources,
        first_exact[path_index],
        first_slots[path_index] % params.link_resources,
    )
    large_slot = jnp.where(
        last_exact[path_index] < params.link_resources,
        last_exact[path_index],
        last_slots[path_index] % params.link_resources,
    )
    slot_index = jnp.where(is_large, large_slot, small_slot)
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def ksp_mscl(state: RSAEnvState, params: RSAEnvParams) -> Array:
    """K-Shortest Path, Minimum Slot-continuity Capacity Loss (MSCL).
    Only suitable for RSA/RMSA.

    Path selection follows KSP order (first candidate path with any valid slot,
    exactly as in ksp_ff); the slot on that path is then chosen to minimise the
    slot-continuity capacity loss. In short: every placement destroys some of the
    network's remaining ability to host future contiguous-slot requests - on the
    chosen route and on every route sharing a link with it. MSCL evaluates that
    destruction exactly, one step ahead, for every candidate slot, and picks the
    placement that destroys least. See capacity_loss for the metric definition,
    the closed-form computation, and full references.

    Reference: R. C. Almeida Jr. et al., "Slot assignment strategy to reduce loss
    of capacity of contiguous-slot path requests in flexible grid optical
    networks", Electronics Letters 49(5), 2013, doi:10.1049/el.2012.4247.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    loss, mask = capacity_loss(state, params)
    # Chosen path is the first one with an available slot
    available_paths = jnp.max(mask, axis=1)
    path_index = jnp.argmax(available_paths)
    slot_index = jnp.argmin(loss[path_index])
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def mscl_ksp(state: RSAEnvState, params: RSAEnvParams) -> Array:
    """Minimum Slot-continuity Capacity Loss (MSCL) across K-Shortest Paths.
    Only suitable for RSA/RMSA.

    Jointly selects the (path, slot) pair minimising the slot-continuity capacity
    loss over all k candidate paths and all slots - i.e. routing and spectrum
    assignment are decided together by the same one-step-lookahead metric, rather
    than fixing the path first as ksp_mscl does. Ties break to the shortest path,
    then the lowest slot index. See capacity_loss for the metric definition, the
    closed-form computation, and full references.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    loss, _ = capacity_loss(state, params)
    # argmin over path-major flattened array ties-breaks to shortest path, lowest slot
    action = jnp.argmin(loss.reshape(-1))
    return action


@partial(jax.jit, static_argnums=(1,))
def ff_ksp(state: RSAEnvState, params: RSAEnvParams) -> Array:
    """Get the first available slot from all paths
    Method: Go through action mask and find the first available slot on all paths

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    first_slots = first_fit(state, params)
    # Chosen path is the one with the lowest index of first available slot
    path_index = jnp.argmin(first_slots)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def lf_ksp(state: EnvState, params: RSAEnvParams) -> Array:
    """Get the last available slot from all paths
    Method: Go through action mask and find the last available slot on all paths

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    last_slots = last_fit(state, params)
    # Normalise the GN band-order "no slot" sentinel (params.link_resources) to -1
    # (the standard branch sentinel) so fully-occupied paths lose the argmax below
    last_slots = jnp.where(last_slots >= params.link_resources, -1, last_slots)
    # Chosen path is the one with the highest index of last available slot
    # (treat the band-ordered GN-model no-slot sentinel of link_resources as invalid)
    last_slots = jnp.where(last_slots < params.link_resources, last_slots, -1)
    path_index = jnp.argmax(last_slots)
    slot_index = last_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def ksp_bf(state: EnvState, params: RSAEnvParams) -> Array:
    """Get the first available slot from all k-shortest paths
    Method: Go through action mask and find the first available slot, starting from shortest path

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    best_slots, fitness = best_fit(state, params)
    # Chosen path is the first one with an available slot
    path_index = jnp.argmin(jnp.where(fitness < jnp.inf, 0, 1))
    slot_index = best_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def bf_ksp(state: EnvState, params: RSAEnvParams) -> Array:
    """Get the first available slot from the first k-shortest paths
    Method: Go through action mask and find the first available slot on all paths

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Action
    """
    best_slots, fitness = best_fit(state, params)
    # Chosen path is the one with the best fit
    path_index = jnp.argmin(fitness)
    slot_index = best_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1, 2, 3))
def ksp_mu(state: EnvState, params: RSAEnvParams, unique_lightpaths: bool, relative: bool) -> Array:
    """Get the most-used slot on the shortest available path.
    Method: Go through action mask and find the utilisation of available slots on each path.
    Find the shortest available path and choose the most utilised slot on that path.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
        unique_lightpaths (bool): Whether to consider unique lightpaths
        relative (bool): Whether to return relative utilisation

    Returns:
        Array: Action
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
def mu_ksp(state: EnvState, params: RSAEnvParams, unique_lightpaths: bool, relative: bool) -> Array:
    """Use the most-used available slot on any path.
    The most-used slot is that which has the most unique lightpaths (if unique_lightpaths=True) or active lightpaths.
    Method: Go through action mask and find the usage of available slots, choose available slot that is most utilised.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters
        unique_lightpaths (bool): Whether to consider unique lightpaths
        relative (bool): Whether to return relative utilisation

    Returns:
        Array: Action
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
def kmc_ff(state: EnvState, params: RSAEnvParams) -> Array:
    """K-Minimum Cut. Only suitable for RSA/RMSA.
    Method:
    1. Go through action mask and find the first available slot on all paths.
    2. For each path, allocate the first available slot.
    3. Sum number of new consecutive zero regions (cuts) created by assignment (on each link)
    4. Choose path that creates the fewest cuts.
    """
    mask = get_action_mask(state, params)
    first_slots = first_fit(state, params)
    link_slot_array = jnp.where(state.link_slot_array < 0, 1.0, state.link_slot_array)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)
    block_sizes = jax.vmap(partial(find_block_sizes, differentiable=False), in_axes=(0,))(
        link_slot_array
    )
    block_sizes_mask = jnp.where(
        block_sizes > 0, 1, 0.0
    )  # Binary array showing initial block starts
    block_count = jnp.sum(block_sizes_mask, axis=1)

    def get_cuts_on_path(i, result):
        initial_slot_index = first_slots[i] % params.link_resources
        path = get_paths(params, nodes_sd)[i]
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        num_slots = required_slots(requested_bw, se, params.slot_size, guardband=params.guardband)
        affected_slots_mask = get_affected_slots_mask(initial_slot_index, num_slots, path, params)
        # Make link-slot_array positive
        updated_slots = set_path_links(link_slot_array, affected_slots_mask, 1.0)
        updated_block_sizes = jax.vmap(
            partial(find_block_sizes, differentiable=False), in_axes=(0,)
        )(updated_slots)
        updated_block_sizes_mask = jnp.where(
            updated_block_sizes > 0, 1, 0
        )  # Binary array showing updated block starts
        updated_block_count = jnp.sum(updated_block_sizes_mask, axis=1)
        num_cuts = jax.lax.cond(
            mask[i][initial_slot_index] == 0.0,  # If true, no valid action for path
            lambda x: jnp.full((1,), params.link_resources * params.num_links).astype(
                jnp.float32
            ),  # Return max no. of cuts
            lambda x: jnp.sum(jnp.maximum(updated_block_count - block_count, 0.0)).reshape(
                (1,)
            ),  # Else, return number of cuts
            1.0,
        )
        result = jax.lax.dynamic_update_slice(result, num_cuts, (i,))
        return result

    # Initialise array to hold number of cuts on each path
    path_cuts_array = jnp.full((mask.shape[0],), 0.0)
    path_cuts_array = jax.lax.fori_loop(0, mask.shape[0], get_cuts_on_path, path_cuts_array)
    path_index = jnp.argmin(path_cuts_array)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def kmf_ff(state: RSAEnvState, params: RSAEnvParams) -> Array:
    """K-Minimum Frag-size. Only suitable for RSA/RMSA.
    Method:
    1. Go through action mask and find the first available slot on all paths.
    2. For each path, allocate the first available slot.
    3. Sum number of new consecutive zero regions (cuts) created by assignment (on each link)
    4. Choose path that creates the fewest cuts.
    """
    mask = get_action_mask(state, params)
    first_slots = first_fit(state, params)
    link_slot_array = jnp.where(state.link_slot_array < 0, 1.0, state.link_slot_array)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)
    blocks = jax.vmap(partial(find_block_sizes, differentiable=False), in_axes=(0,))(
        link_slot_array
    )

    def get_frags_on_path(i, result):
        initial_slot_index = first_slots[i] % params.link_resources
        path = get_paths(params, nodes_sd)[i]
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        num_slots = required_slots(requested_bw, se, params.slot_size, guardband=params.guardband)
        affected_slots_mask = get_affected_slots_mask(initial_slot_index, num_slots, path, params)
        # Mask on path links
        block_sizes = jax.vmap(lambda x, y: jnp.where(x > 0, y, 0.0), in_axes=(0, 0))(path, blocks)
        updated_slots = set_path_links(state.link_slot_array, affected_slots_mask, -1)
        updated_block_sizes = jax.vmap(
            partial(find_block_sizes, differentiable=False), in_axes=(0,)
        )(updated_slots)
        # Mask on path links
        updated_block_sizes = jax.vmap(lambda x, y: jnp.where(x > 0, y, 0.0), in_axes=(0, 0))(
            path, updated_block_sizes
        )
        difference = updated_block_sizes - block_sizes
        new_frags = jnp.where(difference != 0, block_sizes + difference, 0.0)
        # Slice new frags up to initial slot index (so as to only consider frags to the left)
        new_frags = jnp.where(
            jnp.arange(params.link_resources) < initial_slot_index, new_frags, 0.0
        )
        new_frag_size = jnp.sum(new_frags)
        num_frags = jax.lax.cond(
            mask[i][initial_slot_index] == 0.0,  # If true, no valid action for path
            lambda x: jnp.full(
                (1,), float(params.link_resources * params.num_links)
            ),  # Return max frag size
            lambda x: new_frag_size.reshape((1,)),
            # Else, return number of cuts
            1.0,
        )
        result = jax.lax.dynamic_update_slice(result, num_frags, (i,))
        return result

    # Initialise array to hold number of cuts on each path
    path_frags_array = jnp.full((mask.shape[0],), 0.0)
    path_frags_array = jax.lax.fori_loop(0, mask.shape[0], get_frags_on_path, path_frags_array)
    path_index = jnp.argmin(path_frags_array)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def kme_ff(state: EnvState, params: RSAEnvParams) -> Array:
    """K-Minimum Entropy. Only suitable for RSA/RMSA.
    Method:
    1. Go through action mask and find the first available slot on all paths.
    2. For each path, allocate the first available slot.
    3. Sum the change in Shannon fragmentation entropy of the path's links
       caused by the assignment (Wright, Parker & Lord, JOCN 2015).
    4. Choose path whose assignment increases entropy the least.
    """
    mask = get_action_mask(state, params)
    first_slots = first_fit(state, params)
    link_slot_array = jnp.where(state.link_slot_array < 0, 1.0, state.link_slot_array)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)
    max_entropy = jnp.sum(jnp.log(params.link_resources)) * params.num_links

    def get_link_entropy(blocks):
        ent = jax.vmap(
            lambda x: jnp.sum(x / params.link_resources * jnp.log(params.link_resources / x)),
            in_axes=0,
        )(blocks)
        return jnp.sum(jnp.where(blocks > 0, ent, 0))

    # Baseline per-link entropy before allocation, so paths are ranked by the
    # entropy change their allocation causes rather than the absolute
    # post-allocation entropy (which conflates the allocation's impact with the
    # path's length and existing fragmentation)
    base_block_sizes = jax.vmap(find_block_sizes, in_axes=(0,))(link_slot_array)
    base_entropy = jax.vmap(get_link_entropy, in_axes=(0,))(base_block_sizes)

    def get_entropy_on_path(i, result):
        initial_slot_index = first_slots[i] % params.link_resources
        path = get_paths(params, nodes_sd)[i]
        se = get_paths_se(params, nodes_sd)[i] if params.consider_modulation_format else 1
        num_slots = required_slots(requested_bw, se, params.slot_size, guardband=params.guardband)
        affected_slots_mask = get_affected_slots_mask(initial_slot_index, num_slots, path, params)
        # Make link-slot_array positive
        updated_slots = set_path_links(link_slot_array, affected_slots_mask, 1.0)
        updated_block_sizes = jax.vmap(
            partial(find_block_sizes, differentiable=False), in_axes=(0,)
        )(updated_slots)
        updated_entropy = jax.vmap(get_link_entropy, in_axes=(0,))(updated_block_sizes)
        new_path_entropy = jnp.sum(jnp.dot(path, updated_entropy - base_entropy)).reshape((1,))
        new_path_entropy = jax.lax.cond(
            mask[i][initial_slot_index] == 0.0,  # If true, no valid action for path
            lambda x: max_entropy.astype(jnp.float32).reshape((1,)),  # Return maximum entropy
            lambda x: new_path_entropy,  # Else, return number of cuts
            1.0,
        )
        result = jax.lax.dynamic_update_slice(result, new_path_entropy, (i,))
        return result

    path_entropy_array = jnp.full((mask.shape[0],), 0.0)
    path_entropy_array = jax.lax.fori_loop(
        0, mask.shape[0], get_entropy_on_path, path_entropy_array
    )
    path_index = jnp.argmin(path_entropy_array)
    slot_index = first_slots[path_index] % params.link_resources
    # Convert indices to action
    action = path_index * params.link_resources + slot_index
    return action


@partial(jax.jit, static_argnums=(1,))
def kca_ff(state: EnvState, params: RSAEnvParams) -> Array:
    """Congestion-aware First Fit. Only suitable for RSA/RMSA.
    Method:

    """
    mask = get_action_mask(state, params)
    # Get index of first available slots for each path
    first_slots = first_fit(state, params)
    # Get nodes
    nodes_sd, _ = read_rsa_request(state.request_array)
    # Initialise array to hold congestion on each path
    path_congestion_array = jnp.full((mask.shape[0],), 0.0)
    link_weights = get_link_weights(state, params)

    def get_path_congestion(i, val):
        # Get links on path
        path = get_paths(params, nodes_sd)[i]
        # Get congestion
        path_link_congestion = jnp.multiply(link_weights, path)
        path_congestion = jnp.sum(path_link_congestion).reshape((1,))
        return jax.lax.dynamic_update_slice(val, path_congestion, (i,))

    path_congestion_array = jax.lax.fori_loop(
        0, mask.shape[0], get_path_congestion, path_congestion_array
    )
    # Penalise infeasible paths (mirroring kmc_ff/kmf_ff/kme_ff) so congestion only
    # ranks paths that can actually fit the request. first_fit returns
    # params.link_resources for a path with no valid block, which maps to slot 0
    # below, where the mask is guaranteed 0 for such a path.
    slot_indices = first_slots % params.link_resources
    first_slot_valid = jnp.take_along_axis(mask, slot_indices[:, None], axis=1).squeeze(1)
    path_congestion_array = jnp.where(first_slot_valid > 0, path_congestion_array, jnp.inf)
    path_index = jnp.argmin(path_congestion_array)
    slot_index = first_slots[path_index] % params.link_resources
    action = path_index * params.link_resources + slot_index
    return action


def get_link_weights(state: EnvState, params: RSAEnvParams):
    """Get link weights based on occupancy for use in congestion-aware routing heuristics.

    Args:
        state (EnvState): Environment state
        params (EnvParams): Environment parameters

    Returns:
        Array: Link weights
    """
    if isinstance(params, RWALightpathReuseEnvParams):
        rwalr_state = cast(RWALightpathReuseEnvState, state)
        initial_path_capacity = init_path_capacity_array(
            params.link_length_array.val, params.path_link_array.val, scale_factor=1.0
        )
        initial_path_capacity = jnp.squeeze(
            jax.vmap(lambda x: initial_path_capacity[x])(rwalr_state.path_index_array)
        )
        utilisation = (
            jnp.where(
                initial_path_capacity - rwalr_state.link_capacity_array < 0,
                0,
                initial_path_capacity - rwalr_state.link_capacity_array,
            )
            / initial_path_capacity
        )
        link_occupancy = jnp.sum(utilisation, axis=1)
    else:
        link_occupancy = jnp.count_nonzero(state.link_slot_array, axis=1)
    link_weights = jnp.multiply(
        params.link_length_array.val.T, (1 / (1 - link_occupancy / (params.link_resources + 1)))
    )[0]
    return link_weights


def get_action_mask(state: EnvState, params: RSAEnvParams) -> Array:
    if isinstance(params, RWALightpathReuseEnvParams):
        _, full_mask = jit_profiler.call(
            params.profile, mask_slots_rwalr, state, params, state.request_array
        )
    elif isinstance(params, RMSAGNModelEnvParams):
        updated_state = jit_profiler.call(
            params.profile, mask_slots_rmsa_gn_model, state, params, state.request_array
        )
        full_mask = jnp.where(updated_state.mod_format_mask >= 0, 1.0, 0.0)
    else:
        _, full_mask = jit_profiler.call(params.profile, mask_slots, state, params)
    mask = jnp.reshape(full_mask, (params.k_paths, -1))
    return mask


def best_fit(state: EnvState, params: RSAEnvParams) -> Tuple[Array, Array]:
    """Best-Fit Spectrum Allocation. Returns the best fit slot for each path."""
    mask = get_action_mask(state, params)
    link_slot_array = jnp.where(state.link_slot_array < 0, 1.0, state.link_slot_array)
    nodes_sd, requested_bw = read_rsa_request(state.request_array)

    # We need to define a wrapper function in order to vmap with named arguments
    def _find_block_sizes(arr, starts_only=False, reverse=True):
        return jax.vmap(partial(find_block_sizes, differentiable=False), in_axes=(0, None, None))(
            arr, starts_only, reverse
        )

    block_sizes_right = _find_block_sizes(link_slot_array, starts_only=False, reverse=False)
    block_sizes_left = _find_block_sizes(link_slot_array, starts_only=False, reverse=True)
    block_sizes = jnp.maximum((block_sizes_left + block_sizes_right) - 1, 0)
    paths = get_paths(params, nodes_sd)
    se = (
        get_paths_se(params, nodes_sd)
        if params.consider_modulation_format
        else jnp.ones((params.k_paths,))
    )
    num_slots = jax.vmap(required_slots, in_axes=(None, 0, None, None))(
        requested_bw, se, params.slot_size, params.guardband
    )

    # Quantify how well the request fits within a free spectral block
    def get_bf_on_path(path, blocks, req_slots):
        fits = jax.vmap(lambda x: x - req_slots, in_axes=0)(blocks)
        fits = jnp.where(fits >= 0, fits, params.link_resources)
        path_fit = jnp.dot(path, fits) / jnp.sum(path)
        return path_fit

    fits_block = jax.vmap(lambda x, y, z: get_bf_on_path(x, y, z), in_axes=(0, None, 0))(
        paths, block_sizes, num_slots
    )

    # Quantity much of a gap there is between the assigned slots and the next occupied slots on the left
    def get_bf_on_path_left(path, blocks, req_slots):
        fits = jax.vmap(lambda x: x - req_slots, in_axes=0)(blocks)
        fits = jnp.where(fits >= 0, fits, params.link_resources)
        fits_shift = jax.lax.dynamic_slice(fits, (0, 1), (fits.shape[0], fits.shape[1] - 1))
        fits_shift = jnp.concatenate(
            (jnp.full((fits.shape[0], 1), params.link_resources), fits_shift), axis=1
        )
        fits = fits + 1 / jnp.maximum(fits_shift, 1)
        path_fit = jnp.dot(path, fits) / jnp.sum(path)
        return path_fit

    fits_left = jax.vmap(lambda x, y, z: get_bf_on_path_left(x, y, z), in_axes=(0, None, 0))(
        paths, block_sizes_left, num_slots
    )

    # Quantity much of a gap there is between the assigned slots and the next occupied slots on the right
    def get_bf_on_path_right(path, blocks, req_slots):
        fits = jax.vmap(lambda x: x - req_slots, in_axes=0)(blocks)
        fits = jnp.where(fits >= 0, fits, params.link_resources)
        fits_shift = jax.lax.dynamic_slice(fits, (0, 0), (fits.shape[0], fits.shape[1] - 1))
        fits_shift = jnp.concatenate(
            (fits_shift, jnp.full((fits.shape[0], 1), params.link_resources)), axis=1
        )
        fits = fits + 1 / jnp.maximum(fits_shift, 1)
        path_fit = jnp.dot(path, fits) / jnp.sum(path)
        return path_fit

    fits_right = jax.vmap(lambda x, y, z: get_bf_on_path_right(x, y, z), in_axes=(0, None, 0))(
        paths, block_sizes_right, num_slots
    )

    # Sum the contribution to the overall quality of fit, and scale down the left/right contributions
    fits = jnp.sum(
        jnp.stack(
            (fits_block, fits_left / params.link_resources, fits_right / params.link_resources),
            axis=0,
        ),
        axis=0,
    )
    # Mask out occupied lightpaths (in case the quality of fit on some links is good enough to be considered, even if the overall path is invalid)
    fits = jnp.where(mask == 0, jnp.inf, fits)
    best_slots = jnp.argmin(fits, axis=1)
    best_fits = jnp.min(fits, axis=1)
    return best_slots, best_fits


def first_fit(state: EnvState, params: RSAEnvParams) -> Array:
    """First-Fit Spectrum Allocation. Returns the first fit slot for each path.

    When band_slot_order_ff is set (GN model envs with --band_preference),
    slots are searched in band preference order rather than raw index order.
    """
    mask = get_action_mask(state, params)
    if isinstance(params, GNModelEnvParams) and len(params.band_slot_order_ff.val) > 0:
        order = params.band_slot_order_ff.val
        reordered = mask[:, order]
        reordered = jnp.concatenate((reordered, jnp.full((reordered.shape[0], 1), 1)), axis=1)
        idx = jnp.argmax(reordered, axis=1)
        safe_idx = jnp.clip(idx, 0, params.link_resources - 1)
        first_slots = jnp.where(idx < params.link_resources, order[safe_idx], params.link_resources)
    else:
        # Add a column of ones to make sure occupied paths have non-zero index in "first_slots"
        mask = jnp.concatenate((mask, jnp.full((mask.shape[0], 1), 1)), axis=1)
        first_slots = jnp.argmax(mask, axis=1)
    return first_slots


def last_fit(state: EnvState, params: RSAEnvParams) -> Array:
    """Last-Fit Spectrum Allocation. Returns the last fit slot for each path.

    When band_slot_order_lf is set (GN model envs with --band_preference),
    slots are searched in band preference order (descending within each band).
    """
    mask = get_action_mask(state, params)
    if isinstance(params, GNModelEnvParams) and len(params.band_slot_order_lf.val) > 0:
        order = params.band_slot_order_lf.val
        reordered = mask[:, order]
        reordered = jnp.concatenate((reordered, jnp.full((reordered.shape[0], 1), 1)), axis=1)
        idx = jnp.argmax(reordered, axis=1)
        safe_idx = jnp.clip(idx, 0, params.link_resources - 1)
        last_slots = jnp.where(idx < params.link_resources, order[safe_idx], params.link_resources)
    else:
        # Add a column of ones to make sure occupied paths have non-zero index in "last_slots"
        mask = jnp.concatenate((jnp.full((mask.shape[0], 1), 1), mask), axis=1)
        last_slots = jnp.argmax(mask[:, ::-1], axis=1)
        last_slots = params.link_resources - last_slots - 1
    return last_slots


def is_large_request(state: EnvState, params: RSAEnvParams) -> Array:
    """Classify the current request as large (True) or small (False) by comparing its
    datarate to the midpoint of the requestable datarate range. Used by the
    first-last-fit family to segregate request size classes at opposite spectrum ends."""
    _, requested_bw = read_rsa_request(state.request_array)
    threshold = (jnp.min(params.values_bw.val) + jnp.max(params.values_bw.val)) / 2
    return requested_bw > threshold


def get_request_num_slots(state: EnvState, params: RSAEnvParams) -> Array:
    """Required slots (incl. guardband) for the current request on each of the k paths."""
    nodes_sd, requested_bw = read_rsa_request(state.request_array)
    se = (
        get_paths_se(params, nodes_sd)
        if params.consider_modulation_format
        else jnp.ones((params.k_paths,))
    )
    num_slots = jax.vmap(required_slots, in_axes=(None, 0, None, None))(
        requested_bw, se, params.slot_size, params.guardband
    )
    return num_slots.astype(jnp.int32)


def get_path_free_arrays(link_slot_array: Array, path_links: Array) -> Array:
    """Aggregate slot occupancy over the links of each path.

    Args:
        link_slot_array: (num_links, num_slots) array; nonzero = occupied
        path_links: (num_paths, num_links) binary path-link incidence

    Returns:
        Array: bool (num_paths, num_slots); True where the slot is free on every link
    """
    occupied = jnp.where(link_slot_array != 0, 1.0, 0.0)
    path_occ = jnp.dot(path_links.astype(jnp.float32), occupied)
    return path_occ == 0


def get_run_lengths_and_bounds(free: Array) -> Tuple[Array, Array, Array]:
    """For each slot position, the length of the free run starting there and the
    [start, end) bounds of the free run containing it.

    Args:
        free: bool (..., num_slots)

    Returns:
        Tuple: (run_len, run_start, run_end), each (..., num_slots) int32.
        For occupied positions run_len = 0 and run_start = run_end = position.
    """
    num_slots = free.shape[-1]
    axis = free.ndim - 1  # associative_scan(reverse=True) requires a non-negative axis
    idx = jnp.arange(num_slots, dtype=jnp.int32)
    prev_occ = jax.lax.associative_scan(jnp.maximum, jnp.where(free, -1, idx), axis=axis)
    next_occ = jax.lax.associative_scan(
        jnp.minimum, jnp.where(free, num_slots, idx), axis=axis, reverse=True
    )
    run_start = jnp.where(free, prev_occ + 1, idx)
    run_end = jnp.where(free, next_occ, idx)
    run_len = jnp.where(free, next_occ - idx, 0)
    return run_len, run_start, run_end


def exact_fit(state: EnvState, params: RSAEnvParams) -> Tuple[Array, Array, Array]:
    """Exact-Fit Spectrum Allocation. For each path, find free blocks whose size
    exactly equals the required slots for the request.

    Returns:
        Tuple: (first_exact, last_exact, mask). first_exact/last_exact hold the
        lowest/highest exactly-fitting block start per path, or link_resources if
        no exact fit exists on that path.
    """
    mask = get_action_mask(state, params)
    nodes_sd, _ = read_rsa_request(state.request_array)
    num_slots = get_request_num_slots(state, params)
    paths = get_paths(params, nodes_sd)
    free = get_path_free_arrays(state.link_slot_array, paths)
    run_len, run_start, _ = get_run_lengths_and_bounds(free)
    slot_indices = jnp.arange(params.link_resources, dtype=jnp.int32)
    is_block_start = free & (run_start == slot_indices[None, :])
    exact = is_block_start & (run_len == num_slots[:, None]) & (mask == 1)
    # First exact-fit block start per path (link_resources if none)
    exact_pad = jnp.concatenate((exact, jnp.full((exact.shape[0], 1), True)), axis=1)
    first_exact = jnp.argmax(exact_pad, axis=1).astype(jnp.int32)
    # Last exact-fit block start per path (link_resources if none)
    last_exact = jnp.where(
        jnp.any(exact, axis=1),
        params.link_resources - jnp.argmax(exact[:, ::-1], axis=1) - 1,
        params.link_resources,
    ).astype(jnp.int32)
    return first_exact, last_exact, mask


def capacity_loss(state: EnvState, params: RSAEnvParams) -> Tuple[Array, Array]:
    """Slot-continuity capacity loss (the MSCL metric) of every candidate
    (path, slot) assignment for the current request.

    Capacity
        A future request needing w contiguous slots fits on path p at starting
        position i iff slots i..i+w-1 are free on every link of p. Writing
        run_p[i] for the length of the contiguous free run starting at slot i of
        p's link-aggregated spectrum (0 if slot i is occupied on any link), the
        number of feasible placements for size w is N_p(w) = #{i : run_p[i] >= w},
        and the slot-continuity capacity of p is the total over all demand sizes:

            C_p = sum_{w=1..W} N_p(w) = sum_i min(run_p[i], W)

        W is the largest possible request in slots (from the max datarate, the
        lowest spectral efficiency and the slot size, plus guardband); demand
        sizes are weighted uniformly, as in the original formulation.

    Loss
        Serving the current request on route r from slot s occupies slots
        [s, e) on every link of r (e = s + required slots incl. guardband). That
        reduces C_q for r itself and for every route q sharing >= 1 link with r;
        spectrum elsewhere is untouched. The MSCL loss of candidate (r, s) is

            loss(r, s) = sum_{q affected} [ C_q before - C_q after ]

        and the MSCL heuristics choose the candidate minimising it: a one-step
        lookahead that consumes dead-end fragments and spares large aligned voids
        on heavily-shared links, rather than packing blindly like first-fit.

    Closed form
        Instead of materialising the updated spectrum for each of the k x S
        candidates, the per-route loss decomposes exactly into two terms computed
        from run-length arrays and a single prefix sum:

        - inside the block: each free position i in [s, e) drops from
          min(run[i], W) to 0; summed as a difference of the prefix sum of
          min(run, W) at e and s.
        - left of the block: positions i in the free run containing s (run start
          a, run end b) are truncated from run length b - i to s - i, losing
          min(b - i, W) - min(s - i, W); summing over d = s - i = 1..D with
          D = s - a and gap g = b - s gives sum_min(D, g) - sum_min(D, 0), where
          sum_min(D, off) = sum_{d=1..D} min(d + off, W) has the closed form
          t*off + t(t+1)/2 + (D-t)*W with t = clip(W - off, 0, D).

        Positions at or beyond e keep their runs (a free run starting there
        cannot reach back across the newly occupied block), and runs not touching
        [s, e) are unaffected. If s is already occupied for a route then a = b = s
        and both terms vanish for it, which is exactly right. This closed form is
        verified against a brute-force before/after recount in
        heuristics_test.py::CapacityLossBruteforceTest.

    Interfering route set
        The affected routes are the shortest path of every node pair that shares
        a link with the candidate route (the single-route-per-pair route set of
        the original 2013 formulation), plus the candidate route itself - counted
        once, since when the candidate is its pair's k=0 path it already is that
        pair's shortest. Multi-route interfering sets (every stored path of every
        pair) are a published extension (Santos et al., SBrT 2021) and would cost
        k times more here.

    References
        R. C. Almeida Jr., A. F. dos Santos, K. D. R. Assis, H. Waldman &
        J. F. Martins-Filho, "Slot assignment strategy to reduce loss of capacity
        of contiguous-slot path requests in flexible grid optical networks",
        Electronics Letters 49(5), pp. 358-360, 2013. doi:10.1049/el.2012.4247
        X. Zhang & C. Qiao, "Wavelength assignment for dynamic traffic in
        multi-fiber WDM networks", ICCCN 1998 - the relative-capacity-loss RWA
        metric that MSCL generalises to contiguous-spectrum RSA/RMSA.
        M. L. Santos, R. C. Almeida Jr. & D. R. B. Araujo, "Multi-route spectrum
        assignment by slot-continuity capacity loss in elastic optical networks",
        SBrT 2021 - multi-route interfering sets.

    Returns:
        Tuple: (loss, mask). loss is (k_paths, link_resources) float32 with jnp.inf
        at invalid actions; mask is the (k_paths, link_resources) action mask.
    """
    mask = get_action_mask(state, params)
    nodes_sd, _ = read_rsa_request(state.request_array)
    num_slots = get_request_num_slots(state, params)
    num_resources = params.link_resources

    # Maximum future demand size (in slots, incl. guardband) to account capacity for.
    # params fields are static under jit so this is computed at trace time.
    values_bw = np.asarray(params.values_bw.val)
    min_se = (
        float(np.min(np.asarray(params.path_se_array.val)))
        if params.consider_modulation_format
        else 1.0
    )
    w_cap = int(np.ceil(float(np.max(values_bw)) / (min_se * float(params.slot_size))))
    w_cap = max(w_cap + int(params.guardband), 1)

    # Interfering route set: the shortest path of every node pair (rows of the
    # path-link array are pair-major with k consecutive rows per pair)
    shortest_paths = params.path_link_array.val[:: params.k_paths]
    if params.pack_path_bits:
        shortest_paths = jnp.unpackbits(shortest_paths, axis=1)[:, : params.num_links]
    shortest_paths = jnp.asarray(shortest_paths, dtype=jnp.float32)
    cand_paths = jnp.asarray(get_paths(params, nodes_sd), dtype=jnp.float32)

    def prep(paths):
        free = get_path_free_arrays(state.link_slot_array, paths)
        run_len, run_start, run_end = get_run_lengths_and_bounds(free)
        # Capacity contribution of each start position, capped at the max demand size
        capped = jnp.minimum(run_len, w_cap)
        cum_cap = jnp.concatenate(
            (jnp.zeros((capped.shape[0], 1), dtype=jnp.int32), jnp.cumsum(capped, axis=-1)),
            axis=-1,
        )
        return run_start, run_end, cum_cap

    slot_indices = jnp.arange(num_resources, dtype=jnp.int32)
    # Block end (exclusive) of each candidate assignment; clipped for safe gathering
    # (assignments that overrun the spectrum are already invalid in the mask)
    ends = jnp.minimum(slot_indices[None, :] + num_slots[:, None], num_resources)

    def sum_min(d, offset):
        # sum_{i=1..d} min(i + offset, w_cap), for the left-of-block truncation term
        t = jnp.clip(w_cap - offset, 0, d)
        return t * offset + t * (t + 1) // 2 + (d - t) * w_cap

    def truncation_loss(run_start, run_end):
        # Capacity lost by positions left of the assignment whose free run is cut at s
        d = slot_indices[None, :] - run_start
        gap = run_end - slot_indices[None, :]
        return sum_min(d, gap) - sum_min(d, 0)

    short_start, short_end, short_cum = prep(shortest_paths)
    cand_start, cand_end, cand_cum = prep(cand_paths)

    # Loss on interfering (shortest-per-pair) routes: truncation left of s plus the
    # capacity of positions inside [s, e) that become occupied. Since the block end
    # e = s + w_r depends on the candidate path only through its width w_r, the sum
    # over interfering routes commutes with the gather at e: everything reduces to
    # matmuls over the pair dimension plus one shifted gather of the aggregate,
    # avoiding a (num_pairs, k, S) intermediate. Aggregates can reach ~1e7 on
    # 100+-node topologies, so float32 rounding of ~1 capacity unit is possible in
    # near-tied candidates; the ranking is unaffected for practical purposes (the
    # brute-force equality test runs on small topologies where sums stay exact).
    short_trunc = truncation_loss(short_start, short_end)  # (num_pairs, S)
    shares = (jnp.dot(cand_paths, shortest_paths.T) > 0).astype(jnp.float32)  # (k, P)
    base = jnp.dot(
        shares, (short_trunc - short_cum[:, :num_resources]).astype(jnp.float32)
    )  # (k, S)
    cum_agg = jnp.dot(shares, short_cum.astype(jnp.float32))  # (k, S + 1)
    loss = base + jnp.take_along_axis(cum_agg, ends, axis=1)

    # Same loss terms on the candidate route itself. The candidate is already in
    # the interfering set iff it is its pair's shortest path (k index 0);
    # otherwise add its own loss explicitly
    cand_trunc = truncation_loss(cand_start, cand_end)  # (k, S)
    cand_inside = jnp.take_along_axis(cand_cum, ends, axis=1) - cand_cum[:, :num_resources]
    cand_delta = cand_trunc + cand_inside
    not_shortest = (jnp.arange(params.k_paths) != 0).astype(jnp.float32)[:, None]
    loss = loss + cand_delta.astype(jnp.float32) * not_shortest
    loss = jnp.where(mask == 0, jnp.inf, loss)
    return loss, mask


@partial(jax.jit, static_argnums=(1, 2, 3))
def most_used(state: EnvState, params: RSAEnvParams, unique_lightpaths, relative) -> Array:
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
        Array: Most used slots (array length = link_resources)
    """
    if isinstance(params, RWALightpathReuseEnvParams) and unique_lightpaths:
        rwalr_state = cast(RWALightpathReuseEnvState, state)
        most_used_slots = jnp.count_nonzero(rwalr_state.path_index_array + 1, axis=0) + 1
    elif isinstance(params, RWALightpathReuseEnvParams) and not unique_lightpaths:
        rwalr_state = cast(RWALightpathReuseEnvState, state)
        # Get initial path capacity
        initial_path_capacity = init_path_capacity_array(
            params.link_length_array.val, params.path_link_array.val, scale_factor=1.0
        )
        initial_path_capacity = jnp.squeeze(
            jax.vmap(lambda x: initial_path_capacity[x])(rwalr_state.path_index_array)
        )
        utilisation = jnp.where(
            initial_path_capacity - rwalr_state.link_capacity_array < 0,
            0,
            initial_path_capacity - rwalr_state.link_capacity_array,
        )
        if relative:
            utilisation = utilisation / initial_path_capacity
        # Get most used slots by summing the utilisation along the slots
        most_used_slots = jnp.sum(utilisation, axis=0) + 1
    else:
        most_used_slots = jnp.count_nonzero(state.link_slot_array, axis=0) + 1
    return most_used_slots
