import wandb
import sys
import os
import time
import pathlib
import matplotlib.pyplot as plt
from absl import app, flags
import xlron.train.parameter_flags
import numpy as np
import jax
from xlron.environments.env_funcs import *
from xlron.environments import isrs_gn_model
from xlron.environments.rsa import *

FLAGS = flags.FLAGS


# TODO - use make_rsa_env to pull out the flags and stick them in params

def calculate_maximum_reach_per_mod_format(params):
    """
    Calculate maximum reach (in number of spans) per modulation format.
    """
    snr_func = lambda x: isrs_gn_model.to_db(isrs_gn_model.get_snr(ch_power_W_i=10 ** (x.reshape((420, 1)) / 10) * 0.001)[0])
    objective = lambda x: -jnp.sum(jnp.log2(1+isrs_gn_model.get_snr(ch_power_W_i=10 ** (x.reshape((420, 1)) / 10) * 0.001)[0]))
    # Print out range of uniform launch powers and resulting SNRs
    for i in range(-20, 2, 1):
        params = jnp.full((420,), i)
        print(f"initial snr {i}: {snr_func(params)}")
        print(f"initial_info {i}: {objective(params)}")
    params = jnp.arange(-8.4, -4.2, 0.01)  # Create a slope so that lower frequencies have lower power initially
    # (to compensate for ISRS shifting power to lower freqs)
    key = jax.random.PRNGKey(758493)  # Random seed is explicit in JAX
    params = jax.random.uniform(key, shape=(420,)) + params
    print(f"initial_info: {objective(params)}")
    print(f"initial snr: {snr_func(params)}")
    # Optimise with jaxopt
    lbfgsb = jaxopt.LBFGSB(fun=objective, stepsize=0.001, maxiter=100000)
    lower_bounds = jnp.full((420,), -20.)
    upper_bounds = jnp.full((420,), 2.)
    bounds = (lower_bounds, upper_bounds)
    sol, info = lbfgsb.run(params, bounds=bounds)
    grad_fun = jax.grad(objective)
    grad = grad_fun(sol)
    print(f"grad: {grad}")
    print(f"final_info: {objective(sol)}")
    print(f"final snr: {snr_func(sol)}")
    return sol, info


def main(argv):
    # Set up config options
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)

    config_dict = {k: v.value for k, v in FLAGS.__flags.items()}
    env, env_params = make_rsa_env(config_dict)

    # TODO - alternatively, define the objective as get_link_snr_array, then get_snr_for_path(path, link_snr_array, params)

    # Get each path
    for path_id in range(len(env_params.path_link_array)):
        path_links = env_params.path_link_array[path_id]
        print(f"Path {path_id}: {path_links}")
        # Get number of links and set number of roadms
        num_links = jnp.sum(path_links)
        num_roadms = num_links
        env_params = env_params.replace(num_roadms=num_roadms)
        #config_dict = config_dict.replace(num_roadms=num_roadms)
        #env, env_params = make_rsa_env(config_dict)
        # Get length array
        length_array = env_params.length_array
        indices = jnp.where(path_links == 1)[0]
        link_lengths = jnp.take(env_params.link_length_array, indices, axis=0)
        # Remove zeros and combine length array
        link_lengths = jnp.take(link_lengths, jnp.where(link_lengths.reshape((-1,)) != 0))
        path_length = jnp.sum(link_lengths)
        # For each modulation format, set channel centres, bandwidths, mod. formats
        for mod_format_row in env_params.modulations_array:
            # Get spectral efficiency
            spectral_efficiency = mod_format_row[0]
            # Get required slots
            # TODO - allow 400 Gbps to be investigated also
            required_slots = jnp.ceil(100 / (spectral_efficiency*env_params.slot_width))
            # Get channel start indices
            start_indices = jnp.where(jnp.arange(env_params.num_channels) % required_slots == 0, 1, 0)
            # Get channel centres
            channel_centres = jax.vmap(lambda x: jax.lax.cond(get_centre_frequency(start_indices, ))



    sol, info = calculate_maximum_reach_per_mod_format(env_params)
    print(f"sol: {sol}, info: {info}")

    # TODO - save data to CSV and name from EXPERIMENT_NAME flag
    # Add path_id, path length, mean span length, request bandwidth, modulation format,



if __name__ == "__main__":
  app.run(main)
