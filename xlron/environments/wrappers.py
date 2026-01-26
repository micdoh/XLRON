import timeit
from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper
from jax import Array, tree_util

from xlron import dtype_config
from xlron.environments.dataclasses import (
    LogEnvState,
    RSAEnvParams,
    RSAEnvState,
    RSAGNModelEnvParams,
)
from xlron.environments.env_funcs import (
    get_path_indices,
    get_snr_for_path,
    process_path_action,
    read_rsa_request,
)


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths.
    Modified from: https://github.com/RobertTLange/gymnax/blob/master/gymnax/wrappers/purerl.py
    """

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        params: Optional[RSAEnvParams] = None,
        state: Optional[RSAEnvState] = None,
    ) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key, params, state)
        log_state = LogEnvState(
            env_state=env_state,
            lengths=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            returns=jnp.array(0, dtype=dtype_config.REWARD_DTYPE),
            cum_returns=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            accepted_services=jnp.array(0, dtype=dtype_config.LARGE_INT_DTYPE),
            accepted_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            total_bitrate=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            utilisation=jnp.array(0, dtype=dtype_config.LARGE_FLOAT_DTYPE),
            terminal=jnp.array(False),
            truncated=jnp.array(False),
        )
        return obs, log_state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        log_state: LogEnvState,
        action: Union[int, float] | Tuple[Union[int, float], Union[int, float]],
        params: RSAEnvParams,
    ) -> Tuple[Array, LogEnvState, float, bool, bool, dict]:
        obs, env_state, reward, terminal, truncated, info = self._env.step(
            key, log_state.env_state, action, params
        )
        done = jnp.logical_or(terminal, truncated)
        log_state = LogEnvState(
            env_state=env_state,
            lengths=log_state.lengths * (1 - done) + 1,
            returns=reward,
            cum_returns=log_state.cum_returns * (1 - done) + reward,
            accepted_services=env_state.accepted_services,
            accepted_bitrate=env_state.accepted_bitrate,
            total_bitrate=env_state.total_bitrate,
            utilisation=jnp.count_nonzero(env_state.link_slot_array) / env_state.link_slot_array.size,
            terminal=terminal,
            truncated=truncated,
        )
        info["lengths"] = log_state.lengths
        info["returns"] = log_state.returns
        info["cum_returns"] = log_state.cum_returns
        info["accepted_services"] = log_state.accepted_services
        info["accepted_bitrate"] = log_state.accepted_bitrate
        info["total_bitrate"] = log_state.total_bitrate
        info["utilisation"] = log_state.utilisation
        info["terminal"] = terminal
        info["truncated"] = truncated
        # First check if we're dealing with RSAGNModelEnvParams
        is_gn_params = isinstance(params, RSAGNModelEnvParams)

        # For RSA params, unpack the action
        if is_gn_params:
            action, power_action = action
            info["launch_power"] = power_action

        # Now, if we need to log actions OR we have RSA params, compute the common fields
        if is_gn_params or params.log_actions:
            # Compute common fields
            nodes_sd, dr_request = read_rsa_request(log_state.env_state.request_array)
            source, dest = nodes_sd
            i = get_path_indices(source, dest, params.k_paths, params.num_nodes, directed=params.directed_graph).astype(
                jnp.int32)
            path_index, slot_index = process_path_action(log_state.env_state, params, action)

            # Set common info
            info["path_index"] = i + path_index
            info["slot_index"] = slot_index
            info["source"] = source
            info["dest"] = dest
            info["data_rate"] = dr_request[0]

            # RSA-specific throughput info
            if is_gn_params:
                info["throughput"] = env_state.throughput

            # Logging-specific info
            if params.log_actions:
                # RSA-specific logging
                if is_gn_params:
                    path = params.path_link_array.val[path_index.astype(jnp.int32)]
                    info["path_snr"] = get_snr_for_path(path, env_state.link_snr_array, params)[
                        slot_index.astype(jnp.int32)]
                # Common logging fields
                info["arrival_time"] = env_state.current_time[0]
                info["departure_time"] = env_state.current_time[0] + env_state.holding_time[0]
        return obs, log_state, reward, terminal, truncated, info

    def _tree_flatten(self) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        children = ()  # arrays / dynamic values
        aux_data = (self._env,)  # static values, e.g. env params
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data: Tuple[Any, ...], children: Tuple[Any, ...]) -> "LogWrapper":
        return cls(*children, *aux_data)


class TimeIt:
    """Context manager for timing execution of code blocks."""

    def __init__(self, tag, frames=None):
        self.tag = tag
        self.frames = frames

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (': Elapsed time=%.2fs' % self.elapsed_secs)
        if self.frames:
            msg += ', FPS=%.2e' % (self.frames / self.elapsed_secs)
        print(msg)


tree_util.register_pytree_node(LogWrapper,
                               LogWrapper._tree_flatten,
                               LogWrapper._tree_unflatten)
