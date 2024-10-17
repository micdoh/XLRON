from itertools import combinations, islice
from functools import partial
from typing import Sequence, Union, Optional, Tuple
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper
from absl import flags
import math
import pathlib
import itertools
import networkx as nx
import jax.numpy as jnp
import chex
import jax
import timeit
import json
import numpy as np
import jraph
from flax import struct
from jax._src import core
from jax._src import dtypes
from jax._src import prng
from jax._src.typing import Array, ArrayLike, DTypeLike
from typing import Generic, TypeVar
from collections import defaultdict
from jax import tree_util
from xlron.environments.dataclasses import *
from xlron.environments.env_funcs import get_path_indices, process_path_action, read_rsa_request, get_path_slots, get_paths


Shape = Sequence[int]
T = TypeVar('T')      # Declare type variable


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths.
    Modified from: https://github.com/RobertTLange/gymnax/blob/master/gymnax/wrappers/purerl.py
    """

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0, 0, 0, False)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        state = LogEnvState(
            env_state=env_state,
            lengths=state.lengths * (1 - done) + 1,
            returns=reward,
            cum_returns=state.cum_returns * (1 - done) + reward,
            accepted_services=env_state.accepted_services,
            accepted_bitrate=env_state.accepted_bitrate,
            total_bitrate=env_state.total_bitrate,
            utilisation=jnp.count_nonzero(env_state.link_slot_array) / env_state.link_slot_array.size,
            done=done,
        )
        info["lengths"] = state.lengths
        info["returns"] = state.returns
        info["cum_returns"] = state.cum_returns
        info["accepted_services"] = state.accepted_services
        info["accepted_bitrate"] = state.accepted_bitrate
        info["total_bitrate"] = state.total_bitrate
        info["utilisation"] = state.utilisation
        # Log path length if required, to calculate average path length and number of hops
        if params.log_actions:
            nodes_sd, dr_request = read_rsa_request(state.env_state.request_array)
            source, dest = nodes_sd
            i = get_path_indices(source, dest, params.k_paths, params.num_nodes, directed=params.directed_graph).astype(
                jnp.int32)
            path_index, slot_index = process_path_action(state.env_state, params, action)
            info["path_index"] = i + path_index
            info["slot_index"] = slot_index
            info["source"] = source
            info["dest"] = dest
            info["data_rate"] = dr_request[0]
        info["done"] = done
        return obs, state, reward, done, info

    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {"env": self._env}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class HashableArrayWrapper(Generic[T]):
    """Wrapper for making arrays hashable.
    In order to access pre-computed data, such as shortest paths between node-pairs or the constituent links of a path,
    within a jitted function, we need to make the arrays containing this data hashable. By defining this wrapper, we can
    define a __hash__ method that returns a hash of the array's bytes, thus making the array hashable.
    From: https://github.com/google/jax/issues/4572#issuecomment-709677518
    """
    def __init__(self, val: T):
        self.val = val

    def __getattribute__(self, prop):
        if prop == 'val' or prop == "__hash__" or prop == "__eq__":
            return super(HashableArrayWrapper, self).__getattribute__(prop)
        return getattr(self.val, prop)

    def __getitem__(self, key):
        return self.val[key]

    def __setitem__(self, key, val):
        self.val[key] = val

    def __hash__(self):
        return hash(self.val.tobytes())

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return self.__hash__() == other.__hash__()

        f = getattr(self.val, "__eq__")
        return f(self, other)


class RolloutWrapper:
    """Wrapper to define batch evaluation for generation parameters. Used for genetic algorithm.
    From: https://github.com/RobertTLange/gymnax/
    """
    def __init__(
        self,
        model_forward=None,
        env: environment.Environment = None,
        num_env_steps: Optional[int] = None,
        env_params: EnvParams = None,
    ):
        """Wrapper to define batch evaluation for generation parameters."""
        self.env = env
        # Define the RL environment & network forward function
        self.env_params = env_params
        self.model_forward = model_forward

        if num_env_steps is None:
            self.num_env_steps = self.env_params.max_requests
        else:
            self.num_env_steps = num_env_steps

    @partial(jax.jit, static_argnums=(0, 2))
    def population_rollout(self, rng_eval, policy_params):
        """Reshape parameter vector and evaluate the generation."""
        # Evaluate population of nets on gymnax task - vmap over rng & params
        pop_rollout = jax.vmap(self.batch_rollout, in_axes=(None, 0))
        return pop_rollout(rng_eval, policy_params)

    @partial(jax.jit, static_argnums=(0, 2))
    def batch_rollout(self, rng_eval, policy_params):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None))
        return batch_rollout(rng_eval, policy_params)

    @partial(jax.jit, static_argnums=(0, 2))
    def single_rollout(self, rng_input, policy_params):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            if self.model_forward is not None:
                action = self.model_forward(policy_params, obs, rng_net)
            else:
                action = self.env.action_space(self.env_params).sample(rng_net)
            next_obs, next_state, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                policy_params,
                rng,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_obs, done]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                policy_params,
                rng_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.num_env_steps,
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        return obs, action, reward, next_obs, done, cum_return

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape


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
