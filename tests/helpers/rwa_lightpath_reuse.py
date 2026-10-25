"""Cached lightpath-reuse fixtures shared across source-module tests."""

import jax

from xlron.environments.make_env import make

_rwa_lr_cache = {}


def _rwa_lr_cached_setup(cache_key, settings):
    """Return (key, env, obs, state, params) using cached env/params."""
    key = jax.random.PRNGKey(0)
    if cache_key not in _rwa_lr_cache:
        env, params = make(settings, log_wrapper=False)
        _rwa_lr_cache[cache_key] = (env, params)
    env, params = _rwa_lr_cache[cache_key]
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_lightpath_reuse_4_nsfnet_test_setup(**kwargs):
    settings = dict(
        k=5,
        topology_name="nsfnet_deeprmsa_undirected",
        link_resources=4,
        max_requests=1000,
        values_bw=[100],
        incremental_loading=True,
        env_type="rwa_lightpath_reuse",
        scale_factor=1.0,
    )
    if not kwargs:
        return _rwa_lr_cached_setup("rwa_lr_nsfnet_4", settings)
    settings.update(kwargs)
    key = jax.random.PRNGKey(0)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_lightpath_reuse_4node_test_setup(**kwargs):
    settings = dict(
        k=2,
        topology_name="4node",
        link_resources=4,
        max_requests=1000,
        values_bw=[100],
        incremental_loading=True,
        env_type="rwa_lightpath_reuse",
    )
    if not kwargs:
        return _rwa_lr_cached_setup("rwa_lr_4node", settings)
    settings.update(kwargs)
    key = jax.random.PRNGKey(0)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params
