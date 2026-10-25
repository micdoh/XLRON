"""Cached RSA/RWA fixtures shared across source-module tests."""

import jax

from xlron.environments.make_env import make


def keys_test_setup():
    rng = jax.random.PRNGKey(0)  # N.B. all test rely on 0 seed for reproducibility
    rng, key_init, key_reset, key_policy, key_step = jax.random.split(rng, 5)
    return rng, key_init, key_reset, key_policy, key_step


def settings_rwa_4node():
    return dict(
        load=100,
        k=2,
        topology_name="4node",
        link_resources=4,
        max_requests=10,
        mean_service_holding_time=10,
        env_type="rwa",
        values_bw=[1],
        slot_size=1,
        guardband=0,
    )


_cache = {}


def _cached_setup(cache_key, settings, seed=0):
    """Return (key, env, obs, state, params) using cached env/params."""
    key = jax.random.PRNGKey(seed)
    if cache_key not in _cache:
        env, params = make(settings, log_wrapper=False)
        _cache[cache_key] = (env, params)
    env, params = _cache[cache_key]
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_4node_test_setup(**kwargs):
    if not kwargs:
        return _cached_setup("rwa_4node", settings_rwa_4node())
    key = jax.random.PRNGKey(0)
    settings = settings_rwa_4node()
    settings.update(kwargs)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rwa_4node_agg_slots_test_setup(**kwargs):
    if not kwargs:
        settings = settings_rwa_4node()
        settings["aggregate_slots"] = 2
        settings["link_resources"] = 6
        return _cached_setup("rwa_4node_agg_slots", settings)
    key = jax.random.PRNGKey(0)
    settings = settings_rwa_4node()
    settings["aggregate_slots"] = 2
    settings["link_resources"] = 6
    settings.update(kwargs)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_4node_3_slot_request_test_setup(**kwargs):
    if not kwargs:
        settings = settings_rwa_4node()
        settings["env_type"] = "rsa"
        settings["values_bw"] = [3]
        settings["link_resources"] = 5
        return _cached_setup("rsa_4node_3_slot", settings)
    key = jax.random.PRNGKey(0)
    settings = settings_rwa_4node()
    settings["env_type"] = "rsa"
    settings["values_bw"] = [3]
    settings["link_resources"] = 5
    settings.update(kwargs)
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_16_test_setup(**kwargs):
    base_settings = dict(
        load=100,
        k=5,
        topology_name="nsfnet_deeprmsa_undirected",
        link_resources=16,
        max_requests=10,
        values_bw=[1, 2, 3],
        slot_size=1,
        mean_service_holding_time=10,
        env_type="rsa",
    )
    if not kwargs:
        return _cached_setup("rsa_nsfnet_16", base_settings)
    key = jax.random.PRNGKey(0)
    base_settings.update(kwargs)
    env, params = make(base_settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_16_mod_test_setup(**kwargs):
    base_settings = dict(
        load=100,
        k=5,
        topology_name="nsfnet_deeprmsa_undirected",
        link_resources=16,
        max_requests=10,
        consider_modulation_format=True,
        slot_size=12.5,
        mean_service_holding_time=10,
        env_type="rsa",
        # Expectations in the tests below were computed with this modulations table
        # (the pre-unification implicit default); the flag default is now
        # modulations_deeprmsa.csv, so pin it explicitly.
        modulations_csv_filepath="./xlron/data/modulations/modulations.csv",
    )
    if not kwargs:
        return _cached_setup("rsa_nsfnet_16_mod", base_settings)
    key = jax.random.PRNGKey(0)
    base_settings.update(kwargs)
    env, params = make(base_settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_nsfnet_4_test_setup(**kwargs):
    base_settings = dict(
        load=1000,
        k=5,
        topology_name="nsfnet_deeprmsa_undirected",
        link_resources=4,
        max_requests=10,
        values_bw=[1, 2, 3],
        slot_size=1,
        mean_service_holding_time=10,
        env_type="rsa",
    )
    if not kwargs:
        return _cached_setup("rsa_nsfnet_4", base_settings)
    key = jax.random.PRNGKey(0)
    base_settings.update(kwargs)
    env, params = make(base_settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params
