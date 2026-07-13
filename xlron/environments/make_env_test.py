"""Tests for lightweight run-configuration validation (validate_config)
and configuration processing (process_config / make)."""

import jax
import jax.numpy as jnp
import pytest
from box import Box

from xlron import dtype_config
from xlron.environments.make_env import make, process_config, validate_config


def _cfg(**overrides):
    base = dict(
        NUM_ENVS=4,
        ROLLOUT_LENGTH=32,
        NUM_MINIBATCHES=1,
        TOTAL_TIMESTEPS=4096,
        NUM_UPDATES=4,
        continuous_operation=True,
        end_first_blocking=False,
        max_requests=4,
    )
    base.update(overrides)
    return Box(base)


def test_structural_error_on_zero_envs():
    with pytest.raises(ValueError):
        validate_config(_cfg(NUM_ENVS=0), is_eval=False)


def test_sane_continuous_config_is_quiet(capsys):
    validate_config(_cfg(), is_eval=False)
    assert "WARNING" not in capsys.readouterr().out


def test_warns_when_window_shorter_than_episode(capsys):
    # Episodic (not continuous, not end_first_blocking): window 8 < episode 1000.
    validate_config(
        _cfg(continuous_operation=False, NUM_UPDATES=1, ROLLOUT_LENGTH=8, max_requests=1000),
        is_eval=False,
    )
    assert "episode-end metrics will be empty" in capsys.readouterr().out


def test_eval_skips_episode_window_check(capsys):
    validate_config(
        _cfg(continuous_operation=False, NUM_UPDATES=1, ROLLOUT_LENGTH=8, max_requests=1000),
        is_eval=True,
    )
    assert "WARNING" not in capsys.readouterr().out


def test_warns_on_indivisible_minibatches(capsys):
    validate_config(_cfg(NUM_MINIBATCHES=3), is_eval=False)
    assert "not divisible" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# process_config / make config plumbing
# ---------------------------------------------------------------------------


def _rwa_4node_settings(**overrides):
    settings = dict(
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
    settings.update(overrides)
    return settings


def test_maximise_throughput_flag_reaches_params():
    _, params = make(_rwa_4node_settings(maximise_throughput=True), log_wrapper=False)
    assert params.maximise_throughput is True


def test_deprecated_maximise_throughout_spelling_still_works():
    with pytest.deprecated_call():
        _, params = make(_rwa_4node_settings(maximise_throughout=True), log_wrapper=False)
    assert params.maximise_throughput is True


def test_node_probs_builds_traffic_matrix():
    probs = [0.4, 0.3, 0.2, 0.1]
    env, params = make(_rwa_4node_settings(node_probs="0.4,0.3,0.2,0.1"), log_wrapper=False)
    _, state = env.reset(jax.random.PRNGKey(0), params)
    # Expected: outer product with zeroed diagonal, normalised to sum to 1.
    expected = jnp.outer(jnp.array(probs), jnp.array(probs))
    expected = jnp.where(jnp.eye(expected.shape[0]) == 1, 0, expected)
    expected = (expected / expected.sum()).astype(dtype_config.SMALL_FLOAT_DTYPE)
    assert jnp.allclose(state.traffic_matrix, expected)


def test_disable_node_features_uppercase_flag_reaches_params():
    # disable_node_features lives on RSAEnvParams, not the EnvParams base: use getattr.
    _, params = make(_rwa_4node_settings(DISABLE_NODE_FEATURES=True), log_wrapper=False)
    assert getattr(params, "disable_node_features") is True


def test_disable_node_features_lowercase_key_still_works():
    _, params = make(_rwa_4node_settings(disable_node_features=True), log_wrapper=False)
    assert getattr(params, "disable_node_features") is True


def _eval_cfg(**overrides):
    base = dict(
        NUM_ENVS=10,
        ROLLOUT_LENGTH=32,
        NUM_MINIBATCHES=1,
        STEPS_PER_INCREMENT=5000,
        TOTAL_TIMESTEPS=5000,
        EVAL_HEURISTIC=True,
        continuous_operation=False,
        end_first_blocking=False,
        max_requests=1000,
    )
    base.update(overrides)
    return base


def test_episodic_eval_warns_when_steps_per_increment_inflated(capsys):
    # One episode = max_requests * NUM_ENVS = 10000 steps > requested SPI/TOTAL of 5000,
    # so both STEPS_PER_INCREMENT and TOTAL_TIMESTEPS are silently doubled without a warning.
    config = process_config(_eval_cfg())
    out = capsys.readouterr().out
    assert config.STEPS_PER_INCREMENT == 10000
    assert config.TOTAL_TIMESTEPS == 10000
    assert "Increasing STEPS_PER_INCREMENT" in out
    assert "TOTAL_TIMESTEPS adjusted from 5000 to 10000" in out


def test_continuous_eval_warns_when_max_requests_overridden(capsys):
    config = process_config(
        _eval_cfg(
            continuous_operation=True,
            NUM_ENVS=1,
            STEPS_PER_INCREMENT=1000,
            TOTAL_TIMESTEPS=1000,
            max_requests=999,
        )
    )
    out = capsys.readouterr().out
    assert config.max_requests == 1000
    assert "max_requests (999) is overridden to 1000" in out


def test_matching_eval_config_is_quiet(capsys):
    # SPI exactly fits one episode and divides TOTAL_TIMESTEPS: no warnings expected.
    process_config(_eval_cfg(STEPS_PER_INCREMENT=10000, TOTAL_TIMESTEPS=10000))
    assert "WARNING" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Flag defaults as single source of truth for dict-config callers
# ---------------------------------------------------------------------------


def test_process_config_empty_dict_uses_flag_defaults():
    """Dict-config callers omitting a key must get the CLI (parameter_flags) default.

    Regression test: make_env used to hardcode config.get() fallbacks that diverged
    from the flag defaults (e.g. include_no_op True vs flag False), so tests and
    library users silently got different behaviour from an identical CLI run.
    """
    from xlron.parameter_flags import get_flag_defaults

    defaults = get_flag_defaults()
    config = process_config({})
    # The historically divergent keys (fallback != flag default before the fix):
    for key in [
        "include_no_op",
        "calc_minimum_osnr",
        "relative_arrival_times",
        "path_sort_criteria",
        "mean_service_holding_time",
        "num_spectral_features",
        "max_power_per_fibre",
        "max_snr",
        "snr_margin",
        "max_power",
        "step_power",
        "enforce_band_gaps",
        "coherent",
        "mod_format_correction",
        "temperature",
        "topology_name",
        "load",
        "link_resources",
        "env_type",
        "node_resources",
        "max_node_resources",
        "modulations_csv_filepath",
        "transformer_obs_type",
        "dispersion_slope",
    ]:
        assert config[key] == defaults[key], (
            f"process_config default for {key!r} ({config[key]!r}) does not match "
            f"the parameter_flags default ({defaults[key]!r})"
        )
    # Every flag key must be present after layering (no more silent fallbacks).
    missing = set(defaults) - set(config)
    assert not missing, f"flag keys missing from processed config: {sorted(missing)}"


def test_process_config_user_config_overrides_flag_defaults():
    config = process_config(
        dict(mean_service_holding_time=10, include_no_op=True, relative_arrival_times=False)
    )
    assert config.mean_service_holding_time == 10
    assert config.include_no_op is True
    assert config.relative_arrival_times is False


def test_dict_config_flag_defaults_reach_params():
    """make() with a minimal dict resolves omitted keys to the CLI flag defaults."""
    settings = dict(
        env_type="rwa",
        topology_name="4node",
        link_resources=4,
        k=2,
        load=100,
        max_requests=10,
        values_bw=[1],
        slot_size=1,
    )
    _, params = make(settings, log_wrapper=False)
    assert params.include_no_op is False  # flag default; old dict fallback was True
    assert params.relative_arrival_times is True  # flag default; old dict fallback was False
    assert params.mean_service_holding_time == 25.0  # flag default; old dict fallback was 10
    assert params.num_spectral_features == 8  # flag default; old dict fallback was 3


def test_make_env_config_get_fallbacks_match_flag_defaults():
    """Any literal config.get() fallback in make_env.py for a flag-backed key must
    equal the parameter_flags default.

    The fallbacks are dead code for configs that pass through process_config (which
    layers in every flag default), but keeping them aligned prevents silent divergence
    for any future code path that bypasses the layering. Fallbacks of None are treated
    as presence checks and skipped, as are non-literal (computed) fallbacks.
    """
    import ast
    import inspect

    from xlron.environments import make_env
    from xlron.parameter_flags import get_flag_defaults

    defaults = get_flag_defaults()
    tree = ast.parse(inspect.getsource(make_env))

    def matches(fallback, default):
        if fallback is None:
            return True  # presence-check idiom
        if isinstance(fallback, bool) != isinstance(default, bool):
            return False
        try:
            return float(fallback) == float(default)
        except (TypeError, ValueError):
            return fallback == default

    violations = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "config"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
        ):
            continue
        key = node.args[0].value
        if key not in defaults:
            continue  # not a CLI flag (e.g. lowercase aliases, script-only keys)
        try:
            fallback = ast.literal_eval(node.args[1])
        except (ValueError, SyntaxError, TypeError):
            continue  # computed fallback, cannot compare statically
        if defaults[key] is None and fallback is not None:
            violations.append((node.lineno, key, fallback, defaults[key]))
        elif defaults[key] is not None and not matches(fallback, defaults[key]):
            violations.append((node.lineno, key, fallback, defaults[key]))
    assert not violations, (
        "config.get() fallbacks in make_env.py diverge from parameter_flags defaults "
        f"(line, key, fallback, flag default): {violations}"
    )
