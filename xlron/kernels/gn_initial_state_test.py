"""Bootstrap keeps live services and independent RNGs, without counting warmup."""

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xlron.environments.transformer_snr_test import setup
from xlron.environments.wrappers import LogWrapper
from xlron.kernels.gn_initial_state import restore_physical_state


@pytest.mark.slow
def test_bootstrap_checkpoint_upgrade_and_actual_ppo_entrypoint(tmp_path):
    from xlron.environments.make_env import make
    from xlron.train.train_utils import init_network, save_model

    c, _, _, _ = setup("rmsa_gn_model", "occupancy_snr")
    c.continuous_operation = True
    c.incremental_loading = False
    c.MODEL_PATH = str(tmp_path / "initial.eqx")
    c.WANDB = False
    save_model(init_network(c, jax.random.PRNGKey(30)), c)
    raw, params = make(c, log_wrapper=False)
    _, state = raw.reset(jax.random.PRNGKey(31), params)
    batched = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    snapshot = tmp_path / "physical.npz"
    keys = (
        "env_type",
        "topology_name",
        "link_resources",
        "k",
        "values_bw",
        "slot_size",
        "guardband",
        "load",
        "coherent",
        "continuous_operation",
        "uniform_spans",
        "mod_format_correction",
    )
    np.savez(
        snapshot,
        workload_key=json.dumps({k: c[k] for k in keys}),
        **{f"state_{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(batched))},
    )
    output = tmp_path / "trained.eqx"
    command = [
        sys.executable,
        "-m",
        "xlron.kernels.gn_train_benchmark",
        "--xpm-backend",
        "jax",
        "--initial-model",
        c.MODEL_PATH,
        "--upgrade-modulation-head",
        "--initial-env-state",
        str(snapshot),
        "--bootstrap-warmup-steps",
        "1",
        "--env_type=rmsa_gn_model",
        "--topology_name=5node_directed",
        "--link_resources=8",
        "--k=2",
        "--values_bw=100",
        "--slot_size=25",
        "--guardband=0",
        "--power_per_channel=-10",
        "--launch_power_type=fixed",
        "--continuous_operation",
        "--incremental_loading=false",
        "--USE_TRANSFORMER",
        "--transformer_obs_type=occupancy_snr",
        "--transformer_embedding_size=8",
        "--transformer_intermediate_size=16",
        "--transformer_num_heads=2",
        "--num_wire_features=2",
        "--transformer_modulation_features=true",
        "--gn_mask_mode=all_slots",
        "--NUM_ENVS=2",
        "--ROLLOUT_LENGTH=2",
        "--NUM_MINIBATCHES=2",
        "--UPDATE_EPOCHS=1",
        "--TOTAL_TIMESTEPS=8",
        "--STEPS_PER_INCREMENT=4",
        "--ENV_WARMUP_STEPS=0",
        "--RETRAIN_MODEL",
        "--SAVE_MODEL",
        f"--MODEL_PATH={output}",
    ]
    result = subprocess.run(
        command,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "Restored admitted physical bootstrap" in result.stdout
    assert "Discarded 1 bootstrap requests" in result.stdout
    with output.open("rb") as f:
        assert json.loads(f.readline())["transformer_modulation_features"] is True


@pytest.mark.parametrize("envs", [1, 3])
def test_physical_bootstrap_resets_metrics_preserves_services_and_fresh_keys(tmp_path, envs):
    c, raw, p, state = setup("rmsa_gn_model", "occupancy_snr")
    c.continuous_operation = True
    c.NUM_ENVS = envs
    mask, full, mods = raw.action_mask(state, p)
    state = state.replace(link_slot_mask=mask, full_link_slot_mask=full, mod_format_mask=mods)
    _, state, *_ = raw.step(jax.random.PRNGKey(4), state, jnp.argmax(mask), p)
    source = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    path = tmp_path / "physical.npz"
    workload = {k: c[k] for k in ("env_type", "topology_name", "link_resources", "load")}
    np.savez(
        path,
        workload_key=json.dumps(workload),
        **{f"state_{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(source))},
    )
    wrapped = LogWrapper(raw)
    key = jax.random.PRNGKey(8)
    keys = jax.random.split(key, envs)
    _, outer = jax.vmap(lambda k: wrapped.reset(k, p))(keys) if envs > 1 else wrapped.reset(key, p)
    original = (None, outer, None, key, jax.random.PRNGKey(9))
    restored = restore_physical_state(original, p, c, path)
    for a, b in zip(original[3:], restored[3:], strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.any(restored[1].env_state.accepted_services)
    assert not np.any(restored[1].lengths)
    expected = np.stack([state.link_slot_array] * envs) if envs > 1 else state.link_slot_array
    np.testing.assert_array_equal(restored[1].env_state.link_slot_array, expected)
    assert restored[2][0] is restored[1].env_state
    c.load += 1
    with pytest.raises(ValueError, match="load"):
        restore_physical_state(original, p, c, path)
