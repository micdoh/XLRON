"""Unit tests for `reconfigurable_routing_bounds.py`."""

import types

import jax
import jax.numpy as jnp
from absl.testing import absltest

from xlron.bounds.reconfigurable_routing_bounds import _get_select_action, get_eval_fn
from xlron.environments.make_env import make
from xlron.heuristics.heuristics import ff_ksp, ksp_ff


def _rsa_4node_setup():
    """Small RSA env on the 4-node ring for defrag tests."""
    settings = dict(
        load=2,
        k=2,
        topology_name="4node",
        link_resources=5,
        max_requests=2,
        mean_service_holding_time=10,
        env_type="rsa",
        values_bw=[1, 2],
        slot_size=1,
        guardband=0,
    )
    env, params = make(settings, log_wrapper=True)
    params_det = params.replace(deterministic_requests=True)
    key = jax.random.PRNGKey(0)
    init_obs, env_state = env.reset(key, params)
    return env, params_det, init_obs, env_state


# Two requests, both active at the time of request index 1 (current_time=2):
# req0: 0->1, bw=2 (2 slots on 1-hop path), arrived t=1, holds 10 (departs t=11)
# req1: 1->2, bw=1 (1 slot on 1-hop path),  arrived t=2, holds 10 (departs t=12)
# Columns: [source, bitrate, dest, arrival_time, holding_time, current_time]
_REQUESTS = jnp.array(
    [
        [0.0, 2.0, 1.0, 1.0, 10.0, 1.0],
        [1.0, 1.0, 2.0, 1.0, 10.0, 2.0],
    ],
    dtype=jnp.float32,
)


class GetSelectActionTest(absltest.TestCase):
    def test_supported_heuristics(self):
        self.assertIs(_get_select_action("ksp_ff"), ksp_ff)
        self.assertIs(_get_select_action("ff_ksp"), ff_ksp)

    def test_unsupported_heuristic_raises(self):
        with self.assertRaisesRegex(ValueError, "ksp_bf"):
            _get_select_action("ksp_bf")


class RunDefragmentationTest(absltest.TestCase):
    """Non-compiled defrag path must place ALL active requests (incl. index 0)."""

    def test_all_active_requests_placed(self):
        env, params_det, init_obs, env_state = _rsa_4node_setup()
        config = types.SimpleNamespace(path_heuristic="ksp_ff", TOTAL_TIMESTEPS=2, load=2)
        _, run_defrag = get_eval_fn(config, env, params_det, compile_defrag=False)

        rng = jax.random.PRNGKey(1)
        sort_index = jnp.array(1)  # both requests are active at requests[1]'s time
        _, new_env_state, blocking = run_defrag(rng, _REQUESTS, sort_index, init_obs, env_state)

        self.assertFalse(bool(blocking))
        lsa = new_env_state.env_state.link_slot_array
        # 2 slots (req0) + 1 slot (req1), both on 1-hop paths
        self.assertEqual(float(jnp.sum(lsa)), 3.0)
        # Departure times reconstructed to the real departures of the two requests
        lsda = new_env_state.env_state.link_slot_departure_array
        departures = set(jnp.unique(lsda[lsa > 0]).tolist())
        self.assertEqual(departures, {11.0, 12.0})


class RunDefragTrimmedTest(absltest.TestCase):
    """Compiled defrag path must place trimmed[0] (the largest active service)
    and must not implement the stale request carried in defrag_initial_state."""

    def test_all_active_requests_placed_no_stale_leak(self):
        env, params_det, init_obs, env_state = _rsa_4node_setup()
        config = types.SimpleNamespace(path_heuristic="ksp_ff", TOTAL_TIMESTEPS=2, load=2)
        _, run_defrag_trimmed = get_eval_fn(config, env, params_det, compile_defrag=True)

        max_active = max(1, int(config.load * 2))
        # Plant a distinguishable stale request (2->3, bw=1) in the carried state:
        # pre-fix, step 0 of the defrag episode would implement it.
        defrag_initial_state = env_state.replace(
            env_state=env_state.env_state.replace(
                list_of_requests=jnp.zeros((max_active, 6), dtype=jnp.float32),
                request_array=jnp.array([2.0, 1.0, 3.0], dtype=jnp.float32),
                holding_time=jnp.array([5.0], dtype=jnp.float32),
                current_time=jnp.array([0.0], dtype=jnp.float32),
            )
        )

        rng = jax.random.PRNGKey(1)
        sort_index = jnp.array(1)
        _, lsa, lsda, blocking = run_defrag_trimmed(
            rng, _REQUESTS, sort_index, init_obs, defrag_initial_state
        )

        self.assertFalse(bool(blocking))
        # Both active requests placed (2 + 1 slots) and no stale-request slots leaked
        self.assertEqual(float(jnp.sum(lsa)), 3.0)
        departures = set(jnp.unique(lsda[lsa > 0]).tolist())
        self.assertEqual(departures, {11.0, 12.0})


if __name__ == "__main__":
    absltest.main()
