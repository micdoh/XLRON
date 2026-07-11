"""Regression tests for neural network model construction and forward passes.

GNN + GN-model envs: ``init_graph_tuple``/``update_graph_tuple`` build ``graph.edges``
as ``stack([normalized_snr, normalized_power], axis=-1)`` -> shape (E, S, 2), which
``GraphNet.__call__`` flattens to 2*link_resources features per edge. ``init_network``
must size the GNN edge embedder from the same rule (previously it hardcoded
``link_resources``, crashing at trace time with a dot_general contracting-dimension
mismatch for any ``--USE_GNN`` + GN-model run).
"""

import chex
import distrax
import jax
from absl import flags
from absl.testing import absltest

import xlron.parameter_flags  # noqa: F401  (registers all XLRON flags)
from xlron.environments.make_env import make, process_config
from xlron.train.train_utils import init_network


def _flag_defaults() -> dict:
    """Full config dict from the registered absl flag defaults."""
    f = flags.FLAGS
    defaults = {}
    for name in f:
        try:
            defaults[name] = f[name].default
        except Exception:
            continue
    return defaults


def _gnn_env_setup(env_type: str, **overrides):
    """Tiny env + processed config for GNN model construction tests."""
    cfg = _flag_defaults()
    cfg.update(
        env_type=env_type,
        topology_name="nsfnet_deeprmsa_directed",
        link_resources=10,
        k=4,
        values_bw=[100],
        incremental_loading=True,
        slot_size=12.5,
        guardband=0,
        ROLLOUT_LENGTH=10,
        TOTAL_TIMESTEPS=20,
        STEPS_PER_INCREMENT=10,
        NUM_ENVS=1,
        ENV_WARMUP_STEPS=0,
        USE_GNN=True,
    )
    cfg.update(overrides)
    config = process_config(cfg)
    env, params = make(config, log_wrapper=False)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    return config, env, params, state, key


class GNNModelConstructionTest(chex.TestCase):
    """GNN model construction + forward pass on GN-model and plain RSA envs."""

    def _build_and_forward(self, env_type: str, **overrides):
        config, env, params, state, key = _gnn_env_setup(env_type, **overrides)
        model = init_network(config, key)
        pi, value = model(state, params)
        return config, params, state, model, pi, value

    def test_rsa_gn_model_edge_embedder_width(self):
        """GN-model graph edges are (E, S, 2); embedder input must be 2*link_resources."""
        config, params, state, model, pi, value = self._build_and_forward("rsa_gn_model")
        # The env builds stacked [snr, power] per-slot edge features
        self.assertEqual(
            state.graph.edges.shape,
            (params.num_links, config.link_resources, 2),
        )
        # The edge embedder must accept the flattened width
        self.assertEqual(
            model.actor.graph_net.edge_embedder.weight.shape[1],
            2 * config.link_resources,
        )
        self.assertEqual(
            model.critic.graph_net.edge_embedder.weight.shape[1],
            2 * config.link_resources,
        )

    def test_rsa_gn_model_fixed_power_forward(self):
        """With fixed launch power the actor returns a bare path distribution."""
        config, params, state, model, pi, value = self._build_and_forward("rsa_gn_model")
        self.assertIsInstance(pi, distrax.Categorical)
        expected_actions = config.k * -(-config.link_resources // config.aggregate_slots)
        self.assertEqual(pi.logits.shape, (expected_actions + int(params.include_no_op),))
        chex.assert_tree_all_finite(pi.logits)
        chex.assert_tree_all_finite(value)

    def test_rsa_gn_model_rl_power_forward(self):
        """With RL launch power the actor returns (path_dist, power_dist)."""
        config, params, state, model, pi, value = self._build_and_forward(
            "rsa_gn_model", launch_power_type="rl"
        )
        self.assertIsInstance(pi, tuple)
        path_dist, power_dist = pi
        self.assertIsInstance(path_dist, distrax.Categorical)
        self.assertIsNotNone(power_dist)
        chex.assert_tree_all_finite(path_dist.logits)

    def test_rmsa_gn_model_forward(self):
        """RMSA GN-model envs use the same stacked edge features."""
        config, params, state, model, pi, value = self._build_and_forward("rmsa_gn_model")
        self.assertEqual(
            model.actor.graph_net.edge_embedder.weight.shape[1],
            2 * config.link_resources,
        )
        self.assertIsInstance(pi, distrax.Categorical)
        chex.assert_tree_all_finite(pi.logits)
        chex.assert_tree_all_finite(value)

    def test_rsa_forward_unchanged(self):
        """Plain RSA envs keep link_resources-wide edge features (no regression)."""
        config, params, state, model, pi, value = self._build_and_forward("rsa")
        self.assertEqual(
            state.graph.edges.shape,
            (params.num_links, config.link_resources),
        )
        self.assertEqual(
            model.actor.graph_net.edge_embedder.weight.shape[1],
            config.link_resources,
        )
        self.assertIsInstance(pi, distrax.Categorical)
        chex.assert_tree_all_finite(pi.logits)
        chex.assert_tree_all_finite(value)


if __name__ == "__main__":
    absltest.main()
