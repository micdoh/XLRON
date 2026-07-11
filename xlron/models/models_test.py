"""Tests for xlron.models: init scales, launch-power heads, and action sampling.

Covers regressions fixed in the models bundle:
- actor/critic output-head orthogonal init scales (0.01 / 1.0, matching the original
  Flax models) and independent actor/critic init keys
- LaunchPowerActorCriticMLP construction (undeclared eqx fields) and critic forward pass
- degenerate GNN continuous launch-power Beta distribution (alpha == beta)
- distrax .probs property called as a method in deterministic sampling
- shared PRNG key for joint path/power sampling
- init_network dispatch for launch_power_type="rl" with the MLP model
"""

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from box import Box

from xlron import dtype_config
from xlron.environments.gn_model.isrs_gn_model import from_dbm
from xlron.environments.make_env import make
from xlron.models.gnn import ActorCriticGNN
from xlron.models.mlp import ActorCriticMLP, LaunchPowerActorCriticMLP
from xlron.models.transformer import ActorCriticTransformer
from xlron.train.train_utils import TrainState, init_network, select_action

# Module-level caches for expensive make() calls (mirrors gn_model_test.py)
_env_cache = {}


def _cached_env(cache_key, settings, seed=0):
    key = jax.random.PRNGKey(seed)
    if cache_key not in _env_cache:
        _env_cache[cache_key] = make(settings)  # LogWrapper-wrapped
    env, params = _env_cache[cache_key]
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_gn_model_rl_setup():
    return _cached_env(
        "rsa_gn_model_rl",
        dict(
            k=4,
            topology_name="nsfnet_deeprmsa_undirected",
            link_resources=8,
            max_requests=10,
            values_bw=[100],
            incremental_loading=True,
            env_type="rsa_gn_model",
            interband_gap=0,
            slot_size=25,
            mod_format_correction=False,
            launch_power_type="rl",
            launch_power=0.0,
        ),
    )


def rmsa_gn_model_rl_setup():
    return _cached_env(
        "rmsa_gn_model_rl",
        dict(
            k=4,
            topology_name="nsfnet_deeprmsa_directed",
            link_resources=10,
            max_requests=100,
            values_bw=[100],
            incremental_loading=True,
            env_type="rmsa_gn_model",
            slot_size=12.5,
            guardband=0,
            mod_format_correction=False,
            max_power_per_fibre=10.0,
            coherent=False,
            include_no_op=False,
            launch_power_type="rl",
        ),
        seed=3,
    )


def make_small_gnn(state, params, discrete, key=1):
    """ActorCriticGNN sized to the cached rsa_gn_model env's graph tuple."""
    edge_feat = int(np.prod(state.env_state.graph.edges.shape[1:]))
    node_feat = int(state.env_state.graph.nodes.shape[-1])
    return ActorCriticGNN(
        input_edge_features=edge_feat,
        input_node_features=node_feat,
        input_global_features=1,
        num_layers=1,
        num_units=8,
        message_passing_steps=1,
        mlp_layers=1,
        mlp_latent=8,
        edge_embedding_size=8,
        edge_mlp_layers=1,
        edge_mlp_latent=8,
        edge_output_size_actor=8,
        edge_output_size_critic=0,
        global_embedding_size=4,
        global_mlp_layers=0,
        global_mlp_latent=0,
        global_output_size_actor=0,
        global_output_size_critic=1,
        node_embedding_size=4,
        node_mlp_layers=1,
        node_mlp_latent=8,
        node_output_size_actor=0,
        node_output_size_critic=0,
        attn_mlp_layers=0,
        attn_mlp_latent=0,
        use_attention=False,
        normalise_by_link_length=False,
        gnn_layer_norm=True,
        mlp_layer_norm=False,
        vmap=False,
        discrete=discrete,
        min_power_dbm=-5.0,
        max_power_dbm=0.5,
        step_power_dbm=0.1,
        key=jax.random.PRNGKey(key),
    )


def _gn_select_action_config(**overrides):
    config = Box(
        dict(
            env_type="rsa_gn_model",
            launch_power_type="rl",
            USE_GNN=True,
            USE_TRANSFORMER=False,
            GNN_OUTPUT_RSA=True,
            GNN_OUTPUT_LP=True,
            global_output_size_actor=0,
            deterministic=False,
        )
    )
    config.update(overrides)
    return config


class MLPInitScaleTest(chex.TestCase):
    """Orthogonal init: hidden layers sqrt(2), actor head 0.01, critic head 1.0."""

    def setUp(self):
        super().setUp()
        self.model = ActorCriticMLP(10, 7, num_layers=2, num_units=16, key=jax.random.PRNGKey(0))

    def test_head_and_hidden_init_scales(self):
        # Orthogonal init scaled by `scale` has all singular values == scale
        actor_head_sv = jnp.linalg.svd(self.model.actor.layers[-1].weight, compute_uv=False)
        critic_head_sv = jnp.linalg.svd(self.model.critic.layers[-1].weight, compute_uv=False)
        hidden_sv = jnp.linalg.svd(self.model.actor.layers[0].weight, compute_uv=False)
        chex.assert_trees_all_close(actor_head_sv, jnp.full_like(actor_head_sv, 0.01), atol=1e-4)
        chex.assert_trees_all_close(critic_head_sv, jnp.full_like(critic_head_sv, 1.0), atol=1e-4)
        chex.assert_trees_all_close(hidden_sv, jnp.full_like(hidden_sv, np.sqrt(2)), atol=1e-4)

    def test_actor_critic_hidden_layers_differ(self):
        # Actor and critic must not start with byte-identical hidden weights
        self.assertFalse(
            bool(
                jnp.array_equal(
                    self.model.actor.layers[0].weight, self.model.critic.layers[0].weight
                )
            )
        )

    def test_sample_action_deterministic_is_mode(self):
        dist = distrax.Categorical(logits=jnp.array([0.1, 2.0, -1.0, 0.5]))
        action = self.model.sample_action(jax.random.PRNGKey(0), dist, deterministic=True)
        self.assertEqual(int(action), 1)


class TransformerInitTest(chex.TestCase):
    def _make(self, share_layers):
        return ActorCriticTransformer(
            input_size=12,
            embedding_size=8,
            intermediate_size=16,
            num_slot_actions=4,
            num_layers=1,
            num_heads=2,
            enable_dropout=False,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            share_layers=share_layers,
            num_wire_features=2,
            actor_mlp_width=8,
            critic_mlp_width=8,
            actor_mlp_depth=1,
            critic_mlp_depth=1,
            num_request_specific_cols=5,
            key=jax.random.PRNGKey(0),
        )

    def test_encoders_differ_when_not_shared(self):
        model = self._make(share_layers=False)
        actor_enc, critic_enc = model.actor_critic
        # Transformer blocks have identical shapes for actor/critic; with independent init
        # keys they must not start byte-identical.
        actor_leaves = jax.tree_util.tree_leaves(actor_enc.layers)
        critic_leaves = jax.tree_util.tree_leaves(critic_enc.layers)
        any_diff = any(not bool(jnp.array_equal(a, c)) for a, c in zip(actor_leaves, critic_leaves))
        self.assertTrue(any_diff)

    def test_shared_encoder_is_same_object(self):
        model = self._make(share_layers=True)
        self.assertIs(model.actor_critic[0], model.actor_critic[1])

    def test_sample_action_deterministic_is_mode(self):
        model = self._make(share_layers=True)
        dist = distrax.Categorical(logits=jnp.array([0.1, -2.0, 3.0, 0.5]))
        action = model.sample_action(jax.random.PRNGKey(0), dist, deterministic=True)
        self.assertEqual(int(action), 2)


class LaunchPowerMLPTest(chex.TestCase):
    """Construction + forward + sampling for LaunchPowerActorCriticMLP."""

    K_PATHS = 4
    OBS_DIM = 4 + 4 * 7  # num_base_features + k_paths * num_path_features

    def _make(self, discrete):
        return LaunchPowerActorCriticMLP(
            1,
            self.OBS_DIM,
            num_layers=2,
            num_units=16,
            discrete=discrete,
            min_power_dbm=-5.0,
            max_power_dbm=0.5,
            step_power_dbm=0.1,
            k_paths=self.K_PATHS,
            key=jax.random.PRNGKey(0),
        )

    def test_discrete_forward_and_sampling(self):
        model = self._make(discrete=True)
        obs = jax.random.normal(jax.random.PRNGKey(1), (self.OBS_DIM,))
        (path_dist, dist), value = model(obs)
        self.assertIsNone(path_dist)
        self.assertIsInstance(dist, distrax.Categorical)
        chex.assert_shape(dist.logits, (self.K_PATHS, model.num_power_levels))
        chex.assert_shape(value, ())
        action, log_prob = model.sample_action(jax.random.PRNGKey(2), dist, log_prob=True)
        chex.assert_shape(action, (self.K_PATHS,))
        chex.assert_shape(log_prob, (self.K_PATHS,))
        # Powers are linear-unit conversions of the discrete dBm levels
        valid_powers = from_dbm(model.power_levels)
        self.assertTrue(bool(jnp.all(jnp.isin(action, valid_powers))))
        det_action = model.sample_action(jax.random.PRNGKey(2), dist, deterministic=True)
        chex.assert_shape(det_action, (self.K_PATHS,))
        chex.assert_shape(model.get_action_probs(dist), (self.K_PATHS, model.num_power_levels))

    def test_continuous_forward_and_sampling(self):
        model = self._make(discrete=False)
        obs = jax.random.normal(jax.random.PRNGKey(1), (self.OBS_DIM,))
        (_, dist), value = model(obs)
        self.assertIsInstance(dist, distrax.Beta)
        self.assertEqual(dist.batch_shape, (self.K_PATHS,))
        # Separate alpha/beta heads: generically not equal, so the mean is learnable
        self.assertFalse(bool(jnp.all(dist.alpha == dist.beta)))
        action, log_prob = model.sample_action(jax.random.PRNGKey(2), dist, log_prob=True)
        chex.assert_shape(action, (self.K_PATHS,))
        chex.assert_shape(log_prob, (self.K_PATHS,))
        min_power, max_power = from_dbm(-5.0), from_dbm(0.5)
        self.assertTrue(bool(jnp.all((action >= min_power) & (action <= max_power))))
        det_action = model.sample_action(jax.random.PRNGKey(2), dist, deterministic=True)
        chex.assert_shape(det_action, (self.K_PATHS,))


class InitNetworkDispatchTest(chex.TestCase):
    def test_launch_power_rl_dispatches_to_launch_power_mlp(self):
        config = Box(
            dict(
                env_type="rmsa_gn_model",
                USE_TRANSFORMER=False,
                USE_GNN=False,
                launch_power_type="rl",
                ACTION_DIM=1,
                INPUT_DIM=32,
                include_no_op=False,
                ACTIVATION="tanh",
                NUM_LAYERS=2,
                NUM_UNITS=16,
                mlp_layer_norm=False,
                discrete_launch_power=True,
                min_power=-5.0,
                max_power=0.5,
                step_power=0.1,
                k=4,  # config key is "k" (experiment_data_setup syncs it to env k_paths)
            )
        )
        network = init_network(config, jax.random.PRNGKey(0))
        self.assertIsInstance(network, LaunchPowerActorCriticMLP)
        self.assertEqual(network.k_paths, 4)  # ty: ignore[unresolved-attribute]


class GNNPathPowerSamplingTest(chex.TestCase):
    """Path/power joint sampling must draw from independent keys."""

    def setUp(self):
        super().setUp()
        _, _, _, state, params = rsa_gn_model_rl_setup()
        self.model = make_small_gnn(state, params, discrete=True)

    def test_sample_action_path_deterministic_is_mode(self):
        dist = distrax.Categorical(logits=jnp.array([0.1, 2.0, -1.0]))
        action = self.model.sample_action_path(jax.random.PRNGKey(0), dist, deterministic=True)
        self.assertEqual(int(action), 1)
        self.assertEqual(action.dtype, dtype_config.INDEX_DTYPE)

    def test_path_power_samples_decorrelated(self):
        num_paths, num_levels = 30, self.model.num_power_levels
        path_dist = distrax.Categorical(logits=jnp.zeros(num_paths))
        power_dist = distrax.Categorical(logits=jnp.zeros(num_levels))

        def draw(key):
            path_action, power_action, _ = self.model.sample_action_path_power(
                key, (path_dist, power_dist), log_prob=True
            )
            return path_action, power_action

        keys = jax.random.split(jax.random.PRNGKey(42), 8000)
        path_actions, power_actions = jax.vmap(draw)(keys)
        first_power = from_dbm(self.model.power_levels)[0]
        picked_path0 = path_actions == 0
        picked_power0 = jnp.isclose(power_actions, first_power)
        # With a shared key the power draw is (nearly) a deterministic function of the path
        # draw: P(power==levels[0] | path==0) == 1.0. Independent keys give the marginal.
        conditional = jnp.sum(picked_path0 & picked_power0) / jnp.maximum(jnp.sum(picked_path0), 1)
        self.assertLess(float(conditional), 0.5)
        self.assertGreater(float(conditional), 1.0 / num_levels / 3)


class GNNLaunchPowerIntegrationTest(chex.TestCase):
    """End-to-end select_action for the GN-model RL launch-power modes."""

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_gn_model_rl_setup()

    def _train_state(self, model):
        return TrainState.create(model=model, tx=optax.adam(1e-3))

    def test_continuous_power_dist_is_per_path_beta(self):
        model = make_small_gnn(self.state, self.params, discrete=False)
        (path_dist, power_dist), value = model(self.state.env_state, self.params)
        self.assertIsInstance(power_dist, distrax.Beta)
        self.assertEqual(power_dist.batch_shape, (self.params.k_paths,))
        self.assertFalse(bool(jnp.all(power_dist.alpha == power_dist.beta)))

    def test_select_action_path_and_power(self):
        model = make_small_gnn(self.state, self.params, discrete=False)
        config = _gn_select_action_config()
        for seed, deterministic in [(3, False), (4, True)]:
            config.deterministic = deterministic
            env_state, action, log_prob, value = select_action(
                (jax.random.PRNGKey(seed), self.state, None),
                self.env,
                self.params,
                self._train_state(model),
                config,
            )
            chex.assert_shape(action, (2,))  # [path_action, power]
            chex.assert_shape(log_prob, ())
            # Carried launch power array keeps its (k_paths,) shape and LARGE_FLOAT dtype
            chex.assert_shape(env_state.env_state.launch_power_array, (self.params.k_paths,))
            self.assertEqual(
                env_state.env_state.launch_power_array.dtype,
                jnp.dtype(dtype_config.LARGE_FLOAT_DTYPE),
            )

    def test_select_action_power_only(self):
        model = make_small_gnn(self.state, self.params, discrete=False)
        config = _gn_select_action_config(GNN_OUTPUT_RSA=False, GNN_OUTPUT_LP=True)
        env_state, action, log_prob, value = select_action(
            (jax.random.PRNGKey(3), self.state, None),
            self.env,
            self.params,
            self._train_state(model),
            config,
        )
        chex.assert_shape(action, (2,))
        chex.assert_shape(env_state.env_state.launch_power_array, (self.params.k_paths,))

    def test_select_action_discrete_power(self):
        model = make_small_gnn(self.state, self.params, discrete=True)
        config = _gn_select_action_config(deterministic=True)
        env_state, action, log_prob, value = select_action(
            (jax.random.PRNGKey(3), self.state, None),
            self.env,
            self.params,
            self._train_state(model),
            config,
        )
        chex.assert_shape(action, (2,))
        valid_powers = from_dbm(model.power_levels)
        self.assertTrue(bool(jnp.any(jnp.isclose(action[1], valid_powers, rtol=1e-5))))


class LaunchPowerMLPSelectActionTest(chex.TestCase):
    """Power-only RL launch power with the (non-GNN) MLP model, path via heuristic."""

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rmsa_gn_model_rl_setup()

    def test_select_action_power_only_mlp(self):
        model = LaunchPowerActorCriticMLP(
            1,
            int(self.obs.shape[-1]),
            num_layers=2,
            num_units=16,
            discrete=True,
            min_power_dbm=-5.0,
            max_power_dbm=0.5,
            step_power_dbm=0.1,
            k_paths=self.params.k_paths,
            key=jax.random.PRNGKey(0),
        )
        train_state = TrainState.create(model=model, tx=optax.adam(1e-3))
        config = Box(
            dict(
                env_type="rmsa_gn_model",
                launch_power_type="rl",
                USE_GNN=False,
                USE_TRANSFORMER=False,
                GNN_OUTPUT_RSA=False,
                GNN_OUTPUT_LP=False,
                global_output_size_actor=0,
                deterministic=False,
            )
        )
        env_state, action, log_prob, value = select_action(
            (jax.random.PRNGKey(3), self.state, (self.obs,)),
            self.env,
            self.params,
            train_state,
            config,
        )
        chex.assert_shape(action, (2,))
        chex.assert_shape(log_prob, ())
        chex.assert_shape(env_state.env_state.launch_power_array, (self.params.k_paths,))
        self.assertEqual(
            env_state.env_state.launch_power_array.dtype,
            jnp.dtype(dtype_config.LARGE_FLOAT_DTYPE),
        )


if __name__ == "__main__":
    absltest.main()
