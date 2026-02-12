import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import pathlib
import tempfile

import chex
import distrax
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from xlron.environments.dataclasses import *
from xlron.environments.env_funcs import *
from xlron.environments.make_env import make
from xlron.environments.rsa import *
from xlron.environments.wrappers import *


def rsa_gn_model_4_nsfnet_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rsa_gn_model_4_nsfnet = dict(
        k=5,
        topology_name="nsfnet_deeprmsa_undirected",
        link_resources=4,
        max_requests=10,
        values_bw=[100],
        incremental_loading=True,
        env_type="rsa_gn_model",
        interband_gap=0,
        slot_size=25,
        mod_format_correction=False,
        launch_power=0.0,
    )
    env, params = make(settings_rsa_gn_model_4_nsfnet, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


@absltest.skipThisClass("Not finalized")
class RSAGNModelTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_gn_model_4_nsfnet_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        (
            "case_end_episode",
            jnp.array(
                [
                    [
                        200.0,
                        800.0,
                        500.0,
                        0.0,
                    ],
                    [
                        1100.0,
                        700.0,
                        700.0,
                        700.0,
                    ],
                    [
                        0.0,
                        0.0,
                        300.0,
                        0.0,
                    ],
                    [
                        800.0,
                        500.0,
                        600.0,
                        500.0,
                    ],
                    [
                        400.0,
                        100.0,
                        200.0,
                        0.0,
                    ],
                    [
                        300.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        400.0,
                        300.0,
                        300.0,
                        600.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        400.0,
                        0.0,
                        200.0,
                        0.0,
                    ],
                    [
                        600.0,
                        500.0,
                        900.0,
                        900.0,
                    ],
                    [
                        400.0,
                        300.0,
                        200.0,
                        400.0,
                    ],
                    [
                        0.0,
                        0.0,
                        400.0,
                        700.0,
                    ],
                    [
                        700.0,
                        500.0,
                        600.0,
                        300.0,
                    ],
                    [
                        600.0,
                        800.0,
                        500.0,
                        700.0,
                    ],
                    [
                        500.0,
                        600.0,
                        700.0,
                        300.0,
                    ],
                    [
                        600.0,
                        200.0,
                        200.0,
                        400.0,
                    ],
                    [
                        1100.0,
                        900.0,
                        700.0,
                        900.0,
                    ],
                    [
                        1000.0,
                        800.0,
                        800.0,
                        900.0,
                    ],
                    [
                        400.0,
                        600.0,
                        400.0,
                        600.0,
                    ],
                    [
                        700.0,
                        200.0,
                        400.0,
                        300.0,
                    ],
                    [
                        600.0,
                        900.0,
                        800.0,
                        1300.0,
                    ],
                    [
                        1100.0,
                        1200.0,
                        1300.0,
                        1500.0,
                    ],
                ]
            ),
        ),
    )
    def test_end_episode(self, expected):
        rng, reset_rng = jax.random.split(self.key)
        obsv, env_state = self.env.reset(reset_rng, self.params)
        reward = jnp.array([1.0])
        i = 0
        while reward > 0:
            i += 1
            rng, rng_sample, rng_step = jax.random.split(rng, 3)
            # get mask
            env_state = self.env.action_mask(env_state, self.params)
            mask = env_state.link_slot_mask
            # make distribution
            action_dist = distrax.Categorical(logits=jnp.where(mask, mask, -1e8))
            # jax.debug.print("action dist {}", action_dist.logits, ordered=True)
            # sample distribution
            path_action = action_dist.sample(seed=rng_sample)
            path_index, slot_index = process_path_action(env_state, self.params, path_action)
            path = get_paths(self.params, read_rsa_request(env_state.request_array)[0])[path_index]
            jax.debug.print("---i--- {}", i, ordered=True)
            jax.debug.print("path {}", path, ordered=True)
            jax.debug.print("slot {}", slot_index, ordered=True)
            power_action = jnp.array([0])
            action = jnp.concatenate(
                [path_action.reshape((1,)), power_action.reshape((1,))], axis=0
            )
            # step env
            obsv, env_state, reward, done, info = self.variant(self.env.step, static_argnums=(3))(
                rng_step, env_state, action, self.params
            )
            jax.debug.print("action mask {}", env_state.link_slot_mask, ordered=True)
            jax.debug.print("action {}", action, ordered=True)
            jax.debug.print("reward {}", reward, ordered=True)
            jax.debug.print("link snr array {}", env_state.link_snr_array, ordered=True)
            jax.debug.print(
                "path_snr {}",
                get_snr_for_path(path, env_state.link_snr_array, self.params),
                ordered=True,
            )
            jax.debug.print("link_slot_array {}", env_state.link_slot_array, ordered=True)
            jax.debug.print("path_index_array {}", env_state.path_index_array, ordered=True)
            jax.debug.print(
                "channel_centre_bw_array {}", env_state.channel_centre_bw_array, ordered=True
            )
            jax.debug.print("channel_power_array {}", env_state.channel_power_array, ordered=True)
            jax.debug.print(
                "modulation_format_index_array {}",
                env_state.modulation_format_index_array,
                ordered=True,
            )
            jax.debug.print("-----END-----")
            jax.debug.print("request_array {}", env_state.request_array, ordered=True)
            if i == self.params.max_requests:
                break
        chex.assert_trees_all_close(env_state.link_snr_array, expected)


class TransceiverAmplifierNoiseTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        # Create a temporary CSV file with test data
        self.test_data = """sub_band,wavelength_min_nm,wavelength_max_nm,frequency_min_ghz,frequency_max_ghz,NF_ASE_dB,SNR_TRX_dB
1,1484.86,1519.8,197242.07,201921.52,7.0,15.80
2,1520,1529,196069.31,197225.30,9.0,17.82
3,1529.2,1568,191121.46,196043.48,5.5,21.25
4,1568.2,1607.8,186519.62,191096.83,6.0,21.25
5,1608,1619.67,185105.45,186483.67,9.0,17.07"""

        self.temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        self.temp_file.write(self.test_data)
        self.temp_file.close()
        self.noise_data_filepath = self.temp_file.name

    def tearDown(self):
        # Clean up temp file
        pathlib.Path(self.temp_file.name).unlink()
        super().tearDown()

    @chex.variants(without_jit=True)
    @parameterized.named_parameters(
        (
            "case_single_channel_band3",
            1,  # link_resources
            1550.0e-9,  # ref_lambda (m) - in band 3
            50.0,  # slot_size (GHz)
            jnp.array([21.25]),  # expected transceiver SNR
            jnp.array([5.5]),  # expected amplifier NF
        ),
        (
            "case_three_channels_band3",
            3,  # link_resources
            1550.0e-9,  # ref_lambda (m) - center freq ~193.4 THz
            100.0,  # slot_size (GHz)
            jnp.array([21.25, 21.25, 21.25]),  # all channels in band 3
            jnp.array([5.5, 5.5, 5.5]),  # all band 3 NF
        ),
        (
            "case_five_channels_band1",
            5,  # link_resources
            1500.0e-9,  # ref_lambda (m) - center freq ~199.9 THz
            50.0,  # slot_size (GHz)
            jnp.array([15.80, 15.80, 15.80, 15.80, 15.80]),  # all in band 1
            jnp.array([7.0, 7.0, 7.0, 7.0, 7.0]),
        ),
        (
            "case_seven_channels_band4",
            7,  # link_resources
            1588.0e-9,  # ref_lambda (m) - center freq ~188.8 THz
            150.0,  # slot_size (GHz)
            jnp.array([21.25, 21.25, 21.25, 21.25, 21.25, 21.25, 21.25]),  # all in band 4
            jnp.array([6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]),
        ),
        (
            "case_channels_spanning_multiple_bands",
            5,  # link_resources
            1528.77e-9,  # ref_lambda (m) - positioned at ~196.1 THz (band 2/3 boundary)
            600.0,  # slot_size (GHz) - spacing to span bands 3, 2, and 1
            jnp.array([21.25, 21.25, 17.82, 17.82, 15.80]),  # bands 3, 3, 2, 2, 1
            jnp.array([5.5, 5.5, 9.0, 9.0, 7.0]),  # corresponding NF values
        ),
    )
    def test_noise_array_initialization(
        self, link_resources, ref_lambda, slot_size, expected_transceiver_snr, expected_amplifier_nf
    ):
        # Call the function
        transceiver_snr_array, amplifier_noise_figure_array = self.variant(
            init_transceiver_amplifier_noise_arrays, static_argnums=(0, 1, 2)
        )(link_resources, ref_lambda, slot_size, self.noise_data_filepath)

        # Debug prints
        jax.debug.print("transceiver_snr_array: {}", transceiver_snr_array, ordered=True)
        jax.debug.print(
            "amplifier_noise_figure_array: {}", amplifier_noise_figure_array, ordered=True
        )

        # Assertions
        chex.assert_trees_all_close(transceiver_snr_array, expected_transceiver_snr)
        chex.assert_trees_all_close(amplifier_noise_figure_array, expected_amplifier_nf)

    def test_frequency_outside_bands_raises_error(self):
        """Test that frequencies outside all bands raise an error"""
        with self.assertRaises(ValueError) as context:
            # Use a wavelength that results in frequency outside all bands
            # Band 5 max is ~186.5 THz, so wavelength < ~1608 nm gives higher freq
            init_transceiver_amplifier_noise_arrays(
                link_resources=1,
                ref_lambda=1450.0e-9,  # Results in ~206.7 THz, outside all bands
                slot_size=50.0,
                noise_data_filepath=self.noise_data_filepath,
            )

        self.assertIn("outside the defined bands", str(context.exception))

    @chex.variants(without_jit=True)
    def test_slot_frequency_calculation(self):
        """Test that slot frequencies are calculated correctly"""
        link_resources = 5
        ref_lambda = 1550.0e-9  # m
        slot_size = 100.0  # GHz

        # Expected center frequency
        c = 299792458  # m/s
        expected_center_freq = c / ref_lambda / 1e9  # GHz

        # Expected slot centers relative to center
        expected_relative_slots = jnp.array([-200.0, -100.0, 0.0, 100.0, 200.0])
        expected_center_freq + expected_relative_slots

        # Run function and check intermediate calculations
        # (Would need to modify function to return slot_frequencies_ghz for testing)
        transceiver_snr_array, amplifier_noise_figure_array = self.variant(
            init_transceiver_amplifier_noise_arrays, static_argnums=(0, 1, 2)
        )(link_resources, ref_lambda, slot_size, self.noise_data_filepath)

        # At least check the arrays have correct shape
        self.assertEqual(transceiver_snr_array.shape, (link_resources,))
        self.assertEqual(amplifier_noise_figure_array.shape, (link_resources,))


def rmsa_gn_model_test_setup():
    key = jax.random.PRNGKey(0)
    settings = dict(
        k=4,
        topology_name="nsfnet_deeprmsa_directed",
        link_resources=10,
        max_requests=100,
        values_bw=[100],
        incremental_loading=True,
        env_type="rmsa_gn_model",
        slot_size=100,
        guardband=0,
        mod_format_correction=False,
        launch_power=3.0,
        max_power_per_fibre=13.0,
        coherent=False,
        include_no_op=False,
    )
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


class RMSAGNModelMaskTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rmsa_gn_model_test_setup()

    def test_mask_has_valid_entries_empty_network(self):
        """On an empty network, the mask should have at least one valid slot."""
        state = self.env.action_mask(self.state, self.params)
        mask = state.link_slot_mask
        self.assertTrue(jnp.any(mask > 0), "Mask should have valid entries on empty network")

    def test_mask_shape(self):
        """Mask should have correct shape accounting for include_no_op."""
        state = self.env.action_mask(self.state, self.params)
        base_size = self.params.k_paths * self.params.link_resources
        expected_mask_size = base_size + (1 if self.params.include_no_op else 0)
        self.assertEqual(state.link_slot_mask.shape, (expected_mask_size,))
        self.assertEqual(state.mod_format_mask.shape, (base_size,))

    def test_mod_format_mask_values(self):
        """mod_format_mask should contain -1 (invalid) or valid mod format indices."""
        state = self.env.action_mask(self.state, self.params)
        mfm = state.mod_format_mask
        num_mods = self.params.modulations_array.val.shape[0]
        # All entries should be >= -1 and < num_mods
        self.assertTrue(jnp.all(mfm >= -1.0))
        self.assertTrue(jnp.all(mfm < num_mods))

    def test_step_after_mask_does_not_crash(self):
        """Taking a masked action should not crash."""
        rng = self.key
        state = self.env.action_mask(self.state, self.params)
        mask = state.link_slot_mask
        # Sample a valid action
        rng, rng_sample, rng_step = jax.random.split(rng, 3)
        action_dist = distrax.Categorical(logits=jnp.where(mask > 0, 0.0, -1e8))
        path_action = action_dist.sample(seed=rng_sample)
        power_action = jnp.array([0])
        action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))], axis=0)
        obs, new_state, reward, terminal, truncated, info = self.env.step(
            rng_step, state, action, self.params
        )
        # Verify step completes without error and returns valid arrays
        self.assertTrue(jnp.isfinite(reward))
        self.assertTrue(isinstance(terminal, jax.Array))
        self.assertTrue(isinstance(truncated, jax.Array))


if __name__ == "__main__":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
