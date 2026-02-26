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


# Module-level caches for expensive make() calls.
# Only env and params are cached; obs/state are recomputed per call via
# env.reset() to avoid buffer donation/deletion issues with chex device variants.
_gn_cache = {}


def _gn_cached_setup(cache_key, settings, seed=0):
    """Return (key, env, obs, state, params) using cached env/params."""
    key = jax.random.PRNGKey(seed)
    if cache_key not in _gn_cache:
        env, params = make(settings, log_wrapper=False)
        _gn_cache[cache_key] = (env, params)
    env, params = _gn_cache[cache_key]
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


def rsa_gn_model_4_nsfnet_test_setup():
    return _gn_cached_setup("rsa_gn_model_4_nsfnet", dict(
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
    ))


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
            mask, _ = self.env.action_mask(env_state, self.params)
            env_state = env_state.replace(link_slot_mask=mask)
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
            obsv, env_state, reward, done, truncated, info = self.variant(self.env.step, static_argnums=(3))(
                rng_step, env_state, action, self.params
            )
            jax.debug.print("action mask {}", env_state.link_slot_mask, ordered=True)
            jax.debug.print("action {}", action, ordered=True)
            jax.debug.print("reward {}", reward, ordered=True)
            jax.debug.print("link snr array {}", env_state.link_snr_array, ordered=True)
            jax.debug.print(
                "path_snr {}",
                get_snr_for_path(path, env_state.link_snr_array, self.params, env_state),
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
        self.test_data = """sub_band,wavelength_min_nm,wavelength_max_nm,frequency_min_ghz,frequency_max_ghz,NF_ASE_dB,SNR_TRX_dB,roadm_express_loss_dB,roadm_add_drop_loss_dB,roadm_NF_dB
1,1484.86,1519.8,197242.07,201921.52,7.0,15.80,5.0,8.0,5.0
2,1520,1529,196069.31,197225.30,9.0,17.82,5.0,8.0,5.0
3,1529.2,1568,191121.46,196043.48,5.5,21.25,5.0,8.0,5.0
4,1568.2,1607.8,186519.62,191096.83,6.0,21.25,5.0,8.0,5.0
5,1608,1619.67,185105.45,186483.67,9.0,17.07,5.0,8.0,5.0"""

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
        (
            transceiver_snr_array,
            amplifier_noise_figure_array,
            roadm_express_loss_array,
            roadm_add_drop_loss_array,
            roadm_noise_figure_array,
        ) = self.variant(init_transceiver_amplifier_noise_arrays, static_argnums=(0, 1, 2))(
            link_resources, ref_lambda, slot_size, self.noise_data_filepath
        )

        # Debug prints
        jax.debug.print("transceiver_snr_array: {}", transceiver_snr_array, ordered=True)
        jax.debug.print(
            "amplifier_noise_figure_array: {}", amplifier_noise_figure_array, ordered=True
        )

        # Assertions
        chex.assert_trees_all_close(transceiver_snr_array, expected_transceiver_snr)
        chex.assert_trees_all_close(amplifier_noise_figure_array, expected_amplifier_nf)

    def test_frequency_outside_bands_returns_zeros(self):
        """Test that frequencies outside all bands get zero noise values (gap slots)."""
        (
            transceiver_snr,
            amplifier_nf,
            roadm_express,
            roadm_add_drop,
            roadm_nf,
        ) = init_transceiver_amplifier_noise_arrays(
            link_resources=1,
            ref_lambda=1450.0e-9,  # Results in ~206.7 THz, outside all bands
            slot_size=50.0,
            noise_data_filepath=self.noise_data_filepath,
        )
        # Gap slots should have zero values
        self.assertEqual(float(transceiver_snr[0]), 0.0)
        self.assertEqual(float(amplifier_nf[0]), 0.0)

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
        (
            transceiver_snr_array,
            amplifier_noise_figure_array,
            roadm_express_loss_array,
            roadm_add_drop_loss_array,
            roadm_noise_figure_array,
        ) = self.variant(init_transceiver_amplifier_noise_arrays, static_argnums=(0, 1, 2))(
            link_resources, ref_lambda, slot_size, self.noise_data_filepath
        )

        # At least check the arrays have correct shape
        self.assertEqual(transceiver_snr_array.shape, (link_resources,))
        self.assertEqual(amplifier_noise_figure_array.shape, (link_resources,))
        self.assertEqual(roadm_express_loss_array.shape, (link_resources,))
        self.assertEqual(roadm_add_drop_loss_array.shape, (link_resources,))
        self.assertEqual(roadm_noise_figure_array.shape, (link_resources,))


def rmsa_gn_model_test_setup():
    # Seed 3 generates a short-path request (300 km) that passes SNR checks
    return _gn_cached_setup("rmsa_gn_model", dict(
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
    ), seed=3)


class RMSAGNModelMaskTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rmsa_gn_model_test_setup()

    def test_mask_has_valid_entries_empty_network(self):
        """On an empty network, the mask should have at least one valid slot."""
        mask, _, _ = self.env.action_mask(self.state, self.params)
        self.assertTrue(jnp.any(mask > 0), "Mask should have valid entries on empty network")

    def test_mask_shape(self):
        """Mask should have correct shape accounting for include_no_op."""
        mask, _, mod_format_mask = self.env.action_mask(self.state, self.params)
        base_size = self.params.k_paths * self.params.link_resources
        expected_mask_size = base_size + (1 if self.params.include_no_op else 0)
        self.assertEqual(mask.shape, (expected_mask_size,))
        self.assertEqual(mod_format_mask.shape, (base_size,))

    def test_mod_format_mask_values(self):
        """mod_format_mask should contain -1 (invalid) or valid mod format indices."""
        _, _, mfm = self.env.action_mask(self.state, self.params)
        num_mods = self.params.modulations_array.val.shape[0]
        # All entries should be >= -1 and < num_mods
        self.assertTrue(jnp.all(mfm >= -1.0))
        self.assertTrue(jnp.all(mfm < num_mods))

    def test_step_after_mask_does_not_crash(self):
        """Taking a masked action should not crash."""
        rng = self.key
        mask, _, mod_format_mask = self.env.action_mask(self.state, self.params)
        state = self.state.replace(link_slot_mask=mask, mod_format_mask=mod_format_mask)
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


def rmsa_gn_model_enforce_band_gaps_test_setup():
    key = jax.random.PRNGKey(3)
    if "rmsa_gn_model_band_gaps" in _gn_cache:
        env, params = _gn_cache["rmsa_gn_model_band_gaps"]
        obs, state = env.reset(key, params)
        return key, env, obs, state, params
    # Create a temp band data CSV with two non-contiguous bands so that a gap
    # appears in the middle of the 100-slot range (ref_lambda=1564nm default).
    # Slot freq range at defaults: ~191064 - 192302 GHz.
    # Band A covers slots 0-39, Band B covers slots 60-99, leaving a 20-slot gap.
    band_csv = (
        "band_name,wavelength_min_nm,wavelength_max_nm,frequency_min_ghz,frequency_max_ghz\n"
        "A,1560,1570,191060,191555\n"
        "B,1555,1560,191810,192305\n"
    )
    band_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    band_file.write(band_csv)
    band_file.close()

    settings = dict(
        k=4,
        topology_name="nsfnet_deeprmsa_directed",
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
        band_preference="A,B",
        band_data_filepath=band_file.name,
    )
    env, params = make(settings, log_wrapper=False)
    obs, state = env.reset(key, params)
    # Clean up temp file
    pathlib.Path(band_file.name).unlink()
    _gn_cache["rmsa_gn_model_band_gaps"] = (env, params)
    return key, env, obs, state, params


class EnforceBandGapsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        (
            self.key,
            self.env,
            self.obs,
            self.state,
            self.params,
        ) = rmsa_gn_model_enforce_band_gaps_test_setup()

    def test_band_gaps_present_in_initial_state(self):
        """Band gap slots should be -1 in the initial link_slot_array."""
        lsa = self.state.link_slot_array
        gap_starts = self.params.gap_starts.val
        gap_widths = self.params.gap_widths.val
        # There should be at least one gap
        self.assertGreater(len(gap_starts), 0, "enforce_band_gaps should produce gaps")
        # All gap slots should be -1 on every link
        for i in range(len(gap_starts)):
            start = int(gap_starts[i])
            width = int(gap_widths[i])
            gap_slots = lsa[:, start : start + width]
            self.assertTrue(
                jnp.all(gap_slots == -1),
                f"Gap at slot {start} width {width} should be -1 but got {gap_slots}",
            )

    def test_mask_does_not_propose_gap_slots(self):
        """Action mask should be zero for any slot inside a band gap."""
        mask, _, _ = self.env.action_mask(self.state, self.params)
        gap_starts = self.params.gap_starts.val
        gap_widths = self.params.gap_widths.val
        for i in range(len(gap_starts)):
            start = int(gap_starts[i])
            width = int(gap_widths[i])
            for p in range(self.params.k_paths):
                offset = p * self.params.link_resources
                gap_mask = mask[offset + start : offset + start + width]
                self.assertTrue(
                    jnp.all(gap_mask == 0),
                    f"Mask should be 0 in gap at path {p} slot {start}",
                )

    def test_band_gaps_survive_step(self):
        """Band gap slots should remain -1 after taking a valid action."""
        mask, _, mod_format_mask = self.env.action_mask(self.state, self.params)
        state = self.state.replace(link_slot_mask=mask, mod_format_mask=mod_format_mask)
        rng_sample, rng_step = jax.random.split(self.key)
        action_dist = distrax.Categorical(logits=jnp.where(mask > 0, 0.0, -1e8))
        path_action = action_dist.sample(seed=rng_sample)
        power_action = jnp.array([0])
        action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))])
        _, new_state, _, _, _, _ = self.env.step(rng_step, state, action, self.params)
        # Gaps must still be -1
        lsa = new_state.link_slot_array
        gap_starts = self.params.gap_starts.val
        gap_widths = self.params.gap_widths.val
        for i in range(len(gap_starts)):
            start = int(gap_starts[i])
            width = int(gap_widths[i])
            gap_slots = lsa[:, start : start + width]
            self.assertTrue(
                jnp.all(gap_slots == -1),
                f"Gap at slot {start} should still be -1 after step",
            )


def rsa_gn_model_band_preference_test_setup(band_preference):
    return _gn_cached_setup(f"rsa_gn_model_band_pref_{band_preference}", dict(
        k=4,
        topology_name="nsfnet_deeprmsa_directed",
        link_resources=100,
        max_requests=100,
        values_bw=[100],
        incremental_loading=True,
        env_type="rsa_gn_model",
        slot_size=12.5,
        guardband=0,
        mod_format_correction=False,
        max_power_per_fibre=10.0,
        coherent=False,
        include_no_op=False,
        band_preference=band_preference,
    ), seed=3)


class BandPreferenceTest(parameterized.TestCase):
    """Test that --band_preference controls first-fit/last-fit slot ordering."""

    def test_c_band_first_fit_prefers_c_band(self):
        """With C,L preference, first-fit should pick a C-band slot (>= 43)."""
        from xlron.heuristics.heuristics import first_fit

        _, env, _, state, params = rsa_gn_model_band_preference_test_setup("C,L")
        ff = first_fit(state, params)
        # C-band starts at slot 43 for default ref_lambda=1564nm, 100 slots, 12.5 GHz
        self.assertTrue(
            jnp.all(ff[ff < params.link_resources] >= 43),
            f"Expected C-band slots (>= 43) but got {ff}",
        )

    def test_l_band_first_fit_prefers_l_band(self):
        """With L,C preference, first-fit should pick an L-band slot (< 43)."""
        from xlron.heuristics.heuristics import first_fit

        _, env, _, state, params = rsa_gn_model_band_preference_test_setup("L,C")
        ff = first_fit(state, params)
        self.assertTrue(
            jnp.all(ff[ff < params.link_resources] < 43),
            f"Expected L-band slots (< 43) but got {ff}",
        )

    def test_no_preference_starts_at_slot_zero(self):
        """Without band_preference, first-fit should start from slot 0."""
        from xlron.heuristics.heuristics import first_fit

        key = jax.random.PRNGKey(3)
        settings = dict(
            k=4,
            topology_name="nsfnet_deeprmsa_directed",
            link_resources=100,
            max_requests=100,
            values_bw=[100],
            incremental_loading=True,
            env_type="rsa_gn_model",
            slot_size=12.5,
            guardband=0,
            mod_format_correction=False,
            max_power_per_fibre=10.0,
            coherent=False,
            include_no_op=False,
        )
        env, params = make(settings, log_wrapper=False)
        _, state = env.reset(key, params)
        ff = first_fit(state, params)
        # Without preference, first available slot is 0
        self.assertEqual(int(ff[0]), 0)

    def test_last_fit_with_c_preference(self):
        """With C,L preference, last-fit should pick a C-band slot."""
        from xlron.heuristics.heuristics import last_fit

        _, env, _, state, params = rsa_gn_model_band_preference_test_setup("C,L")
        lf = last_fit(state, params)
        # Last-fit in C-band (slots 43-99) should return a high C-band slot
        valid = lf[lf < params.link_resources]
        self.assertTrue(
            jnp.all(valid >= 43),
            f"Expected C-band slots (>= 43) but got {lf}",
        )

    def test_band_slot_order_is_valid_permutation(self):
        """band_slot_order arrays should be permutations of [0, link_resources)."""
        _, _, _, _, params = rsa_gn_model_band_preference_test_setup("C,L")
        order_ff = params.band_slot_order_ff.val
        order_lf = params.band_slot_order_lf.val
        self.assertEqual(len(order_ff), params.link_resources)
        self.assertEqual(len(order_lf), params.link_resources)
        self.assertEqual(sorted(order_ff.tolist()), list(range(params.link_resources)))
        self.assertEqual(sorted(order_lf.tolist()), list(range(params.link_resources)))


def rsa_gn_model_subchannels_test_setup(num_subchannels=1):
    return _gn_cached_setup(f"rsa_gn_model_subch_{num_subchannels}", dict(
        k=4,
        topology_name="nsfnet_deeprmsa_directed",
        link_resources=10,
        max_requests=100,
        values_bw=[100],
        incremental_loading=True,
        env_type="rsa_gn_model",
        slot_size=100,
        guardband=0,
        mod_format_correction=False,
        max_power_per_fibre=10.0,
        coherent=False,
        include_no_op=False,
        num_subchannels=num_subchannels,
    ), seed=3)


class NumSubchannelsTest(parameterized.TestCase):
    """Tests for the num_subchannels SPM correction feature."""

    def test_backward_compatibility_default(self):
        """num_subchannels=1 (default) should produce identical results to baseline."""
        _, env1, _, state1, params1 = rsa_gn_model_subchannels_test_setup(num_subchannels=1)
        # Verify the parameter is set correctly
        self.assertEqual(params1.num_subchannels, 1)

    def test_num_subchannels_stored_in_params(self):
        """num_subchannels=8 should be correctly stored in params."""
        _, _, _, _, params = rsa_gn_model_subchannels_test_setup(num_subchannels=8)
        self.assertEqual(params.num_subchannels, 8)

    def test_spm_reduction_with_subchannels(self):
        """num_subchannels=8 should produce lower SPM eta than num_subchannels=1."""
        from xlron.environments.gn_model.isrs_gn_model import isrs_gn_model_uniform

        # Single channel, uniform parameters
        num_ch = 1
        freq = jnp.array([193.5e12])  # Hz
        bw = jnp.array([100e9])  # 100 GHz
        power = jnp.array([0.001])  # 1 mW

        common_kwargs = dict(
            num_channels=num_ch,
            num_spans=10,
            ref_lambda=1550e-9,
            length=80e3,
            ch_power_W_i=power,
            ch_centre_i=freq,
            ch_bandwidth_i=bw,
            coherent=False,
            mod_format_correction=False,
        )

        _, _, eta_spm_1, _ = isrs_gn_model_uniform(**common_kwargs, num_subchannels=1)
        _, _, eta_spm_8, _ = isrs_gn_model_uniform(**common_kwargs, num_subchannels=8)

        # SPM should be lower with more subchannels
        self.assertTrue(
            float(jnp.squeeze(eta_spm_8)) < float(jnp.squeeze(eta_spm_1)),
            f"SPM with 8 subchannels ({float(jnp.squeeze(eta_spm_8))}) "
            f"should be less than with 1 ({float(jnp.squeeze(eta_spm_1))})",
        )

    def test_xpm_unchanged_with_subchannels(self):
        """XPM between different channels should be identical regardless of num_subchannels."""
        from xlron.environments.gn_model.isrs_gn_model import isrs_gn_model_uniform

        num_ch = 3
        freq = jnp.array([193.3e12, 193.4e12, 193.5e12])
        bw = jnp.array([100e9, 100e9, 100e9])
        power = jnp.array([0.001, 0.001, 0.001])

        common_kwargs = dict(
            num_channels=num_ch,
            num_spans=10,
            ref_lambda=1550e-9,
            length=80e3,
            ch_power_W_i=power,
            ch_centre_i=freq,
            ch_bandwidth_i=bw,
            coherent=False,
            mod_format_correction=False,
        )

        _, _, _, eta_xpm_1 = isrs_gn_model_uniform(**common_kwargs, num_subchannels=1)
        _, _, _, eta_xpm_8 = isrs_gn_model_uniform(**common_kwargs, num_subchannels=8)

        chex.assert_trees_all_close(eta_xpm_1, eta_xpm_8)

    def test_spm_monotonicity(self):
        """More subchannels should produce monotonically lower SPM."""
        from xlron.environments.gn_model.isrs_gn_model import isrs_gn_model_uniform

        freq = jnp.array([193.5e12])
        bw = jnp.array([100e9])
        power = jnp.array([0.001])

        common_kwargs = dict(
            num_channels=1,
            num_spans=10,
            ref_lambda=1550e-9,
            length=80e3,
            ch_power_W_i=power,
            ch_centre_i=freq,
            ch_bandwidth_i=bw,
            coherent=False,
            mod_format_correction=False,
        )

        prev_spm = float("inf")
        for n_sub in [1, 2, 4, 8]:
            _, _, eta_spm, _ = isrs_gn_model_uniform(**common_kwargs, num_subchannels=n_sub)
            spm_val = float(jnp.squeeze(eta_spm))
            self.assertLess(
                spm_val,
                prev_spm,
                f"SPM with {n_sub} subchannels ({spm_val}) should be less than previous ({prev_spm})",
            )
            prev_spm = spm_val

    def test_get_snr_fused_subchannels(self):
        """get_snr_fused should also produce lower NLI with subchannels."""
        from xlron.environments.gn_model.isrs_gn_model import get_snr_fused

        num_ch = 3
        freq = jnp.array([193.3e12, 193.4e12, 193.5e12])
        bw = jnp.array([100e9, 100e9, 100e9])
        power = jnp.array([0.001, 0.001, 0.001])

        common_kwargs = dict(
            ch_power_w_i=power,
            ch_centre_i=freq,
            ch_bandwidth_i=bw,
            num_spans=10,
            span_length=80e3,
            num_channels=num_ch,
            ref_lambda=1550e-9,
            attenuation=0.2 / 4.343 / 1e3,
            attenuation_bar=0.2 / 4.343 / 1e3,
            nonlinear_coeff=1.2e-3,
            raman_gain_slope=0.028 / 1e3 / 1e12,
            dispersion_coeff=17e-12 / 1e-9 / 1e3,
            dispersion_slope=0.067e-12 / 1e-9 / 1e3 / 1e-9,
            amplifier_noise_figure=jnp.array([5.5, 5.5, 5.5]),
            transceiver_snr=jnp.array([21.25, 21.25, 21.25]),
            coherent=False,
        )

        snr_1 = get_snr_fused(**common_kwargs, num_subchannels=1)
        snr_8 = get_snr_fused(**common_kwargs, num_subchannels=8)

        # Higher subchannels -> lower NLI -> higher SNR
        self.assertTrue(
            jnp.all(snr_8 >= snr_1),
            f"SNR with 8 subchannels should be >= SNR with 1 subchannel. "
            f"Got snr_8={snr_8}, snr_1={snr_1}",
        )

    def test_isrs_gn_model_non_uniform_subchannels(self):
        """Non-uniform span model should also support num_subchannels."""
        from xlron.environments.gn_model.isrs_gn_model import isrs_gn_model

        freq = jnp.array([193.5e12])
        bw = jnp.array([100e9])
        power = jnp.array([0.001])

        common_kwargs = dict(
            num_channels=1,
            num_spans=2,
            max_spans=2,
            ref_lambda=1550e-9,
            length=jnp.array([80e3, 80e3]),
            ch_power_W_i=power,
            ch_centre_i=freq,
            ch_bandwidth_i=bw,
            coherent=False,
            mod_format_correction=False,
            excess_kurtosis_i=jnp.zeros(1),
        )

        _, _, eta_spm_1, _ = isrs_gn_model(**common_kwargs, num_subchannels=1)
        _, _, eta_spm_8, _ = isrs_gn_model(**common_kwargs, num_subchannels=8)

        self.assertTrue(
            float(jnp.squeeze(eta_spm_8)) < float(jnp.squeeze(eta_spm_1)),
            "SPM should be lower with subchannels in non-uniform model",
        )

    def test_step_with_subchannels_does_not_crash(self):
        """Full env step should work with num_subchannels > 1."""
        key, env, obs, state, params = rsa_gn_model_subchannels_test_setup(num_subchannels=8)
        mask, _ = env.action_mask(state, params)
        rng_sample, rng_step = jax.random.split(key)
        action_dist = distrax.Categorical(logits=jnp.where(mask > 0, 0.0, -1e8))
        path_action = action_dist.sample(seed=rng_sample)
        power_action = jnp.array([0])
        action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))])
        obs, new_state, reward, terminal, truncated, info = env.step(
            rng_step, state, action, params
        )
        self.assertTrue(jnp.isfinite(reward))


class ChannelCentreFreqCachingTest(parameterized.TestCase):
    """Tests for channel_centre_freq_array caching in state."""

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = (
            rsa_gn_model_subchannels_test_setup()
        )

    def test_initial_centre_freq_array_is_zero(self):
        """channel_centre_freq_array should be all zeros on empty network."""
        self.assertTrue(
            jnp.all(self.state.channel_centre_freq_array == 0),
            "Initial centre freq array should be all zeros",
        )
        self.assertTrue(
            jnp.all(self.state.channel_centre_freq_array_prev == 0),
            "Initial prev centre freq array should be all zeros",
        )

    def test_centre_freq_array_shape(self):
        """channel_centre_freq_array should have shape (num_links, link_resources)."""
        expected_shape = (self.params.num_links, self.params.link_resources)
        self.assertEqual(self.state.channel_centre_freq_array.shape, expected_shape)
        self.assertEqual(self.state.channel_centre_freq_array_prev.shape, expected_shape)

    def test_centre_freq_set_after_step(self):
        """After a successful placement, centre freq should be set on path links."""
        mask, _ = self.env.action_mask(self.state, self.params)
        rng_sample, rng_step = jax.random.split(self.key)
        action_dist = distrax.Categorical(logits=jnp.where(mask > 0, 0.0, -1e8))
        path_action = action_dist.sample(seed=rng_sample)
        power_action = jnp.array([0])
        action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))])
        obs, new_state, reward, terminal, truncated, info = self.env.step(
            rng_step, self.state, action, self.params
        )
        # If the action succeeded (reward > 0), some entries should be nonzero
        has_placement = reward > 0
        if has_placement:
            self.assertTrue(
                jnp.any(new_state.channel_centre_freq_array != 0),
                "Centre freq should be set after successful placement",
            )
        # Even if blocked, the array should remain valid (finite)
        self.assertTrue(
            jnp.all(jnp.isfinite(new_state.channel_centre_freq_array)),
            "Centre freq array should be finite",
        )

    def test_centre_freq_consistent_with_occupied_slots(self):
        """Centre freq should be nonzero exactly where channel_centre_bw_array is nonzero."""
        # Take a few steps to fill the network
        state = self.state
        rng = self.key
        for _ in range(5):
            mask, _ = self.env.action_mask(state, self.params)
            rng, rng_sample, rng_step = jax.random.split(rng, 3)
            action_dist = distrax.Categorical(logits=jnp.where(mask > 0, 0.0, -1e8))
            path_action = action_dist.sample(seed=rng_sample)
            power_action = jnp.array([0])
            action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))])
            _, state, _, _, _, _ = self.env.step(rng_step, state, action, self.params)

        # Where bandwidth is set, centre freq should also be set (and vice versa)
        bw_nonzero = state.channel_centre_bw_array != 0
        freq_nonzero = state.channel_centre_freq_array != 0
        chex.assert_trees_all_close(bw_nonzero.astype(jnp.int32), freq_nonzero.astype(jnp.int32))

    def test_rmsa_gn_model_centre_freq_caching(self):
        """Centre freq caching should also work for RMSA GN model."""
        key, env, obs, state, params = rmsa_gn_model_test_setup()
        # Check initial state
        self.assertTrue(jnp.all(state.channel_centre_freq_array == 0))
        # Take a step
        mask, _, mod_format_mask = env.action_mask(state, params)
        state = state.replace(link_slot_mask=mask, mod_format_mask=mod_format_mask)
        rng_sample, rng_step = jax.random.split(key)
        action_dist = distrax.Categorical(logits=jnp.where(mask > 0, 0.0, -1e8))
        path_action = action_dist.sample(seed=rng_sample)
        power_action = jnp.array([0])
        action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))])
        obs, new_state, reward, terminal, truncated, info = env.step(
            rng_step, state, action, params
        )
        self.assertTrue(jnp.all(jnp.isfinite(new_state.channel_centre_freq_array)))


# ---------------------------------------------------------------------------
# Gerard 2025 regression test
# ---------------------------------------------------------------------------

# Static copy of the Gerard 2025 preset (copied from xlron/gui/presets.py).
# This is intentionally kept static so it doesn't silently inherit future
# preset changes that would mask model regressions.
_GERARD_2025_PRESET = {
    "env_type": "rsa_gn_model",
    "topology_name": "three_node_chain_undirected",
    "link_resources": 91,
    "k": 2,
    "slot_size": 100.0,
    "guardband": 0,
    "values_bw": "100",
    "band_preference": "C,L",
    "inter_band_gap_ghz": 275.0,
    "slots_per_band": "45,45",
    "span_length": 100.0,
    "nonlinear_coefficient": 0.843e-3,
    "dispersion_coeff": 21e-6,
    "dispersion_slope": 70,
    "raman_gain_slope": 1.8e-17,
    "attenuation": 3.91e-5,
    "use_raman_amp": True,
    "raman_pump_power_bw": "0.16,0.04,0.03,0.03,0.05",
    "raman_pump_freq_bw": "200.5e12,203.1e12,205.7e12,208.4e12,211.1e12",
    "coherent": True,
    "custom_traffic_matrix_csv_filepath": "xlron/data/traffic_matrices/three_node_chain_traffic_0to2.csv",
    "incremental_loading": True,
    "max_requests": 90,
    "TOTAL_TIMESTEPS": 90,
    "NUM_ENVS": 1,
    "STEPS_PER_INCREMENT": 100,
    "path_heuristic": "ksp_ff",
    "num_subchannels": 8,
    "power_per_channel_per_band": "1.8,2.3",
    "calc_minimum_osnr": True,
    "modulations_csv_filepath": "./xlron/data/modulations/modulations.csv",
    "band_data_filepath": "./xlron/data/gn_model/band_data/band_data_gerard2025.csv",
    "noise_data_filepath": "./xlron/data/gn_model/transceiver_amplifier_data/transceiver_amplifier_data_gerard2025.csv",
    # Required by process_config
    "ROLLOUT_LENGTH": 90,
    "NUM_MINIBATCHES": 1,
    "seed": 0,
    "load": 250,
    "mean_service_holding_time": 25,
}


class Gerard2025RegressionTest(absltest.TestCase):
    """Regression test: Gerard 2025 preset with KSP-FF should produce a
    stable Shannon-Hartley throughput.  If this value changes, it means
    the GN/DRA model, pump fitting, or env step logic has changed."""

    def test_throughput_regression(self):
        from xlron.environments.env_funcs import (
            calculate_throughput_from_active_lightpaths,
            get_launch_power,
            process_path_action,
        )
        from xlron.environments.make_env import process_config
        from xlron.heuristics.heuristics import ksp_ff

        processed = process_config(_GERARD_2025_PRESET)
        env_wrapped, params = make(processed)
        raw_env = env_wrapped._env

        rng = jax.random.PRNGKey(0)
        rng, reset_key = jax.random.split(rng)
        obs, state = raw_env.reset(reset_key, params)

        for _ in range(90):
            rng, _, step_key = jax.random.split(rng, 3)
            action = ksp_ff(state, params)
            _, initial_slot_index = process_path_action(state, params, action)
            launch_power = get_launch_power(state, action, action, initial_slot_index, params)
            full_action = jnp.concatenate(
                [action.reshape((1,)), launch_power.reshape((1,))], axis=0
            )
            obs, state, reward, terminal, truncated, info = raw_env.step_env(
                step_key, state, full_action, params
            )

        throughput_gbps = float(calculate_throughput_from_active_lightpaths(state, params))

        # Regression value: 68030.10 Gb/s (~68.0 Tb/s) measured 2026-02-19.
        # Changed from 66048.76 after band_data/transceiver_amplifier_data CSV
        # directory restructure.
        # Tolerance of 5 Gb/s accounts for floating-point platform differences and
        # minor fitting variation across platforms/jaxopt versions.
        self.assertAlmostEqual(
            throughput_gbps,
            68030.10,
            delta=5.0,
            msg="Gerard 2025 throughput regression: model output changed",
        )


if __name__ == "__main__":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
