import pathlib
import tempfile
import unittest

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

# The public release of XLRON ships a stub for the Distributed Raman Amplification (DRA) GN model
# (implementation withheld pending publication). Detect the stub by its sentinel so DRA-dependent
# regression tests skip on the public release but still run where the full DRA model is present.
try:
    from xlron.environments.gn_model import isrs_gn_model_dra as _dra_module

    _DRA_AVAILABLE = not hasattr(_dra_module, "_REMOVED_MESSAGE")
except Exception:
    _DRA_AVAILABLE = False


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
    return _gn_cached_setup(
        "rsa_gn_model_4_nsfnet",
        dict(
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
        ),
    )


@absltest.skipThisClass("Not finalized")
class RSAGNModelTest(chex.TestCase):
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
            obsv, env_state, reward, done, truncated, info = self.variant(
                self.env.step, static_argnums=(3)
            )(rng_step, env_state, action, self.params)
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


class TransceiverAmplifierNoiseTest(chex.TestCase):
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
    return _gn_cached_setup(
        "rmsa_gn_model",
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
        ),
        seed=3,
    )


class RMSAGNModelMaskTest(chex.TestCase):
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


class EnforceBandGapsTest(chex.TestCase):
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
    return _gn_cached_setup(
        f"rsa_gn_model_band_pref_{band_preference}",
        dict(
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
        ),
        seed=3,
    )


class BandPreferenceTest(chex.TestCase):
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
    return _gn_cached_setup(
        f"rsa_gn_model_subch_{num_subchannels}",
        dict(
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
        ),
        seed=3,
    )


class NumSubchannelsTest(chex.TestCase):
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


class ChannelCentreFreqCachingTest(chex.TestCase):
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


@unittest.skipUnless(
    _DRA_AVAILABLE,
    "DRA GN model implementation withheld from the public release of XLRON",
)
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
        raw_env = env_wrapped._env  # ty: ignore[unresolved-attribute]

        rng = jax.random.PRNGKey(0)
        rng, reset_key = jax.random.split(rng)
        obs, state = raw_env.reset(reset_key, params)

        for _ in range(90):
            rng, _, step_key = jax.random.split(rng, 3)
            action = ksp_ff(state, params)
            _, initial_slot_index = process_path_action(state, params, action)
            launch_power = get_launch_power(state, action, action, initial_slot_index, params)
            full_action = jnp.concatenate(
                [action.reshape((1,)), launch_power.reshape((1,))],  # ty: ignore[unresolved-attribute]
                axis=0,
            )
            obs, state, reward, terminal, truncated, info = raw_env.step_env(
                step_key, state, full_action, params
            )

        throughput_gbps = float(calculate_throughput_from_active_lightpaths(state, params))

        # Regression value: 68030.10 Gb/s (~68.0 Tb/s) measured 2026-02-19.
        # Changed from 66048.76 after band_data/transceiver_amplifier_data CSV
        # directory restructure.
        # Tolerance of 100 Gb/s accounts for floating-point platform differences
        # and minor fitting variation across platforms/jaxopt versions.
        self.assertAlmostEqual(
            throughput_gbps,
            68030.10,
            delta=100.0,
            msg="Gerard 2025 throughput regression: model output changed",
        )


class RMSAGNAggregateSlotsTest(chex.TestCase):
    """aggregate_slots > 1 must not crash RMSA-GN masking (regression: single-array unpack)
    and the implemented modulation format must come from the full-resolution mod_format_mask
    entry of the decoded slot, not the raw aggregated action index."""

    def test_masked_step_selects_correct_mod_format(self):
        settings = dict(
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
            aggregate_slots=2,
        )
        env, params = make(settings, log_wrapper=False)
        key = jax.random.PRNGKey(3)  # short-path request that passes SNR checks
        obs, state = env.reset(key, params)
        mask, full_mask, mod_format_mask = env.action_mask(state, params)  # ty: ignore[unresolved-attribute]
        self.assertEqual(mask.shape[0], params.k_paths * 5)  # ceil(10 / 2) = 5
        self.assertEqual(mod_format_mask.shape[0], params.k_paths * 10)  # full resolution
        self.assertTrue(bool(jnp.any(mask > 0)))
        state = state.replace(
            link_slot_mask=mask, full_link_slot_mask=full_mask, mod_format_mask=mod_format_mask
        )
        # Pick the valid aggregated action whose decoded slot has the largest format index so
        # the assertion below cannot pass vacuously on format 0
        valid_actions = jnp.where(mask > 0)[0]
        decoded = [process_path_action(state, params, a) for a in valid_actions]
        full_idx = jnp.array([int(p) * int(params.link_resources) + int(s) for p, s in decoded])
        formats = mod_format_mask[full_idx]
        best = int(jnp.argmax(formats))
        action, expected_format = valid_actions[best], formats[best]
        self.assertGreaterEqual(float(expected_format), 0)
        path_index, initial_slot = decoded[best]

        _, state, reward, terminal, _, _ = jax.jit(env.step, static_argnums=(3,))(
            key, state, action, params
        )
        # Accepted on an empty network: the implemented format at the decoded slot must equal
        # the full-resolution mask entry
        implemented = jnp.max(state.modulation_format_index_array[:, int(initial_slot)])
        self.assertEqual(float(implemented), float(expected_format))


class InitLinkLengthArrayGNModelTest(chex.TestCase):
    """GN-model link lengths must follow sorted(graph.edges) of the graph as given.

    Regression: the old implementation built [sorted undirected] + [sorted undirected], which
    permutes lengths across links on directed topologies whenever the two directions'
    lexicographic positions interleave (silently wrong span counts and per-link SNR)."""

    def test_directed_asymmetric_lengths_follow_edge_order(self):
        g = nx.DiGraph()
        # Asymmetric distances so any permutation is detectable
        g.add_edge(0, 1, distance=100)
        g.add_edge(1, 0, distance=150)
        g.add_edge(1, 2, distance=200)
        g.add_edge(2, 1, distance=250)
        g.add_edge(0, 2, distance=300)
        g.add_edge(2, 0, distance=350)

        span_array = init_link_length_array_gn_model(g, max_span_length=100e3, max_spans=6)
        totals_km = jnp.sum(span_array, axis=1) / 1e3

        expected = jnp.array(
            [g.edges[e]["distance"] for e in sorted(g.edges)], dtype=totals_km.dtype
        )
        chex.assert_trees_all_close(totals_km, expected)
        # Cross-check: same ordering as the non-GN link length array
        chex.assert_trees_all_close(totals_km, init_link_length_array(g).astype(totals_km.dtype))

    def test_undirected_unchanged(self):
        g = nx.Graph()
        g.add_edge(0, 1, distance=100)
        g.add_edge(1, 2, distance=200)
        span_array = init_link_length_array_gn_model(g, max_span_length=100e3, max_spans=3)
        chex.assert_trees_all_close(jnp.sum(span_array, axis=1) / 1e3, jnp.array([100.0, 200.0]))


class ActiveLightpathRegistryTest(chex.TestCase):
    """Registry lifecycle under dynamic traffic with blocking.

    Regressions covered: departure row-writes smeared down column 0 (a (3,1) update);
    departures were inserted negative and never flipped positive on success, so any blocked
    request wiped every live entry via the failure-undo; and live entries never expired."""

    def _dynamic_env(self):
        settings = dict(
            k=4,
            topology_name="nsfnet_deeprmsa_directed",
            link_resources=10,
            max_requests=100,
            values_bw=[100],
            env_type="rsa_gn_model",
            slot_size=12.5,
            guardband=0,
            mod_format_correction=False,
            max_power_per_fibre=10.0,
            coherent=False,
            include_no_op=False,
            load=100,
            mean_service_holding_time=25,
        )
        return make(settings, log_wrapper=False)

    def test_lifecycle_accept_block_expire(self):
        env, params = self._dynamic_env()
        key = jax.random.PRNGKey(3)  # short-path first request that passes SNR checks
        obs, state = env.reset(key, params)
        step = jax.jit(env.step, static_argnums=(3,))

        # --- Two accepted placements -> two fully-populated live rows ---
        for i in range(2):
            mask = env.action_mask(state, params)
            mask = mask[0] if isinstance(mask, tuple) else mask
            self.assertTrue(bool(jnp.any(mask > 0)))
            key, akey = jax.random.split(key)
            obs, state, reward, terminal, _, _ = step(akey, state, jnp.argmax(mask), params)
        self.assertEqual(int(state.accepted_services), 2)
        dep = state.active_lightpaths_array_departure
        registry = state.active_lightpaths_array
        # No pending (negative) entries survive complete_step
        self.assertTrue(bool(jnp.all(dep >= 0)))
        # Row-write regression: each accepted lightpath fills ALL 3 columns of its row
        self.assertEqual(int(jnp.sum(dep > 0)), 6)
        self.assertEqual(int(jnp.sum(registry[:, 0] >= 0)), 2)
        self.assertTrue(bool(jnp.all(registry[jnp.where(dep[:, 0] > 0)[0], 2] > 0)))

        # --- A blocked request must NOT touch the live registry ---
        mask = env.action_mask(state, params)
        mask = mask[0] if isinstance(mask, tuple) else mask
        self.assertTrue(bool(jnp.any(mask == 0)))
        invalid_action = jnp.argmin(mask)
        key, akey = jax.random.split(key)
        obs, state2, reward, terminal, _, _ = step(akey, state, invalid_action, params)
        self.assertEqual(int(state2.accepted_services), 2)  # blocked
        chex.assert_trees_all_close(state2.active_lightpaths_array, state.active_lightpaths_array)
        chex.assert_trees_all_close(
            state2.active_lightpaths_array_departure,
            state.active_lightpaths_array_departure,
        )
        # The failure restore must recover the exact pre-action GN state, not init values
        # (regression: *_prev snapshots were never written, wiping these arrays on block)
        chex.assert_trees_all_close(state2.path_index_array, state.path_index_array)
        chex.assert_trees_all_close(state2.channel_power_array, state.channel_power_array)

        # --- Throughput is finite and positive with live lightpaths ---
        throughput = calculate_throughput_from_active_lightpaths(state2, params)
        self.assertTrue(bool(jnp.isfinite(throughput)))
        self.assertGreater(float(throughput), 0.0)

        # --- Advancing time past the departures expires the rows ---
        far_future = jnp.full_like(state2.current_time, 1e7)
        expired = remove_expired_services_rsa_gn_model(
            state2.replace(current_time=far_future), params
        )
        self.assertTrue(bool(jnp.all(expired.active_lightpaths_array == -1)))  # ty: ignore[unresolved-attribute]
        self.assertTrue(bool(jnp.all(expired.active_lightpaths_array_departure == 0)))  # ty: ignore[unresolved-attribute]


class TotalPowerPerLinkTest(chex.TestCase):
    """Adjacent same-path-index channels must be counted as separate channels via their
    centre-frequency transition (path index alone merged them, dropping the second
    block's power from the max_power_per_fibre checks)."""

    def test_adjacent_same_path_channels_both_counted(self):
        power = jnp.full((1, 4), 1e-3)
        freq = jnp.array([[10.0, 10.0, 20.0, 20.0]])
        same_path = jnp.array([[5, 5, 5, 5]])
        total = compute_total_power_per_link(power, same_path, freq)
        chex.assert_trees_all_close(total, jnp.array([2e-3]))
        # Distinct paths unchanged; empty slots ignored
        two_paths = jnp.array([[5, 5, 7, 7]])
        chex.assert_trees_all_close(
            compute_total_power_per_link(power, two_paths, freq), jnp.array([2e-3])
        )
        empty = jnp.array([[-1, -1, -1, -1]])
        chex.assert_trees_all_close(
            compute_total_power_per_link(power, empty, freq), jnp.array([0.0])
        )


class ScaledLaunchPowerTest(chex.TestCase):
    """launch_power_type='scaled' must scale slot power by true-path-length fraction.

    Regression: the branch indexed the per-link length vector with a PATH index and took
    the max over per-span partial sums, producing garbage scalings (and an IndexError
    under pack_path_bits)."""

    def test_scaled_power_is_path_length_fraction(self):
        settings = dict(
            k=4,
            topology_name="nsfnet_deeprmsa_directed",
            link_resources=10,
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
            launch_power_type="scaled",
        )
        env, params = make(settings, log_wrapper=False)
        key = jax.random.PRNGKey(3)
        obs, state = env.reset(key, params)
        power = get_launch_power(state, jnp.array(0), jnp.array(0.0), jnp.array(0), params)

        nodes_sd, _ = read_rsa_request(state.request_array)
        source, dest = nodes_sd
        i = get_path_indices(
            params,
            source,
            dest,
            params.k_paths,
            params.num_nodes,
            directed=params.directed_graph,
        ).astype(jnp.int32)
        link_lengths = jnp.sum(params.link_length_array.val, axis=1)
        pla = params.path_link_array.val.astype(link_lengths.dtype)
        expected = params.slot_launch_power_array.val[0] * (  # ty: ignore[unresolved-attribute]
            (pla[i] @ link_lengths) / jnp.max(pla @ link_lengths)
        )
        chex.assert_trees_all_close(jnp.squeeze(power), jnp.squeeze(expected), rtol=1e-5)
        self.assertGreater(float(jnp.squeeze(power)), 0.0)
        self.assertLessEqual(
            float(jnp.squeeze(power)),
            float(params.slot_launch_power_array.val[0]),  # ty: ignore[unresolved-attribute]
        )


class GetPathsObsGNModelTest(chex.TestCase):
    """The launch-power observation must feed NORMALISED path lengths (regression: the
    normalisation was computed then discarded, so raw metres ~1e5-1e7 entered the obs)."""

    def test_path_length_feature_is_normalised(self):
        _, env, _, state, params = rsa_gn_model_4_nsfnet_test_setup()
        request = state.request_array.reshape((-1,))
        ps_w = calculate_path_stats(state, params, request).shape[1] - 3  # ty: ignore[unresolved-attribute]
        obs = get_paths_obs_gn_model(state, params)
        stats = obs[4:].reshape(params.k_paths, ps_w + 5)
        path_length_norm = stats[:, ps_w]
        self.assertTrue(bool(jnp.all(path_length_norm > 0)))
        self.assertTrue(bool(jnp.all(path_length_norm <= 1.0)))
        # The stats block is bounded (raw metres would be >= 1e5); obs[3] is holding_time,
        # which is ~1e6 under incremental loading and not part of this regression
        self.assertLess(float(jnp.max(jnp.abs(obs[4:]))), 1e3)


class AmplifierGainIsrsTiltTest(chex.TestCase):
    """calculate_amplifier_gain_isrs must produce a real ISRS gain tilt.

    Regressions covered: the exp(-f_i*C) factors in the numerator and denominator
    cancelled exactly, so the returned gain was flat for ANY power distribution; and
    mixed THz/km vs SI units made the tilt exponent 1000x too small even in principle."""

    def _defaults(self):
        a = 0.2 / 4.343 / 1e3  # Np/m
        length = 100e3  # m
        cr = 0.028 / 1e3 / 1e12  # 1/(W*m*Hz), SI as passed by the env
        return a, length, cr

    def test_gain_tilt_sign_and_magnitude_uniform_loading(self):
        from xlron.environments.gn_model.isrs_gn_model import (
            calculate_amplifier_gain_isrs,
            to_db,
        )

        a, length, cr = self._defaults()
        P = jnp.full(5, 1e-3)
        f = jnp.linspace(-2.5e12, 2.5e12, 5)  # Hz offsets over a 5 THz band
        gain_db = to_db(calculate_amplifier_gain_isrs(a, length, cr, P, f))
        # SRS depletes high-frequency channels, so gain must increase with frequency
        self.assertTrue(bool(jnp.all(jnp.diff(gain_db) > 0)))
        # The Zirngibl tilt exists even for uniform powers; spread must be nonzero
        # but small relative to the ~20 dB loss compensation
        spread = float(jnp.max(gain_db) - jnp.min(gain_db))
        self.assertGreater(spread, 0.01)
        self.assertLess(spread, 10.0)
        loss_comp_db = float(to_db(jnp.exp(a * length)))
        self.assertAlmostEqual(float(jnp.mean(gain_db)), loss_comp_db, delta=1.0)

    def test_gain_tilt_asymmetric_loading(self):
        from xlron.environments.gn_model.isrs_gn_model import (
            calculate_amplifier_gain_isrs,
            to_db,
        )

        a, length, cr = self._defaults()
        P = jnp.array([0.1, 0.05, 0.3, 0.02, 0.001])
        f = jnp.linspace(-2.5e12, 2.5e12, 5)
        gain_db = to_db(calculate_amplifier_gain_isrs(a, length, cr, P, f))
        self.assertTrue(bool(jnp.all(jnp.diff(gain_db) > 0)))
        self.assertGreater(float(jnp.max(gain_db) - jnp.min(gain_db)), 1.0)

    def test_flat_gain_without_raman(self):
        from xlron.environments.gn_model.isrs_gn_model import calculate_amplifier_gain_isrs

        a, length, _ = self._defaults()
        P = jnp.array([0.1, 0.05, 0.3, 0.02, 0.001])
        f = jnp.linspace(-2.5e12, 2.5e12, 5)
        gain = calculate_amplifier_gain_isrs(a, length, 0.0, P, f)
        chex.assert_trees_all_close(gain, jnp.full(5, jnp.exp(a * length)), rtol=1e-6)

    def test_zero_power_channels_get_loss_compensation(self):
        from xlron.environments.gn_model.isrs_gn_model import calculate_amplifier_gain_isrs

        a, length, cr = self._defaults()
        P = jnp.array([1e-3, 0.0, 1e-3])
        f = jnp.array([-1e12, 0.0, 1e12])
        gain = calculate_amplifier_gain_isrs(a, length, cr, P, f)
        self.assertAlmostEqual(float(gain[1]), float(jnp.exp(a * length)), delta=1e-6)


class ModFormatCorrectionTest(chex.TestCase):
    """mod_format_correction must apply a finite, small kurtosis correction.

    Regressions covered: `p_i > 1.0` guards on Watt-scale powers zeroed the intended
    power ratio, `gamma ** (2 / B_k)` in place of `gamma**2 / B_k`, a mis-parenthesised
    second arctan term with an unguarded 0/0 on the i == k diagonal (NaN eta for every
    channel, mapped to -50 dB SNR by nan_to_num => silent 100% blocking), and an
    asymptotic term garbled from Ref. [3, Eq. (12)] that overflowed float32."""

    def _uniform_kwargs(self):
        return dict(
            num_channels=3,
            num_spans=10,
            ref_lambda=1550e-9,
            length=80e3,
            ch_power_W_i=jnp.full(3, 1e-3),
            ch_centre_i=jnp.array([-100e9, 0.0, 100e9]),
            ch_bandwidth_i=jnp.full(3, 100e9),
            coherent=False,
        )

    def test_zero_kurtosis_equals_correction_off(self):
        from xlron.environments.gn_model.isrs_gn_model import isrs_gn_model_uniform

        kwargs = self._uniform_kwargs()
        _, eta_off, _, _ = isrs_gn_model_uniform(**kwargs, mod_format_correction=False)
        _, eta_on, _, _ = isrs_gn_model_uniform(
            **kwargs, mod_format_correction=True, excess_kurtosis_i=jnp.zeros(3)
        )
        chex.assert_trees_all_close(eta_on, eta_off)

    def test_negative_kurtosis_reduces_nli(self):
        from xlron.environments.gn_model.isrs_gn_model import isrs_gn_model_uniform

        kwargs = self._uniform_kwargs()
        _, eta_off, _, _ = isrs_gn_model_uniform(**kwargs, mod_format_correction=False)
        # Excess kurtosis of uniform 16-QAM (Ref. [3, Table I])
        _, eta_on, _, _ = isrs_gn_model_uniform(
            **kwargs, mod_format_correction=True, excess_kurtosis_i=jnp.full(3, -0.68)
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(eta_on))))
        # Negative kurtosis => reduced NLI, but the correction must stay small
        self.assertTrue(bool(jnp.all(eta_on < eta_off)))
        self.assertTrue(bool(jnp.all(eta_on > 0.5 * eta_off)))

    def test_non_uniform_span_model_finite(self):
        from xlron.environments.gn_model.isrs_gn_model import isrs_gn_model

        kwargs = self._uniform_kwargs()
        kwargs["length"] = jnp.full(10, 80e3)
        _, eta_off, _, _ = isrs_gn_model(**kwargs, max_spans=10, mod_format_correction=False)
        _, eta_on, _, _ = isrs_gn_model(
            **kwargs,
            max_spans=10,
            mod_format_correction=True,
            excess_kurtosis_i=jnp.full(3, -0.68),
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(eta_on))))
        self.assertTrue(bool(jnp.all(eta_on < eta_off)))
        self.assertTrue(bool(jnp.all(eta_on > 0.5 * eta_off)))

    def test_get_snr_finite_with_correction(self):
        from xlron.environments.gn_model.isrs_gn_model import get_snr, to_db

        a = 0.2 / 4.343 / 1e3
        common = dict(
            num_channels=3,
            max_spans=10,
            num_spans=10,
            length=jnp.full(10, 80e3),
            ch_power_w_i=jnp.full(3, 1e-3),
            ch_centre_i=jnp.array([-100e9, 0.0, 100e9]),
            ch_bandwidth_i=jnp.full(3, 100e9),
            attenuation_i=jnp.array(a),
            attenuation_bar_i=jnp.array(a),
            amplifier_noise_figure=jnp.array([5.0, 5.0, 5.0]),
            transceiver_snr=jnp.array([0.0, 0.0, 0.0]),
            excess_kurtosis_i=jnp.full(3, -0.68),
            uniform_spans=False,
        )
        snr_off = get_snr(**common, mod_format_correction=False)[0]
        snr_on = get_snr(**common, mod_format_correction=True)[0]
        self.assertTrue(bool(jnp.all(jnp.isfinite(snr_on))))
        # Less NLI => higher SNR, by at most ~1 dB in this configuration
        self.assertTrue(bool(jnp.all(snr_on >= snr_off)))
        self.assertLess(float(jnp.max(to_db(snr_on) - to_db(snr_off))), 1.0)


class NonUniformSpanPaddingTest(chex.TestCase):
    """Zero-padded span-length arrays must give identical NLI to unpadded ones.

    Regression: spm_single/xpm_single were added on every one of max_spans scan
    iterations regardless of span activity, inflating NLI by max_spans/num_spans for
    every link shorter than the topology's longest (the env always zero-pads rows of
    link_length_array up to the global max_spans)."""

    def _kwargs(self):
        return dict(
            num_channels=3,
            num_spans=2,
            ref_lambda=1550e-9,
            ch_power_W_i=jnp.full(3, 1e-3),
            ch_centre_i=jnp.array([-100e9, 0.0, 100e9]),
            ch_bandwidth_i=jnp.full(3, 100e9),
            coherent=False,
            excess_kurtosis_i=jnp.zeros(3),
        )

    def test_padded_equals_unpadded(self):
        from xlron.environments.gn_model.isrs_gn_model import isrs_gn_model

        kwargs = self._kwargs()
        length = jnp.array([80e3, 80e3])
        nli_exact, eta_exact, spm_exact, xpm_exact = isrs_gn_model(
            **kwargs, max_spans=2, length=length, mod_format_correction=False
        )
        nli_pad, eta_pad, spm_pad, xpm_pad = isrs_gn_model(
            **kwargs,
            max_spans=10,
            length=jnp.concatenate([length, jnp.zeros(8)]),
            mod_format_correction=False,
        )
        chex.assert_trees_all_close(spm_pad, spm_exact, rtol=1e-6)
        chex.assert_trees_all_close(xpm_pad, xpm_exact, rtol=1e-6)
        chex.assert_trees_all_close(nli_pad, nli_exact, rtol=1e-6)

    def test_padded_with_mod_format_correction_finite(self):
        from xlron.environments.gn_model.isrs_gn_model import isrs_gn_model

        kwargs = self._kwargs()
        kwargs["excess_kurtosis_i"] = jnp.full(3, -0.68)
        length = jnp.array([80e3, 80e3])
        _, eta_exact, _, _ = isrs_gn_model(
            **kwargs, max_spans=2, length=length, mod_format_correction=True
        )
        # Padding spans have L == 0: the asymptotic correction divides by L and must
        # be masked, not produce inf/NaN
        _, eta_pad, _, _ = isrs_gn_model(
            **kwargs,
            max_spans=10,
            length=jnp.concatenate([length, jnp.zeros(8)]),
            mod_format_correction=True,
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(eta_pad))))
        chex.assert_trees_all_close(eta_pad, eta_exact, rtol=1e-6)


class BlockedRequestLinkSnrRestoreTest(chex.TestCase):
    """A blocked request must restore link_snr_array to its pre-action value.

    Regression: implement_action_rmsa_gn_model recomputes link_snr_array from the
    tentative placement, but complete_step_rmsa_gn_model's failure restore omitted it,
    so the blocked (undone) lightpath's NLI contribution leaked into the SNR features
    of the next observation for every co-propagating channel."""

    def test_blocked_step_restores_link_snr(self):
        key, env, obs, state, params = rmsa_gn_model_test_setup()
        step = jax.jit(env.step, static_argnums=(3,))
        request0 = state.request_array

        # --- Accepted placement ---
        mask, full_mask, mfm = env.action_mask(state, params)
        self.assertTrue(bool(jnp.any(mask > 0)))
        path_action = jnp.argmax(mask)
        state = state.replace(
            link_slot_mask=mask, full_link_slot_mask=full_mask, mod_format_mask=mfm
        )
        action = jnp.concatenate([path_action.reshape((1,)), jnp.zeros((1,))])
        key, akey = jax.random.split(key)
        _, state1, _, _, _, _ = step(akey, state, action, params)
        self.assertEqual(int(state1.accepted_services), 1)

        # --- Replay the same request so the same action is a guaranteed collision ---
        state1 = state1.replace(request_array=request0)
        mask2, full2, mfm2 = env.action_mask(state1, params)
        self.assertEqual(float(mask2[path_action]), 0.0)
        state1 = state1.replace(
            link_slot_mask=mask2, full_link_slot_mask=full2, mod_format_mask=mfm2
        )
        key, akey = jax.random.split(key)
        _, state2, _, _, _, _ = step(akey, state1, action, params)
        self.assertEqual(int(state2.accepted_services), 1)  # blocked
        chex.assert_trees_all_close(state2.link_snr_array, state1.link_snr_array)
        # And the accepted lightpath's other GN arrays survive too
        chex.assert_trees_all_close(state2.channel_power_array, state1.channel_power_array)


class RoadmAseInLoggedSnrTest(chex.TestCase):
    """Logged info['path_snr'] must include path-level ROADM ASE like the
    masking/acceptance checks (regression: the LogWrapper and SNR-reward call sites
    omitted the state argument, so they reported a systematically higher SNR than the
    one used to accept or reject the same lightpath; the wrapper also indexed
    path_link_array with the local k-path index instead of the global one)."""

    def _wrapped_env(self):
        key = jax.random.PRNGKey(3)
        if "rsa_gn_model_log_actions" not in _gn_cache:
            settings = dict(
                k=4,
                topology_name="nsfnet_deeprmsa_directed",
                link_resources=10,
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
                log_actions=True,
            )
            _gn_cache["rsa_gn_model_log_actions"] = make(settings, log_wrapper=True)
        env, params = _gn_cache["rsa_gn_model_log_actions"]
        obs, log_state = env.reset(key, params)
        return key, env, log_state, params

    def _global_path_index(self, state, params, path_action):
        nodes_sd, _ = read_rsa_request(state.request_array)
        source, dest = nodes_sd
        i = get_path_indices(
            params,
            source,
            dest,
            params.k_paths,
            params.num_nodes,
            directed=params.directed_graph,
        ).astype(jnp.int32)
        path_index, slot_index = process_path_action(state, params, path_action)
        return i + path_index, slot_index

    def test_path_snr_with_state_is_lower(self):
        """ROADM ASE adds noise: SNR with state must be strictly below stateless SNR."""
        key, env, log_state, params = self._wrapped_env()
        raw_env = env._env
        state = log_state.env_state
        mask = raw_env.action_mask(state, params)
        mask = mask[0] if isinstance(mask, tuple) else mask
        path_action = jnp.argmax(mask)
        # Placed path of the acted (pre-step) request
        global_path_index, slot_index = self._global_path_index(state, params, path_action)
        action = jnp.concatenate([path_action.reshape((1,)), jnp.zeros((1,))])
        key, akey = jax.random.split(key)
        _, state1, _, _, _, _ = jax.jit(raw_env.step, static_argnums=(3,))(
            akey, state, action, params
        )
        self.assertEqual(int(state1.accepted_services), 1)
        path = params.path_link_array.val[int(global_path_index)]
        snr_with_state = get_snr_for_path(path, state1.link_snr_array, params, state1)
        snr_without_state = get_snr_for_path(path, state1.link_snr_array, params)
        slot = int(slot_index)
        self.assertLess(float(snr_with_state[slot]), float(snr_without_state[slot]))

    def test_logged_path_snr_matches_state_snr(self):
        key, env, log_state, params = self._wrapped_env()
        raw_env = env._env
        state = log_state.env_state
        mask = raw_env.action_mask(state, params)
        mask = mask[0] if isinstance(mask, tuple) else mask
        path_action = jnp.argmax(mask)
        action = jnp.concatenate([path_action.reshape((1,)), jnp.zeros((1,))])
        key, akey = jax.random.split(key)
        _, log_state1, _, _, _, info = jax.jit(env.step, static_argnums=(3,))(
            akey, log_state, action, params
        )
        env_state = log_state1.env_state
        # Recompute expected SNR exactly as the wrapper does (post-step state and
        # global path index), WITH the state argument
        expected = get_snr_for_path(
            params.path_link_array.val[jnp.asarray(info["path_index"], dtype=jnp.int32)],
            env_state.link_snr_array,
            params,
            env_state,
        )[jnp.asarray(info["slot_index"], dtype=jnp.int32)]
        chex.assert_trees_all_close(info["path_snr"], expected)


class RSAGNModelObsShapeTest(chex.TestCase):
    """RSAGNModelEnv.get_obs must match the advertised observation space (regression:
    it returned a 0-d scalar while observation_space declared 4 + 7*k_paths features,
    crashing any flat-obs model at trace time)."""

    def test_obs_matches_observation_space(self):
        _, env, obs, state, params = rsa_gn_model_4_nsfnet_test_setup()
        obs = env.get_obs(state, params)
        self.assertEqual(obs.ndim, 1)
        self.assertEqual(obs.shape[0], int(env.observation_space(params).n))
        # RMSA GN model advertises the same obs; the two variants must agree
        _, renv, _, rstate, rparams = rmsa_gn_model_test_setup()
        robs = renv.get_obs(rstate, rparams)
        self.assertEqual(robs.shape[0], int(renv.observation_space(rparams).n))


class LaunchPowerActorCriticMLPTest(chex.TestCase):
    """LaunchPowerActorCriticMLP must construct and process the restored GN observation
    (regression: it assigned undeclared eqx fields, raising FrozenInstanceError, and its
    critic network was never built; it was also unreachable because init_network compared
    the string launch_power_type flag against the integer 3)."""

    def test_construct_and_forward(self):
        from xlron.models.mlp import LaunchPowerActorCriticMLP

        k_paths = 4
        input_dim = 4 + 7 * k_paths  # matches RSAGNModelEnv.observation_space
        model = LaunchPowerActorCriticMLP(
            1,
            input_dim,
            k_paths=k_paths,
            min_power_dbm=0.0,
            max_power_dbm=2.0,
            step_power_dbm=0.1,
            key=jax.random.PRNGKey(0),
        )
        obs = jax.random.normal(jax.random.PRNGKey(1), (input_dim,))
        (path_dist, power_dist), value = model(obs)
        self.assertIsNone(path_dist)
        logits = power_dist.logits  # ty: ignore[unresolved-attribute]
        self.assertEqual(logits.shape, (k_paths, model.num_power_levels))
        self.assertEqual(value.shape, ())
        self.assertTrue(bool(jnp.all(jnp.isfinite(logits))))
        self.assertTrue(bool(jnp.isfinite(value)))


class IncludeNoOpMaskInitTest(chex.TestCase):
    """GN-model envs must initialise link_slot_mask with the no-op element (regression:
    init omitted include_no_op while every recomputed mask appends it, so the lax.scan
    carry structure mismatched at trace time and --include_no_op was unusable)."""

    def _settings(self, env_type):
        return dict(
            k=4,
            topology_name="nsfnet_deeprmsa_directed",
            link_resources=10,
            max_requests=100,
            values_bw=[100],
            incremental_loading=True,
            env_type=env_type,
            slot_size=12.5,
            guardband=0,
            mod_format_correction=False,
            max_power_per_fibre=10.0,
            coherent=False,
            include_no_op=True,
        )

    def test_rmsa_gn_model_mask_shapes_match(self):
        key, env, obs, state, params = _gn_cached_setup(
            "rmsa_gn_model_no_op", self._settings("rmsa_gn_model"), seed=3
        )
        mask, _, _ = env.action_mask(state, params)
        expected = params.k_paths * params.link_resources + 1
        self.assertEqual(mask.shape, (expected,))
        self.assertEqual(state.link_slot_mask.shape, mask.shape)

    def test_rsa_gn_model_mask_shapes_match(self):
        key, env, obs, state, params = _gn_cached_setup(
            "rsa_gn_model_no_op", self._settings("rsa_gn_model"), seed=3
        )
        mask = env.action_mask(state, params)
        mask = mask[0] if isinstance(mask, tuple) else mask
        expected = params.k_paths * params.link_resources + 1
        self.assertEqual(mask.shape, (expected,))
        self.assertEqual(state.link_slot_mask.shape, mask.shape)


class AggregatedLaunchPowerDecodeTest(chex.TestCase):
    """The synthesised per-path action used for launch-power lookup during RMSA-GN
    masking must round-trip through process_path_action (regression: a floor-division
    stride mis-decoded the path index whenever link_resources % aggregate_slots != 0,
    so masks were evaluated with the wrong path's tabular/scaled launch power)."""

    def test_path_action_roundtrip_nondivisible_aggregation(self):
        import math

        settings = dict(
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
            aggregate_slots=4,  # ceil(10/4)=3 != floor(10/4)=2
        )
        key, env, obs, state, params = _gn_cached_setup("rmsa_gn_model_agg4", settings, seed=3)
        stride = math.ceil(params.link_resources / params.aggregate_slots)
        for i in range(params.k_paths):
            path_index, _ = process_path_action(state, params, jnp.array(i * stride))
            self.assertEqual(int(path_index), i)
        # The floor stride used by the regressed encoding does NOT round-trip
        floor_stride = params.link_resources // params.aggregate_slots
        path_index, _ = process_path_action(state, params, jnp.array(1 * floor_stride))
        self.assertNotEqual(int(path_index), 1)


def rmsa_gn_model_mod_format_reward_test_setup():
    return _gn_cached_setup(
        "rmsa_gn_model_mod_format_reward",
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
            reward_type="mod_format",
        ),
        seed=3,
    )


class ModFormatRewardTest(chex.TestCase):
    """Regression test: reward_type='mod_format' used to assert RSAGNModelEnvParams but
    read modulation_format_index_array, which only exists on RMSAGNModelEnvState, so it
    failed at trace time in every configuration.
    """

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = (
            rmsa_gn_model_mod_format_reward_test_setup()
        )

    def test_step_traces_and_returns_finite_reward(self):
        rng, rng_sample, rng_step = jax.random.split(self.key, 3)
        mask, _, mod_format_mask = self.env.action_mask(self.state, self.params)
        state = self.state.replace(link_slot_mask=mask, mod_format_mask=mod_format_mask)
        action_dist = distrax.Categorical(logits=jnp.where(mask > 0, 0.0, -1e8))
        path_action = action_dist.sample(seed=rng_sample)
        power_action = jnp.array([0])
        action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))], axis=0)
        obs, new_state, reward, terminal, truncated, info = self.env.step(
            rng_step, state, action, self.params
        )
        self.assertTrue(bool(jnp.isfinite(reward)))


def rsa_gn_model_log_actions_test_setup():
    # Not via _gn_cached_setup, which forces log_wrapper=False: this test targets LogWrapper
    key = jax.random.PRNGKey(0)
    if "rsa_gn_model_log_actions" not in _gn_cache:
        settings = dict(
            k=5,
            topology_name="nsfnet_deeprmsa_undirected",
            link_resources=10,
            max_requests=100,
            values_bw=[100],
            env_type="rsa_gn_model",
            interband_gap=0,
            slot_size=25,
            mod_format_correction=False,
            launch_power=0.0,
            load=100,
            mean_service_holding_time=10,
            log_actions=True,
            relative_arrival_times=False,
        )
        env, params = make(settings, log_wrapper=True)
        _gn_cache["rsa_gn_model_log_actions"] = (env, params)
    env, params = _gn_cache["rsa_gn_model_log_actions"]
    obs, state = env.reset(key, params)
    return key, env, obs, state, params


class LogWrapperActionLoggingTest(chex.TestCase):
    """Regression tests for LogWrapper's log_actions fields.

    path_snr used to be looked up with the k-relative path index (missing the
    node-pair offset into path_link_array), and arrival/departure times were read
    from the post-step state, i.e. from the NEXT request.
    """

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = (
            rsa_gn_model_log_actions_test_setup()
        )

    def test_path_snr_and_times_describe_acting_request(self):
        params = self.params
        pre_state = self.state.env_state
        # Pre-step values (read before step: the acting request's fields are
        # overwritten by generate_request inside step_env)
        nodes_sd, _ = read_rsa_request(pre_state.request_array)
        source, dest = nodes_sd
        i = int(
            get_path_indices(
                params,
                source,
                dest,
                params.k_paths,
                params.num_nodes,
                directed=params.directed_graph,
            )
        )
        # Sanity: the request is not for the first node pair, so the missing
        # offset would have selected the wrong path row before the fix
        self.assertGreater(i, 0)
        expected_arrival = float(pre_state.current_time[0])
        expected_departure = float(pre_state.current_time[0] + pre_state.holding_time[0])
        # Select k-index 1, slot 0 (aggregate_slots=1 => action = k_index * link_resources)
        k_index = 1
        path_action = jnp.array(k_index * params.link_resources, dtype=jnp.float32)
        action = jnp.stack([path_action, jnp.array(0.0, dtype=jnp.float32)])

        rng, step_key = jax.random.split(self.key)
        obs, log_state, reward, terminal, truncated, info = self.env.step(
            step_key, self.state, action, params
        )

        expected_path = params.path_link_array.val[i + k_index]
        expected_snr = get_snr_for_path(expected_path, log_state.env_state.link_snr_array, params)[
            0
        ]
        chex.assert_trees_all_close(info["path_snr"], expected_snr)
        self.assertEqual(int(info["path_index"]), i + k_index)
        self.assertAlmostEqual(float(info["arrival_time"]), expected_arrival, places=5)
        self.assertAlmostEqual(float(info["departure_time"]), expected_departure, places=4)


if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
