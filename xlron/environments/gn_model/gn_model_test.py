import os
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4"
import distrax
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from xlron.environments.env_funcs import *
from xlron.environments.rsa import *
from xlron.environments.wrappers import *
from xlron.environments.dataclasses import *
from xlron.environments.make_env import make


def rsa_gn_model_4_nsfnet_test_setup():
    key = jax.random.PRNGKey(0)
    settings_rsa_gn_model_4_nsfnet = dict(
        k=5, topology_name="nsfnet_deeprmsa_undirected", link_resources=4, max_requests=10,
        values_bw=[100], incremental_loading=True, env_type="rsa_gn_model",
        interband_gap=0, slot_size=25, mod_format_correction=False, launch_power=0.0
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
        ("case_end_episode",
         jnp.array([[ 200.,  800.,  500.,    0.,],
                    [1100.,  700.,  700.,  700.,],
                    [   0.,    0.,  300.,    0.,],
                    [ 800.,  500.,  600.,  500.,],
                    [ 400.,  100.,  200.,    0.,],
                    [ 300.,    0.,    0.,    0.,],
                    [ 400.,  300.,  300.,  600.,],
                    [   0.,    0.,    0.,    0.,],
                    [ 400.,    0.,  200.,    0.,],
                    [ 600.,  500.,  900.,  900.,],
                    [ 400.,  300.,  200.,  400.,],
                    [   0.,    0.,  400.,  700.,],
                    [ 700.,  500.,  600.,  300.,],
                    [ 600.,  800.,  500.,  700.,],
                    [ 500.,  600.,  700.,  300.,],
                    [ 600.,  200.,  200.,  400.,],
                    [1100.,  900.,  700.,  900.,],
                    [1000.,  800.,  800.,  900.,],
                    [ 400.,  600.,  400.,  600.,],
                    [ 700.,  200.,  400.,  300.,],
                    [ 600.,  900.,  800., 1300.,],
                    [1100., 1200., 1300., 1500.,]])
         ),
    )
    def test_end_episode(self, expected):
        rng, reset_rng = jax.random.split(self.key)
        obsv, env_state = self.env.reset(reset_rng, self.params)
        reward = jnp.array([1.])
        i = 0
        while reward > 0:
            i += 1
            rng, rng_sample, rng_step = jax.random.split(rng, 3)
            # get mask
            env_state = self.env.action_mask(env_state, self.params)
            mask = env_state.link_slot_mask
            # make distribution
            action_dist = distrax.Categorical(logits=jnp.where(mask, mask, -1e8))
            #jax.debug.print("action dist {}", action_dist.logits, ordered=True)
            # sample distribution
            path_action = action_dist.sample(seed=rng_sample)
            path_index, slot_index = process_path_action(env_state, self.params, path_action)
            path = get_paths(self.params, read_rsa_request(env_state.request_array)[0])[path_index]
            jax.debug.print("---i--- {}", i, ordered=True)
            jax.debug.print("path {}", path, ordered=True)
            jax.debug.print("slot {}", slot_index, ordered=True)
            power_action = jnp.array([0])
            action = jnp.concatenate([path_action.reshape((1,)), power_action.reshape((1,))], axis=0)
            # step env
            obsv, env_state, reward, done, info = self.variant(self.env.step, static_argnums=(3))(
                rng_step, env_state, action, self.params
            )
            jax.debug.print("action mask {}", env_state.link_slot_mask, ordered=True)
            jax.debug.print("action {}", action, ordered=True)
            jax.debug.print("reward {}", reward, ordered=True)
            jax.debug.print("link snr array {}", env_state.link_snr_array, ordered=True)
            jax.debug.print("path_snr {}", get_snr_for_path(path, env_state.link_snr_array, self.params), ordered=True)
            jax.debug.print("link_slot_array {}", env_state.link_slot_array, ordered=True)
            jax.debug.print("path_index_array {}", env_state.path_index_array, ordered=True)
            jax.debug.print("channel_centre_bw_array {}", env_state.channel_centre_bw_array, ordered=True)
            jax.debug.print("channel_power_array {}", env_state.channel_power_array, ordered=True)
            jax.debug.print("modulation_format_index_array {}", env_state.modulation_format_index_array, ordered=True)
            jax.debug.print("-----END-----")
            jax.debug.print("request_array {}", env_state.request_array, ordered=True)
            if i == self.params.max_requests:
                break
        chex.assert_trees_all_close(env_state.link_snr_array, expected)

if __name__ == '__main__':
    os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4"
    jax.config.update('jax_numpy_rank_promotion', 'raise')
    absltest.main()
