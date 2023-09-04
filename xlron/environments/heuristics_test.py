import os
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
from xlron.environments.env_funcs import *
from xlron.environments.vone import *
from xlron.environments.rsa import *
from xlron.environments.heuristics import *
from xlron.environments.env_test import *


# Set the number of (emulated) host devices
num_devices = 4
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={num_devices}"


class KspffTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_empty", jnp.array([0, 1, 1]),
         jnp.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0], ]),
         jnp.array(0)),
        ("case_full", jnp.array([0, 1, 1]),
         jnp.array([[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1], ]),
         jnp.array(0)),
        ("case_start_edge", jnp.array([0, 1, 1]),
         jnp.array([[0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1], ]),
         jnp.array(0)),
        ("case_end_edge", jnp.array([0, 1, 1]),
         jnp.array([[1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0], ]),
         jnp.array(3)),
        ("case_ksp", jnp.array([0, 1, 1]),
         jnp.array([[1, 1, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1], ]),
         jnp.array(3)),
    )
    def test_ksp_ff(self, request_array, link_slot_array, expected):
        self.state = self.state.replace(request_array=request_array, link_slot_array=link_slot_array)
        action = self.variant(ksp_ff, static_argnums=(1,))(self.state, self.params)
        chex.assert_trees_all_close(action, expected)


class FfkspTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.key, self.env, self.obs, self.state, self.params = rsa_4node_test_setup()

    @chex.all_variants()
    @parameterized.named_parameters(
        ("case_empty", jnp.array([0, 1, 1]),
         jnp.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0], ]),
         jnp.array(0)),
        ("case_full", jnp.array([0, 1, 1]),
         jnp.array([[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1], ]),
         jnp.array(0)),
        ("case_start_edge", jnp.array([0, 1, 1]),
         jnp.array([[0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1], ]),
         jnp.array(0)),
        ("case_end_edge", jnp.array([0, 1, 1]),
         jnp.array([[1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 0], ]),
         jnp.array(3)),
        ("case_ff", jnp.array([0, 1, 1]),
         jnp.array([[1, 1, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 1], ]),
         jnp.array(4)),
    )
    def test_ff_ksp(self, request_array, link_slot_array, expected):
        self.state = self.state.replace(request_array=request_array, link_slot_array=link_slot_array)
        action = self.variant(ff_ksp, static_argnums=(1,))(self.state, self.params)
        chex.assert_trees_all_close(action, expected)


if __name__ == '__main__':
    jax.config.update('jax_numpy_rank_promotion', 'raise')
    absltest.main()
