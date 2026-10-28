"""Time the transformer policy's forward/backward pass on one PPO minibatch.

Builds the RMSA-GN environment and ActorCriticTransformer of a training config,
replicates a reset state to a minibatch (the cost does not depend on the
values), and times value_and_grad of a PPO-shaped loss: the part of every
minibatch update that is not the environment. Set XLRON_DUMP_HLO=path to save
the optimised HLO.

Example (full C+L+S NSFNET, the transformer of the GN sweep scripts):
  python -m experimental.benchmarks.policy_update_bench --minibatch 512
"""

import argparse
import ctypes
import os
import sysconfig
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from xlron.environments.make_env import make, process_config
from xlron.train.train_utils import init_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--slots", type=int, default=1668)
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--modulation-features", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--profile", action="store_true", help="cudaProfilerStart/Stop around the timed calls"
    )
    parser.add_argument(
        "--forward-only", action="store_true", help="time the rollout's policy forward pass"
    )
    args = parser.parse_args()

    config = process_config(
        dict(
            env_type="rmsa_gn_model",
            topology_name="nsfnet_deeprmsa_directed",
            band_preference="C,L,S" if args.slots == 1668 else None,
            inter_band_gap_ghz=25,
            link_resources=args.slots,
            slot_size=12.5,
            k=5,
            values_bw=[100, 200, 400],
            load=5000,
            guardband=0,
            coherent=True,
            power_per_channel=-10,
            max_power_per_fibre=21,
            launch_power_type="fixed",
            continuous_operation=True,
            gn_mask_mode="all_slots",
            gn_kernel_backend="pallas",
            USE_TRANSFORMER=True,
            transformer_obs_type="occupancy_snr",
            transformer_num_layers=args.layers,
            transformer_num_heads=args.heads,
            transformer_embedding_size=args.embedding,
            transformer_intermediate_size=args.intermediate,
            transformer_actor_pooling="mean",
            transformer_modulation_features=args.modulation_features,
            transformer_spectrum_encoder="dense",
            NUM_ENVS=1,
            ROLLOUT_LENGTH=args.minibatch,
            TOTAL_TIMESTEPS=args.minibatch,
            STEPS_PER_INCREMENT=args.minibatch,
            ENV_WARMUP_STEPS=0,
        )
    )
    env, params = make(config, log_wrapper=False)
    config.ACTION_DIM = env.num_actions(params)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, params)
    model = init_network(config, key)
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (args.minibatch,) + jnp.shape(x)), state)
    batch = jax.tree.map(jnp.array, batch)  # materialise, as a trajectory buffer is
    actions = jnp.zeros((args.minibatch,), jnp.int32)
    params_, static = eqx.partition(model, eqx.is_inexact_array)

    def loss(p, states):
        m = eqx.combine(p, static)
        pi, value = jax.vmap(m, in_axes=(0, None))(states, params)
        return jnp.mean(pi.log_prob(actions)) + jnp.mean(value**2) + jnp.mean(pi.entropy())

    if args.forward_only:

        def forward(p, states):
            pi, value = jax.vmap(eqx.combine(p, static), in_axes=(0, None))(states, params)
            return pi.sample(seed=key), value

        fn = jax.jit(forward)
    else:
        fn = jax.jit(jax.value_and_grad(loss))
    start = time.perf_counter()
    executable = fn.lower(params_, batch).compile()
    print(f"compile {time.perf_counter() - start:.1f}s", flush=True)
    if path := os.environ.get("XLRON_DUMP_HLO"):
        Path(path).write_text(executable.as_text())
    for _ in range(3):
        jax.block_until_ready(executable(params_, batch))
    runtime = None
    if args.profile:
        library = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime/lib/libcudart.so.12"
        runtime = ctypes.CDLL(str(library) if library.is_file() else "libcudart.so.12")
        runtime.cudaProfilerStart()
    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        jax.block_until_ready(executable(params_, batch))
        samples.append(time.perf_counter() - start)
    if runtime is not None:
        runtime.cudaProfilerStop()
    mem = executable.memory_analysis()
    print(
        f"minibatch {args.minibatch}: median {1000 * np.median(samples):.2f} ms, "
        f"min {1000 * min(samples):.2f} ms, temp {mem.temp_size_in_bytes / 2**20:.0f} MiB"
    )


if __name__ == "__main__":
    main()
