"""Synchronized forward/backward costs on identical saved physical states.

This is a model microbenchmark, not a PPO or environment throughput estimate.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--only",
        choices=[
            "occupancy-dense128",
            "lifetime-dense128",
            "lifetime-conv128",
            "lifetime-dense256",
        ],
    )
    parser.add_argument("--profile-phase", choices=["forward", "forward_backward"])
    args = parser.parse_args()
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from absl import flags
    import xlron.parameter_flags
    from xlron.environments.make_env import make, process_config
    from xlron.train.train_utils import init_network

    report = dict(
        device=jax.devices()[0].device_kind,
        jax=jax.__version__,
        envs=args.envs,
        state_sha256=hashlib.sha256(args.state_input.read_bytes()).hexdigest(),
        xla_flags=os.environ.get("XLA_FLAGS"),
        results=[],
    )
    for name, obs, encoder, width in (
        ("occupancy-dense128", "occupancy_snr", "dense", 128),
        ("lifetime-dense128", "lifetime_snr", "dense", 128),
        ("lifetime-conv128", "lifetime_snr", "conv", 128),
        ("lifetime-dense256", "lifetime_snr", "dense", 256),
    ):
        if args.only and args.only != name:
            continue
        c = {k: flags.FLAGS[k].default for k in flags.FLAGS}
        c.update(
            env_type="rmsa_gn_model",
            topology_name="nsfnet_deeprmsa_directed",
            k=5,
            link_resources=1668,
            slot_size=12.5,
            band_preference="C,L,S",
            inter_band_gap_ghz=25,
            load=5000,
            guardband=0,
            coherent=True,
            power_per_channel=-10,
            max_power_per_fibre=21,
            launch_power_type="fixed",
            continuous_operation=True,
            values_bw=[100, 200, 400],
            USE_TRANSFORMER=True,
            transformer_obs_type=obs,
            transformer_spectrum_encoder=encoder,
            transformer_embedding_size=width,
            transformer_intermediate_size=2 * width,
            transformer_num_layers=2,
            transformer_num_heads=4,
            transformer_actor_pooling="sum",
            aggregate_slots=1,
            NUM_ENVS=args.envs,
            TOTAL_TIMESTEPS=32768,
            ROLLOUT_LENGTH=64,
        )
        c = process_config(c)
        env, p = make(c, log_wrapper=False)
        c.ACTION_DIM = env.num_actions(p)
        _, s = env.reset(jax.random.PRNGKey(3), p)
        leaves, td = jax.tree.flatten(s)
        with np.load(args.state_input) as archive:
            restored = []
            for i, leaf in enumerate(leaves):
                value = archive[f"state_{i}"][: args.envs]
                if value.shape[1:] != leaf.shape or value.dtype != leaf.dtype:
                    raise ValueError(
                        f"Saved leaf {i} mismatch: {value.shape}/{value.dtype} vs {leaf.shape}/{leaf.dtype}"
                    )
                restored.append(jnp.asarray(value))
        states = jax.tree.unflatten(td, restored)
        model = init_network(c, jax.random.PRNGKey(42))

        def loss(m, states):
            def one(state):
                pi, v = m(state, p)
                return -pi.log_prob(jnp.array(0)) + 0.5 * v * v

            return jnp.mean(jax.vmap(one)(states))

        tests = (
            ("forward", eqx.filter_jit(loss)),
            ("forward_backward", eqx.filter_jit(eqx.filter_value_and_grad(loss))),
        )
        row = dict(
            name=name,
            parameters=sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))),
            actions=c.ACTION_DIM,
            aggregate_slots=p.aggregate_slots,
            timings={},
        )
        for phase, fn in tests:
            started = time.perf_counter()
            compiled = fn.lower(model, states).compile()
            compile_s = time.perf_counter() - started
            out = jax.block_until_ready(compiled(model, states))
            runtime = None
            if args.profile_phase == phase:
                import ctypes
                import sysconfig

                library = (
                    Path(sysconfig.get_paths()["purelib"])
                    / "nvidia/cuda_runtime/lib/libcudart.so.12"
                )
                runtime = ctypes.CDLL(str(library) if library.is_file() else "libcudart.so.12")
                if runtime.cudaProfilerStart():
                    raise RuntimeError("cudaProfilerStart failed")
            samples = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                out = jax.block_until_ready(compiled(model, states))
                samples.append(1000 * (time.perf_counter() - started))
            if runtime is not None and runtime.cudaProfilerStop():
                raise RuntimeError("cudaProfilerStop failed")
            assert all(np.isfinite(x).all() for x in jax.tree.leaves(out))
            row["timings"][phase] = dict(
                compile_s=compile_s, median_ms=float(np.median(samples)), samples_ms=samples
            )
            print(json.dumps(dict(name=name, phase=phase, **row["timings"][phase])), flush=True)
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del model, states, compiled, out
        jax.clear_caches()


if __name__ == "__main__":
    main()
