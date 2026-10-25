"""Compare matched policy rollouts and full PPO from a saved physical state."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

from experimental.benchmarks.gn_cls_tuning_campaign import wait_for_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--state-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=("dense", "conv"), default="dense")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--pack-old-predicates", action="store_true")
    parser.add_argument("--reuse-screened-power", action="store_true")
    parser.add_argument("--narrow-new-snr", action="store_true")
    parser.add_argument("--reuse-unaffected-snr", action="store_true")
    parser.add_argument("--compact-snr-updates", action="store_true")
    parser.add_argument("--screen-batch-size", type=int)
    parser.add_argument("--group-widths", action="store_true")
    parser.add_argument("--fused-endpoint-snr", action="store_true")
    parser.add_argument("--generic-row-programs", type=int)
    parser.add_argument("--endpoint-xpm-backend", choices=["jax", "pallas"], default="jax")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--name")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "source_sha256.json").write_text(
        json.dumps(
            {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path("xlron").rglob("*.py"))
            },
            indent=2,
        )
        + "\n"
    )
    lease = wait_for_gpu(
        args.gpu,
        lambda state, **kw: print(json.dumps(dict(state=state, **kw)), flush=True),
        args.gpu_memory_fraction,
    )
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_DEFAULT_MATMUL_PRECISION="highest",
        PYTHONPATH=str(Path.cwd()),
        XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0",
    )
    if args.gpu_memory_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.gpu_memory_fraction)
    try:
        # Check the previously failing rewrite before allocating the PPO model.
        with (args.output / "strict-fp32-test.log").open("w") as log:
            check = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-x",
                    "xlron/models/spectrum_transformer_test.py::test_fused_convolution_value_and_gradient_match_offset_reference",
                ],
                stdout=log,
                check=False,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
        if check.returncode:
            raise RuntimeError("Strict FP32 convolution test failed")
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from absl import flags

        import xlron.parameter_flags  # noqa: F401 -- register the shared absl flags
        from xlron.environments import env_funcs
        from xlron.environments.gn_model import isrs_gn_model as gn
        from xlron.environments.gn_model import rmsa_gn_model
        from xlron.environments.make_env import process_config
        from xlron.train import train_utils
        from xlron.train.ppo import _env_rollout_advantages, get_learner_fn

        if args.generic_row_programs:
            from xlron.kernels import gn_xpm_legacy

            gn_xpm_legacy.xpm_legacy_pallas = partial(
                gn_xpm_legacy.xpm_legacy_pallas, persistent_rows=args.generic_row_programs
            )
        if args.endpoint_xpm_backend != "jax":
            env_funcs.get_snr_link_array = partial(
                env_funcs.get_snr_link_array, output_xpm_backend=args.endpoint_xpm_backend
            )
        gn.get_snr = partial(gn.get_snr, xpm_backend="pallas_grouped")
        env_funcs.get_candidate_baseline_snr = partial(
            env_funcs.get_candidate_baseline_snr, backend="pallas"
        )
        env_funcs.get_candidate_snr_evidence = partial(
            env_funcs.get_candidate_snr_evidence, backend="pallas"
        )
        env_funcs.get_lightpath_snr = partial(
            env_funcs.get_lightpath_snr, nsr_backend="pallas_precomputed"
        )
        env_funcs.mask_slots_rmsa_gn_model = partial(
            env_funcs.mask_slots_rmsa_gn_model,
            prefilter_new=True,
            pack_old=args.pack_old_predicates,
            reuse_screened_power=args.reuse_screened_power,
            narrow_new_snr=args.narrow_new_snr,
            reuse_unaffected_snr=args.reuse_unaffected_snr,
            compact_snr_updates=args.compact_snr_updates,
            screen_batch_size=args.screen_batch_size,
            group_widths=args.group_widths,
            fused_endpoint_snr=args.fused_endpoint_snr,
        )
        rmsa_gn_model.mask_slots_rmsa_gn_model = env_funcs.mask_slots_rmsa_gn_model
        c = {k: flags.FLAGS[k].default for k in flags.FLAGS}
        batch = args.envs * args.rollout
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
            transformer_obs_type="lifetime_snr",
            transformer_spectrum_encoder=args.encoder,
            transformer_embedding_size=args.width,
            transformer_intermediate_size=2 * args.width,
            transformer_num_layers=2,
            transformer_num_heads=4,
            transformer_actor_pooling="sum",
            transformer_modulation_features=True,
            aggregate_slots=1,
            gn_mask_mode="all_slots",
            gn_mask_batch_size=64,
            gn_mask_env_batch_size=1,
            NUM_ENVS=args.envs,
            TOTAL_TIMESTEPS=batch * 1000,
            STEPS_PER_INCREMENT=batch,
            ROLLOUT_LENGTH=args.rollout,
            UPDATE_EPOCHS=1,
            NUM_MINIBATCHES=8,
            LR=0.0005,
            LR_SCHEDULE="constant",
            OFF_POLICY_IAM=True,
            VALID_MASS_LOSS_COEF=0.002,
            VML_SCHEDULE="constant",
            GAMMA=0.996,
            GAE_LAMBDA=0.99,
            CLIP_EPS=0.04,
            ENT_COEF=0.005,
            SEPARATE_VF_OPTIMIZER=True,
            VF_LR=0.00005,
            VF_COEF=0.1,
            ENV_WARMUP_STEPS=0,
            EVAL_DURING_TRAINING=False,
            WANDB=False,
            ENHANCED_LOGGING=True,
        )
        c = process_config(c)
        runner, env, p = train_utils.experiment_data_setup(c, jax.random.PRNGKey(42))
        leaves, definition = jax.tree.flatten(runner[1].env_state)
        restored = []
        with np.load(args.state_input) as archive:
            for i, leaf in enumerate(leaves):
                value = archive[f"state_{i}"][: args.envs]
                if value.shape != leaf.shape or value.dtype != leaf.dtype:
                    raise ValueError(f"Saved physical state leaf {i} mismatch")
                restored.append(jnp.asarray(value))
        states = jax.tree.unflatten(definition, restored)
        states = states.replace(
            link_snr_array=jax.vmap(lambda s: env_funcs.get_snr_link_array(s, p))(states)
        )
        outer = runner[1].replace(env_state=states)
        runner = (runner[0], outer, (states, p), runner[3], runner[4])
        assert c.NUM_UPDATES == 1

        def rollout_with_advantages(initial_runner):
            final_runner, trajectory, advantages, targets, _, _ = _env_rollout_advantages(
                initial_runner, env, p, c
            )
            # Keep the actual policy rollout and advantage calculation live, without
            # writing the entire learning-only observation buffer to host outputs.
            return {
                "runner_state": final_runner,
                "reward_sum": jnp.sum(trajectory.reward),
                "advantage_sum": jnp.sum(advantages),
                "target_sum": jnp.sum(targets),
            }

        def measure(name, function):
            start = time.perf_counter()
            executable = jax.jit(function).lower(runner).compile()
            compile_s = time.perf_counter() - start
            print(json.dumps({"phase": name, "compile_s": compile_s}), flush=True)
            for _ in range(2):
                result = jax.block_until_ready(executable(runner))
            samples = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                result = jax.block_until_ready(executable(runner))
                samples.append(time.perf_counter() - start)
            median = float(np.median(samples))
            phase = {
                "compile_s": compile_s,
                "median_s": median,
                "environment_transitions_per_s": batch / median,
                "samples_s": samples,
                "memory": str(executable.memory_analysis()),
            }
            print(json.dumps(dict(phase=name, **phase)), flush=True)
            return result, phase

        rollout_result, rollout_phase = measure("rollout_with_advantages", rollout_with_advantages)
        result, ppo_phase = measure("full_ppo", get_learner_fn(env, p, runner[0], c))
        # The learning phase must follow exactly the same physical trajectory.
        for expected, actual in zip(
            jax.tree.leaves(rollout_result["runner_state"][1]),
            jax.tree.leaves(result["runner_state"][1]),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
        optimizer_updates = int(c.UPDATE_EPOCHS * c.NUM_MINIBATCHES * c.NUM_UPDATES)
        ppo_phase["optimizer_updates_per_s"] = optimizer_updates / ppo_phase["median_s"]
        model = result["runner_state"][0].model_params
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(result["loss_info"]))
        final_states = {
            f"state_{i}": np.asarray(x)
            for i, x in enumerate(jax.tree.leaves(result["runner_state"][1]))
        }
        np.savez_compressed(args.output / "final-states.npz", **final_states)
        if args.reference:
            with np.load(args.reference) as expected:
                for name, value in final_states.items():
                    np.testing.assert_array_equal(value, expected[name])
        report = {
            "device": jax.devices()[0].device_kind,
            "jax": jax.__version__,
            "encoder": args.encoder,
            "width": args.width,
            "envs": args.envs,
            "rollout": args.rollout,
            "epochs": 1,
            "minibatches": 8,
            "observations": "lifetime_snr",
            "modulation_features": True,
            "mask": "all_slots",
            "candidate_batch": 64,
            "mask_env_chunk": 1,
            "fp32_matmul": "highest",
            "transitions_per_rollout": batch,
            "optimizer_minibatch_updates_per_rollout": optimizer_updates,
            "rollout_with_advantages": rollout_phase,
            "full_ppo": ppo_phase,
            "additional_learning_s": ppo_phase["median_s"] - rollout_phase["median_s"],
            "final_environment_states_exact": True,
            "state_sha256": hashlib.sha256(args.state_input.read_bytes()).hexdigest(),
            "reference_final_states_exact": True if args.reference else None,
            "gpu_memory_fraction": args.gpu_memory_fraction,
            "pack_old_predicates": args.pack_old_predicates,
            "reuse_screened_power": args.reuse_screened_power,
            "narrow_new_snr": args.narrow_new_snr,
            "reuse_unaffected_snr": args.reuse_unaffected_snr,
            "compact_snr_updates": args.compact_snr_updates,
            "screen_batch_size": args.screen_batch_size,
            "group_widths": args.group_widths,
            "fused_endpoint_snr": args.fused_endpoint_snr,
            "generic_row_programs": args.generic_row_programs,
            "endpoint_xpm_backend": args.endpoint_xpm_backend,
            "loss": {k: np.asarray(v).tolist() for k, v in result["loss_info"].items()},
            "note": "Same initial policy, physical states, random keys and rollout length for both phases. Rollout includes advantages; full PPO also retains learning buffers, computes losses/gradients and applies Adam updates. Excludes compilation, warmup and W&B. Repeats fixed initial inputs; not a learning-quality trial.",
        }
        (args.output / "ppo.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in report.items() if k not in ("loss", "memory")}), flush=True
        )
        import wandb

        with wandb.init(
            entity="micdoh",
            project="XLRON-GN-CLS",
            group="gn-cls-20261025-fullslot-observations",
            name=args.name or f"full-mask-ppo-{args.encoder}{args.width}",
            job_type="benchmark",
            config={k: v for k, v in report.items() if k not in ("loss", "memory", "samples_s")},
        ) as run:
            run.summary["benchmark/environment_transitions_per_s"] = ppo_phase[
                "environment_transitions_per_s"
            ]
            run.summary["benchmark/optimizer_minibatch_updates_per_s"] = ppo_phase[
                "optimizer_updates_per_s"
            ]
            run.summary["benchmark/rollout_with_advantages_transitions_per_s"] = rollout_phase[
                "environment_transitions_per_s"
            ]
            run.save(str(args.output / "ppo.json"), base_path=str(args.output))
            run.save(str(args.output / "source_sha256.json"), base_path=str(args.output))
            (args.output / "wandb-report.json").write_text(
                json.dumps({"id": run.id, "url": run.url}) + "\n"
            )
    finally:
        lease.close()


if __name__ == "__main__":
    main()
