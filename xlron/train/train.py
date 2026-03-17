import os
import sys

# Must set XLA env vars before importing JAX, so check sys.argv directly
# (absl flags aren't parsed yet at import time)
if "--DETERMINISTIC_OPS" in sys.argv:
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_enable_triton_gemm=false "
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
else:
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )

import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags

import wandb
from xlron import dtype_config
from xlron.environments.dataclasses import ActionInfo
from xlron.environments.env_funcs import create_run_name
from xlron.environments.make_env import make, process_config
from xlron.environments.wrappers import Profiler, jit_profiler
from xlron.heuristics.eval_heuristic import get_eval_fn
from xlron.parameter_flags import *  # noqa: F403,F401  # Ignore linter warnings for * import
from xlron.train.ppo import get_learner_fn
from xlron.train.train_utils import (
    experiment_data_setup,
    get_user_flags,
    load_model,
    log_actions,
    log_metrics,
    metrics,
    plot_metrics,
    print_experiment_summary,
    print_metrics,
    run_eval_during_training,
    select_action_eval,
    save_model,
    setup_wandb,
)

FLAGS = flags.FLAGS

# Create a global mutable container to collect data
collected_states = []


def _default_render_output_path(config: Dict[str, Any], run_name: str) -> Path:
    output = config.get("RENDER_OUTPUT_FILE")
    if output:
        return Path(output)
    run_dir = os.environ.get("XLRON_RUN_DIR")
    if run_dir:
        return Path(run_dir) / f"{run_name}_eval_render.mp4"
    data_out = config.get("DATA_OUTPUT_FILE")
    if data_out:
        return Path(data_out).expanduser().resolve().parent / f"{run_name}_eval_render.mp4"
    return Path.cwd() / f"{run_name}_eval_render.mp4"


def run_eval_render(config: Dict[str, Any], eval_state: Any, run_name: str):
    mode = str(config.get("RENDER_EVAL_MODE", "off")).lower()
    if mode not in {"save", "human"}:
        return
    if not (config.get("EVAL_HEURISTIC") or config.get("EVAL_MODEL")):
        return

    gui_run = bool(os.environ.get("XLRON_RUN_DIR"))
    if gui_run and mode == "human":
        # GUI launches detached subprocesses; OS pop-up windows are often unavailable there.
        # Ensure the render is still viewable in the GUI output pane by recording a file.
        print(
            "GUI run detected: 'human' render mode switched to 'save' so playback appears in GUI."
        )
        mode = "save"

    backend_name = ""
    if mode == "human":
        try:
            import matplotlib.pyplot as plt

            backend_name = str(plt.get_backend()).lower()
        except Exception:
            backend_name = "unknown"
        non_interactive = ("agg" in backend_name) or ("pdf" in backend_name) or ("svg" in backend_name) or ("ps" in backend_name)
        if non_interactive:
            fallback_mode = "save"
            print(
                f"Render mode '{mode}' requested but matplotlib backend '{backend_name}' is non-interactive. "
                f"Falling back to '{fallback_mode}' for this eval run."
            )
            mode = fallback_mode

    try:
        render_config = config.copy()
        render_config.log_wrapper = True
        render_config.NUM_ENVS = 1
        render_config.ENV_WARMUP_STEPS = 0
        env, env_params = make(render_config)
        base_env = env._env if hasattr(env, "_env") else env

        key = jax.random.PRNGKey(int(config.SEED) + 1337)
        key, reset_key = jax.random.split(key)
        obsv, env_state = env.reset(reset_key, env_params)
        last_obs = (
            (env_state.env_state, env_params)
            if config.USE_GNN or config.USE_TRANSFORMER
            else tuple([obsv])
        )

        run_type = "MODEL EVAL" if config.EVAL_MODEL else f"{str(config.path_heuristic).upper()} EVAL"
        render_context = {
            "run_type": run_type,
            "path_heuristic": config.path_heuristic,
            "render_scale": float(config.RENDER_SCALE),
            "topology_name": config.topology_name,
            "topology_directory": config.get("topology_directory", None),
        }
        max_steps = int(config.RENDER_MAX_STEPS) if int(config.RENDER_MAX_STEPS) > 0 else int(
            config.max_requests * getattr(config, "scale_factor", 1)
        )
        fps = max(float(config.RENDER_FPS), 0.1)
        frames = []
        output_path = None
        writer = None
        writer_error = None

        if mode == "save":
            output_path = _default_render_output_path(config, run_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix.lower() not in {".gif", ".mp4"}:
                output_path = output_path.with_suffix(".mp4")
            try:
                import imageio.v2 as imageio

                writer = imageio.get_writer(str(output_path), fps=fps)
                print(f"Streaming eval render frames to {output_path}")
            except Exception as e:
                writer_error = e
                print(
                    f"Streaming writer unavailable ({e}); falling back to in-memory frame buffer."
                )

        select_action_eval_jit = jax.jit(
            partial(
                select_action_eval,
                env=env,
                env_params=env_params,
                eval_state=eval_state,
                config=config,
            )
        )

        for step_idx in range(max_steps):
            key, action_key, step_key = jax.random.split(key, 3)
            prev_log_state = env_state
            prev_state = (
                prev_log_state.env_state if hasattr(prev_log_state, "env_state") else prev_log_state
            )

            select_action_state = (action_key, env_state, last_obs)
            env_state, action, _, _ = select_action_eval_jit(select_action_state)
            action_info = base_env.process_action(prev_state, action, env_params)

            obsv, env_state, _, terminal, truncated, info = env.step(
                step_key, env_state, action, env_params
            )
            curr_state = env_state.env_state if hasattr(env_state, "env_state") else env_state

            render_action_info = action_info
            if isinstance(info, dict) and "_render_affected_slots_mask" in info:
                render_action_info = ActionInfo(
                    action=info["_render_action"],
                    path_index=info["_render_path_index"],
                    initial_slot_index=info["_render_initial_slot_index"],
                    nodes_sd=info["_render_nodes_sd"],
                    requested_datarate=info["_render_requested_datarate"],
                    num_slots=info["_render_num_slots"],
                    path=info["_render_path"],
                    se=info["_render_se"],
                    affected_slots_mask=info["_render_affected_slots_mask"],
                    power_action=info["_render_power_action"],
                )
            if isinstance(info, dict) and "_render_check" in info:
                check = jnp.asarray(info["_render_check"])
            else:
                accepted = int(jnp.asarray(curr_state.accepted_services).reshape(-1)[0]) > int(
                    jnp.asarray(prev_state.accepted_services).reshape(-1)[0]
                )
                check = jnp.array(0 if accepted else 1, dtype=jnp.float32)

            # Render/debug sanity check: compare requested slots with actual newly allocated width.
            try:
                accepted_now = float(np.asarray(jax.device_get(check)).reshape(-1)[0]) < 0.5
                if accepted_now:
                    prev_slots = np.asarray(jax.device_get(prev_state.link_slot_array), dtype=float)
                    curr_slots = np.asarray(jax.device_get(curr_state.link_slot_array), dtype=float)
                    if prev_slots.shape == curr_slots.shape:
                        delta_mask = (curr_slots - prev_slots) > 0.5
                        path_mask = (
                            np.asarray(jax.device_get(render_action_info.path), dtype=float) > 0.5
                        )
                        if path_mask.shape[0] == delta_mask.shape[0]:
                            delta_mask = delta_mask & path_mask[:, None]
                            per_link_w = np.sum(delta_mask, axis=1)
                            on_path_w = per_link_w[per_link_w > 0]
                            if on_path_w.size > 0:
                                alloc_w = int(np.min(on_path_w))
                                req_w = int(
                                    round(
                                        float(
                                            np.asarray(
                                                jax.device_get(render_action_info.num_slots)
                                            ).reshape(-1)[0]
                                        )
                                    )
                                )
                                if alloc_w != req_w:
                                    step_id = int(
                                        np.asarray(jax.device_get(curr_state.total_timesteps)).reshape(
                                            -1
                                        )[0]
                                    )
                                    print(
                                        f"RENDER_FSU_MISMATCH step={step_id} req={req_w} alloc={alloc_w} "
                                        f"action={int(np.asarray(jax.device_get(render_action_info.action)).reshape(-1)[0])}"
                                    )
            except Exception:
                pass

            if mode == "human":
                base_env.render(
                    curr_state,
                    params=env_params,
                    action_info=render_action_info,
                    check=check,
                    mode="human",
                    render_context=render_context,
                )
                import matplotlib.pyplot as plt

                plt.waitforbuttonpress(timeout=-1)

            if mode == "save":
                frame = base_env.render(
                    curr_state,
                    params=env_params,
                    action_info=render_action_info,
                    check=check,
                    mode="rgb_array",
                    render_context=render_context,
                )
                frame_np = np.array(frame, copy=True)
                if writer is not None:
                    writer.append_data(frame_np)
                else:
                    frames.append(frame_np)

            last_obs = (
                (env_state.env_state, env_params)
                if config.USE_GNN or config.USE_TRANSFORMER
                else tuple([obsv])
            )
            if bool(jnp.asarray(terminal)) or bool(jnp.asarray(truncated)):
                # env.step auto-resets; continue to fill requested render budget.
                pass

        if mode == "save":
            if writer is not None:
                try:
                    writer.close()
                    print(f"Saved eval render: {output_path}")
                except Exception as e:
                    print(f"Failed to finalize eval render at {output_path}: {e}")
            elif frames:
                print(
                    f"Captured {len(frames)} render frame(s) for eval recording"
                    + (f" (streaming unavailable: {writer_error})" if writer_error else ".")
                )
                try:
                    if output_path.suffix.lower() == ".mp4":
                        import imageio.v2 as imageio

                        imageio.mimsave(str(output_path), frames, fps=fps)
                    else:
                        try:
                            import imageio.v2 as imageio

                            imageio.mimsave(str(output_path), frames, fps=fps)
                        except Exception:
                            from PIL import Image

                            pil_frames = [Image.fromarray(frame) for frame in frames]
                            duration_ms = int(round(1000.0 / fps))
                            pil_frames[0].save(
                                str(output_path),
                                save_all=True,
                                append_images=pil_frames[1:],
                                duration=duration_ms,
                                loop=0,
                                optimize=False,
                                disposal=2,
                            )
                    print(f"Saved eval render: {output_path}")
                except Exception as e:
                    print(f"Failed to save eval render to {output_path}: {e}")
            else:
                print("No frames captured for eval rendering; skipping file save.")
        base_env.close()
    finally:
        pass


def identify_default_device(
    gpu_index: int | None = None, auto_select: bool = False
) -> List[jax.Device]:
    """
    Identifies and sets the default JAX device, preferring the GPU with most free memory.

    Args:
        gpu_index: Specific GPU index to use (0-indexed). If None and auto_select=False, uses GPU 0
        auto_select: If True, automatically select the GPU with most free memory
                     (overrides gpu_index if both are provided)

    Returns:
        jax_device: The selected JAX device object
    """
    # Check if running on TPU
    if os.environ.get("COLAB_TPU_ADDR", False):
        print("Running on TPU")
        device = jax.devices()[0]
        jax.config.update("jax_default_device", device)
        return device

    def get_gpu_memory_info():
        """Get memory information for NVIDIA GPUs using nvidia-smi."""
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,nounits,noheader",
                ],
                encoding="utf-8",
            )

            gpu_info = []
            for line in result.strip().split("\n"):
                idx, free_mem = line.split(",")
                gpu_info.append((int(idx.strip()), int(free_mem.strip())))

            return gpu_info
        except Exception as e:
            print(f"Warning: Could not query GPU info: {e}")
            return []

    # Get available JAX devices
    jax_devices = jax.devices()

    # If no GPUs available, use default device (likely CPU)
    if not any(d.platform == "gpu" for d in jax_devices):
        print("No GPUs detected, using default device (CPU)")
        device = jax_devices[0]
        jax.config.update("jax_default_device", device)
        return device

    # Get GPU memory information
    gpu_memory_info = get_gpu_memory_info()

    # Auto-select GPU with most free memory
    if auto_select and gpu_memory_info:
        selected_gpu_idx, free_mem = max(gpu_memory_info, key=lambda x: x[1])
        print(f"Auto-selected GPU {selected_gpu_idx} with {free_mem} MB free memory")

        # Display all GPU memory info for context
        print("All GPUs:")
        for idx, mem in gpu_memory_info:
            print(f"  GPU {idx}: {mem} MB free")

        gpu_index = selected_gpu_idx

    # Default to GPU 0 if nothing specified
    if gpu_index is None:
        gpu_index = 0
        print("No GPU specified, defaulting to GPU 0")

    # Find the corresponding JAX device
    try:
        device = [d for d in jax_devices if d.platform == "gpu"][gpu_index]
        print(f"Setting default device to: {device}")
        jax.config.update("jax_default_device", device)
        return device
    except IndexError:
        print(f"Warning: GPU {gpu_index} not found, using first available GPU")
        device = [d for d in jax_devices if d.platform == "gpu"][0]
        jax.config.update("jax_default_device", device)
        return device


def _update_experiment_input_load(experiment_input, load_val, mean_service_holding_time, num_learners):
    """Update arrival_rate and mean_service_holding_time in experiment_input for a new load.

    experiment_input is (runner_state, env_state, obsv, rng_step, rng_epoch) where
    env_state is a LogEnvState wrapping an inner EnvState. With NUM_ENVS > 1 the
    inner state has vmapped (batched) leaves, so arrival_rate has shape [NUM_ENVS].
    We use jnp.full_like to preserve the existing shape and dtype.
    """
    runner_state, env_state, obsv, rng_step, rng_epoch = experiment_input
    arrival_rate = load_val / mean_service_holding_time
    # Use full_like to match shape (scalar or [NUM_ENVS] or [NUM_LEARNERS, NUM_ENVS])
    ar = jnp.full_like(env_state.env_state.arrival_rate, arrival_rate)
    msht = jnp.full_like(env_state.env_state.mean_service_holding_time, mean_service_holding_time)

    inner = env_state.env_state.replace(arrival_rate=ar, mean_service_holding_time=msht)
    env_state = env_state.replace(env_state=inner)
    return (runner_state, env_state, obsv, rng_step, rng_epoch)


def train(argv: list[str], config: Dict[str, Any] = {}) -> None:
    flags = FLAGS if not config else config
    # Capture explicitly-set flags before process_config converts them
    user_flags = get_user_flags(FLAGS) if flags is FLAGS else dict(config)
    config = process_config(flags)
    render_mode = str(config.get("RENDER_EVAL_MODE", "off")).lower()
    if render_mode != "off" and int(config.NUM_ENVS) != 1:
        old_envs = int(config.NUM_ENVS)
        old_total_timesteps = float(config.TOTAL_TIMESTEPS)
        new_total_timesteps = max(1, int(old_total_timesteps / old_envs))
        print(
            f"WARNING: render mode '{render_mode}' supports one environment. "
            f"Forcing NUM_ENVS from {old_envs} to 1 and scaling TOTAL_TIMESTEPS "
            f"from {int(old_total_timesteps)} to {new_total_timesteps}."
        )
        config.NUM_ENVS = 1
        config.TOTAL_TIMESTEPS = new_total_timesteps
        config = process_config(config)
    config.log_wrapper = True  # Always use log wrapper for training
    # Initialize dtypes based on flags
    dtype_config.initialize_dtypes(FLAGS)

    # Identify and set the default JAX device
    # If user specifies VISIBLE_DEVICES, use the first one; otherwise auto-select
    if config.VISIBLE_DEVICES:
        # Parse comma-separated GPU indices
        gpu_indices = [int(x) for x in config.VISIBLE_DEVICES.split(",")]
        default_device = identify_default_device(gpu_index=gpu_indices[0], auto_select=False)
    else:
        # Auto-select GPU with most free memory
        default_device = identify_default_device(auto_select=True)

    print(f"Default device set to: {default_device}")

    def merge_func(x):
        # Original dims: (learner, num_updates, rollout_length, num_envs)
        # Compute the new shape
        # If X has only 2 or 3 dims then add more leading dims until it has 4
        if x.ndim == 2:
            x = x[None, :, :, None]
        elif x.ndim == 3 and config.NUM_LEARNERS == 1:
            x = x[None, :, :, :]
        elif x.ndim == 3 and config.NUM_ENVS > 1:
            x = x[:, :, :, None]
        learner, num_updates, rollout_length, num_envs = x.shape
        new_shape = (learner * num_envs, num_updates, rollout_length)
        # Perform a single transpose operation
        x = jnp.transpose(x, (0, 3, 1, 2))
        # Reshape to merge learner and num_envs dimensions
        return x.reshape(new_shape)

    # Check device count
    print(f"Available devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")
    num_devices = len(jax.devices())
    config.NUM_DEVICES = num_devices

    # Set flags for debugging
    jax.config.update("jax_debug_nans", config.DEBUG_NANS)
    jax.config.update("jax_disable_jit", config.DISABLE_JIT)
    jax.config.update("jax_enable_x64", config.ENABLE_X64)
    # Option to print memory usage for debugging OOM errors
    if config.PRINT_MEMORY_USE:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        os.environ["TF_CPP_VMODULE"] = "bfc_allocator=1"
    # Set the fraction of memory to pre-allocate
    if config.PREALLOCATE_MEM:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = config.PREALLOCATE_MEM_FRACTION
        print(f"XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    print(f"XLA_PYTHON_CLIENT_PREALLOCATE={os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']}")

    # Setup the project name, experiment name for wandb and plots
    run_name = create_run_name(config)
    project_name = config.PROJECT if config.PROJECT else run_name
    experiment_name = config.EXPERIMENT_NAME if config.EXPERIMENT_NAME else run_name

    # Setup wandb
    if config.WANDB:
        setup_wandb(config, project_name, experiment_name)

    # Print every flag and its name
    if config.DEBUG:
        print("non-flag arguments:", argv)
        jax.config.update("jax_debug_nans", True)
    if config.NO_TRUNCATE:
        jax.numpy.set_printoptions(threshold=sys.maxsize)  # Don't truncate printed arrays
        # increase line length for numpy print options
        jax.numpy.set_printoptions(linewidth=220)

    if config.PRINT_FLAGS:
        for name in config:
            print(name, config[name])

    if config.RETRAIN_MODEL or config.EVAL_MODEL:
        model = load_model(config, jax.random.PRNGKey(config.SEED))
    else:
        model = None

    # Print experiment summary (env_params not yet available; will print after setup)
    print_experiment_summary(config)

    profiler = Profiler(enabled=True)

    # Build load sweep array
    if (
        config.get("min_load") is not None
        and config.get("max_load") is not None
        and config.get("step_load") is not None
    ):
        loads = np.arange(
            config.min_load, config.max_load + config.step_load / 2, config.step_load
        )
    else:
        loads = np.array([config.load])

    with profiler.section("COMPILATION"):
        print("\n---BEGINNING COMPILATION---\n")

        rng = jax.random.PRNGKey(config.SEED)
        if config.NUM_LEARNERS > 1:
            rng = jax.random.split(rng, config.NUM_LEARNERS)
            experiment_fn = (
                get_learner_fn if not (config.EVAL_HEURISTIC or config.EVAL_MODEL) else get_eval_fn
            )
            experiment_input, env, env_params = jax.vmap(
                experiment_data_setup, axis_name="learner", in_axes=(None, 0)
            )(config, rng)
            experiment_fn = experiment_fn(env, env_params, experiment_input[0], config)
            run_experiment = (
                jax.jit(jax.vmap(experiment_fn, axis_name="learner"))
                .lower(experiment_input)
                .compile()
            )
        else:
            experiment_fn = (
                get_learner_fn if not (config.EVAL_HEURISTIC or config.EVAL_MODEL) else get_eval_fn
            )
            experiment_input, env, env_params = experiment_data_setup(config, rng)
            experiment_fn = experiment_fn(env, env_params, experiment_input, config)
            run_experiment = jax.jit(experiment_fn).lower(experiment_input).compile()

    # Set up eval during training: fresh single-env setup without warmup
    if config.EVAL_DURING_TRAINING and not (config.EVAL_HEURISTIC or config.EVAL_MODEL):
        with profiler.section("EVAL_COMPILATION"):
            # Create eval config: single env, no warmup, custom timestep budget
            eval_config = config.copy()
            eval_config.NUM_ENVS = min(config.NUM_ENVS, 10)
            eval_config.ENV_WARMUP_STEPS = 0
            eval_timesteps = (
                config.EVAL_TIMESTEPS * eval_config.NUM_ENVS
                if config.EVAL_TIMESTEPS > 0
                else config.STEPS_PER_INCREMENT
            )
            # Ensure eval runs long enough for warmup + measurement
            eval_timesteps += config.ENV_WARMUP_STEPS * eval_config.NUM_ENVS
            eval_config.STEPS_PER_INCREMENT = eval_timesteps
            # With continuous_operation, make_env sets max_requests=TOTAL_TIMESTEPS.
            # Set eval TOTAL_TIMESTEPS so each env runs eval_timesteps/NUM_ENVS steps
            # as a single continuous episode.
            eval_config.TOTAL_TIMESTEPS = eval_timesteps
            eval_config.log_wrapper = True

            # Fresh env setup for eval (no warmup, single env)
            eval_rng = jax.random.PRNGKey(config.SEED + 1)
            eval_input, eval_env, eval_env_params = experiment_data_setup(eval_config, eval_rng)

            # Sync eval_config with what make_env actually computed.
            # process_config (called inside make_env) rounds TOTAL_TIMESTEPS
            # down to be divisible by ROLLOUT_LENGTH*NUM_ENVS, which changes
            # max_requests when continuous_operation=True. The eval scan in
            # get_eval_fn reads config.max_requests for its scan length, so a
            # mismatch causes the env to auto-reset mid-eval.
            eval_config.max_requests = int(eval_env_params.max_requests) // eval_config.NUM_ENVS
            eval_config.STEPS_PER_INCREMENT = eval_config.TOTAL_TIMESTEPS

            eval_compile_input = (
                experiment_input[0],
                eval_input[1],
                eval_input[2],
                eval_input[3],
                eval_input[4],
            )
            eval_fn = get_eval_fn(eval_env, eval_env_params, eval_input, eval_config)
            # Run eval on the same device as training to avoid transfer overhead.
            # The eval is lightweight (single env) so GPU contention is minimal.
            run_eval = jax.jit(eval_fn).lower(eval_compile_input).compile()
        print(
            f"Eval during training enabled: {eval_config.STEPS_PER_INCREMENT} timesteps every {config.EVAL_FREQUENCY} increment(s)"
        )
    else:
        run_eval = eval_input = None

    # START TRAINING
    # Save base experiment input for load sweep (reused for each load)
    base_experiment_input = experiment_input
    mean_service_holding_time = float(env_params.mean_service_holding_time)

    for load_idx, load_val in enumerate(loads):
        load_val = float(load_val)
        if len(loads) > 1:
            print(f"\n{'='*70}")
            print(f"LOAD SWEEP: load={load_val:.1f} ({load_idx + 1}/{len(loads)})")
            print(f"{'='*70}")
            # Update config.load and user_flags for logging/JSONL output
            config.load = load_val
            config.arrival_rate = load_val / mean_service_holding_time
            user_flags["load"] = load_val
            # Update experiment_input with new arrival_rate for this load
            experiment_input = _update_experiment_input_load(
                base_experiment_input, load_val, mean_service_holding_time,
                config.NUM_LEARNERS,
            )
            # Update run/experiment names for this load
            run_name = create_run_name(config)
            experiment_name = config.EXPERIMENT_NAME if config.EXPERIMENT_NAME else run_name
        else:
            experiment_input = base_experiment_input

        # Setup wandb per load (each load gets its own run)
        # For single-load case, wandb was already set up before compilation
        if config.WANDB and len(loads) > 1:
            if load_idx > 0:
                wandb.finish()
            setup_wandb(config, project_name, experiment_name)

        start_time = time.time()
        log_time = 0.0
        total_run_time = 0.0
        best_eval_metric = float("inf")
        first_save = True
        episode_count = update_count = step_count = 0
        merged_out: Dict[str, Dict[str, jax.Array]] = {}
        processed_data: Dict[str, Dict[str, jax.Array]] = {}
        processed_data_all: Dict[str, Dict[str, jax.Array]] = {}
        for i in range(config.NUM_INCREMENTS):
            print(f"\n---INCREMENT {i + 1}/{config.NUM_INCREMENTS}---")
            # Run the increment
            with profiler.section("EXECUTION", frames=config.STEPS_PER_INCREMENT * config.NUM_LEARNERS):
                out = run_experiment(experiment_input)
                out = jax.tree_util.tree_map(
                    lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out
                )  # Wait for all devices to finish
            prev_total = total_run_time
            total_run_time = time.time() - start_time - log_time  # Update total first
            increment_run_time = total_run_time - prev_total  # Increment = difference
            log_start_time = time.time()
            render_mode = str(config.RENDER_EVAL_MODE).lower()
            if (config.EVAL_HEURISTIC or config.EVAL_MODEL) and render_mode != "off":
                if render_mode in {"save", "human"}:
                    eval_state_for_render = (
                        experiment_input[0] if isinstance(experiment_input, tuple) else experiment_input
                    )
                    run_eval_render(config, eval_state_for_render, experiment_name)
            merged_out, processed_data = log_metrics(
                config,
                out,
                total_run_time,
                increment_run_time,
                merge_func,
                episode_count=episode_count,
                update_count=update_count,
                step_count=step_count,
            )
            # Save model params (skip if EVAL_DURING_TRAINING, which saves only the best model)
            if config.SAVE_MODEL and not config.EVAL_DURING_TRAINING:
                train_state = out["runner_state"][0]  # Get TrainState from the first learner
                # Determine current metric value to decide whether to save
                if config.continuous_operation:
                    if config.reward_type == "bitrate":
                        current_metric = float(
                            processed_data["bitrate_blocking_probability"]["mean"][-1]
                        )
                    else:
                        current_metric = float(
                            processed_data["service_blocking_probability"]["mean"][-1]
                        )
                else:
                    if config.reward_type == "bitrate":
                        current_metric = float(
                            processed_data["bitrate_blocking_probability"]["episode_end_mean"][-1]
                        )
                    else:
                        current_metric = float(
                            processed_data["service_blocking_probability"]["episode_end_mean"][-1]
                        )
                if current_metric <= best_eval_metric:
                    best_eval_metric = current_metric
                    model = eqx.combine(train_state.model_params, train_state.model_static)
                    saved_path = save_model(model, config, first_save=first_save)
                    if first_save:
                        config.MODEL_PATH = str(saved_path)
                        first_save = False

            # Extend every item in processed data with new data
            episode_count += len(processed_data["service_blocking_probability"]["episode_end_mean"])
            step_count += config.STEPS_PER_INCREMENT // config.NUM_ENVS
            update_count += config.NUM_UPDATES * config.NUM_MINIBATCHES
            # Concatenate arrays for each key
            processed_data_all = (
                processed_data
                if i == 0
                else jax.tree.map(
                    lambda x, y: jnp.concatenate([x, y]), processed_data_all, processed_data
                )
            )

            # Run eval during training
            if (
                config.EVAL_DURING_TRAINING
                and not (config.EVAL_HEURISTIC or config.EVAL_MODEL)
                and (i + 1) % config.EVAL_FREQUENCY == 0
            ):
                with profiler.section("EVAL"):
                    best_eval_metric, first_save = run_eval_during_training(
                        config,
                        run_eval,
                        eval_input,
                        out,
                        best_eval_metric,
                        step_count,
                        first_save=first_save,
                    )

            log_time += time.time() - log_start_time
            # Update the experiment input for the next increment
            experiment_input = out["runner_state"]  # TrainState, EnvState, Obs, key, key
            for metric in metrics:
                if processed_data.get(metric):
                    if config.continuous_operation:
                        print(
                            f"{metric}: {float(processed_data[metric]['mean'][-1]):.3f} ± {float(processed_data[metric]['std'][-1]):.3f}"
                        )
                    else:
                        print(
                            f"Episode end {metric}: {float(processed_data[metric]['episode_end_mean'][-1])} ± {float(processed_data[metric]['episode_end_std'][-1]):.3f}"
                        )

        # END OF INCREMENTS FOR THIS LOAD
        # Build timing info from profiler records
        timing = {}
        if "COMPILATION" in profiler._records:
            timing["compilation_time_s"] = sum(e for e, _ in profiler._records["COMPILATION"])
        execution_entries = [
            (e, f)
            for tag, entries in profiler._records.items()
            if tag.startswith("EXECUTION")
            for e, f in entries
        ]
        if execution_entries:
            timing["execution_time_s"] = sum(e for e, _ in execution_entries)
            total_frames = sum(f for _, f in execution_entries if f is not None)
            if total_frames and timing["execution_time_s"] > 0:
                timing["fps"] = total_frames / timing["execution_time_s"]

        print_metrics(processed_data_all, config, user_flags=user_flags, timing=timing or None)
        if config.PLOTTING:
            plot_metrics(experiment_name, processed_data_all, config)
        if config.log_actions:
            log_actions(merged_out, processed_data, config)

    # END OF LOAD SWEEP
    profiler.summary()
    if config.PROFILE:
        jit_profiler.summary()

    if config.WANDB:
        wandb.finish()


if __name__ == "__main__":
    FLAGS(sys.argv)
    app.run(train)
