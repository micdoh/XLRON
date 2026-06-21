"""Benchmark env-state memory and throughput for float32 vs --mixed_precision.

Standalone script (no neural network): builds an ensemble of parallel envs, measures the carried
env-state memory and the per-step throughput of a jitted scan rollout, for both precisions.

  uv run python -m xlron.environments.benchmark_mixed_precision \
      --env_type=rmsa --topology_name=nsfnet_deeprmsa_directed \
      --link_resources=100 --k=5 --load=250 --NUM_ENVS=2000 --bench_steps=200

On the CPU backend, fp16/bf16 *compute* is emulated, so the headline signal here is the env-state
MEMORY reduction; the speed column is indicative only (the real speedup shows on GPU/TPU). The
script prints jax.default_backend() so you know which regime you are in.
"""

import time

import jax
import jax.numpy as jnp
from absl import app, flags
from box import Box

import xlron.parameter_flags  # noqa: F401  (defines all flags)
from xlron.environments.make_env import make

flags.DEFINE_integer(
    "bench_steps", 200, "Number of scanned env steps per timed run", flag_values=flags.FLAGS
)
flags.DEFINE_integer(
    "bench_warmup", 1, "Number of warmup (compile) runs before timing", flag_values=flags.FLAGS
)
FLAGS = flags.FLAGS


def _pytree_bytes(tree):
    return sum(int(x.nbytes) for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "nbytes"))


def _unwrap(state):
    return state.env_state if hasattr(state, "env_state") else state


def _config(mixed):
    cfg = {k: v.value for k, v in FLAGS.__flags.items()}
    cfg["mixed_precision"] = mixed
    # Keep the rollout self-contained / divisible.
    cfg["NUM_MINIBATCHES"] = cfg.get("NUM_MINIBATCHES") or 1
    return Box(cfg)


def _make_rollout(env, params, n_envs, n_steps):
    raw = env._env
    mask_v = jax.vmap(raw.action_mask, in_axes=(0, None))
    step_v = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    @jax.jit
    def rollout(keys):
        _, state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)

        def body(carry, _):
            state, key = carry
            key, sub = jax.random.split(key)
            es = _unwrap(state)
            mask = mask_v(es, params)
            mask = mask[0] if isinstance(mask, tuple) else mask
            actions = jnp.argmax(mask, axis=-1)
            skeys = jax.random.split(sub, n_envs)
            out = step_v(skeys, state, actions, params)
            return (out[1], key), None

        (state, _), _ = jax.lax.scan(body, (state, keys[0]), None, length=n_steps)
        return state

    return rollout


def _bench_one(mixed, n_envs, n_steps, warmup):
    env, params = make(_config(mixed))
    keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
    _, state0 = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    es0 = _unwrap(state0)

    # Per-env env-state memory (single env slice -> multiply by n_envs for the ensemble).
    single = jax.tree_util.tree_map(lambda x: x[0] if hasattr(x, "ndim") and x.ndim else x, es0)
    bytes_per_env = _pytree_bytes(single)

    rollout = _make_rollout(env, params, n_envs, n_steps)
    for _ in range(max(1, warmup)):
        jax.block_until_ready(rollout(keys))
    t0 = time.perf_counter()
    jax.block_until_ready(rollout(keys))
    dt = time.perf_counter() - t0
    fps = (n_envs * n_steps) / dt

    dtypes = {
        "link_slot_array": str(es0.link_slot_array.dtype),
        "departure": str(es0.link_slot_departure_array.dtype),
        "current_time": str(es0.current_time.dtype),
    }
    return bytes_per_env, fps, dt, dtypes


def main(argv):
    del argv
    n_envs = int(FLAGS.NUM_ENVS)
    n_steps = int(FLAGS.bench_steps)
    warmup = int(FLAGS.bench_warmup)
    print(f"backend={jax.default_backend()}  devices={jax.devices()}")
    print(
        f"env_type={FLAGS.env_type}  topology={FLAGS.topology_name}  "
        f"link_resources={FLAGS.link_resources}  k={FLAGS.k}  NUM_ENVS={n_envs}  steps={n_steps}"
    )
    print("-" * 78)

    results = {}
    for mixed in (False, True):
        bpe, fps, dt, dtypes = _bench_one(mixed, n_envs, n_steps, warmup)
        results[mixed] = (bpe, fps, dt)
        label = "MIXED  " if mixed else "DEFAULT"
        print(
            f"[{label}] env_state: {bpe:,} B/env x {n_envs} = {bpe * n_envs / 1e6:.2f} MB   "
            f"FPS={fps:.3e}   rollout={dt:.3f}s"
        )
        print(f"          dtypes: {dtypes}")

    bpe_d, fps_d, _ = results[False]
    bpe_m, fps_m, _ = results[True]
    print("-" * 78)
    print(
        f"env-state memory:  {bpe_d * n_envs / 1e6:.2f} MB -> {bpe_m * n_envs / 1e6:.2f} MB  "
        f"({100 * (bpe_d - bpe_m) / bpe_d:.1f}% reduction)"
    )
    print(
        f"throughput:        {fps_d:.3e} -> {fps_m:.3e} FPS  "
        f"({100 * (fps_m - fps_d) / fps_d:+.1f}%)  "
        f"[NB: fp16 compute is emulated on CPU; speed signal is GPU-only]"
    )


if __name__ == "__main__":
    app.run(main)
