"""
Gradient Sense-Check for Differentiable RSA Environment.

This script verifies that gradients from the differentiable environment
point towards better action combinations. It uses a minimal 2-step rollout
on the 5-node undirected topology with RWA (1 slot per request).

Setup:
  - 5-node undirected topology, k=3 paths, 4 link resources
  - RWA mode (guardband=0, each request uses 1 slot)
  - Action space: 3 paths * 4 slots = 12 actions (0..11)
  - 2 deterministic requests: both from node 0 → node 1
  - Incremental loading (no expiry)
  - Empty network at start

Expected reward landscape:
  - Both succeed (no collision): total reward = 0
  - One collides: total reward = -1
  - Both fail (e.g. overflow): total reward = -2

The test checks that:
  1. Gradients are non-zero at suboptimal action points
  2. Gradients point toward higher-reward regions
  3. Adam optimization converges to an optimal action pair
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from absl import flags
from functools import partial

# Must parse flags before importing xlron modules that use them
FLAGS = flags.FLAGS
# Parse with allow_unparse so we can run standalone
if not FLAGS.is_parsed():
    FLAGS(["gradient_sense_check"])

from xlron import dtype_config
from xlron.environments.env_funcs import *
from xlron.environments.dataclasses import EnvState
from xlron.train.train_utils import *
from xlron.environments.make_env import *


def make_config(list_of_requests, total_timesteps):
    """Create a minimal differentiable RWA config."""
    env_config = {
        "env_type": "rwa",
        "k": 3,
        "link_resources": 4,
        "incremental_loading": True,
        "end_first_blocking": False,
        "topology_name": "5node_undirected",
        "values_bw": [1],
        "slot_size": 1,
        "max_requests": total_timesteps,
        "temperature": 1.0,
        "deterministic_requests": True,
        "differentiable": True,
    }
    train_config = {
        "SEED": 0,
        "NUM_LEARNERS": 1,
        "TOTAL_TIMESTEPS": total_timesteps,
        "STEPS_PER_INCREMENT": total_timesteps,
        "NUM_MINIBATCHES": 1,
        "NUM_ENVS": 1,
        "ROLLOUT_LENGTH": total_timesteps,
        "UPDATE_EPOCHS": 1,
        "LR": 1e-3,
        "GAMMA": 1.0,
        "GAE_LAMBDA": 1.0,
        "LR_SCHEDULE": "constant",
    }
    config = process_config(FLAGS, **env_config, **train_config)
    config.list_of_requests = jnp.array(list_of_requests)
    return config


def init_environment(config):
    """Initialize env and get initial state."""
    dtype_config.initialize_dtypes(config)
    env, env_params = make(config, log_wrapper=False)
    rng = jax.random.PRNGKey(config.SEED)
    rng, key = jax.random.split(rng)
    obsv, env_state = env.reset(key, env_params)
    # Cast to float32 for differentiability
    env_state = jax.tree.map(lambda x: x.astype(jnp.float32), env_state)
    return env, env_params, key, env_state


def create_env_step(env):
    """Create a step function compatible with jax.lax.scan."""
    def env_step(runner_state, action):
        key, env_state, env_params = runner_state
        # step_env returns 6 values: obs, state, reward, terminal, truncated, info
        obs, state, reward, terminal, truncated, info = env.step_env(
            key, env_state, action, env_params
        )
        return (key, state, env_params), reward
    return env_step


def create_rollout_fn(env, env_step, key, env_params):
    """Create a jitted rollout function."""
    def rollout(runner_state, actions):
        # Reset to initial state each time
        _, st = env.reset(key, env_params)
        st = jax.tree.map(lambda x: x.astype(jnp.float32), st)
        r_state = (runner_state[0], st, runner_state[2])
        out, rew = jax.lax.scan(env_step, r_state, actions)
        return out, rew
    return jax.jit(rollout)


def create_loss_fn(rollout_fn, key, env_state, env_params):
    """Loss = negative total reward (we want to maximize reward = minimize loss)."""
    def get_loss(action_pair):
        actions = jnp.array(action_pair, dtype=jnp.float32)
        runner_state = (key, env_state, env_params)
        _, rewards = rollout_fn(runner_state, actions)
        return -jnp.sum(rewards)  # Negate: minimize loss = maximize reward
    return jax.jit(get_loss)


def create_reward_fn(rollout_fn, key, env_state, env_params):
    """Raw total reward (for landscape visualization)."""
    def get_reward(action_pair):
        actions = jnp.array(action_pair, dtype=jnp.float32)
        runner_state = (key, env_state, env_params)
        _, rewards = rollout_fn(runner_state, actions)
        return jnp.sum(rewards)
    return jax.jit(get_reward)


def compute_landscape(reward_fn, loss_fn, num_actions=12, step=1.0):
    """Compute reward and gradient for every integer action pair."""
    action_vals = np.arange(0, num_actions, step)
    results = []
    for a1 in action_vals:
        for a2 in action_vals:
            pair = jnp.array([a1, a2], dtype=jnp.float32)
            reward = float(reward_fn(pair))
            loss, grad = jax.value_and_grad(loss_fn)(pair)
            results.append({
                'a1': float(a1), 'a2': float(a2),
                'reward': reward, 'loss': float(loss),
                'grad_a1': float(grad[0]), 'grad_a2': float(grad[1]),
            })
    return results


def print_reward_landscape(results, num_actions):
    """Print reward as a grid."""
    print("\n=== REWARD LANDSCAPE (rows=action1, cols=action2) ===")
    header = "     " + "".join(f"{a:>6.0f}" for a in range(num_actions))
    print(header)
    for a1 in range(num_actions):
        row_data = [r for r in results if int(r['a1']) == a1]
        row_data.sort(key=lambda r: r['a2'])
        row_str = f"{a1:>4.0f} " + "".join(f"{r['reward']:>6.1f}" for r in row_data)
        print(row_str)


def print_gradient_landscape(results, num_actions):
    """Print gradient magnitudes as a grid."""
    print("\n=== GRADIENT MAGNITUDE (rows=action1, cols=action2) ===")
    header = "     " + "".join(f"{a:>8.0f}" for a in range(num_actions))
    print(header)
    for a1 in range(num_actions):
        row_data = [r for r in results if int(r['a1']) == a1]
        row_data.sort(key=lambda r: r['a2'])
        mag = [np.sqrt(r['grad_a1']**2 + r['grad_a2']**2) for r in row_data]
        row_str = f"{a1:>4.0f} " + "".join(f"{m:>8.3f}" for m in mag)
        print(row_str)


def print_gradient_directions(results, num_actions):
    """Print gradient direction arrows."""
    print("\n=== GRADIENT DIRECTION (rows=action1, cols=action2) ===")
    print("  (arrows show direction loss DECREASES = reward INCREASES)")
    header = "     " + "".join(f"{a:>4.0f}" for a in range(num_actions))
    print(header)
    for a1 in range(num_actions):
        row_data = [r for r in results if int(r['a1']) == a1]
        row_data.sort(key=lambda r: r['a2'])
        arrows = []
        for r in row_data:
            g1, g2 = r['grad_a1'], r['grad_a2']
            mag = np.sqrt(g1**2 + g2**2)
            if mag < 1e-8:
                arrows.append("  . ")
            else:
                # Gradient points uphill in loss space; negate for reward direction
                # (we want to show where reward increases)
                angle = np.arctan2(-g2, -g1) * 180 / np.pi
                if -22.5 <= angle < 22.5:
                    arrows.append("  > ")  # a1 increases
                elif 22.5 <= angle < 67.5:
                    arrows.append("  / ")
                elif 67.5 <= angle < 112.5:
                    arrows.append("  ^ ")  # a2 increases
                elif 112.5 <= angle < 157.5:
                    arrows.append("  \\ ")
                elif angle >= 157.5 or angle < -157.5:
                    arrows.append("  < ")  # a1 decreases
                elif -157.5 <= angle < -112.5:
                    arrows.append("  / ")
                elif -112.5 <= angle < -67.5:
                    arrows.append("  v ")  # a2 decreases
                else:
                    arrows.append("  \\ ")
        row_str = f"{a1:>4.0f} " + "".join(arrows)
        print(row_str)


def check_gradient_quality(results, num_actions):
    """Analyze whether gradients are sensible."""
    print("\n=== GRADIENT QUALITY ANALYSIS ===")

    # Find optimal reward
    max_reward = max(r['reward'] for r in results)
    optimal_points = [(r['a1'], r['a2']) for r in results if r['reward'] == max_reward]
    print(f"Optimal reward: {max_reward}")
    print(f"Optimal action pairs: {optimal_points[:10]}...")

    # Check 1: Are gradients non-zero at suboptimal points?
    suboptimal = [r for r in results if r['reward'] < max_reward]
    if suboptimal:
        grad_mags = [np.sqrt(r['grad_a1']**2 + r['grad_a2']**2) for r in suboptimal]
        nonzero = sum(1 for m in grad_mags if m > 1e-8)
        print(f"\nSuboptimal points: {len(suboptimal)}")
        print(f"  With non-zero gradient: {nonzero}/{len(suboptimal)} ({100*nonzero/len(suboptimal):.0f}%)")
        print(f"  Mean gradient magnitude: {np.mean(grad_mags):.6f}")
        print(f"  Max gradient magnitude: {np.max(grad_mags):.6f}")

    # Check 2: At optimal points, are gradients near zero?
    optimal_results = [r for r in results if r['reward'] == max_reward]
    if optimal_results:
        grad_mags = [np.sqrt(r['grad_a1']**2 + r['grad_a2']**2) for r in optimal_results]
        near_zero = sum(1 for m in grad_mags if m < 0.1)
        print(f"\nOptimal points: {len(optimal_results)}")
        print(f"  With near-zero gradient: {near_zero}/{len(optimal_results)}")
        print(f"  Mean gradient magnitude: {np.mean(grad_mags):.6f}")

    # Check 3: Do gradients at suboptimal points move toward better rewards?
    # For each suboptimal point, check if following the negative gradient improves reward
    if suboptimal:
        improved = 0
        total_checked = 0
        for r in suboptimal:
            g1, g2 = r['grad_a1'], r['grad_a2']
            mag = np.sqrt(g1**2 + g2**2)
            if mag < 1e-8:
                continue
            # Step in negative gradient direction (minimize loss = maximize reward)
            step_size = 0.5
            new_a1 = r['a1'] - step_size * g1 / mag
            new_a2 = r['a2'] - step_size * g2 / mag
            # Find nearest integer action pair to the target
            nearest_a1 = int(np.clip(np.round(new_a1), 0, num_actions - 1))
            nearest_a2 = int(np.clip(np.round(new_a2), 0, num_actions - 1))
            # Look up reward at nearest point
            target = [t for t in results if int(t['a1']) == nearest_a1 and int(t['a2']) == nearest_a2]
            if target:
                total_checked += 1
                if target[0]['reward'] >= r['reward']:
                    improved += 1

        print(f"\nGradient direction check:")
        print(f"  Points where gradient points toward equal/better reward: "
              f"{improved}/{total_checked} ({100*improved/max(total_checked,1):.0f}%)")


def run_optimization(loss_fn, reward_fn, num_actions, n_iters=200, lr=0.05):
    """Run Adam optimization from various starting points."""
    print("\n=== OPTIMIZATION TEST ===")

    # Find true optimal
    best_reward = -float('inf')
    best_pair = None
    for a1 in range(num_actions):
        for a2 in range(num_actions):
            r = float(reward_fn(jnp.array([float(a1), float(a2)])))
            if r > best_reward:
                best_reward = r
                best_pair = (a1, a2)
    print(f"True optimal: actions={best_pair}, reward={best_reward}")

    # Test 1: Symmetric starts (challenging — gradient symmetry)
    print("\n  --- Symmetric starts (both actions identical) ---")
    symmetric_starts = [
        jnp.array([0.0, 0.0]),
        jnp.array([5.0, 5.0]),
        jnp.array([11.0, 11.0]),
    ]
    for start in symmetric_starts:
        result = _optimize_once(loss_fn, reward_fn, start, num_actions, n_iters, lr, best_reward)
        _print_result(start, result)

    # Test 2: Asymmetric starts (should reliably converge)
    print("\n  --- Asymmetric starts (different initial actions) ---")
    asymmetric_starts = [
        jnp.array([0.0, 3.0]),
        jnp.array([5.0, 8.0]),
        jnp.array([1.0, 11.0]),
        jnp.array([3.0, 7.0]),
        jnp.array([2.0, 9.0]),
    ]
    for start in asymmetric_starts:
        result = _optimize_once(loss_fn, reward_fn, start, num_actions, n_iters, lr, best_reward)
        _print_result(start, result)

    # Test 3: Symmetric starts with small perturbation to break symmetry
    print("\n  --- Symmetric starts + tiny perturbation (eps=0.1) ---")
    for base in [0.0, 5.0, 11.0]:
        start = jnp.array([base, base + 0.1])
        result = _optimize_once(loss_fn, reward_fn, start, num_actions, n_iters, lr, best_reward)
        _print_result(start, result)

    # Test 4: Random starts (10 trials)
    print("\n  --- Random starts (10 trials) ---")
    rng = jax.random.PRNGKey(42)
    n_success = 0
    n_trials = 10
    for i in range(n_trials):
        rng, key = jax.random.split(rng)
        start = jax.random.uniform(key, shape=(2,)) * (num_actions - 1)
        result = _optimize_once(loss_fn, reward_fn, start, num_actions, n_iters, lr, best_reward)
        n_success += int(result['converged'])
    print(f"  Random starts success rate: {n_success}/{n_trials} ({100*n_success/n_trials:.0f}%)")


def _optimize_once(loss_fn, reward_fn, start, num_actions, n_iters, lr, best_reward):
    """Run a single optimization from a starting point."""
    actions = start
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(actions)

    best_found_reward = -float('inf')
    best_found_actions = actions

    for i in range(n_iters):
        loss_val, grads = jax.value_and_grad(loss_fn)(actions)
        updates, opt_state = optimizer.update(grads, opt_state)
        actions = optax.apply_updates(actions, updates)
        actions = jnp.clip(actions, 0, num_actions - 1)

        reward = float(reward_fn(jnp.round(actions)))
        if reward > best_found_reward:
            best_found_reward = reward
            best_found_actions = jnp.round(actions)

    rounded = jnp.round(actions)
    final_reward = float(reward_fn(rounded))
    return {
        'actions': actions,
        'rounded': rounded,
        'final_reward': final_reward,
        'best_reward': best_found_reward,
        'converged': final_reward == best_reward,
    }


def _print_result(start, result):
    a = result['actions']
    r = result['rounded']
    print(f"  [{start[0]:5.1f},{start[1]:5.1f}] -> [{float(a[0]):6.3f},{float(a[1]):6.3f}] "
          f"(rounded [{float(r[0]):2.0f},{float(r[1]):2.0f}]) "
          f"reward={result['final_reward']:5.1f} "
          f"{'OK' if result['converged'] else 'FAIL'}")


def main():
    """Run the gradient sense check."""
    # 2 deterministic requests: both 0→1, bw=1
    # Format: [source, bw, dest, arrival_time, holding_time, current_time]
    list_of_requests = [
        [0., 1., 1., 0., 1e8, 0.],
        [0., 1., 1., 0., 1e8, 0.],
    ]
    total_timesteps = 2

    print("=" * 60)
    print("GRADIENT SENSE CHECK FOR DIFFERENTIABLE RSA")
    print("=" * 60)
    print(f"Topology: 5-node undirected")
    print(f"k paths: 3, link resources: 4")
    print(f"Mode: RWA (1 slot per request, no guardband)")
    print(f"Requests: 2 identical (node 0 → node 1)")
    print(f"Action space: 12 (3 paths x 4 slots)")

    config = make_config(list_of_requests, total_timesteps)
    env, env_params, key, env_state = init_environment(config)

    print(f"\nEnvironment created successfully")
    print(f"  num_links: {env_params.num_links}")
    print(f"  link_resources: {env_params.link_resources}")
    print(f"  k_paths: {env_params.k_paths}")
    print(f"  differentiable: {env_params.differentiable}")
    print(f"  temperature: {env_params.temperature}")

    # Create stepping and rollout functions
    env_step = create_env_step(env)
    rollout_fn = create_rollout_fn(env, env_step, key, env_params)

    # Create loss and reward functions
    loss_fn = create_loss_fn(rollout_fn, key, env_state, env_params)
    reward_fn = create_reward_fn(rollout_fn, key, env_state, env_params)

    # Warm up JIT
    print("\nCompiling...")
    dummy = loss_fn(jnp.zeros(2))
    print(f"  Compiled. Dummy loss: {dummy}")

    num_actions = 12

    # Compute reward and gradient for all integer action pairs
    print(f"\nComputing reward/gradient landscape for {num_actions}x{num_actions} grid...")
    results = compute_landscape(reward_fn, loss_fn, num_actions)

    # Print results
    print_reward_landscape(results, num_actions)
    print_gradient_landscape(results, num_actions)
    print_gradient_directions(results, num_actions)
    check_gradient_quality(results, num_actions)

    # Run optimization
    run_optimization(loss_fn, reward_fn, num_actions, n_iters=500, lr=0.05)

    # Now test with higher temperature
    for temp in [5.0, 10.0]:
        print(f"\n{'=' * 60}")
        print(f"TEMPERATURE = {temp}")
        print(f"{'=' * 60}")

        config_ht = make_config(list_of_requests, total_timesteps)
        config_ht.temperature = temp
        env_ht, env_params_ht, key_ht, env_state_ht = init_environment(config_ht)

        env_step_ht = create_env_step(env_ht)
        rollout_fn_ht = create_rollout_fn(env_ht, env_step_ht, key_ht, env_params_ht)
        loss_fn_ht = create_loss_fn(rollout_fn_ht, key_ht, env_state_ht, env_params_ht)
        reward_fn_ht = create_reward_fn(rollout_fn_ht, key_ht, env_state_ht, env_params_ht)

        # Warm up
        _ = loss_fn_ht(jnp.zeros(2))

        # Quick gradient check at suboptimal and optimal points
        print(f"\nGradient samples at temperature={temp}:")
        for pair in [(0,0), (1,1), (5,5), (0,1), (2,5)]:
            p = jnp.array(pair, dtype=jnp.float32)
            r = float(reward_fn_ht(p))
            _, g = jax.value_and_grad(loss_fn_ht)(p)
            print(f"  ({pair[0]:2d},{pair[1]:2d}): reward={r:5.1f}, grad=({g[0]:+.6f}, {g[1]:+.6f}), |grad|={float(jnp.linalg.norm(g)):.6f}")

        run_optimization(loss_fn_ht, reward_fn_ht, num_actions, n_iters=500, lr=0.05)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
