"""
Plot the 2-action test case for the differentiable RSA environment.

Generates publication-quality 3D surface plots showing:
  1. Reward landscape (total reward as a function of two actions)
  2. Gradient magnitude landscape
  3. 2D heatmap with gradient direction arrows
  4. Optimization trajectories from multiple starting points

Two cases are produced:
  A) "Simple" (from differentiable.ipynb / case_1.py): constrained network where
     only slot 0 is free on each link, so only 3 valid actions exist per request.
  B) "Full": empty network with all 4 slots available on all links.

Both use:
  - 5-node undirected topology, k=3 paths, 3 link resources
  - RSA mode (2 FSU per request, no guardband)
  - Action space: 9 actions (3 paths x 3 slots = 0..8)
  - 2 deterministic requests
  - Incremental loading (no expiry)

Usage:
  uv run python experimental/differentiable/plot_test_case.py
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Parse flags before importing xlron
from absl import flags
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(["plot_test_case"])

import jax
import jax.numpy as jnp
import optax
from xlron import dtype_config
from xlron.environments.env_funcs import *
from xlron.environments.dataclasses import EnvState
from xlron.train.train_utils import *
from xlron.environments.make_env import *

# Import project style
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plot_style import configure_style, PALETTE, ACCENT_COLORS, PRIMARY_COLORS

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

# Colors from palette for trajectories
TRAJ_COLORS = [
    PRIMARY_COLORS[0],  # teal
    ACCENT_COLORS[1],   # coral
    ACCENT_COLORS[0],   # purple
    ACCENT_COLORS[3],   # orange
    PRIMARY_COLORS[3],  # dark teal
]


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def make_config(list_of_requests, total_timesteps, temperature=5.0):
    env_config = {
        "env_type": "rsa",
        "k": 3,
        "link_resources": 3,
        "incremental_loading": True,
        "end_first_blocking": False,
        "topology_name": "5node_undirected",
        "values_bw": [2],
        "slot_size": 1,
        "max_requests": total_timesteps,
        "temperature": temperature,
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
    dtype_config.initialize_dtypes(config)
    env, env_params = make(config, log_wrapper=False)
    rng = jax.random.PRNGKey(config.SEED)
    rng, key = jax.random.split(rng)
    _, env_state = env.reset(key, env_params)
    env_state = jax.tree.map(lambda x: x.astype(jnp.float32), env_state)
    return env, env_params, key, env_state


def setup_case_simple():
    """Constrained case from differentiable.ipynb / case_1.py.

    Only slot 0 is free on each link; slots 1-2 are occupied.
    Requests: node 1 -> node 3 (using the 3-value format).
    Valid actions: 0, 3, 6 (one per path, all on slot 0).
    """
    def reset_links(state):
        state = state.replace(
            link_slot_array=state.link_slot_array.at[:, 0].set(0.),
        )
        state = state.replace(
            link_slot_array=state.link_slot_array.at[:, 1:].set(-1.),
        )
        state = state.replace(
            link_slot_departure_array=state.link_slot_departure_array.at[:, 0].set(0.),
        )
        state = state.replace(
            link_slot_departure_array=state.link_slot_departure_array.at[:, 1:].set(1e8),
        )
        state = jax.tree.map(lambda x: x.astype(jnp.float32), state)
        return state

    list_of_requests = [
        [1., 1., 3.],
        [1., 1., 3.],
    ]
    return list_of_requests, reset_links


def create_env_step(env):
    """Step function compatible with jax.lax.scan."""
    def env_step(runner_state, action):
        key, env_state, env_params = runner_state
        obs, state, reward, terminal, truncated, info = env.step(
            key, env_state, action, env_params
        )
        return (key, state, env_params), reward
    return env_step


def create_rollout_fn(env, env_step, key, env_params, reset_fn=None):
    def rollout(runner_state, actions):
        _, st = env.reset(key, env_params)
        st = jax.tree.map(lambda x: x.astype(jnp.float32), st)
        if reset_fn is not None:
            st = reset_fn(st)
        r_state = (runner_state[0], st, runner_state[2])
        out, rew = jax.lax.scan(env_step, r_state, actions)
        return out, rew
    return jax.jit(rollout)


def create_reward_fn(rollout_fn, key, env_state, env_params):
    def get_reward(action_pair):
        actions = jnp.array(action_pair, dtype=jnp.float32)
        runner_state = (key, env_state, env_params)
        _, rewards = rollout_fn(runner_state, actions)
        return jnp.sum(rewards)
    return jax.jit(get_reward)


def create_loss_fn(rollout_fn, key, env_state, env_params):
    def get_loss(action_pair):
        actions = jnp.array(action_pair, dtype=jnp.float32)
        runner_state = (key, env_state, env_params)
        _, rewards = rollout_fn(runner_state, actions)
        return -jnp.sum(rewards)
    return jax.jit(get_loss)


# ---------------------------------------------------------------------------
# Landscape computation
# ---------------------------------------------------------------------------

def compute_landscape(reward_fn, loss_fn, num_actions=9, step=0.5):
    """Compute reward and gradient on a fine grid."""
    action_vals = np.arange(0, num_actions, step)
    n = len(action_vals)
    rewards = np.zeros((n, n))
    grad_a1 = np.zeros((n, n))
    grad_a2 = np.zeros((n, n))

    for i, a1 in enumerate(action_vals):
        for j, a2 in enumerate(action_vals):
            pair = jnp.array([a1, a2], dtype=jnp.float32)
            rewards[i, j] = float(reward_fn(pair))
            loss, grad = jax.value_and_grad(loss_fn)(pair)
            grad_a1[i, j] = -float(grad[0])
            grad_a2[i, j] = -float(grad[1])
        print(f"  Row {i+1}/{n} done")

    grad_mag = np.sqrt(grad_a1**2 + grad_a2**2)
    return action_vals, rewards, grad_a1, grad_a2, grad_mag


def gaussian_smooth_reward(reward_fn, action_pair, num_actions=9, sigma=0.8):
    """Compute Gaussian-smoothed reward by evaluating all integer neighbours."""
    # Evaluate reward at every integer action pair
    all_a1 = jnp.arange(num_actions, dtype=jnp.float32)
    all_a2 = jnp.arange(num_actions, dtype=jnp.float32)
    grid_a1, grid_a2 = jnp.meshgrid(all_a1, all_a2, indexing='ij')
    int_pairs = jnp.stack([grid_a1.ravel(), grid_a2.ravel()], axis=1)  # (N^2, 2)

    int_rewards = jax.vmap(reward_fn)(int_pairs)  # (N^2,)

    # Gaussian weights based on distance from action_pair
    diffs = int_pairs - action_pair[None, :]  # (N^2, 2)
    sq_dist = jnp.sum(diffs ** 2, axis=1)
    weights = jnp.exp(-0.5 * sq_dist / (sigma ** 2))
    weights = weights / jnp.sum(weights)

    return jnp.sum(weights * int_rewards)


def compute_landscape_gaussian(reward_fn, num_actions=9, step=0.5, sigma=0.8):
    """Compute Gaussian-smoothed reward and its gradient on a fine grid."""
    action_vals = np.arange(0, num_actions, step)
    n = len(action_vals)
    rewards = np.zeros((n, n))
    grad_a1 = np.zeros((n, n))
    grad_a2 = np.zeros((n, n))

    smooth_fn = jax.jit(lambda pair: gaussian_smooth_reward(
        reward_fn, pair, num_actions=num_actions, sigma=sigma
    ))
    smooth_loss_fn = jax.jit(lambda pair: -gaussian_smooth_reward(
        reward_fn, pair, num_actions=num_actions, sigma=sigma
    ))

    # Compile once
    _ = smooth_fn(jnp.zeros(2))
    _ = jax.value_and_grad(smooth_loss_fn)(jnp.zeros(2))

    for i, a1 in enumerate(action_vals):
        for j, a2 in enumerate(action_vals):
            pair = jnp.array([a1, a2], dtype=jnp.float32)
            rewards[i, j] = float(smooth_fn(pair))
            loss, grad = jax.value_and_grad(smooth_loss_fn)(pair)
            grad_a1[i, j] = -float(grad[0])
            grad_a2[i, j] = -float(grad[1])
        print(f"  Row {i+1}/{n} done")

    grad_mag = np.sqrt(grad_a1**2 + grad_a2**2)
    return action_vals, rewards, grad_a1, grad_a2, grad_mag


# ---------------------------------------------------------------------------
# Optimization trajectory
# ---------------------------------------------------------------------------

def run_optimization_trajectory(loss_fn, reward_fn, start, num_actions=9,
                                 n_iters=300, lr=0.05):
    """Run Adam and return the full trajectory."""
    actions = jnp.array(start, dtype=jnp.float32)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(actions)

    trajectory = [np.array(actions)]
    rewards_hist = [float(reward_fn(jnp.round(actions)))]

    for _ in range(n_iters):
        loss_val, grads = jax.value_and_grad(loss_fn)(actions)
        updates, opt_state = optimizer.update(grads, opt_state)
        actions = optax.apply_updates(actions, updates)
        actions = jnp.clip(actions, 0, num_actions - 1)
        trajectory.append(np.array(actions))
        rewards_hist.append(float(reward_fn(jnp.round(actions))))

    return np.array(trajectory), np.array(rewards_hist)


# ---------------------------------------------------------------------------
# Plotting (uses plot_style.py colors + Arial)
# ---------------------------------------------------------------------------

def plot_reward_surface(action_vals, rewards, suffix="", save=True):
    """3D surface plot of the reward landscape."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(action_vals, action_vals)
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=-1.0, vmax=0.0)

    surf = ax.plot_surface(
        X, Y, rewards.T,
        cmap='RdYlGn', norm=norm,
        linewidth=0.2, edgecolor='k', alpha=0.92,
        antialiased=True, rcount=100, ccount=100,
    )

    ax.set_xlabel('Action 1 (path x slot)', labelpad=12, fontsize=16)
    ax.set_ylabel('Action 2 (path x slot)', labelpad=12, fontsize=16)
    ax.set_zlabel('Total Reward', labelpad=10, fontsize=16)
    ax.set_title('Reward Landscape: 2-Request RWA Test Case', fontsize=18, pad=16)

    ax.text2D(0.02, 0.92, 'Reward = 0: both succeed',
              transform=ax.transAxes, fontsize=12, color='#2d8a2d')
    ax.text2D(0.02, 0.87, 'Reward = \u22121: one fail',
              transform=ax.transAxes, fontsize=12, color='#c4a000')
    ax.text2D(0.02, 0.82, 'Reward = \u22122: both fail',
              transform=ax.transAxes, fontsize=12, color='#cc0000')

    ax.view_init(elev=25, azim=-50)
    ax.set_zlim(-2.1, 0.1)
    ax.tick_params(labelsize=11)
    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.08, label='Total Reward')

    if save:
        fname = f"reward_landscape_3d{suffix}.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname))
        print(f"  Saved {fname}")
    return fig, ax


def plot_gradient_surface(action_vals, grad_mag, suffix="", save=True):
    """3D surface plot of gradient magnitude."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(action_vals, action_vals)

    surf = ax.plot_surface(
        X, Y, grad_mag.T,
        cmap='inferno', linewidth=0.2, edgecolor='k',
        alpha=0.92, antialiased=True, rcount=100, ccount=100,
    )

    ax.set_xlabel('Action 1', labelpad=12, fontsize=16)
    ax.set_ylabel('Action 2', labelpad=12, fontsize=16)
    ax.set_zlabel('|$\\nabla$R|', labelpad=10, fontsize=16)
    ax.set_title('Gradient Magnitude Landscape', fontsize=18, pad=16)
    ax.view_init(elev=25, azim=-50)
    ax.tick_params(labelsize=11)
    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.08,
                 label='Gradient Magnitude')

    if save:
        fname = f"gradient_magnitude_3d{suffix}.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname))
        print(f"  Saved {fname}")
    return fig, ax


def plot_heatmap_with_arrows(action_vals, rewards, grad_a1, grad_a2, grad_mag,
                              loss_fn=None, reward_fn=None, num_actions=9,
                              suffix="", save=True):
    """2D heatmap of reward with gradient direction arrows overlaid.

    If loss_fn and reward_fn are provided, optimization trajectories are
    overlaid on the left (reward) panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    X, Y = np.meshgrid(action_vals, action_vals)
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=-1.0, vmax=0.0)

    # --- Left: Reward heatmap (+ trajectories if available) ---
    ax = axes[0]
    im = ax.pcolormesh(X, Y, rewards.T, cmap='RdYlGn', norm=norm, shading='auto')
    cb1 = fig.colorbar(im, ax=ax, label='Total Reward')
    cb1.ax.tick_params(labelsize=20)
    cb1.ax.yaxis.label.set_size(24)

    if loss_fn is not None and reward_fn is not None:
        starts = [
            [1.0, 1.0],
            [0.0, 4.0],
            [6.0, 2.0],
            [4.0, 7.0],
            [8.0, 8.0],
        ]
        for i, start in enumerate(starts):
            traj, _ = run_optimization_trajectory(
                loss_fn, reward_fn, start, num_actions, n_iters=500, lr=0.05
            )
            color = TRAJ_COLORS[i]
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color, lw=1.5, alpha=0.6)
            step = max(1, len(traj) // 30)
            ax.plot(traj[::step, 0], traj[::step, 1], '.', color=color, ms=4, alpha=0.7)
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, ms=12, mec='k',
                    mew=1.0, zorder=5)
            ax.plot(traj[-1, 0], traj[-1, 1], '*', color=color, ms=16, mec='k',
                    mew=1.0, zorder=5)
        ax.plot([], [], 'o', color='gray', ms=10, mec='k', mew=1.0, label='Start')
        ax.plot([], [], '*', color='gray', ms=14, mec='k', mew=1.0, label='Finish')
        ax.legend(loc='center right', framealpha=0.9, fontsize=20)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)

    ax.set_xlabel('Action 1 (path x slot)', fontsize=26)
    ax.set_ylabel('Action 2 (path x slot)', fontsize=26)
    ax.set_title('Reward Landscape', fontsize=28)
    ax.tick_params(labelsize=20)
    ax.set_aspect('equal')

    # --- Right: Reward heatmap + gradient arrows ---
    ax = axes[1]
    im2 = ax.pcolormesh(X, Y, rewards.T, cmap='RdYlGn', norm=norm, shading='auto',
                         alpha=0.6)
    cb2 = fig.colorbar(im2, ax=ax, label='Total Reward')
    cb2.ax.tick_params(labelsize=20)
    cb2.ax.yaxis.label.set_size(24)

    stride = 2
    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]
    U = grad_a1[::stride, ::stride].T
    V = grad_a2[::stride, ::stride].T
    M = grad_mag[::stride, ::stride].T

    max_mag = M.max() + 1e-12
    U_norm = U / max_mag * 0.6
    V_norm = V / max_mag * 0.6

    ax.quiver(Xs, Ys, U_norm, V_norm, M,
              cmap='plasma', scale=1.0, scale_units='xy',
              width=0.004, headwidth=4, headlength=4,
              clim=[0, max_mag])
    ax.set_xlabel('Action 1 (path x slot)', fontsize=26)
    ax.set_ylabel('Action 2 (path x slot)', fontsize=26)
    ax.set_title('Gradient Direction (arrows point to higher reward)', fontsize=28)
    ax.tick_params(labelsize=20)
    ax.set_aspect('equal')

    plt.tight_layout()
    if save:
        fname = f"reward_heatmap_with_gradients{suffix}.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname))
        print(f"  Saved {fname}")
    return fig


def plot_optimization_trajectories(action_vals, rewards, loss_fn, reward_fn,
                                    num_actions=9, suffix="", save=True):
    """Show optimization trajectories on the reward heatmap."""
    starts = [
        ([1.0, 1.0], TRAJ_COLORS[0], 'o'),
        ([0.0, 4.0], TRAJ_COLORS[1], 's'),
        ([6.0, 2.0], TRAJ_COLORS[2], '^'),
        ([4.0, 7.0], TRAJ_COLORS[3], 'D'),
        ([8.0, 8.0], TRAJ_COLORS[4], 'v'),
    ]

    # Pre-compute trajectories (avoid computing twice)
    trajectories = []
    for start, color, marker in starts:
        traj, rhist = run_optimization_trajectory(
            loss_fn, reward_fn, start, num_actions, n_iters=500, lr=0.05
        )
        trajectories.append((start, color, marker, traj, rhist))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    X, Y = np.meshgrid(action_vals, action_vals)
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=-1.0, vmax=0.0)

    ax.pcolormesh(X, Y, rewards.T, cmap='RdYlGn', norm=norm, shading='auto',
                  alpha=0.7)

    for start, color, marker, traj, rhist in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, lw=1.5, alpha=0.6)
        # Show per-step dots (subsample for clarity)
        step = max(1, len(traj) // 30)
        ax.plot(traj[::step, 0], traj[::step, 1], '.', color=color, ms=4, alpha=0.7)
        # Start marker
        ax.plot(traj[0, 0], traj[0, 1], marker, color=color, ms=12, mec='k',
                mew=1.0, zorder=5,
                label=f'Start ({start[0]:.0f},{start[1]:.0f})')
        # End marker
        ax.plot(traj[-1, 0], traj[-1, 1], '*', color=color, ms=16, mec='k',
                mew=1.0, zorder=5)

    ax.set_xlabel('Action 1', fontsize=16)
    ax.set_ylabel('Action 2', fontsize=16)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)

    plt.tight_layout()
    if save:
        fname = f"optimization_trajectories{suffix}.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname))
        print(f"  Saved {fname}")
    return fig


def plot_heatmap_with_trajectories(action_vals, rewards, loss_fn, reward_fn,
                                    num_actions=9, suffix="", save=True):
    """Reward heatmap with optimization trajectories overlaid."""
    starts = [
        [1.0, 1.0],
        [0.0, 4.0],
        [6.0, 2.0],
        [4.0, 7.0],
        [8.0, 8.0],
    ]

    trajectories = []
    for i, start in enumerate(starts):
        traj, rhist = run_optimization_trajectory(
            loss_fn, reward_fn, start, num_actions, n_iters=500, lr=0.05
        )
        trajectories.append((traj, rhist, TRAJ_COLORS[i]))

    fig, ax = plt.subplots()
    X, Y = np.meshgrid(action_vals, action_vals)
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=-1.0, vmax=0.0)

    im = ax.pcolormesh(X, Y, rewards.T, cmap='RdYlGn', norm=norm, shading='auto')
    fig.colorbar(im, ax=ax, label='Total Reward')

    # Plot trajectories
    for traj, rhist, color in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, lw=1.5, alpha=0.6)
        step = max(1, len(traj) // 30)
        ax.plot(traj[::step, 0], traj[::step, 1], '.', color=color, ms=4, alpha=0.7)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, ms=12, mec='k',
                mew=1.0, zorder=5)
        ax.plot(traj[-1, 0], traj[-1, 1], '*', color=color, ms=16, mec='k',
                mew=1.0, zorder=5)

    # Legend with just shape labels
    ax.plot([], [], 'o', color='gray', ms=10, mec='k', mew=1.0, label='Start')
    ax.plot([], [], '*', color='gray', ms=14, mec='k', mew=1.0, label='Finish')

    ax.set_xlabel('Action 1 (path x slot)')
    ax.set_ylabel('Action 2 (path x slot)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)

    plt.tight_layout()
    if save:
        fname = f"reward_heatmap_with_trajectories{suffix}.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname))
        print(f"  Saved {fname}")
    return fig


def plot_combined_3d(action_vals, rewards, grad_mag, suffix="", save=True):
    """Stacked 3D: reward surface (top) + gradient magnitude surface (bottom)."""
    fig = plt.figure(figsize=(11, 18))

    X, Y = np.meshgrid(action_vals, action_vals)
    norm_reward = TwoSlopeNorm(vmin=-2.0, vcenter=-1.0, vmax=0.0)

    ax1 = fig.add_subplot(211, projection='3d')
    surf1 = ax1.plot_surface(
        X, Y, rewards.T, cmap='RdYlGn', norm=norm_reward,
        linewidth=0.15, edgecolor='k', alpha=0.92,
        antialiased=True, rcount=100, ccount=100,
    )
    ax1.set_xlabel('Action 1', labelpad=14, fontsize=24)
    ax1.set_ylabel('Action 2', labelpad=14, fontsize=24)
    ax1.set_zlabel('Total Reward', labelpad=12, fontsize=24)
    ax1.set_title('Reward Landscape', fontsize=28, pad=16)
    ax1.view_init(elev=25, azim=-50)
    ax1.set_zlim(-2.1, 0.1)
    ax1.tick_params(labelsize=18)
    cb1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=12, pad=0.1, label='Total Reward')
    cb1.ax.tick_params(labelsize=18)
    cb1.ax.yaxis.label.set_size(22)

    ax2 = fig.add_subplot(212, projection='3d')
    surf2 = ax2.plot_surface(
        X, Y, grad_mag.T, cmap='inferno',
        linewidth=0.15, edgecolor='k', alpha=0.92,
        antialiased=True, rcount=100, ccount=100,
    )
    ax2.set_xlabel('Action 1', labelpad=14, fontsize=24)
    ax2.set_ylabel('Action 2', labelpad=14, fontsize=24)
    ax2.set_zlabel('|$\\nabla$R|', labelpad=12, fontsize=24)
    ax2.set_title('Gradient Magnitude', fontsize=28, pad=16)
    ax2.view_init(elev=25, azim=-50)
    ax2.tick_params(labelsize=18)
    cb2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=12, pad=0.1,
                 label='Gradient Magnitude')
    cb2.ax.tick_params(labelsize=18)
    cb2.ax.yaxis.label.set_size(22)

    plt.tight_layout()

    if save:
        fname = f"combined_3d{suffix}.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname))
        print(f"  Saved {fname}")
    return fig


def plot_combined_full_view(action_vals, rewards, grad_a1, grad_a2, grad_mag,
                             loss_fn=None, reward_fn=None, num_actions=9,
                             suffix="", save=True):
    """Two-column 2x2 combined view used in the JOCN paper.

    Top row:    3D reward landscape | 3D gradient magnitude (each with colorbar).
    Bottom row: 2D reward heatmap with optimisation trajectories | 2D reward
                heatmap with gradient-direction arrows.  No colorbars on the
                bottom row -- they reuse the top-row colorbars (RdYlGn for
                reward, inferno for gradient magnitude).
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.15, 1.00],
        hspace=0.20, wspace=0.02,
        left=0.05, right=0.95, top=0.97, bottom=0.07,
    )

    X, Y = np.meshgrid(action_vals, action_vals)
    norm_reward = TwoSlopeNorm(vmin=-2.0, vcenter=-1.0, vmax=0.0)

    # ---- Top-left: 3D reward landscape (no title, no z-axis label) ----
    ax_tl = fig.add_subplot(gs[0, 0], projection='3d')
    surf1 = ax_tl.plot_surface(
        X, Y, rewards.T, cmap='RdYlGn', norm=norm_reward,
        linewidth=0.15, edgecolor='k', alpha=0.92,
        antialiased=True, rcount=100, ccount=100,
    )
    ax_tl.set_xlabel('Action 1', labelpad=14, fontsize=24)
    ax_tl.set_ylabel('Action 2', labelpad=14, fontsize=24)
    ax_tl.set_zlabel('')
    ax_tl.view_init(elev=25, azim=-50)
    ax_tl.set_zlim(-2.1, 0.1)
    ax_tl.tick_params(labelsize=18)
    cb1 = fig.colorbar(surf1, ax=ax_tl, shrink=0.6, aspect=14, pad=0.08)
    cb1.set_label('Total Reward', fontsize=22)
    cb1.ax.tick_params(labelsize=18)

    # ---- Top-right: 3D gradient magnitude (no title, no z-axis label) ----
    ax_tr = fig.add_subplot(gs[0, 1], projection='3d')
    surf2 = ax_tr.plot_surface(
        X, Y, grad_mag.T, cmap='inferno',
        linewidth=0.15, edgecolor='k', alpha=0.92,
        antialiased=True, rcount=100, ccount=100,
    )
    ax_tr.set_xlabel('Action 1', labelpad=14, fontsize=24)
    ax_tr.set_ylabel('Action 2', labelpad=14, fontsize=24)
    ax_tr.set_zlabel('')
    ax_tr.view_init(elev=25, azim=-50)
    ax_tr.tick_params(labelsize=18)
    cb2 = fig.colorbar(surf2, ax=ax_tr, shrink=0.6, aspect=14, pad=0.08)
    cb2.set_label('$|\\Delta\\mathrm{Total\\ Reward}|$', fontsize=22)
    cb2.ax.tick_params(labelsize=18)

    # ---- Bottom-left: reward heatmap with trajectories (no colorbar) ----
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bl.pcolormesh(X, Y, rewards.T, cmap='RdYlGn', norm=norm_reward, shading='auto')

    if loss_fn is not None and reward_fn is not None:
        starts = [
            ([1.0, 1.0], TRAJ_COLORS[0]),
            ([0.0, 4.0], TRAJ_COLORS[1]),
            ([6.0, 2.0], TRAJ_COLORS[2]),
            ([4.0, 7.0], TRAJ_COLORS[3]),
            ([8.0, 8.0], TRAJ_COLORS[4]),
        ]
        for start, color in starts:
            traj, _ = run_optimization_trajectory(
                loss_fn, reward_fn, start, num_actions, n_iters=500, lr=0.05
            )
            ax_bl.plot(traj[:, 0], traj[:, 1], '-', color=color, lw=1.8, alpha=0.65)
            step = max(1, len(traj) // 30)
            ax_bl.plot(traj[::step, 0], traj[::step, 1], '.', color=color,
                       ms=5, alpha=0.75)
            ax_bl.plot(traj[0, 0], traj[0, 1], 'o', color=color, ms=14,
                       mec='k', mew=1.0, zorder=5)
            ax_bl.plot(traj[-1, 0], traj[-1, 1], '*', color=color, ms=20,
                       mec='k', mew=1.0, zorder=5)
        ax_bl.plot([], [], 'o', color='gray', ms=12, mec='k', mew=1.0, label='Start')
        ax_bl.plot([], [], '*', color='gray', ms=18, mec='k', mew=1.0, label='Finish')
        ax_bl.legend(loc='center right', framealpha=0.9, fontsize=20)

    ax_bl.set_xlabel('Action 1 (path x slot)', fontsize=26)
    ax_bl.set_ylabel('Action 2 (path x slot)', fontsize=26)
    ax_bl.set_title('Reward Landscape with Trajectories', fontsize=28)
    ax_bl.tick_params(labelsize=20)
    ax_bl.set_xlim(-0.5, 8.5)
    ax_bl.set_ylim(-0.5, 8.5)
    ax_bl.set_aspect('equal', adjustable='box')

    # ---- Bottom-right: reward heatmap with gradient arrows (no colorbar) ----
    ax_br = fig.add_subplot(gs[1, 1])
    ax_br.pcolormesh(X, Y, rewards.T, cmap='RdYlGn', norm=norm_reward,
                     shading='auto', alpha=0.6)

    stride = 2
    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]
    U = grad_a1[::stride, ::stride].T
    V = grad_a2[::stride, ::stride].T
    M = grad_mag[::stride, ::stride].T

    max_mag = M.max() + 1e-12
    U_norm = U / max_mag * 0.6
    V_norm = V / max_mag * 0.6

    ax_br.quiver(Xs, Ys, U_norm, V_norm, M,
                 cmap='inferno', scale=1.0, scale_units='xy',
                 width=0.005, headwidth=4, headlength=4,
                 clim=[0, max_mag])
    ax_br.set_xlabel('Action 1 (path x slot)', fontsize=26)
    ax_br.set_ylabel('Action 2 (path x slot)', fontsize=26)
    ax_br.set_title('Gradient Direction (toward higher reward)', fontsize=28)
    ax_br.tick_params(labelsize=20)
    ax_br.set_xlim(-0.5, 8.5)
    ax_br.set_ylim(-0.5, 8.5)
    ax_br.set_aspect('equal', adjustable='box')

    # Shift the left column slightly toward the centre by translating both
    # left subplots (and the top-left colorbar that sits beside the 3D plot)
    # ~4% of figure width to the right.  The right column stays where it is.
    _LEFT_SHIFT = 0.04
    for _ax in (ax_tl, cb1.ax, ax_bl):
        _pos = _ax.get_position()
        _ax.set_position(
            [_pos.x0 + _LEFT_SHIFT, _pos.y0, _pos.width, _pos.height]
        )

    if save:
        fname = f"combined_landscape_view{suffix}.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname))
        print(f"  Saved {fname}")
    return fig


# ---------------------------------------------------------------------------
# Generate all plots for one case
# ---------------------------------------------------------------------------

def generate_all_plots(reward_fn, loss_fn, action_vals, rewards, grad_a1,
                       grad_a2, grad_mag, suffix="",
                       gaussian_smooth_fn=None, gaussian_reward_fn=None):
    """Generate the full suite of plots for a given case."""
    # Use Gaussian-smoothed loss/reward for trajectories if provided
    traj_loss = gaussian_smooth_fn if gaussian_smooth_fn is not None else loss_fn
    traj_reward = gaussian_reward_fn if gaussian_reward_fn is not None else reward_fn

    plot_reward_surface(action_vals, rewards, suffix=suffix)
    plot_gradient_surface(action_vals, grad_mag, suffix=suffix)
    plot_heatmap_with_arrows(action_vals, rewards, grad_a1, grad_a2, grad_mag,
                              loss_fn=traj_loss, reward_fn=traj_reward,
                              suffix=suffix)
    plot_combined_3d(action_vals, rewards, grad_mag, suffix=suffix)
    plot_combined_full_view(action_vals, rewards, grad_a1, grad_a2, grad_mag,
                             loss_fn=traj_loss, reward_fn=traj_reward,
                             suffix=suffix)
    print(f"\n  Running optimization trajectories...")
    plot_optimization_trajectories(action_vals, rewards, traj_loss, traj_reward,
                                    suffix=suffix)
    plot_heatmap_with_trajectories(action_vals, rewards, traj_loss, traj_reward,
                                    suffix=suffix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(temperature=5.0, include_simple=False):
    # Apply project-wide style with publication-sized fonts
    configure_style(
        font_size=22,
        axes_label_size=26,
        tick_size=20,
        legend_size=18,
        figure_dpi=150,
    )
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if include_simple:
        # ===================================================================
        # Case A: Simple constrained case (from differentiable.ipynb)
        # Only slot 0 free on each link; requests node 1 -> node 3
        # ===================================================================
        print("=" * 60)
        print("CASE A: SIMPLE (constrained network, 1 free slot per link)")
        print("=" * 60)

        list_of_requests_simple, reset_links = setup_case_simple()
        total_timesteps = len(list_of_requests_simple)

        print("\nSetting up environment...")
        config_s = make_config(list_of_requests_simple, total_timesteps, temperature=temperature)
        env_s, env_params_s, key_s, env_state_s = init_environment(config_s)

        env_step_s = create_env_step(env_s)
        rollout_fn_s = create_rollout_fn(env_s, env_step_s, key_s, env_params_s,
                                          reset_fn=reset_links)
        reward_fn_s = create_reward_fn(rollout_fn_s, key_s, env_state_s, env_params_s)
        loss_fn_s = create_loss_fn(rollout_fn_s, key_s, env_state_s, env_params_s)

        print("Compiling...")
        _ = loss_fn_s(jnp.zeros(2))
        _ = reward_fn_s(jnp.zeros(2))
        print("  Done.")

        print("\nComputing reward/gradient landscape (step=0.5)...")
        vals_s, rew_s, g1_s, g2_s, gm_s = compute_landscape(
            reward_fn_s, loss_fn_s, num_actions=9, step=0.5
        )

        print("\nGenerating plots...")
        generate_all_plots(reward_fn_s, loss_fn_s, vals_s, rew_s, g1_s, g2_s, gm_s,
                           suffix="_simple")

        print("\nComputing Gaussian-smoothed landscape (sigma=0.8)...")
        vals_sg, rew_sg, g1_sg, g2_sg, gm_sg = compute_landscape_gaussian(
            reward_fn_s, num_actions=9, step=0.5, sigma=0.8
        )
        print("\nGenerating Gaussian-smoothed plots...")
        generate_all_plots(reward_fn_s, loss_fn_s, vals_sg, rew_sg, g1_sg, g2_sg,
                           gm_sg, suffix="_simple_gaussian",
                           gaussian_smooth_fn=lambda pair: -gaussian_smooth_reward(
                               reward_fn_s, pair, num_actions=9, sigma=0.8),
                           gaussian_reward_fn=lambda pair: gaussian_smooth_reward(
                               reward_fn_s, pair, num_actions=9, sigma=0.8))

    # ===================================================================
    # Case B: Full empty network (from gradient_sense_check.py)
    # All slots available; requests node 0 -> node 1
    # ===================================================================
    print("\n" + "=" * 60)
    print("CASE B: FULL (empty network, all slots available)")
    print("=" * 60)

    list_of_requests_full = [
        [0., 1., 1., 0., 1e8, 0.],
        [0., 1., 1., 0., 1e8, 0.],
    ]
    total_timesteps = 2

    print("\nSetting up environment...")
    config_f = make_config(list_of_requests_full, total_timesteps, temperature=temperature)
    env_f, env_params_f, key_f, env_state_f = init_environment(config_f)

    env_step_f = create_env_step(env_f)
    rollout_fn_f = create_rollout_fn(env_f, env_step_f, key_f, env_params_f)
    reward_fn_f = create_reward_fn(rollout_fn_f, key_f, env_state_f, env_params_f)
    loss_fn_f = create_loss_fn(rollout_fn_f, key_f, env_state_f, env_params_f)

    print("Compiling...")
    _ = loss_fn_f(jnp.zeros(2))
    _ = reward_fn_f(jnp.zeros(2))
    print("  Done.")

    print("\nComputing reward/gradient landscape (step=0.5)...")
    vals_f, rew_f, g1_f, g2_f, gm_f = compute_landscape(
        reward_fn_f, loss_fn_f, num_actions=9, step=0.5
    )

    print("\nGenerating plots...")
    generate_all_plots(reward_fn_f, loss_fn_f, vals_f, rew_f, g1_f, g2_f, gm_f,
                       suffix="_full")

    print("\nComputing Gaussian-smoothed landscape (sigma=0.8)...")
    vals_fg, rew_fg, g1_fg, g2_fg, gm_fg = compute_landscape_gaussian(
        reward_fn_f, num_actions=9, step=0.5, sigma=0.8
    )
    print("\nGenerating Gaussian-smoothed plots...")
    generate_all_plots(reward_fn_f, loss_fn_f, vals_fg, rew_fg, g1_fg, g2_fg,
                       gm_fg, suffix="_full_gaussian",
                       gaussian_smooth_fn=lambda pair: -gaussian_smooth_reward(
                           reward_fn_f, pair, num_actions=9, sigma=0.8),
                       gaussian_reward_fn=lambda pair: gaussian_smooth_reward(
                           reward_fn_f, pair, num_actions=9, sigma=0.8))

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot differentiable test case")
    parser.add_argument("--temperature", type=float, default=5.0,
                        help="Temperature for differentiable approximations (default: 5.0)")
    parser.add_argument("--include_simple", action="store_true",
                        help="Also plot the simple constrained case (disabled by default)")
    args = parser.parse_args()
    main(temperature=args.temperature, include_simple=args.include_simple)
