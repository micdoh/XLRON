import jax
import optax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from absl import flags
from gymnax.wrappers import GymnaxToGymWrapper

from xlron.environments.env_funcs import *
from xlron.environments.dataclasses import EnvState
from xlron.train.train_utils import *
from xlron.environments.make_env import *

# We load the default flags
FLAGS = flags.FLAGS


def make_config(list_of_requests, total_timesteps):
    """Create a configuration for the environment and training."""
    # Define our experiment
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
        "temperature": 0.1,
        "deterministic_requests": True,
    }

    # Define training details
    train_config = {
        "SEED": 0,
        "NUM_LEARNERS": 1,
        "TOTAL_TIMESTEPS": total_timesteps,
        "NUM_ENVS": 1,
        "ROLLOUT_LENGTH": 3,
        "UPDATE_EPOCHS": 2,
        "LR": 5e-5,
        "GAMMA": 1.0,  # For pre-training the VF, we don't use a discount factor
        "GAE_LAMBDA": 1.0,  # For pre-training the VF, we don't use GAE
        "LR_SCHEDULE": "linear",
        "LAYER_NORM": True,
    }

    # We convert config to a Box container to make items accessible through both dict and dot notation
    config = process_config(FLAGS, **env_config, **train_config)
    config.list_of_requests = jnp.array(list_of_requests)

    return config


def init_environment(config):
    """Initialize the environment and set up the initial state."""
    env, env_params = make(config, log_wrapper=False)
    gym_env = GymnaxToGymWrapper(env, env_params)
    rng = jax.random.PRNGKey(config.SEED)
    rng, key = jax.random.split(rng)
    obsv, env_state = env.reset(key, env_params)

    # Cast everything to float32
    env_state = jax.tree.map(lambda x: x.astype(jnp.float32), env_state)

    return env, env_params, key, env_state


def set_link_slot_array(state):
    """Set up the link slot array in a specific way."""
    state = state.replace(
        link_slot_array=state.link_slot_array.at[:, 0].set(0.),
    )
    state = state.replace(
        link_slot_array=state.link_slot_array.at[:, 1:].set(-1.),
    )
    state = state.replace(
        link_slot_departure_array=state.link_slot_array.at[:, 0].set(0.),
    )
    state = state.replace(
        link_slot_departure_array=state.link_slot_array.at[:, 1:].set(1e8),
    )
    # Convert to float32
    state = jax.tree.map(lambda x: x.astype(jnp.float32), state)
    return state


def create_env_step(env):
    """Create a function for stepping through the environment."""

    def env_step(runner_state, action):
        key, env_state, env_params = runner_state
        obs, state, reward, done, info = env.step(key, env_state, action, env_params)
        return (key, state, env_params), reward

    # def env_step(runner_state, action):
    #     key, env_state, env_params = runner_state
    #     floor_action = jnp.floor(action)
    #     ceil_action = jnp.ceil(action)
    #     # Get interpolation weight based on distance
    #     weight = action - floor_action  # 0.0 at floor, 1.0 at ceil
    #     # Evaluate both integer actions (without rounding in the state transition)
    #     floor_obs, floor_state, floor_reward, floor_done, floor_info = env.step(key, env_state, floor_action, env_params)
    #     ceil_obs, ceil_state, ceil_reward, ceil_done, ceil_info = env.step(key, env_state, ceil_action, env_params)
    #     # Interpolate reward
    #     reward =  (1 - weight) * floor_reward + weight * ceil_reward
    #     obs, state, _, done, info = env.step(key, env_state, action, env_params)
    #     return (key, state, env_params), reward

    # def env_step(runner_state, action):
    #     key, env_state, env_params = runner_state
    #
    #     # Sample several nearby actions
    #     neighbor_range = 2.5
    #     actions = action - jnp.arange(-neighbor_range, neighbor_range, 0.5)
    #
    #     # Apply Gaussian weighting
    #     sigma = 0.8  # Controls smoothness
    #     weights = jnp.exp(-0.5 * ((actions - action) / sigma) ** 2)
    #     weights = weights / jnp.sum(weights)
    #
    #     # Evaluate all actions and compute weighted average reward
    #     rewards = jnp.array([env.step(key, env_state, a, env_params)[2] for a in actions])
    #     reward = jnp.sum(weights * rewards)
    #
    #     # Use original action for state transition
    #     obs, state, _, done, info = env.step(key, env_state, action, env_params)
    #     return (key, state, env_params), reward

    return jax.jit(env_step, static_argnums=(0,))


def create_rollout_fn(env, env_step, key, env_params):
    """Create a function for rolling out a sequence of actions."""

    @partial(jax.jit, static_argnums=(0,))
    def rollout(runner_state, actions):
        _, st = env.reset(key, env_params)
        st = set_link_slot_array(runner_state[1])
        r_state = (runner_state[0], st, runner_state[2])
        out, rew = jax.lax.scan(env_step, r_state, actions)
        return out, rew

    return jax.jit(rollout)


def create_loss_fn(rollout_fn, key, env_state, env_params):
    """Create a function to compute the reward for a given action pair."""

    def get_loss(action_pair):
        actions = jnp.array(action_pair, dtype=jnp.float32)
        runner_state = (key, env_state, env_params)
        _, rewards = rollout_fn(runner_state, actions)
        return jnp.sum(rewards)

    return get_loss


def optimize_actions(loss_fn, initial_actions, n_iterations=100, learning_rate=0.01):
    """
    Optimize actions using gradient descent to maximize reward.

    Args:
        loss_fn: The loss function to minimize
        initial_actions: Starting point for actions optimization
        n_iterations: Number of optimization steps
        learning_rate: Learning rate for gradient descent

    Returns:
        The optimized actions and loss history
    """
    # Initialize actions
    actions = initial_actions

    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(actions)

    # Define update step
    def update_step(actions, opt_state):
        loss_value, grads = jax.value_and_grad(loss_fn)(actions)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_actions = optax.apply_updates(actions, updates)

        # Optional: Project actions back to valid range if needed
        new_actions = jnp.clip(new_actions, 0, 11)

        return None, (new_actions, new_opt_state, loss_value, grads, updates)

    # # Optimization loop
    # print("Starting scan...")
    # _, out = jax.lax.scan(
    #     lambda _, i: update_step(actions, opt_state), None, jnp.arange(n_iterations)
    # )
    # actions, opt_state, loss_value, grads, updates = out

    for i in range(n_iterations):
        _, out = update_step(actions, opt_state)
        actions, opt_state, loss_value, grads, updates = out
        if i % 1000 == 0:
            print(i)
            print(f"Iteration {i}, Loss: {loss_value}")
            jax.debug.print("new actions: {}", jnp.round(actions, 3))
            jax.debug.print("grads: {}", jnp.round(grads, 5))
            jax.debug.print("updates: {}", jnp.round(updates, 4))

    return actions


def generate_action_grid(action_min=0, action_max=12.0, action_step=0.5):
    """Generate a grid of action pairs for visualization."""
    action_vals = jnp.arange(action_min, action_max, action_step)
    action_grid = jnp.meshgrid(action_vals, action_vals)
    action_pairs = jnp.stack([action_grid[0].flatten(), action_grid[1].flatten()], axis=1)

    return action_vals, action_grid, action_pairs


def compute_reward_landscape(reward_fn, action_pairs):
    """Compute rewards and gradients for a grid of action pairs."""
    # v_reward_fn = jax.vmap(reward_fn)
    # v_grad_fn = jax.vmap(jax.grad(reward_fn))
    #
    # rewards = v_reward_fn(action_pairs)
    # gradients = v_grad_fn(action_pairs)

    rewards = []
    gradients = []
    for i, pair in enumerate(action_pairs):
        reward = reward_fn(pair)
        gradient = jax.grad(reward_fn)(pair)
        print(f"Pair: {pair}, Reward: {reward}, Gradient: {gradient}")
        rewards.append(reward)
        gradients.append(gradient)
    rewards = jnp.array(rewards)
    gradients = jnp.array(gradients)

    return rewards, gradients


def plot_reward_landscape(action_vals, action_grid, rewards, gradients):
    """Visualize the reward landscape and gradients."""
    # Reshape for plotting
    rewards_grid = rewards.reshape(len(action_vals), len(action_vals))
    grad_x_grid = gradients[:, 0].reshape(len(action_vals), len(action_vals))
    grad_y_grid = gradients[:, 1].reshape(len(action_vals), len(action_vals))
    grad_magnitude = jnp.sqrt(grad_x_grid ** 2 + grad_y_grid ** 2).reshape(len(action_vals), len(action_vals))

    # Convert to numpy for plotting
    rewards_np = np.array(rewards_grid)
    grad_x_np = np.array(grad_x_grid)
    grad_y_np = np.array(grad_y_grid)
    grad_mag_np = np.array(grad_magnitude)
    x_np = np.array(action_grid[0])
    y_np = np.array(action_grid[1])

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))

    # 3D surface plot for rewards
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(x_np, y_np, rewards_np, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax1.set_xlabel('Action 1')
    ax1.set_ylabel('Action 2')
    ax1.set_zlabel('Reward')
    ax1.set_title('Reward Landscape')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # 2D heatmap for rewards
    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(rewards_np, origin='lower', extent=[0, 12, 0, 12], cmap='coolwarm', aspect='auto')
    ax2.set_xlabel('Action 1')
    ax2.set_ylabel('Action 2')
    ax2.set_title('Reward Heatmap')
    fig.colorbar(im, ax=ax2)

    # Gradient quiver plot
    ax3 = fig.add_subplot(2, 2, 3)
    scale_array = lambda arr: arr * (0.1 / (10 ** np.floor(np.log10(np.max(np.abs(arr[arr != 0]))) + 1))) if np.any(
        arr != 0) else arr
    quiv = ax3.quiver(x_np, y_np, scale_array(grad_x_np), scale_array(grad_y_np), scale_array(grad_mag_np),
                      cmap='viridis', scale=1)
    ax3.set_xlabel('Action 1')
    ax3.set_ylabel('Action 2')
    ax3.set_title('Gradient Direction and Magnitude')
    fig.colorbar(quiv, ax=ax3)

    # Gradient magnitude heatmap
    ax4 = fig.add_subplot(2, 2, 4)
    im2 = ax4.imshow(grad_mag_np, origin='lower', extent=[0, 12, 0, 12], cmap='viridis', aspect='auto')
    ax4.set_xlabel('Action 1')
    ax4.set_ylabel('Action 2')
    ax4.set_title('Gradient Magnitude')
    fig.colorbar(im2, ax=ax4)

    plt.tight_layout()
    plt.show()

    # Print information about extreme points
    print("Maximum reward:", np.max(rewards_np))
    max_idx = np.unravel_index(np.argmax(rewards_np), rewards_np.shape)
    print("Maximum reward action pair:", (action_vals[max_idx[0]], action_vals[max_idx[1]]))
    print("Minimum reward:", np.min(rewards_np))
    min_idx = np.unravel_index(np.argmin(rewards_np), rewards_np.shape)
    print("Minimum reward action pair:", (action_vals[min_idx[0]], action_vals[min_idx[1]]))
    print("Maximum gradient magnitude:", np.max(grad_mag_np))
    max_grad_idx = np.unravel_index(np.argmax(grad_mag_np), grad_mag_np.shape)
    print("Maximum gradient action pair:", (action_vals[max_grad_idx[0]], action_vals[max_grad_idx[1]]))

    return rewards_np, grad_x_np, grad_y_np, grad_mag_np, x_np, y_np


def plot_gradient_landscape(action_vals, grad_x_np, grad_y_np, grad_mag_np, x_np, y_np, smooth=True, sigma=1.0):
    """
    Plot the gradient landscape in 3D, with options for smoothing.

    Args:
        action_vals: Array of action values used for the grid
        grad_x_np: X-component of gradients
        grad_y_np: Y-component of gradients
        grad_mag_np: Magnitude of gradients
        x_np: Grid of x-coordinates
        y_np: Grid of y-coordinates
        smooth: Whether to apply smoothing
        sigma: Smoothing parameter for gaussian filter
    """
    # Create a new figure for gradient landscape
    fig = plt.figure(figsize=(18, 10))

    # Apply smoothing if requested
    if smooth:
        grad_mag_smooth = gaussian_filter(grad_mag_np, sigma=sigma)
        grad_x_smooth = gaussian_filter(grad_x_np, sigma=sigma)
        grad_y_smooth = gaussian_filter(grad_y_np, sigma=sigma)
    else:
        grad_mag_smooth = grad_mag_np
        grad_x_smooth = grad_x_np
        grad_y_smooth = grad_y_np

    # 3D surface plot of gradient magnitude
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(x_np, y_np, grad_mag_smooth, cmap=cm.viridis,
                            linewidth=0, antialiased=True)
    ax1.set_xlabel('Action 1')
    ax1.set_ylabel('Action 2')
    ax1.set_zlabel('Gradient Magnitude')
    ax1.set_title('Gradient Magnitude Landscape' + (' (Smoothed)' if smooth else ''))
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # 3D surface with gradient direction arrows
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(x_np, y_np, grad_mag_smooth, cmap=cm.viridis,
                             alpha=0.8, linewidth=0, antialiased=True)

    # Add arrows to show gradient direction in 3D
    # We'll sample fewer points for clarity
    stride = len(action_vals) // 10
    if stride < 1: stride = 1

    # Scale arrows appropriately
    max_grad = np.max(np.sqrt(grad_x_smooth ** 2 + grad_y_smooth ** 2))
    arrow_scale = 0.5 / max_grad if max_grad > 0 else 1.0

    for i in range(0, len(action_vals), stride):
        for j in range(0, len(action_vals), stride):
            ax2.quiver(x_np[i, j], y_np[i, j], grad_mag_smooth[i, j],
                       grad_x_smooth[i, j] * arrow_scale,
                       grad_y_smooth[i, j] * arrow_scale,
                       0,  # No z-component for gradient
                       color='r', arrow_length_ratio=0.3)

    ax2.set_xlabel('Action 1')
    ax2.set_ylabel('Action 2')
    ax2.set_zlabel('Gradient Magnitude')
    ax2.set_title('Gradient Direction and Magnitude' + (' (Smoothed)' if smooth else ''))

    plt.tight_layout()
    plt.show()


def compute_hessian_eigenvalues(loss_fn, action_pairs):
    """
    Compute Hessian matrices and their eigenvalues for a set of action pairs.

    Args:
        loss_fn: The loss/reward function
        action_pairs: Array of action pairs to analyze

    Returns:
        Array of eigenvalues for each action pair
    """
    eigenvalues_list = []

    for pair in action_pairs:
        try:
            hessian_fn = jax.hessian(lambda x: loss_fn(x))
            hessian = hessian_fn(pair)
            eigenvalues = jnp.linalg.eigvals(hessian)
            eigenvalues_list.append(eigenvalues)
        except Exception as e:
            print(f"Error computing Hessian at {pair}: {e}")
            eigenvalues_list.append(jnp.zeros(2))

    return jnp.array(eigenvalues_list)


def main():
    """Main function to run the optimization and analysis."""
    # Define requests and configure environment
    list_of_requests = [
        [1., 1., 3.],
        [1., 1., 3.],
    ]
    total_timesteps = len(list_of_requests)
    config = make_config(list_of_requests, total_timesteps)

    # Initialize environment
    env, env_params, key, env_state = init_environment(config)
    print(f"Environment initialized")
    env_state = set_link_slot_array(env_state)

    # Define environment stepping and rollout functions
    env_step = create_env_step(env)
    rollout_fn = create_rollout_fn(env, env_step, key, env_params)

    # Create reward and loss functions
    loss_fn = create_loss_fn(rollout_fn, key, env_state, env_params)

    # Generate action grid and compute reward landscape
    action_vals, action_grid, action_pairs = generate_action_grid()

    # compile loss_fn
    print("Loss function compiling...")
    dummy = loss_fn(action_pairs[0])
    dummy = dummy + 1
    print(f"Loss function compiled: {dummy}")

    print(f"Computing reward landscape for {len(action_pairs)} action pairs...")
    rewards, gradients = compute_reward_landscape(loss_fn, action_pairs)

    # Plot reward landscape
    print("Plotting reward landscape...")
    rewards_np, grad_x_np, grad_y_np, grad_mag_np, x_np, y_np = plot_reward_landscape(
        action_vals, action_grid, rewards, gradients
    )

    # Plot gradient landscape
    plot_gradient_landscape(action_vals, grad_x_np, grad_y_np, grad_mag_np, x_np, y_np, smooth=True, sigma=1.0)

    # Initial setup for optimization
    initial_actions = jnp.array([2.0, 4.0])
    initial_actions = jnp.tile(initial_actions, (1,))
    print(f"Initial actions: {initial_actions}")

    # Optimize actions
    print("Starting action optimization...")
    optimized_actions = optimize_actions(
        loss_fn, initial_actions, n_iterations=10000, learning_rate=0.001
    )

    # Run with optimized actions
    env_state = set_link_slot_array(env_state)
    final_state, final_rewards = rollout_fn((key, env_state, env_params), jnp.round(optimized_actions))
    print(f"Total reward with optimized actions: {jnp.sum(final_rewards)}")
    print(f"Final actions: {jnp.round(optimized_actions)}")
    print(f"Final rewards: {final_rewards}")

    # Hessian analysis (optional)
    run_hessian_analysis = False
    if run_hessian_analysis:
        # Sample points from the action space for eigenvalue analysis
        sample_step = 10
        sampled_actions = action_vals[::sample_step]
        sampled_grid = jnp.meshgrid(sampled_actions, sampled_actions)
        sampled_pairs = jnp.stack([sampled_grid[0].flatten(), sampled_grid[1].flatten()], axis=1)

        print(f"Computing eigenvalues for {len(sampled_pairs)} points...")
        eigenvalues_array = compute_hessian_eigenvalues(loss_fn, sampled_pairs)

        # Process and visualize eigenvalues - additional code would go here


if __name__ == "__main__":
    main()