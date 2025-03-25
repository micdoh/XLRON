import jax
import optax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from absl import flags
from gymnax.wrappers import GymnaxToGymWrapper

from xlron.environments.env_funcs import *
from xlron.environments.dataclasses import EnvState
from xlron.train.train_utils import *
from xlron.environments.make_env import *

# We load the default flags
FLAGS = flags.FLAGS


def setup_case_2():
    def reset_links(state):
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
    list_of_requests = [
        [1., 1., 3.],
        [1., 1., 3.],
    ]
    return list_of_requests, reset_links


def setup_case_2_combo():
    # In this case, all slots are occupied except first and last slots on links 2,3,5,6
    # So expect only 2 optimal combinations: 8,3 or 11,0
    def reset_links(state):
        link_slot_array = jnp.array([
            [-1., -1., -1., -1.,],
            [0., -1., -1., 0., ],
            [0., -1., -1., 0., ],
            [-1., -1., -1., -1., ],
            [0., -1., -1., 0., ],
            [0., -1., -1., 0., ],

        ])
        link_slot_departure_array = jnp.array([
            [1e8, 1e8, 1e8, 1e8, ],
            [0., 1e8, 1e8, 0., ],
            [0., 1e8, 1e8, 0., ],
            [1e8, 1e8, 1e8, 1e8, ],
            [0., 1e8, 1e8, 0., ],
            [0., 1e8, 1e8, 0., ],

        ])
        """Set up the link slot array in a specific way."""
        state = state.replace(
            link_slot_array=link_slot_array,
            link_slot_departure_array=link_slot_departure_array,
        )
        # Convert to float32
        state = jax.tree.map(lambda x: x.astype(jnp.float32), state)
        return state
    list_of_requests = [
        [0., 1., 1.],
        [3., 1., 4.],
    ]
    return list_of_requests, reset_links


def setup_case_fail_2():
    # In this case, all slots are occupied except first and last slots on links 2,3,5,6
    # So expect only 2 optimal combinations: 8,3 or 11,0
    def reset_links(state):
        link_slot_array = jnp.array([
            [-1., -1., -1., -1.,],
            [0., -1., -1., -1., ],
            [0., -1., -1., -1., ],
            [-1., -1., -1., -1., ],
            [0., -1., -1., -1., ],
            [0., -1., -1., -1., ],

        ])
        link_slot_departure_array = jnp.array([
            [1e8, 1e8, 1e8, 1e8, ],
            [0., 1e8, 1e8, 1e8, ],
            [0., 1e8, 1e8, 1e8, ],
            [1e8, 1e8, 1e8, 1e8, ],
            [0., 1e8, 1e8, 1e8, ],
            [0., 1e8, 1e8, 1e8, ],

        ])
        """Set up the link slot array in a specific way."""
        state = state.replace(
            link_slot_array=link_slot_array,
            link_slot_departure_array=link_slot_departure_array,
        )
        # Convert to float32
        state = jax.tree.map(lambda x: x.astype(jnp.float32), state)
        return state
    list_of_requests = [
        [3., 1., 4.],
        [0., 1., 1.],
    ]
    return list_of_requests, reset_links


def setup_case_11():
    # In this case, all slots are occupied except first and last slots on links 2,3,5,6
    # So expect only 2 optimal combinations: 8,3 or 11,0
    def reset_links(state):
        link_slot_array = jnp.array([
            [0., 0., 0., 0.,],
            [0., 0., 0., 0., ],
            [0., -1., -1., -1., ],
            [0., 0., 0., 0., ],
            [0., -1., -1., -1., ],
            [0., 0., 0., 0., ],

        ])
        link_slot_departure_array = jnp.array([
            [0., 0., 0., 0.,],
            [0., 0., 0., 0., ],
            [0., 1e8, 1e8, 1e8, ],
            [0., 0., 0., 0., ],
            [0., 1e8, 1e8, 1e8, ],
            [0., 0., 0., 0., ],
        ])
        """Set up the link slot array in a specific way."""
        state = state.replace(
            link_slot_array=link_slot_array,
            link_slot_departure_array=link_slot_departure_array,
        )
        # Convert to float32
        state = jax.tree.map(lambda x: x.astype(jnp.float32), state)
        return state
    # Expect finals actions to be 0,0,0,0,0,1,2,3,5,6,7 or permutation of the final 6 actions
    list_of_requests = [
        [0., 1., 1.],
        [1., 1., 2.],
        [2., 1., 3.],
        [3., 1., 4.],
        [0., 1., 4.],
        [0., 1., 3.],
        [0., 1., 3.],
        [0., 1., 3.],
        [0., 1., 3.],
        [0., 1., 3.],
        [0., 1., 3.],
    ]
    return list_of_requests, reset_links


def setup_case_fail_12():
    # In this case, all slots are occupied except first and last slots on links 2,3,5,6
    # So expect only 2 optimal combinations: 8,3 or 11,0
    def reset_links(state):
        link_slot_array = jnp.array([
            [0., 0., 0., 0.,],
            [0., 0., 0., 0., ],
            [0., -1., -1., -1., ],
            [0., 0., 0., 0., ],
            [0., -1., -1., -1., ],
            [0., 0., 0., 0., ],

        ])
        link_slot_departure_array = jnp.array([
            [0., 0., 0., 0.,],
            [0., 0., 0., 0., ],
            [0., 1e8, 1e8, 1e8, ],
            [0., 0., 0., 0., ],
            [0., 1e8, 1e8, 1e8, ],
            [0., 0., 0., 0., ],
        ])
        """Set up the link slot array in a specific way."""
        state = state.replace(
            link_slot_array=link_slot_array,
            link_slot_departure_array=link_slot_departure_array,
        )
        # Convert to float32
        state = jax.tree.map(lambda x: x.astype(jnp.float32), state)
        return state
    # Expect finals actions to be 0,0,0,0,0,1,2,3,5,6,7 or permutation of the final 6 actions
    list_of_requests = [
        [0., 1., 1.],
        [1., 1., 2.],
        [2., 1., 3.],
        [3., 1., 4.],
        [0., 1., 4.],
        [0., 1., 3.],
        [0., 1., 3.],
        [0., 1., 3.],
        [0., 1., 3.],
        [0., 1., 3.],
        [0., 1., 3.],
        [0., 1., 3.],  # Not enough room for this one!
    ]
    return list_of_requests, reset_links



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
        "temperature": 1.0,
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


def create_env_step(env):
    """Create a function for stepping through the environment."""

    def env_step(runner_state, action):
        key, env_state, env_params = runner_state
        obs, state, reward, done, info = env.step(key, env_state, action, env_params)
        return (key, state, env_params), reward

    # This is unused
    def env_step_interp(runner_state, action):
        key, env_state, env_params = runner_state
        floor_action = jnp.floor(action)
        ceil_action = jnp.ceil(action)
        # Get interpolation weight based on distance
        weight = action - floor_action  # 0.0 at floor, 1.0 at ceil
        # Evaluate both integer actions (without rounding in the state transition)
        floor_obs, floor_state, floor_reward, floor_done, floor_info = env.step(key, env_state, floor_action, env_params)
        ceil_obs, ceil_state, ceil_reward, ceil_done, ceil_info = env.step(key, env_state, ceil_action, env_params)
        # Interpolate reward
        reward =  (1 - weight) * floor_reward + weight * ceil_reward
        obs, state, _, done, info = env.step(key, env_state, action, env_params)
        return (key, state, env_params), reward

    # This is unused
    def env_step_gaussian(runner_state, action):
        key, env_state, env_params = runner_state

        # Sample several nearby actions
        neighbor_range = 2.5
        actions = action - jnp.arange(-neighbor_range, neighbor_range, 0.5)

        # Apply Gaussian weighting
        sigma = 0.8  # Controls smoothness
        weights = jnp.exp(-0.5 * ((actions - action) / sigma) ** 2)
        weights = weights / jnp.sum(weights)

        # Evaluate all actions and compute weighted average reward
        rewards = jnp.array([env.step(key, env_state, a, env_params)[2] for a in actions])
        reward = jnp.sum(weights * rewards)

        # Use original action for state transition
        obs, state, _, done, info = env.step(key, env_state, action, env_params)
        return (key, state, env_params), reward

    return env_step


def create_rollout_fn(env, env_step, key, env_params, reset_fn=None):
    """Create a function for rolling out a sequence of actions."""

    #@partial(jax.jit, static_argnums=(0,))
    def rollout(runner_state, actions):
        _, st = env.reset(key, env_params)
        st = reset_fn(runner_state[1])
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

    return jax.jit(get_loss)


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


def compute_reward_landscape(reward_fn, action_sequence):
    """Compute rewards and gradients for a grid of action pairs."""
    # v_reward_fn = jax.vmap(reward_fn)
    # v_grad_fn = jax.vmap(jax.grad(reward_fn))
    #
    # rewards = v_reward_fn(action_pairs)
    # gradients = v_grad_fn(action_pairs)

    rewards = []
    gradients = []
    for i, seq in enumerate(action_sequence):
        reward = reward_fn(seq)
        gradient = jax.grad(reward_fn)(seq)
        print(f"Action Seq.: {seq}, Reward: {reward}, Gradient: {gradient}")
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

    # Multiply by -1 to represent direction in which actions are updated
    grad_x_np = np.array(grad_x_grid) * -1.
    grad_y_np = np.array(grad_y_grid) * -1.

    grad_mag_np = np.array(grad_magnitude)
    max_grad_x = np.max(grad_x_np)
    min_grad_x = np.min(grad_x_np)
    max_grad_y = np.max(grad_y_np)
    min_grad_y = np.min(grad_y_np)
    # Normalize gradient direction components to [0, 1]
    grad_x_np_norm = (grad_x_np - min_grad_x) / (max_grad_x - min_grad_x + 1e-8)
    grad_y_np_norm = (grad_y_np - min_grad_y) / (max_grad_y - min_grad_y + 1e-8)
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

    # 3D gradient landscape with properly horizontal arrows
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')

    # Plot the surface
    surf3 = ax3.plot_surface(x_np, y_np, grad_mag_np, cmap=cm.viridis,
                             linewidth=0, antialiased=True, alpha=0.9)

    # We'll sample fewer points for clarity
    stride = 2

    # Get the x,y points for the arrows (sparser grid)
    x_points = x_np[::stride, ::stride]
    y_points = y_np[::stride, ::stride]
    z_points = grad_mag_np[::stride, ::stride]

    # Get direction vectors
    u_vectors = grad_x_np[::stride, ::stride]
    v_vectors = grad_y_np[::stride, ::stride]

    # Arrow length parameter - adjust as needed
    arrow_length = 0.8

    for i in range(x_points.shape[0]):
        for j in range(x_points.shape[1]):
            # Start point
            x_start = x_points[i, j]
            y_start = y_points[i, j]
            z_start = z_points[i, j]

            # Direction vector - normalized
            u = u_vectors[i, j]
            v = v_vectors[i, j]
            magnitude = np.sqrt(u ** 2 + v ** 2)

            if magnitude > 0:
                # Normalize and scale
                u_norm = u / magnitude * arrow_length
                v_norm = v / magnitude * arrow_length

                # End point - the key is that z remains the same
                x_end = x_start + u_norm
                y_end = y_start + v_norm
                z_end = z_start  # Same z to ensure horizontal arrow

                # Draw arrow shaft (line)
                ax3.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], 'r-', linewidth=1.5)

                # Draw arrow head (small triangle at end)
                # This is simplified - matplotlib's quiver would handle this better
                head_length = arrow_length * 0.3
                head_width = arrow_length * 0.2

                # Calculate perpendicular direction for arrow head
                perp_u = -v_norm / arrow_length
                perp_v = u_norm / arrow_length

                # Draw arrow head points
                x_head1 = x_end - head_length * u_norm / arrow_length - head_width * perp_u
                y_head1 = y_end - head_length * v_norm / arrow_length - head_width * perp_v
                z_head1 = z_end

                x_head2 = x_end - head_length * u_norm / arrow_length + head_width * perp_u
                y_head2 = y_end - head_length * v_norm / arrow_length + head_width * perp_v
                z_head2 = z_end

                # Draw arrow head
                ax3.plot([x_end, x_head1], [y_end, y_head1], [z_end, z_head1], 'r-', linewidth=1.5)
                ax3.plot([x_end, x_head2], [y_end, y_head2], [z_end, z_head2], 'r-', linewidth=1.5)

    ax3.set_xlabel('Action 1')
    ax3.set_ylabel('Action 2')
    ax3.set_zlabel('Gradient Magnitude')
    ax3.set_title('Gradient Magnitude and Direction')
    fig.colorbar(surf3, ax=ax3)

    # 2D gradient quiver plot
    ax4 = fig.add_subplot(2, 2, 4)
    quiv = ax4.quiver(x_np, y_np, grad_x_np_norm, grad_y_np_norm, grad_mag_np,
                      cmap='viridis', scale=1.0, scale_units='xy')
    ax4.set_xlabel('Action 1')
    ax4.set_ylabel('Action 2')
    ax4.set_title('Gradient Magnitude and Direction')
    fig.colorbar(quiv, ax=ax4)

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
    for setup_case in [setup_case_2, setup_case_2_combo, setup_case_fail_2, setup_case_11, setup_case_fail_12]:
        list_of_requests, reset_links = setup_case()

        total_timesteps = len(list_of_requests)
        print(f"Total timesteps: {total_timesteps}")
        config = make_config(list_of_requests, total_timesteps)

        # Initialize environment
        env, env_params, key, env_state = init_environment(config)
        print(f"Environment initialized")
        env_state = reset_links(env_state)

        # Define environment stepping and rollout functions
        env_step = create_env_step(env)
        rollout_fn = create_rollout_fn(env, env_step, key, env_params, reset_links)

        # Create reward and loss functions
        loss_fn = create_loss_fn(rollout_fn, key, env_state, env_params)

        # Generate action grid and compute reward landscape
        action_vals, action_grid, action_pairs = generate_action_grid()

        # Initial setup for optimization
        initial_actions = jnp.zeros((total_timesteps,), dtype=jnp.float32)
        print(f"Initial actions: {initial_actions}")

        # compile loss_fn
        print("Loss function compiling...")
        dummy = loss_fn(initial_actions)
        dummy = dummy + 1
        print(f"Loss function compiled: {dummy}")

        if len(list_of_requests) == 2:

            print(f"Computing reward landscape for {len(action_pairs)} action pairs...")
            rewards, gradients = compute_reward_landscape(loss_fn, action_pairs)

            # Plot reward landscape
            print("Plotting reward landscape...")
            rewards_np, grad_x_np, grad_y_np, grad_mag_np, x_np, y_np = plot_reward_landscape(
                action_vals, action_grid, rewards, gradients
            )

        optimize = False
        if optimize:
            # Optimize actions
            print("Starting action optimization...")
            optimized_actions = optimize_actions(
                loss_fn, initial_actions, n_iterations=10000, learning_rate=0.01
            )

            # Run with optimized actions
            env_state = reset_links(env_state)
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
