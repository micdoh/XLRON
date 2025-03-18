import jax
import time
import optax
import jax.numpy as jnp
from absl import flags
from flax.training.train_state import TrainState
from gymnax.environments import environment
from scipy.signal.windows import general_hamming
from tensorflow_probability.substrates.jax.distributions.student_t import entropy

from xlron.environments.env_funcs import *
from xlron.environments.gn_model.isrs_gn_model import *
from xlron.environments.dataclasses import EnvState, EnvParams, VONETransition, RSATransition
from xlron.train.train_utils import *
import sys
from xlron.environments.make_env import *
from xlron import parameter_flags
from jax.test_util import check_grads
import optax  # JAX's optimization library
from gymnax.wrappers import GymnaxToGymWrapper
from xlron.models.models import ActorCriticMLP
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm

# We load the default flags
FLAGS = flags.FLAGS

if __name__ == "__main__":

    list_of_requests = [
        [1.,1.,3.],
        [1.,1.,3.],
    ]
    total_timesteps = len(list_of_requests)

    # Then define our experiment...
    env_config = {
        "env_type": "rwa",
        "k": 3,
        "link_resources": 4,
        "incremental_loading" : True,
        "end_first_blocking": False,
        "topology_name": "5node_undirected",
        "values_bw": [1],
        "slot_size": 1,
        "max_requests": total_timesteps,
        "temperature": 0.25,
        "deterministic_requests": True,
    }

    # ... and training details
    train_config = {
        "SEED": 0,
        "NUM_LEARNERS": 1,
        "TOTAL_TIMESTEPS": total_timesteps,
        "NUM_ENVS": 1,
        "ROLLOUT_LENGTH": 3,
        "UPDATE_EPOCHS": 2,
        "LR": 5e-5,
        "GAMMA": 1.0, # For pre-training the VF, we don't use a discount factor
        "GAE_LAMBDA": 1.0, # For pre-training the VF, we don't use GAE
        "LR_SCHEDULE": "linear",
        "LAYER_NORM": True,
    }

    # We convert config to a Box container to make items accessible through both dict and dot notation
    config = process_config(FLAGS, **env_config, **train_config)
    config.list_of_requests = jnp.array(list_of_requests)


    # INIT ENV
    env, env_params = make(config, log_wrapper=False)
    gym_env = GymnaxToGymWrapper(env, env_params)
    rng = jax.random.PRNGKey(config.SEED)
    rng, key = jax.random.split(rng)
    obsv, env_state = env.reset(key, env_params)
    # Update env_state so that link_slot_array is only 0 in the first column
    print(env_state.request_array)

    # Cast everything to float32
    env_state = jax.tree.map(lambda x: x.astype(jnp.float32), env_state)
    action = env.action_space(env_params).sample(key)
    action = action.astype(jnp.float32)

    def set_link_slot_array(_state):
        _state = _state.replace(
            link_slot_array=_state.link_slot_array.at[:, 0].set(0.),
        )
        _state = _state.replace(
            link_slot_array=_state.link_slot_array.at[:, 1:].set(-1.),
        )
        _state = _state.replace(
            link_slot_departure_array=_state.link_slot_array.at[:, 0].set(0.),
        )
        _state = _state.replace(
            link_slot_departure_array=_state.link_slot_array.at[:, 1:].set(1e8),
        )
        # _state = _state.replace(
        #     request_array=jnp.array([1,1,3])
        # )
        _state = jax.tree.map(lambda x: x.astype(jnp.float32), _state)
        return _state
    env_state = set_link_slot_array(env_state)
    actions = jnp.array([4.1, 8.1]) # This should fail for both actions


    # Define function to scan through actions and return the stacked states and rewards
    runner_state = (key, env_state, env_params)
    def env_step(_runner_state, _action):
        _key, _env_state, _env_params = _runner_state
        # Perform the step
        # _env_state = _env_state.replace(
        #     request_array=jnp.array([1,1,3])
        # # )
        # jax.debug.print("requests: {}", _env_state.request_array)
        # jax.debug.print("link_slot_array: {}", _env_state.link_slot_array)
        # jax.debug.print("link_slot_departure_array: {}", _env_state.link_slot_departure_array)
        # jax.debug.print("ACTION: {}", _action, ordered=True)
        #_action = differentiable_round_simple(_action, 0.1)#_env_params.temperature)
        # jax.debug.print("ACTION: {}", _action, ordered=True)
        obs, state, rewrd, done, info = env.step(_key, _env_state, _action, _env_params)
        # jax.debug.print("link_slot_array: {}", state.link_slot_array)
        # jax.debug.print("link_slot_departure_array: {}", state.link_slot_departure_array)
        return (_key, state, env_params), rewrd
    def rollout(r_state, acs):
        _, st = env.reset(key, env_params)
        st = set_link_slot_array(r_state[1])
        r_state = (r_state[0], st, r_state[2])
        out, rew =  jax.lax.scan(env_step, r_state, acs)
        return out, rew

    out_state, rewards = rollout(runner_state, actions)
    print(out_state)
    rewards.sum()


    def optimize_actions(env, env_state, env_params, initial_actions, n_iterations=100, learning_rate=0.01):
        """
        Optimize actions using gradient descent to maximize reward.

        Args:
            env: The environment
            env_state: Initial environment state
            env_params: Environment parameters
            initial_actions: Starting point for actions optimization
            n_iterations: Number of optimization steps
            learning_rate: Learning rate for gradient descent

        Returns:
            The optimized actions
        """
        # Initialize actions as trainable parameters
        actions = initial_actions

        # Create optimizer
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(actions)

        # Define differentiable loss function (negative total reward)
        @jax.jit
        def loss_fn(current_actions):
            e_state = env.reset(key, env_params)[1]
            e_state = set_link_slot_array(e_state)
            r_state = (jax.random.PRNGKey(0), e_state, env_params)  # Fixed seed for deterministic gradients
            _, rewrds = rollout(r_state, current_actions)
            return -jnp.sum(rewrds)

            # Add regularization toward integer values
            # # This encourages actions to move toward x.0 rather than fractional values
            # reg_term = jnp.sum(jnp.abs(current_actions - jnp.round(current_actions)))
            # return reward_loss #+ 0.01 * reg_term  # Small weight for regularization

        # Define update step
        @jax.jit
        def update_step(actions, opt_state):
            loss_value, grads = jax.value_and_grad(loss_fn)(actions)

            # Add noise to gradients for exploration
            # noise = jax.random.normal(jax.random.PRNGKey(0), actions.shape) * 0.0001
            # grads += noise

            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_actions = optax.apply_updates(actions, updates)

            # Optional: Project actions back to valid range if needed
            new_actions = jnp.clip(new_actions, 0, 11)

            return new_actions, new_opt_state, loss_value, grads, updates

        # Optimization loop
        losses = []
        for i in range(n_iterations):

            # Add noise to actions during optimization
            # nkey = jax.random.PRNGKey(i)  # Different seed each iteration
            # noise_scale = max(0.001 * (1.0 - i / n_iterations), 0.01)  # Annealing noise
            # noise = jax.random.normal(nkey, actions.shape) * noise_scale
            # actions = actions + noise

            actions, opt_state, loss_value, grads, updates = update_step(actions, opt_state)
            losses.append(loss_value)
            if i % 1000 == 0:
                print(f"Iteration {i}, Loss: {loss_value}")
                jax.debug.print("new actions: {}", jnp.round(actions, 3))
                jax.debug.print("grads: {}", jnp.round(grads, 5))
                jax.debug.print("updates: {}", jnp.round(updates, 4))

        return actions, losses


    # Get initial random actions
    initial_actions = jnp.array([6.0, 6.1])
    # initial_actions = jnp.full((10,), 2.0)  # Initialize with 3.0
    print(f"intial actions: {initial_actions}")
    # Tile intitial actions 10 times
    initial_actions = jnp.tile(initial_actions, (1,))
    print("total actions: ", initial_actions.shape)

    env_state = set_link_slot_array(env_state)
    jax.debug.print("initial requests: {}", env_state.request_array)

    # Optimize actions
    optimized_actions, loss_history = optimize_actions(
        env, env_state, env_params, initial_actions, n_iterations=100000, learning_rate=0.0001
    )

    # Run the optimized actions to visualize
    env_state = set_link_slot_array(env_state)
    final_state, final_rewards = rollout((key, env_state, env_params), jnp.round(optimized_actions))
    print(f"Total reward with optimized actions: {jnp.sum(final_rewards)}")
    print(f"final actions: {jnp.round(optimized_actions)}")
    print(f"final_rewards: {final_rewards}")

    # INIT ENV (using your existing code)
    env, env_params = make(config, log_wrapper=False)
    gym_env = GymnaxToGymWrapper(env, env_params)
    rng = jax.random.PRNGKey(config.SEED)
    rng, key = jax.random.split(rng)
    obsv, env_state = env.reset(key, env_params)
    env_state = set_link_slot_array(env_state)

    # Define a function to get reward for a single action pair
    def get_reward(action_pair):
        actin = jnp.array(action_pair, dtype=jnp.float32)
        _, rewads = rollout(runner_state, actin)
        return jnp.sum(rewads)

        # # Add regularization toward integer values
        # # This encourages actions to move toward x.0 rather than fractional values
        # reg_term = jnp.sum(jnp.abs(action_pair - jnp.round(action_pair)))
        # return reward_loss #+ 0.01 * reg_term  # Small weight for regularization

    # Create a jitted version of the function and its gradient
    reward_fn = jax.jit(get_reward)
    grad_fn = jax.grad(get_reward)

    # Create a grid of action values from 0 to 8 in steps of 0.5
    action_vals = jnp.arange(0, 12.0, 0.1)
    action_grid = jnp.meshgrid(action_vals, action_vals)
    action_pairs = jnp.stack([action_grid[0].flatten(), action_grid[1].flatten()], axis=1)

    # Vectorize the reward and gradient functions
    v_reward_fn = jax.jit(jax.vmap(reward_fn))
    v_grad_fn = jax.jit(jax.vmap(grad_fn))


    # Calculate rewards and gradients for all action pairs
    rewards = v_reward_fn(action_pairs)
    gradients = v_grad_fn(action_pairs)

    # Reshape for plotting
    rewards_grid = rewards.reshape(len(action_vals), len(action_vals))
    grad_x_grid = gradients[:, 0].reshape(len(action_vals), len(action_vals))
    grad_y_grid = gradients[:, 1].reshape(len(action_vals), len(action_vals))
    grad_magnitude = jnp.sqrt(grad_x_grid**2 + grad_y_grid**2).reshape(len(action_vals), len(action_vals))

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
    scale_array = lambda arr: arr * (0.1 / (10 ** np.floor(np.log10(np.max(np.abs(arr[arr != 0]))) + 1))) if np.any(arr != 0) else arr
    quiv = ax3.quiver(x_np, y_np, scale_array(grad_x_np), scale_array(grad_y_np), scale_array(grad_mag_np), cmap='viridis', scale=1)
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

    # Print out some interesting points
    # Print out some interesting points
    print("Maximum reward:", np.max(rewards_np))
    max_idx = np.unravel_index(np.argmax(rewards_np), rewards_np.shape)
    print("Maximum reward action pair:", (action_vals[max_idx[0]], action_vals[max_idx[1]]))
    print("Minimum reward:", np.min(rewards_np))
    min_idx = np.unravel_index(np.argmin(rewards_np), rewards_np.shape)
    print("Minimum reward action pair:", (action_vals[min_idx[0]], action_vals[min_idx[1]]))
    print("Maximum gradient magnitude:", np.max(grad_mag_np))
    max_grad_idx = np.unravel_index(np.argmax(grad_mag_np), grad_mag_np.shape)
    print("Maximum gradient action pair:", (action_vals[max_grad_idx[0]], action_vals[max_grad_idx[1]]))

    # # Output a table of selected gradient values for verification
    # print("\nGradient values at selected points:")
    print("Action1 Action2  Reward  Grad_X  Grad_Y  Magnitude")
    print("-------------------------------------------------")
    chosen_actions = np.arange(0, 12, 0.5)
    for a1 in list(chosen_actions):
        for a2 in list(chosen_actions):
            idx1 = np.where(action_vals == a1)[0][0]
            idx2 = np.where(action_vals == a2)[0][0]
            r = rewards_np[idx1, idx2]
            gx = grad_x_np[idx1, idx2]
            gy = grad_y_np[idx1, idx2]
            mag = grad_mag_np[idx1, idx2]
            print(f"{a1:.9f} {a2:.9f} {r:.9f} {gx:.9f} {gy:.9f} {mag:.9f}")


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


    # Use the function with your existing data
    plot_gradient_landscape(action_vals, grad_x_np, grad_y_np, grad_mag_np, x_np, y_np, smooth=True, sigma=1.0)

    # Also plot unsmoothed version for comparison
    #plot_gradient_landscape(action_vals, grad_x_np, grad_y_np, grad_mag_np, x_np, y_np, smooth=False)


    if False:
        # HESSIAN
        # Add this code to the end of your file to calculate and visualize Jacobian eigenvalues

        # Define a function to compute Hessian (which is the Jacobian of the gradient) and its eigenvalues
        def compute_hessian_eigenvalues(action_pair):
            """
            Compute the Hessian matrix and its eigenvalues at a given action pair
            """
            hessian_fn = jax.hessian(lambda x: get_reward(x))
            try:
                hessian = hessian_fn(action_pair)
                eigenvalues = jnp.linalg.eigvals(hessian)
                return hessian, eigenvalues
            except Exception as e:
                print(f"Error computing Hessian at {action_pair}: {e}")
                return None, jnp.zeros(2)


        # Sample points from the action space for eigenvalue analysis
        # Use a sparser grid for computational efficiency
        sample_step = 10
        sampled_actions = action_vals[::sample_step]
        sampled_grid = jnp.meshgrid(sampled_actions, sampled_actions)
        sampled_pairs = jnp.stack([sampled_grid[0].flatten(), sampled_grid[1].flatten()], axis=1)

        print(f"Computing eigenvalues for {len(sampled_pairs)} points...")

        # Compute Hessians and eigenvalues for all sampled pairs
        eigenvalues_list = []
        batch_size = 10  # Process in smaller batches

        for i in range(0, len(sampled_pairs), batch_size):
            batch = sampled_pairs[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(sampled_pairs) + batch_size - 1) // batch_size}")

            for j, pair in enumerate(batch):
                _, evals = compute_hessian_eigenvalues(pair)
                eigenvalues_list.append(evals)

        # Convert list to array and reshape
        eigenvalues_array = jnp.array(eigenvalues_list)
        eigenvalues_reshaped = eigenvalues_array.reshape(len(sampled_actions), len(sampled_actions), 2)

        # Extract components
        eigenvalue_magnitudes = jnp.abs(eigenvalues_reshaped)
        eigenvalue_real = jnp.real(eigenvalues_reshaped)
        eigenvalue_imag = jnp.imag(eigenvalues_reshaped)

        # Convert to numpy for plotting
        magnitudes_np = np.array(eigenvalue_magnitudes)
        real_np = np.array(eigenvalue_real)
        imag_np = np.array(eigenvalue_imag)
        x_sampled_np = np.array(sampled_grid[0])
        y_sampled_np = np.array(sampled_grid[1])

        # Create visualization for eigenvalue magnitudes
        fig = plt.figure(figsize=(20, 15))

        # Plot magnitude of largest eigenvalue
        ax1 = fig.add_subplot(2, 2, 1)
        im1 = ax1.imshow(magnitudes_np[:, :, 0], origin='lower',
                         extent=[sampled_actions[0], sampled_actions[-1],
                                 sampled_actions[0], sampled_actions[-1]],
                         cmap='viridis')
        ax1.set_xlabel('Action 1')
        ax1.set_ylabel('Action 2')
        ax1.set_title('Magnitude of Largest Eigenvalue')
        fig.colorbar(im1, ax=ax1)

        # Plot magnitude of smallest eigenvalue
        ax2 = fig.add_subplot(2, 2, 2)
        im2 = ax2.imshow(magnitudes_np[:, :, 1], origin='lower',
                         extent=[sampled_actions[0], sampled_actions[-1],
                                 sampled_actions[0], sampled_actions[-1]],
                         cmap='viridis')
        ax2.set_xlabel('Action 1')
        ax2.set_ylabel('Action 2')
        ax2.set_title('Magnitude of Smallest Eigenvalue')
        fig.colorbar(im2, ax=ax2)

        # Plot real part of largest eigenvalue
        ax3 = fig.add_subplot(2, 2, 3)
        im3 = ax3.imshow(real_np[:, :, 0], origin='lower',
                         extent=[sampled_actions[0], sampled_actions[-1],
                                 sampled_actions[0], sampled_actions[-1]],
                         cmap='coolwarm')
        ax3.set_xlabel('Action 1')
        ax3.set_ylabel('Action 2')
        ax3.set_title('Real Part of Largest Eigenvalue')
        fig.colorbar(im3, ax=ax3)

        # Plot condition number (ratio of largest to smallest eigenvalue magnitudes)
        # This indicates how ill-conditioned the optimization problem is at each point
        condition_number = magnitudes_np[:, :, 0] / jnp.maximum(magnitudes_np[:, :, 1], 1e-10)
        ax4 = fig.add_subplot(2, 2, 4)
        im4 = ax4.imshow(condition_number, origin='lower',
                         extent=[sampled_actions[0], sampled_actions[-1],
                                 sampled_actions[0], sampled_actions[-1]],
                         cmap='plasma', norm=plt.matplotlib.colors.LogNorm())
        ax4.set_xlabel('Action 1')
        ax4.set_ylabel('Action 2')
        ax4.set_title('Condition Number (log scale)')
        fig.colorbar(im4, ax=ax4)

        plt.tight_layout()
        plt.show()

        # 3D visualizations of eigenvalue magnitudes
        fig = plt.figure(figsize=(20, 10))

        # 3D surface of largest eigenvalue magnitude
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(x_sampled_np, y_sampled_np, magnitudes_np[:, :, 0],
                                 cmap=cm.viridis, linewidth=0, antialiased=True)
        ax1.set_xlabel('Action 1')
        ax1.set_ylabel('Action 2')
        ax1.set_zlabel('Magnitude')
        ax1.set_title('Largest Eigenvalue Magnitude')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

        # 3D surface of smallest eigenvalue magnitude
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(x_sampled_np, y_sampled_np, magnitudes_np[:, :, 1],
                                 cmap=cm.viridis, linewidth=0, antialiased=True)
        ax2.set_xlabel('Action 1')
        ax2.set_ylabel('Action 2')
        ax2.set_zlabel('Magnitude')
        ax2.set_title('Smallest Eigenvalue Magnitude')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

        plt.tight_layout()
        plt.show()

        # Create a visualization of eigenvalue spectrum across the action space
        fig = plt.figure(figsize=(16, 10))

        # Create a scatter plot of eigenvalues in the complex plane
        eigenvalues_flat = eigenvalues_array.flatten()
        plt.scatter(np.real(eigenvalues_flat), np.imag(eigenvalues_flat), alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Eigenvalue Spectrum in Complex Plane')

        # Add a circle to represent the unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.5)
        plt.axis('equal')

        plt.tight_layout()
        plt.show()

        # Print statistics about the eigenvalues
        print("\nEigenvalue Statistics:")
        print(f"Max magnitude of largest eigenvalue: {np.max(magnitudes_np[:, :, 0])}")
        print(f"Min magnitude of largest eigenvalue: {np.min(magnitudes_np[:, :, 0])}")
        print(f"Mean magnitude of largest eigenvalue: {np.mean(magnitudes_np[:, :, 0])}")
        print(f"Max magnitude of smallest eigenvalue: {np.max(magnitudes_np[:, :, 1])}")
        print(f"Min magnitude of smallest eigenvalue: {np.min(magnitudes_np[:, :, 1])}")
        print(f"Mean magnitude of smallest eigenvalue: {np.mean(magnitudes_np[:, :, 1])}")

        # Find points with extreme eigenvalues
        max_idx = np.unravel_index(np.argmax(magnitudes_np[:, :, 0]), magnitudes_np[:, :, 0].shape)
        min_idx = np.unravel_index(np.argmin(magnitudes_np[:, :, 1]), magnitudes_np[:, :, 1].shape)

        print(f"\nAction pair with largest eigenvalue: ({sampled_actions[max_idx[0]]}, {sampled_actions[max_idx[1]]})")
        print(f"  Largest eigenvalue: {eigenvalues_reshaped[max_idx[0], max_idx[1], 0]}")
        print(f"  Smallest eigenvalue: {eigenvalues_reshaped[max_idx[0], max_idx[1], 1]}")

        print(f"\nAction pair with smallest eigenvalue: ({sampled_actions[min_idx[0]]}, {sampled_actions[min_idx[1]]})")
        print(f"  Largest eigenvalue: {eigenvalues_reshaped[min_idx[0], min_idx[1], 0]}")
        print(f"  Smallest eigenvalue: {eigenvalues_reshaped[min_idx[0], min_idx[1], 1]}")

        # Create a histogram of eigenvalue magnitudes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.hist(magnitudes_np[:, :, 0].flatten(), bins=30, alpha=0.7)
        ax1.set_xlabel('Magnitude')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Largest Eigenvalue Magnitudes')

        ax2.hist(magnitudes_np[:, :, 1].flatten(), bins=30, alpha=0.7)
        ax2.set_xlabel('Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Smallest Eigenvalue Magnitudes')

        plt.tight_layout()
        plt.show()
