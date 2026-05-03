import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax  # For optimization tests

# Import the differentiable functions
from xlron.environments.diff_utils import *

# Set up plotting
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Initialize random keys
key = jax.random.key(42)


def visualize_function_and_gradients(func, x_range, temp_values, title, exact_func=None, threshold=None):
    """
    Visualize a function and its gradients with different temperature values.

    Args:
        func: The differentiable function to test
        x_range: The range of x values to test
        temp_values: List of temperature values to test
        title: The title for the plot
        exact_func: The exact (non-differentiable) function if available
        threshold: The threshold value to use if the function requires it
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # Function values plot
    ax1 = plt.subplot(gs[0])
    ax1.set_title(f"{title} - Function Values")

    # Gradients plot
    ax2 = plt.subplot(gs[1])
    ax2.set_title(f"{title} - Gradients")

    colors = plt.cm.viridis(np.linspace(0, 1, len(temp_values)))

    for i, temp in enumerate(temp_values):
        # Define the function with the current temperature
        if threshold is not None:
            def f(x):
                return func(x, threshold=threshold, temperature=temp)
        else:
            def f(x):
                return func(x, temperature=temp)

        # Compute the function values and gradients
        y_values = jax.vmap(f)(x_range)
        grad_f = jax.vmap(jax.grad(f))
        grad_values = grad_f(x_range)

        # Plot the function values
        ax1.plot(x_range, y_values, label=f"temp={temp}", color=colors[i])

        # Plot the gradients
        ax2.plot(x_range, grad_values, label=f"temp={temp}", color=colors[i])

    # If the exact function is provided, plot it as a reference
    if exact_func is not None:
        exact_values = jax.vmap(exact_func)(x_range)
        ax1.plot(x_range, exact_values, 'k--', label='Exact', linewidth=2)

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    return fig


def optimize_with_function(func, init_value, target_value, steps=100, learning_rate=0.1, temp_values=[0.1, 1.0, 10.0]):
    """
    Test optimization using the gradients of the function.

    Args:
        func: The differentiable function to test
        init_value: The initial value for optimization
        target_value: The target value to optimize towards
        steps: Number of optimization steps
        learning_rate: Learning rate for gradient descent
        temp_values: List of temperature values to test
    """
    results = {}

    for temp in temp_values:
        # Define loss function using the differentiable function
        def loss_fn(x):
            return ((func(x, temperature=temp) - target_value) ** 2).mean()

        # Initialize the optimization
        x = jnp.array(init_value)
        loss_history = []
        value_history = []

        # Simple gradient descent
        for _ in range(steps):
            value = func(x, temperature=temp)
            loss = loss_fn(x)
            grad_x = jax.grad(loss_fn)(x)

            # Update x using gradient descent
            x = x - learning_rate * grad_x

            loss_history.append(float(loss))
            value_history.append(float(value))

        results[temp] = {
            'final_value': float(func(x, temperature=temp)),
            'final_x': float(x),
            'loss_history': loss_history,
            'value_history': value_history
        }

    return results


def plot_optimization_results(results, title):
    """
    Plot the optimization results.

    Args:
        results: Results from the optimize_with_function function
        title: The title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for temp, res in results.items():
        ax1.plot(res['loss_history'], label=f"temp={temp}")
        ax2.plot(res['value_history'], label=f"temp={temp}")

    ax1.set_title(f"{title} - Loss History")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.set_title(f"{title} - Value History")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Function Value")
    ax2.legend()

    plt.tight_layout()
    return fig


def do_differentiable_indexing():
    """Test the differentiable_index and differentiable_indexing functions."""
    print("Testing differentiable_indexing...")

    # Create a test array
    array = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # Define a range of indices to test
    indices = jnp.linspace(-0.5, 4.5, 100)

    # Define a function to compute non-differentiable indexing
    def exact_index(idx):
        # Clamp idx to valid range and convert to int
        clamped_idx = jnp.clip(jnp.round(idx), 0, len(array) - 1).astype(jnp.int32)
        return array[clamped_idx]

    # Define differentiable indexing function with different temperatures
    def diff_index(idx, temperature=1.0):
        return differentiable_index(array, idx, temperature=temperature)

    # Visualize the function and gradients
    temp_values = [0.1, 1.0, 5.0, 10.0, 20.0]
    fig1 = visualize_function_and_gradients(diff_index, indices, temp_values, "Differentiable Index", exact_index)

    # Test optimization - try to select specific values from the array
    target_idx = 2.5
    print(f"  Testing optimization towards index {target_idx}")

    # Function to optimize
    def index_value(idx, temperature=1.0):
        return differentiable_index(array, idx, temperature=temperature)

    # Target the value at the specified index (interpolating if needed)
    target_value = jnp.interp(target_idx, jnp.arange(len(array)), array)

    # Start from a different index
    init_idx = 0.0

    # Optimize the index to reach the target value
    results = {}
    steps = 100
    learning_rate = 0.05

    for temp in temp_values:
        idx = jnp.array(init_idx)
        loss_history = []
        idx_history = []
        value_history = []

        # Define loss function
        def loss_fn(x):
            value = index_value(x, temperature=temp)
            return (value - target_value) ** 2

        for _ in range(steps):
            value = index_value(idx, temperature=temp)
            loss = loss_fn(idx)
            grad_idx = jax.grad(loss_fn)(idx)

            # Update idx using gradient descent
            idx = idx - learning_rate * grad_idx

            loss_history.append(float(loss))
            idx_history.append(float(idx))
            value_history.append(float(value))

        results[temp] = {
            'final_idx': float(idx),
            'final_value': float(index_value(idx, temperature=temp)),
            'loss_history': loss_history,
            'idx_history': idx_history,
            'value_history': value_history
        }

    # Create optimization plots
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))

    for temp, res in results.items():
        axes[0].plot(res['loss_history'], label=f"temp={temp}")
        axes[1].plot(res['idx_history'], label=f"temp={temp}")
        axes[2].plot(res['value_history'], label=f"temp={temp}")

    axes[0].set_title(f"Loss History (Target idx: {target_idx})")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title(f"Index History (Target idx: {target_idx})")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Index")
    axes[1].axhline(y=target_idx, color='r', linestyle='--', label=f'Target: {target_idx}')
    axes[1].legend()

    axes[2].set_title(f"Value History (Target value: {target_value})")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Value")
    axes[2].axhline(y=target_value, color='r', linestyle='--', label=f'Target: {target_value}')
    axes[2].legend()

    plt.tight_layout()

    # Store the results
    opt_results = {
        'target_idx': target_idx,
        'target_value': float(target_value),
        'optimization_results': results,
        'plot': fig2
    }

    return {
        'function_plot': fig1,
        'optimization_results': opt_results
    }


def do_differentiable_cond():
    """Test the differentiable_cond function."""
    print("Testing differentiable_cond...")

    # Define test functions for true and false branches
    def true_fn(x):
        return x * 2

    def false_fn(x):
        return x / 2

    # Define a range of condition values to test
    x = jnp.linspace(-2, 2, 100)

    # Define a function to evaluate the conditional
    def eval_cond(cond_val, operand=1.0, threshold=0.0, temperature=1.0):
        return differentiable_cond(cond_val, true_fn, false_fn, operand, threshold, temperature)

    # Define the exact (non-differentiable) conditional
    def exact_cond(cond_val, operand=1.0):
        return jax.lax.cond(cond_val >= 0, true_fn, false_fn, operand)

    # Visualize the function and gradients
    temp_values = [0.1, 1.0, 5.0, 10.0, 20.0]

    # We need a different approach to visualize cond since it takes the condition value
    # but returns based on the operand
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    operand = 1.0  # Fixed operand for visualization

    # Function values plot
    axes[0].set_title("Differentiable Cond - Function Values")

    # Plot the exact conditional
    exact_values = jnp.array([exact_cond(c, operand) for c in x])
    axes[0].plot(x, exact_values, 'k--', label='Exact', linewidth=2)

    # Plot for different temperatures
    for temp in temp_values:
        y_values = jnp.array([eval_cond(c, operand, 0.0, temp) for c in x])
        axes[0].plot(x, y_values, label=f"temp={temp}")

    axes[0].legend()
    axes[0].set_xlabel("Condition Value")
    axes[0].set_ylabel("Output")

    # Gradient plot
    axes[1].set_title("Differentiable Cond - Gradients")

    # Compute gradients for different temperatures
    for temp in temp_values:
        grad_values = []
        for c in x:
            # Use finite differences for gradient since it's cleaner for this case
            h = 1e-5
            y1 = eval_cond(c - h, operand, 0.0, temp)
            y2 = eval_cond(c + h, operand, 0.0, temp)
            grad = (y2 - y1) / (2 * h)
            grad_values.append(grad)

        axes[1].plot(x, grad_values, label=f"temp={temp}")

    axes[1].legend()
    axes[1].set_xlabel("Condition Value")
    axes[1].set_ylabel("Gradient")

    plt.tight_layout()

    # Test optimization - changing the condition to get a target output
    target_outputs = [0.5, 2.0]  # Target both branches
    opt_results = {}

    for target in target_outputs:
        print(f"  Testing optimization towards output {target}")

        # Define a loss function
        def loss_fn(cond_val, operand=1.0, temperature=1.0):
            output = eval_cond(cond_val, operand, 0.0, temperature)
            return (output - target) ** 2

        # Start with a condition that gives the wrong branch
        init_cond = -1.0 if target > 1.0 else 1.0

        # Optimize the condition to get the target output
        results = {}
        steps = 100
        learning_rate = 0.1

        for temp in temp_values:
            cond_val = jnp.array(init_cond)
            loss_history = []
            cond_history = []
            output_history = []

            for _ in range(steps):
                output = eval_cond(cond_val, operand, 0.0, temp)
                loss = loss_fn(cond_val, operand, temp)
                grad_cond = jax.grad(lambda c: loss_fn(c, operand, temp))(cond_val)

                # Update condition using gradient descent
                cond_val = cond_val - learning_rate * grad_cond

                loss_history.append(float(loss))
                cond_history.append(float(cond_val))
                output_history.append(float(output))

            results[temp] = {
                'final_cond': float(cond_val),
                'final_output': float(eval_cond(cond_val, operand, 0.0, temp)),
                'loss_history': loss_history,
                'cond_history': cond_history,
                'output_history': output_history
            }

        # Create optimization plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for temp, res in results.items():
            axes[0].plot(res['loss_history'], label=f"temp={temp}")
            axes[1].plot(res['cond_history'], label=f"temp={temp}")
            axes[2].plot(res['output_history'], label=f"temp={temp}")

        axes[0].set_title(f"Loss History (Target output: {target})")
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].set_title(f"Condition Value History")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Condition Value")
        axes[1].axhline(y=0.0, color='r', linestyle='--', label='Threshold: 0.0')
        axes[1].legend()

        axes[2].set_title(f"Output History")
        axes[2].set_xlabel("Steps")
        axes[2].set_ylabel("Output")
        axes[2].axhline(y=target, color='r', linestyle='--', label=f'Target: {target}')
        axes[2].legend()

        plt.tight_layout()

        opt_results[target] = {
            'optimization_results': results,
            'plot': fig
        }

    return {
        'function_plot': fig,
        'optimization_results': opt_results
    }


def main():
    """Main testing function."""
    print("Testing differentiable operations with different temperature values")

    # Create a directory to save results
    import os
    os.makedirs("diff_tests", exist_ok=True)

    # Run all tests
    tests = {
        'indexing': do_differentiable_indexing(),
        'cond': do_differentiable_cond()
    }

    # Save all figures
    for test_name, test_results in tests.items():
        if isinstance(test_results, dict):
            for fig_name, fig in [(n, obj) for n, obj in test_results.items() if isinstance(obj, plt.Figure)]:
                fig.savefig(f"diff_tests/{test_name}_{fig_name}.png")

    return tests


if __name__ == "__main__":
    main()
