# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working with the XLRON codebase.

## Project Overview

XLRON ("ex-el-er-on") is a JAX-based library for simulating optical network resource allocation problems and training reinforcement learning agents. It provides gym-style environments that run entirely on GPU/TPU for massively parallel training.

## Build and Run Commands

```bash
# Install dependencies
uv sync

# Run training
python -m xlron.train.train --env_type=rsa --topology_name=nsfnet_chen --link_resources=100

# Run with heuristic evaluation
python -m xlron.train.train --env_type=rsa --topology_name=4node --EVAL_HEURISTIC --path_heuristic=ksp_ff

# Run tests
pytest tests/

# Run single test file
pytest tests/test_env_funcs.py -v
```

## Architecture

### Core Structure

```
xlron/
├── environments/           # JAX-based optical network environments
│   ├── dataclasses.py     # Flax struct dataclasses for state/params
│   ├── env_funcs.py       # Core environment functions (step, reset, etc.)
│   ├── diff_utils.py      # Differentiable approximations for discrete ops
│   ├── make_env.py        # Environment factory
│   └── wrappers.py        # Gym-style wrappers
├── train/
│   ├── train.py           # Main training entry point
│   ├── train_utils.py     # Training utilities
│   └── ppo.py             # PPO implementation
├── heuristics/            # Classical algorithms (KSP, First-Fit, etc.)
├── dtype_config.py        # Device-aware dtype configuration
└── parameter_flags.py     # Command-line flags (absl)
```

### Key Design Patterns

1. **Flax Struct Dataclasses**: State and parameters use `@struct.dataclass` with `pytree_node=False` for static fields that shouldn't be traced by JAX.

2. **Functional Style**: All environment functions are pure functions that take state/params and return new state. No side effects.

3. **JIT Compilation**: Functions use `@jax.jit` or `@partial(jax.jit, static_argnums=...)` for compilation. Static arguments must be hashable.

4. **Differentiable Operations**: `diff_utils.py` provides differentiable approximations using straight-through gradient estimators and temperature-controlled soft functions.

## Environment Types

- `rsa` - Basic Routing and Spectrum Assignment
- `rmsa` - Routing, Modulation and Spectrum Assignment  
- `deeprmsa` - DeepRMSA with path statistics observation
- `rwa_lightpath_reuse` - RWA with lightpath reuse
- `vone` - Virtual Optical Network Embedding
- `rsa_gn_model` / `rmsa_gn_model` - With GN model physical layer

## Key Configuration Flags

### Environment
- `--env_type` - Environment type (rsa, rmsa, deeprmsa, etc.)
- `--topology_name` - Network topology (nsfnet_chen, 4node, etc.)
- `--link_resources` - Number of frequency slots per link
- `--k_paths` - Number of K-shortest paths
- `--load` - Traffic load in Erlangs

### Training
- `--TOTAL_TIMESTEPS` - Total training timesteps
- `--NUM_ENVS` - Number of parallel environments
- `--LR` - Learning rate
- `--NUM_UPDATES` - Number of PPO updates

### Differentiable Mode
- `--differentiable` - Enable differentiable approximations (default: False)
- `--temperature` - Temperature for soft approximations (default: 1.0)

### Heuristics
- `--EVAL_HEURISTIC` - Run heuristic evaluation instead of RL
- `--path_heuristic` - Heuristic algorithm (ksp_ff, ksp_bf, ff_ksp, etc.)

## Working with Differentiable Operations

The `diff_utils.py` module provides differentiable versions of discrete operations:

```python
# Pattern: all functions accept differentiable parameter
result = differentiable_where(condition, true_val, false_val, 
                               threshold=0.5, temperature=1.0, 
                               differentiable=params.differentiable)

# When differentiable=False: returns hard result directly
# When differentiable=True: uses straight-through estimator
```

Functions available:
- `differentiable_where` - Soft conditional selection
- `differentiable_compare` - Soft comparison (==, >=, <=, >, <, !=)
- `differentiable_argmax` - Soft argmax
- `differentiable_round`, `differentiable_ceil`, `differentiable_floor`
- `differentiable_index`, `differentiable_indexing` - Soft array indexing
- `differentiable_one_hot_index_update` - Soft one-hot update
- `differentiable_cond` - Soft conditional execution

## Common Patterns

### Adding a New Parameter

1. Add flag in `parameter_flags.py`
2. Add field to appropriate dataclass in `dataclasses.py`
3. Wire config to params in `make_env.py`
4. Use `params.field_name` in environment functions

### Static vs Dynamic Arguments

- Use `pytree_node=False` in dataclasses for values that don't change during training
- Use `static_argnames` in JIT decorators for string/bool arguments
- Never use traced values as static arguments (causes unhashable error)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_env_funcs.py::test_function_name -v

# Run with coverage
pytest tests/ --cov=xlron
```

## Dtype Configuration

`dtype_config.py` provides device-aware dtypes:
- `COMPUTE_DTYPE`, `PARAMS_DTYPE` - For neural network computations
- `LARGE_FLOAT_DTYPE`, `SMALL_FLOAT_DTYPE` - For environment state
- `LARGE_INT_DTYPE`, `MED_INT_DTYPE`, `SMALL_INT_DTYPE` - For indices/counters
- `INDEX_DTYPE` - Always int32 for JAX array indexing
