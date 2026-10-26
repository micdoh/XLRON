# Test organization

Unit tests live beside their source module and use its name: `ppo.py` has
`ppo_test.py`, `train_utils.py` has `train_utils_test.py`, and so on. Add coverage
to the matching file; group feature-specific cases by the function or class
under test rather than creating another feature-named test file.

Tests spanning several modules live in `tests/integration/`, named for the source
entry module they exercise. For example, `test_env_funcs.py` checks complete GN
admission using the environment functions and custom kernels.

Shared deterministic inputs and environment builders live in `tests/helpers/`.
Import helpers directly rather than importing another test module. In particular,
wildcard imports of test modules also import test classes and collect them again.

Run tests from the repository root with `uv run --no-sync pytest`. GPU tests keep
their existing skip markers, and full training/checkpoint tests retain `slow`.
