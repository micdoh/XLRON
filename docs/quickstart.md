# Quickstart

## Setup virtual environment

XLRON uses [uv](https://docs.astral.sh/uv/) to manage dependencies. Clone the repository and install:

```bash
git clone https://github.com/micdoh/XLRON.git
cd XLRON

# Install with uv (recommended)
uv sync

# Install dev dependencies
uv sync --group dev

# If running on GPU
uv sync --group gpu

# If running on TPU
uv sync --group tpu
```

## Running experiments

Every experiment in XLRON is defined through command-line options. No need to edit the source code. See [configuration options](./flags_reference.md) for a full list of options.

To run an experiment, use the `xlron.train.train` entry point. For example, to recreate the training of DeepRMSA:

```bash
uv run python -m xlron.train.train --env_type=deeprmsa --continuous_operation --load=250 --k=5 --topology_name=nsfnet_deeprmsa_directed --link_resources=100 --mean_service_holding_time=25 --truncate_holding_time --ROLLOUT_LENGTH=100 --NUM_LAYERS 5 --NUM_UNITS 128 --NUM_ENVS 16 --TOTAL_TIMESTEPS 5000000 --ENV_WARMUP_STEPS 3000 --LR 5e-5 --LR_SCHEDULE linear --UPDATE_EPOCHS 10 --GAE_LAMBDA 0.9 --GAMMA 0.95 --ACTION_MASKING
```

To evaluate a heuristic (e.g. KSP-FF) instead of training an agent:

```bash
uv run python -m xlron.train.train --env_type=rmsa --topology_name=nsfnet_deeprmsa_directed --link_resources=100 --k=50 --load=250 --continuous_operation --ENV_WARMUP_STEPS=3000 --TOTAL_TIMESTEPS=20000000 --NUM_ENVS=2000 --EVAL_HEURISTIC --path_heuristic=ksp_ff
```

You can save models with `--SAVE_MODEL` to the directory specified by `--MODEL_PATH`, and evaluate a saved model with `--EVAL_MODEL --MODEL_PATH <path>`. You can log data to Weights and Biases (see [Understanding XLRON](understanding_xlron.md) for more details) with `--WANDB`. You can also log episode end data to a CSV file with `--EPISODE_DATA_OUTPUT_FILE`.
