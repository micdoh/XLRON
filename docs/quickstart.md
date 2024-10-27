# Quickstart

## Setup virtual environment

We recommend using [poetry](https://python-poetry.org/) to manage dependencies. To install poetry, run:

```bash
pip install poetry
```

Then, clone the repository and install the dependencies:

```bash
git clone
cd XLRON
poetry install
```

If you prefer, you can use requirements.txt (not recommended as it may not be up-to-date):

```bash
pip install -r requirements.txt
```

N.B. if you want to run on GPU or TPU, then you need to edit the `pyproject.toml` file to include the appropriate accelerator-specific dependencies, i.e. comment/uncomment the relevant lines below, then follow the steps above:

```angular2html
jax = {extras = ["tpu"], version = "^0.4.11"}  # for Google Cloud TPU
jax = {extras = ["cuda11_pip"], version = "^0.4.11"} # or cuda12_pip depending on drivers
jax = {extras = ["cpu"], version = "^0.4.11"}  # for CPU-only
```

## Running experiments

Every experiment in XLRON is defined through command-line options. No need to edit the source code. See [configuration options](./flags_reference.md) for a full list of options.

To run an experiment, use the `train.py` script. For example, to recreate the training of DeepRMSA:

```bash
poetry run python train.py --env_type=deeprmsa --env_type=deeprmsa --continuous_operation --load=250 --k=5 --topology_name=nsfnet_deeprmsa --link_resources=100 --max_requests=1e3 --max_timesteps=1e3 --mean_service_holding_time=25 --ROLLOUT_LENGTH=100 --continuous_operation --NUM_LAYERS 5 --NUM_UNITS 128 --NUM_ENVS 16 --TOTAL_TIMESTEPS 5000000 --ENV_WARMUP_STEPS 3000 --LR 5e-5 --WARMUP_PEAK_MULTIPLIER 2 --LR_SCHEDULE linear --UPDATE_EPOCHS 10 --GAE_LAMBDA 0.9 --GAMMA 0.95 --ACTION_MASKING
```

Launch power optimisation using closed-form GN model RMSA environment (experimental feature):
```bash
poetry run python train.py --env_type=rsa_gn_model --PROJECT LAUNCH_POWER --ROLLOUT_LENGTH=50 --NUM_LAYERS 3 --NUM_UNITS 128 --load=100 --k=5 --weight=weight --topology_name=nsfnet_deeprmsa_directed --link_resources=115 --max_requests=10 --max_timesteps=10 --mean_service_holding_time=25 --continuous_operation --ENV_WARMUP_STEPS=0 --TOTAL_TIMESTEPS 20000 --NUM_ENVS 200 --launch_power_type=rl --interband_gap=100 --values_bw=400,600,800,1200 --guardband=0 --coherent --reward_type=bitrate --snr_margin=0.01 --slot_size=100 --max_power=3 --min_power=-1 --VISIBLE_DEVICES=1 --PLOTTING --DOWNSAMPLE_FACTOR=1 --LR 1e-4 --LR_SCHEDULE warmup_cosine --WARMUP_PEAK_MULTIPLIER 2 --GAE_LAMBDA 0.9 --GAMMA 0.9  --EVAL_MODEL --LOAD_MODEL --MODEL_PATH /home/XLRON/models/BEST_LP --WANDB --DATA_OUTPUT_FILE /home/XLRON/data/launch_power_train_out.csv
```

You can save models with `--SAVE_MODEL` to the directory specified by `--MODEL_PATH`. You can log data to Weights and Biases (see [Understanding XLRON](understanding_xlron.md) for more details) with `--WANDB`. You can also log episode end data to a CSV file with `--DATA_OUTPUT_FILE`.
