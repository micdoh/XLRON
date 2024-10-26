# Quickstart

## Setup virtual environment:

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

## Running experiments

Every experiment in XLRON is defined through commnd-line options. No need to edit the source code.

To run an experiment, use the `train.py` script. For example:

```bash
poetry run python train.py --env_type=rsa_gn_model --PROJECT LAUNCH_POWER --ROLLOUT_LENGTH=50 --NUM_LAYERS 3 --NUM_UNITS 128 --load=100 --k=5 --weight=weight --topology_name=nsfnet_deeprmsa_directed --link_resources=115 --max_requests=10 --max_timesteps=10 --mean_service_holding_time=25 --continuous_operation --ENV_WARMUP_STEPS=0 --TOTAL_TIMESTEPS 20000 --NUM_ENVS 200 --launch_power_type=rl --interband_gap=100 --values_bw=400,600,800,1200 --guardband=0 --coherent --reward_type=bitrate --snr_margin=0.01 --slot_size=100 --max_power=3 --min_power=-1 --VISIBLE_DEVICES=1 --PLOTTING --DOWNSAMPLE_FACTOR=1 --LR 1e-4 --LR_SCHEDULE warmup_cosine --WARMUP_PEAK_MULTIPLIER 2 --GAE_LAMBDA 0.9 --GAMMA 0.9  --EVAL_MODEL --LOAD_MODEL --MODEL_PATH /home/XLRON/models/BEST_LP --WANDB --DATA_OUTPUT_FILE /home/XLRON/data/launch_power_train_out.csv
```


