#!/bin/bash

poetry run python ./xlron/train/train.py \
  --env_type=deeprmsa \
  --continuous_operation \
  --load=250 \
  --k=5 \
  --topology_name=nsfnet_deeprmsa_undirected \
  --link_resources=100 \
  --max_requests=1e3 \
  --max_timesteps=1e3 \
  --mean_service_holding_time=25 \
  --ROLLOUT_LENGTH=100 \
  --continuous_operation \
  --NUM_LAYERS=5 \
  --NUM_UNITS=128 \
  --NUM_ENVS=16 \
  --TOTAL_TIMESTEPS=5000000 \
  --ENV_WARMUP_STEPS=3000 \
  --LR=5e-5 \
  --WARMUP_PEAK_MULTIPLIER=2 \
  --LR_SCHEDULE=linear \
  --UPDATE_EPOCHS=10 \
  --GAE_LAMBDA=0.9 \
  --GAMMA=0.95 \
  --ACTION_MASKING \
  --WANDB
  # --SAVE_MODEL
