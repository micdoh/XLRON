#!/bin/bash

python /Users/ianwang/PycharmProjects/XLRON/xlron/train/train.py \
  --env_type=multibandrsa \
  --continuous_operation \
# Change the load to 400
  --load=400 \

  --k=5 \
  --topology_name=nsfnet_deeprmsa_directed \
  --link_resources=100 \
  --max_requests=1e3 \
  --max_timesteps=1e3 \
  --mean_service_holding_time=25 \
  --continuous_operation \
  --EVAL_HEURISTIC \
  --path_heuristic=ksp_ff \
# Change the number of environments to 10
  --NUM_ENVS=10 \

  --TOTAL_TIMESTEPS=100000 \
  --PLOTTING \
  --slot_size=50 \
  --bandgap_start=2500 \
  --bandgap=100 \
  --WANDB
