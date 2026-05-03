#!/bin/bash

# Loop from 0 to 1 in steps of 0.1
for i in {0..10}; do
    # Calculate launch power
    lp=$(echo "scale=1; $i * 0.1" | bc)

    echo "Running with launch power: $lp"

    python xlron/train/train.py \
        --env_type=rsa_gn_model \
        --load=100 \
        --k=5 \
        --topology_name=nsfnet_deeprmsa_directed \
        --link_resources=115 \
        --max_requests=10 \
        --max_timesteps=10 \
        --mean_service_holding_time=25 \
        --continuous_operation \
        --ENV_WARMUP_STEPS=700 \
        --TOTAL_TIMESTEPS 200000 \
        --NUM_ENVS 200 \
        --PLOTTING \
        --launch_power_type=fixed \
        --launch_power "$lp" \
        --interband_gap=100 \
        --values_bw=400,600,800,1200 \
        --guardband=0 \
        --coherent \
        --reward_type=bitrate \
        --snr_margin=0.01 \
        --slot_size=100 \
        --VISIBLE_DEVICES=2 \
        --DOWNSAMPLE_FACTOR 1 \
        --WANDB \
        --PROJECT FIXED_LP_SWEEP \
        --EXPERIMENT_NAME "FIXED_LP_SWEEP_${lp}" \
        --EVAL_HEURISTIC \
        --path_heuristic ksp_ff

    echo "Completed run with launch power: $lp"
    echo "----------------------------------------"
done