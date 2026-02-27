#!/bin/bash

PYTHON_PATH="./.venv/bin/python"
EVAL_PATH="./xlron/train/train.py"
OUTPUT_FILE="experiment_results_eval_bounds.jsonl"

# Clear output file
> $OUTPUT_FILE

run_experiment() {
    local name=$1
    local topology=$2
    local traffic_load=$3
    local k=$4
    local heur=$5
    local additional_args=$6

    echo "Running $name: topology=$topology, load=$traffic_load, k=$k"

    $PYTHON_PATH $EVAL_PATH \
        --load=$traffic_load \
        --k=$k \
        --topology_name=$topology \
        --continuous_operation \
        --ENV_WARMUP_STEPS=3000 \
        --TOTAL_TIMESTEPS 20000000 \
        --NUM_ENVS 2000 \
        --EVAL_HEURISTIC \
        --path_heuristic $heur \
        --modulations_csv_filepath "./xlron/data/modulations/modulations_deeprmsa.csv" \
        --PROJECT "$name" \
        --DATA_OUTPUT_FILE "$OUTPUT_FILE" \
        $additional_args
}

k=50

# Deep/Reward/GCN-RMSA Experiments
args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --truncate_holding_time"
for traffic_load in 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300; do
    run_experiment "DeepRMSA" "nsfnet_deeprmsa_directed" "$traffic_load" "$k" "ksp_ff" "$args"
done
args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --truncate_holding_time"
for traffic_load in 400 410 420 430 440 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630; do
    run_experiment "DeepRMSA" "cost239_deeprmsa_directed" "$traffic_load" "$k" "ksp_ff" "$args"
done
args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --truncate_holding_time"
for traffic_load in 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 510; do
    run_experiment "GCN-RMSA" "usnet_gcnrnn_directed" "$traffic_load" "$k" "ksp_ff" "$args"
done

# MaskRSA Experiments
args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12"
for traffic_load in 90 95 100 105 110 115 120 125 130 135 140 145; do
    run_experiment "MaskRSA" "nsfnet_deeprmsa_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
for traffic_load in 160 170 180 190 200 210 220 230 240 250 260; do
    run_experiment "MaskRSA" "jpn48_undirected" "$traffic_load" "$k" "ff_ksp" "$args"
done

# PtrNet-RSA-40 Experiments
base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10"
var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"
# NSFNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 200 210 220 230 240 250 260 270; do
    run_experiment "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
# COST239 PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 420 430 440 450 460 470 480 490 500; do
    run_experiment "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
# USNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 210 220 230 240 250 260 270 280 290 300 310; do
    run_experiment "PtrNet-RSA-40" "usnet_ptrnet_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done

# NSFNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 210 220 230 240 250 260 270 280 290 300 310 320 330 340; do
    run_experiment "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
# COST239 PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630 640 650 660 670; do
    run_experiment "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
# USNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380; do
    run_experiment "PtrNet-RSA-80" "usnet_ptrnet_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
