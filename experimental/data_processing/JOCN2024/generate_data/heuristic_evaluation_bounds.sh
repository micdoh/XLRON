#!/bin/bash

EVAL_PATH="./xlron/train/train.py"
OUTPUT_FILE="experiment_results_eval_bounds.jsonl"

# Clear output file
> $OUTPUT_FILE

run_experiment() {
    local name=$1
    local topology=$2
    local min_load=$3
    local max_load=$4
    local step_load=$5
    local k=$6
    local heur=$7
    local additional_args=$8

    echo "Running $name: topology=$topology, load=$min_load-$max_load (step $step_load), k=$k"

    uv run $EVAL_PATH \
        --min_load=$min_load \
        --max_load=$max_load \
        --step_load=$step_load \
        --k=$k \
        --topology_name=$topology \
        --continuous_operation \
        --ENV_WARMUP_STEPS=0 \
        --TOTAL_TIMESTEPS 2600000 \
        --STEPS_PER_INCREMENT 130000 \
        --NUM_ENVS 200 \
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

# NSFNET DeepRMSA
run_experiment "DeepRMSA" "nsfnet_deeprmsa_directed" 150 300 10 "$k" "ksp_ff" "$args"

# COST239 DeepRMSA
run_experiment "DeepRMSA" "cost239_deeprmsa_directed" 400 630 10 "$k" "ksp_ff" "$args"

# USNET GCN-RMSA
run_experiment "GCN-RMSA" "usnet_gcnrnn_directed" 310 540 10 "$k" "ksp_ff" "$args"

# MaskRSA Experiments
args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12"

# NSFNET MaskRSA
run_experiment "MaskRSA" "nsfnet_deeprmsa_undirected" 80 175 5 "$k" "ksp_ff" "$args"

# JPN48 MaskRSA
run_experiment "MaskRSA" "jpn48_undirected" 150 280 10 "$k" "ff_ksp" "$args"

# PtrNet-RSA-40 Experiments
base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10"
var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"

# NSFNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_experiment "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" 200 270 10 "$k" "ksp_ff" "$args"

# COST239 PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_experiment "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" 420 500 10 "$k" "ksp_ff" "$args"

# USNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_experiment "PtrNet-RSA-40" "usnet_ptrnet_undirected" 210 310 10 "$k" "ksp_ff" "$args"

# PtrNet-RSA-80 Experiments
# NSFNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_experiment "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" 210 340 10 "$k" "ksp_ff" "$args"

# COST239 PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_experiment "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" 450 670 10 "$k" "ksp_ff" "$args"

# USNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_experiment "PtrNet-RSA-80" "usnet_ptrnet_undirected" 220 380 10 "$k" "ksp_ff" "$args"
