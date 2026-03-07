#!/bin/bash

EVAL_PATH="./xlron/train/train.py"
OUTPUT_FILE="experiment_results_transformer_eval_bounds.jsonl"

# Clear output file
> $OUTPUT_FILE

run_experiment() {
    local name=$1
    local topology=$2
    local min_load=$3
    local max_load=$4
    local step_load=$5
    local k=$6
    local model_path=$7
    local num_heads=$8
    local agg_slots=$9
    local additional_args=${10}

    echo "Running $name: topology=$topology, load=$min_load-$max_load (step $step_load), k=$k, model=$model_path"

    XLA_PYTHON_CLIENT_MEM_FRACTION=.98 uv run $EVAL_PATH \
        --min_load=$min_load \
        --max_load=$max_load \
        --step_load=$step_load \
        --k=$k \
        --topology_name=$topology \
        --continuous_operation \
        --ENV_WARMUP_STEPS=3000 \
        --TOTAL_TIMESTEPS 2000000 \
        --NUM_ENVS 200 \
        --EVAL_MODEL \
        --MODEL_PATH "$model_path" \
        --USE_TRANSFORMER \
        --transformer_num_layers 2 \
        --transformer_num_heads $num_heads \
        --aggregate_slots $agg_slots \
        --SEPARATE_VF_OPTIMIZER \
        --ROLLOUT_LENGTH 64 \
        --modulations_csv_filepath "./xlron/data/modulations/modulations_deeprmsa.csv" \
        --PROJECT "TRANSFORMER_EVAL" \
        --DATA_OUTPUT_FILE "$OUTPUT_FILE" \
        $additional_args
}

k=50

# Deep/Reward/GCN-RMSA Experiments (env_type=rmsa, link_resources=100)
args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --truncate_holding_time"

# NSFNET DeepRMSA (nsfnet_deeprmsa_directed) - note: 4 heads for this model
run_experiment "DeepRMSA" "nsfnet_deeprmsa_directed" 150 300 10 "$k" "./episodic_20_8_10.eqx" 4 20 "$args"

# COST239 DeepRMSA (cost239_deeprmsa_directed)
run_experiment "DeepRMSA" "cost239_deeprmsa_directed" 400 630 10 "$k" "./cost239_deeprmsa_13.eqx" 8 50 "$args"

# USNET GCN-RMSA (usnet_gcnrnn_directed)
run_experiment "GCN-RMSA" "usnet_gcnrnn_directed" 310 510 10 "$k" "./usnet_2.eqx" 8 20 "$args"

# MaskRSA Experiments (env_type=rmsa, link_resources=80)
args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12"

# NSFNET MaskRSA (nsfnet_deeprmsa_undirected) - note: 4 heads for this model
run_experiment "MaskRSA" "nsfnet_deeprmsa_undirected" 90 145 5 "$k" "./nsfnet_maskrsa_43_1.eqx" 4 20 "$args"

# JPN48 MaskRSA (jpn48_undirected)
run_experiment "MaskRSA" "jpn48_undirected" 160 260 10 "$k" "./jpn48_maskrsa.eqx" 8 20 "$args"

# PtrNet-RSA-40 Experiments (env_type=rsa, link_resources=40)
base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10"

# NSFNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_experiment "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" 200 270 10 "$k" "./nsfnet_rsa40.eqx" 8 20 "$args"

# COST239 PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_experiment "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" 420 500 10 "$k" "./cost239_rsa40.eqx" 8 20 "$args"

# USNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_experiment "PtrNet-RSA-40" "usnet_ptrnet_undirected" 210 310 10 "$k" "./usnet_rsa40.eqx" 8 20 "$args"

# PtrNet-RSA-80 Experiments (env_type=rsa, link_resources=80)
var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"

# NSFNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_experiment "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" 210 340 10 "$k" "./nsfnet_rsa80.eqx" 8 20 "$args"

# COST239 PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_experiment "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" 450 670 10 "$k" "./cost239_rsa80_1.eqx" 8 20 "$args"

# USNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_experiment "PtrNet-RSA-80" "usnet_ptrnet_undirected" 220 380 10 "$k" "./usnet_rsa80.eqx" 8 20 "$args"
