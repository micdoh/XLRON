#!/bin/bash

PYTHON_PATH=".venv/bin/python"
SCRIPT_PATH="-m xlron.train.train"
OUTPUT_FILE="experiment_results_eval.jsonl"

# Clear output file
> $OUTPUT_FILE

run_experiment() {
    local name=$1
    local topology=$2
    local min_load=$3
    local max_load=$4
    local step_load=$5
    local k=$6
    local weight=$7
    local additional_args=$8

    echo "Running $name: topology=$topology, load=$min_load-$max_load (step $step_load), k=$k"

    $PYTHON_PATH $SCRIPT_PATH \
        --load=$max_load \
        --min_load=$min_load \
        --max_load=$max_load \
        --step_load=$step_load \
        --k=$k \
        $weight \
        --topology_name=$topology \
        --max_requests=1e3 \
        --continuous_operation \
        --ENV_WARMUP_STEPS=3000 \
        --TOTAL_TIMESTEPS 100000 \
        --NUM_ENVS 10 \
        --EVAL_HEURISTIC \
        --path_heuristic ksp_ff \
        --modulations_csv_filepath "./xlron/data/modulations/modulations_deeprmsa.csv" \
        --PROJECT "$name" \
        --DATA_OUTPUT_FILE "$OUTPUT_FILE" \
        $additional_args
}

# Helper for single-load experiments (no sweep)
run_single() {
    local name=$1
    local topology=$2
    local traffic_load=$3
    local k=$4
    local weight=$5
    local additional_args=$6

    echo "Running $name: topology=$topology, load=$traffic_load, k=$k"

    $PYTHON_PATH $SCRIPT_PATH \
        --load=$traffic_load \
        --k=$k \
        $weight \
        --topology_name=$topology \
        --max_requests=1e3 \
        --continuous_operation \
        --ENV_WARMUP_STEPS=3000 \
        --TOTAL_TIMESTEPS 100000 \
        --NUM_ENVS 10 \
        --EVAL_HEURISTIC \
        --path_heuristic ksp_ff \
        --modulations_csv_filepath "./xlron/data/modulations/modulations_deeprmsa.csv" \
        --PROJECT "$name" \
        --DATA_OUTPUT_FILE "$OUTPUT_FILE" \
        $additional_args
}

for weight in "--weight=weight" ""; do

  for k in $([ -n "$weight" ] && echo "5" || echo "5 50"); do

    # DeepRMSA Experiments (single load points)
    run_single "DeepRMSA" "nsfnet_deeprmsa_directed" "250" "$k" "$weight" "--env_type rmsa --link_resources 100 --mean_service_holding_time 25 --truncate_holding_time"
    run_single "DeepRMSA" "cost239_deeprmsa_directed" "600" "$k" "$weight" "--env_type rmsa --link_resources 100 --mean_service_holding_time 30 --truncate_holding_time"

    # Reward-RMSA
    args="--env_type rmsa --link_resources 100 --mean_service_holding_time 14 --truncate_holding_time $weight"
    run_experiment "Reward-RMSA" "nsfnet_deeprmsa_directed" "168" "210" "14" "$k" "$weight" "$args"

    # MaskRSA NSFNET
    args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12 $weight"
    run_experiment "MaskRSA" "nsfnet_deeprmsa_undirected" "80" "160" "10" "$k" "$weight" "$args"

    # MaskRSA JPN48
    run_experiment "MaskRSA" "jpn48_undirected" "120" "160" "10" "$k" "$weight" "$args"

    # GCN-RMSA NSFNET
    args="--env_type rmsa --link_resources 100 --mean_service_holding_time 14 --truncate_holding_time $weight"
    run_experiment "GCN-RMSA" "nsfnet_deeprmsa_directed" "154" "210" "14" "$k" "$weight" "$args"

    # GCN-RMSA COST239
    args="--env_type rmsa --link_resources 100 --mean_service_holding_time 23 --truncate_holding_time $weight"
    run_experiment "GCN-RMSA" "cost239_deeprmsa_directed" "368" "460" "23" "$k" "$weight" "$args"

    # GCN-RMSA USNET
    args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --truncate_holding_time $weight"
    run_experiment "GCN-RMSA" "usnet_gcnrnn_directed" "320" "400" "20" "$k" "$weight" "$args"

    # PtrNet-RSA-40 Experiments
    base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10 $weight"

    # NSFNET PtrNet-RSA-40
    args="$base_args --link_resources 40 --values_bw 1"
    run_experiment "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" "180" "240" "10" "$k" "$weight" "$args"

    # NSFNET PtrNet-RSA-80
    var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"
    args="$base_args --link_resources 80 --values_bw $var_bw"
    run_experiment "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" "200" "240" "10" "$k" "$weight" "$args"

    # COST239 PtrNet-RSA-40
    args="$base_args --link_resources 40 --values_bw 1"
    run_experiment "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" "340" "420" "20" "$k" "$weight" "$args"

    # COST239 PtrNet-RSA-80
    args="$base_args --link_resources 80 --values_bw $var_bw"
    run_experiment "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" "420" "460" "20" "$k" "$weight" "$args"

    # USNET PtrNet-RSA-40
    args="$base_args --link_resources 40 --values_bw 1"
    run_experiment "PtrNet-RSA-40" "usnet_ptrnet_undirected" "210" "280" "10" "$k" "$weight" "$args"

    # USNET PtrNet-RSA-80
    args="$base_args --link_resources 80 --values_bw $var_bw"
    run_experiment "PtrNet-RSA-80" "usnet_ptrnet_undirected" "260" "330" "10" "$k" "$weight" "$args"

  done

done
