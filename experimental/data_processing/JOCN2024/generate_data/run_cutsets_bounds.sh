#!/bin/bash

PYTHON_PATH="./.venv/bin/python"
SCRIPT_PATH="-m xlron.bounds.cutsets_bounds"
OUTPUT_FILE="experiment_results_cutsets_bounds.jsonl"

# Fixed cutset flags
CUTSET_FLAGS="--CUTSET_EXHAUSTIVE --CUTSET_BATCH_SIZE=512 --CUTSET_ITERATIONS=32 --CUTSET_TOP_K=256 --cutset_link_selection_mode=least_congested" # Optionally add this flag for a looser bound: --NEGLECT_SPECTRUM_CONTINUITY

# Clear output file
> $OUTPUT_FILE

run_experiment() {
    local name=$1
    local topology=$2
    local min_load=$3
    local max_load=$4
    local step_load=$5
    local k=$6
    local additional_args=$7

    echo "Running $name: topology=$topology, load=$min_load-$max_load (step $step_load), k=$k"

    local cutset_flags="${8:-$CUTSET_FLAGS}"

    $PYTHON_PATH $SCRIPT_PATH \
        --topology_name "$topology" \
        --load "$max_load" \
        --min_load "$min_load" \
        --max_load "$max_load" \
        --step_load "$step_load" \
        --k "$k" \
        --max_requests 13000 \
        --num_trials 10 \
        --modulations_csv_filepath "./xlron/data/modulations/modulations_deeprmsa.csv" \
        --PROJECT "$name" \
        --DATA_OUTPUT_FILE "$OUTPUT_FILE" \
        $cutset_flags \
        $additional_args
}

# DeepRMSA, Reward-RMSA, GCN-RMSA Experiments
args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --continuous_operation --truncate_holding_time"
run_experiment "DeepRMSA~Reward-RMSA~GCN-RMSA" "nsfnet_deeprmsa_directed" "150" "300" "10" "50" "$args"
run_experiment "DeepRMSA~Reward-RMSA~GCN-RMSA" "cost239_deeprmsa_directed" "370" "670" "10" "50" "$args"
run_experiment "DeepRMSA~Reward-RMSA~GCN-RMSA" "usnet_gcnrnn_directed" "310" "540" "10" "50" "$args"

# MaskRSA NSFNET
args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12 --continuous_operation"
run_experiment "MaskRSA" "nsfnet_deeprmsa_undirected" "80" "175" "5" "50" "$args"
# MaskRSA JPN48 (too many nodes for exhaustive search, use shortest-paths method)
JPN48_CUTSET_FLAGS="--CUTSET_TOP_K=256 --cutset_link_selection_mode=least_congested"
run_experiment "MaskRSA" "jpn48_undirected" "150" "300" "10" "50" "$args" "$JPN48_CUTSET_FLAGS"

# PtrNet-RSA
base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10 --continuous_operation"
var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"

# NSFNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_experiment "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" "170" "270" "10" "50" "$args"
# COST239 PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_experiment "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" "330" "500" "10" "50" "$args"
# USNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_experiment "PtrNet-RSA-40" "usnet_ptrnet_undirected" "210" "310" "10" "50" "$args"

# NSFNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_experiment "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" "200" "340" "10" "50" "$args"
# COST239 PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_experiment "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" "420" "670" "10" "50" "$args"
# USNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_experiment "PtrNet-RSA-80" "usnet_ptrnet_undirected" "220" "380" "10" "50" "$args"
