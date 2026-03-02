#!/bin/bash

PYTHON_PATH=".venv/bin/python"
DEFRAG_PATH="xlron/bounds/reconfigurable_routing_bounds.py"
OUTPUT_FILE="experiment_results_reconfigurable_bounds.jsonl"

# Clear output file
> $OUTPUT_FILE

run_reconfigurable_routing_bound() {
    local name=$1
    local topology=$2
    local min_load=$3
    local max_load=$4
    local step_load=$5
    local k=$6
    local additional_args=$7
    local heur=$8

    echo "Running $name: topology=$topology, load=$min_load-$max_load (step $step_load), k=$k, heur=$heur"

    $PYTHON_PATH -u $DEFRAG_PATH \
        --topology_name "$topology" \
        --load "$max_load" \
        --min_load "$min_load" \
        --max_load "$max_load" \
        --step_load "$step_load" \
        --k "$k" \
        --TOTAL_TIMESTEPS 13000 \
        --NUM_ENVS 1 \
        --modulations_csv_filepath "./xlron/data/modulations/modulations_deeprmsa.csv" \
        --path_heuristic "$heur" \
        --COMPILE_RR_BOUNDS \
        --PROJECT "$name" \
        --DATA_OUTPUT_FILE "$OUTPUT_FILE" \
        $additional_args
}

# DeepRMSA, Reward-RMSA, GCN-RMSA  Experiments
args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --continuous_operation --truncate_holding_time"
run_reconfigurable_routing_bound "DeepRMSA~Reward-RMSA~GCN-RMSA" "nsfnet_deeprmsa_directed" "150" "300" "10" "50" "$args" "ksp_ff"
run_reconfigurable_routing_bound "DeepRMSA~Reward-RMSA~GCN-RMSA" "cost239_deeprmsa_directed" "400" "670" "10" "50" "$args" "ksp_ff"
run_reconfigurable_routing_bound "DeepRMSA~Reward-RMSA~GCN-RMSA" "usnet_gcnrnn_directed" "310" "540" "10" "50" "$args" "ksp_ff"

# MaskRSA NSFNET
args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12 --continuous_operation"
run_reconfigurable_routing_bound "MaskRSA" "nsfnet_deeprmsa_undirected" "90" "175" "5" "50" "$args" "ksp_ff"
# MaskRSA JPN48
run_reconfigurable_routing_bound "MaskRSA" "jpn48_undirected" "160" "300" "10" "50" "$args" "ff_ksp"

# PtrNet-RSA
base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10 --continuous_operation"
var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"
# NSFNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_reconfigurable_routing_bound "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" "200" "270" "10" "50" "$args" "ksp_ff"
# COST239 PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_reconfigurable_routing_bound "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" "420" "500" "10" "50" "$args" "ksp_ff"
# USNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
run_reconfigurable_routing_bound "PtrNet-RSA-40" "usnet_ptrnet_undirected" "210" "310" "10" "50" "$args" "ksp_ff"

# NSFNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_reconfigurable_routing_bound "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" "210" "340" "10" "50" "$args" "ksp_ff"
# COST239 PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_reconfigurable_routing_bound "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" "450" "670" "10" "50" "$args" "ksp_ff"
# USNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
run_reconfigurable_routing_bound "PtrNet-RSA-80" "usnet_ptrnet_undirected" "220" "380" "10" "50" "$args" "ksp_ff"
