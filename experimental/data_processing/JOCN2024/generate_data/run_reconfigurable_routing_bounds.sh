#!/bin/bash

PYTHON_PATH="/Users/michaeldoherty/Library/Caches/pypoetry/virtualenvs/xlron-QeH3eSKC-py3.11/bin/python"
SCRIPT_PATH="/Users/michaeldoherty/git/XLRON/capacity_bound_estimation/reconfigurable_routing_bounds_sequential.py"
OUTPUT_FILE="experiment_results_bounds.csv"

echo "experiment,topology,load,k,heur,blocking_prob_mean,blocking_prob_std,blocking_prob_iqr_lower,blocking_prob_iqr_upper,block_count_mean,block_count_std,block_count_iqr_lower,block_count_iqr_upper,fix_count_mean,fix_count_std,fix_count_iqr_lower,fix_count_iqr_upper,fix_ratio_mean,fix_ratio_std,fix_ratio_iqr_lower,fix_ratio_iqr_upper" > $OUTPUT_FILE

run_experiment() {
    local name=$1
    local topology=$2
    local traffic_load=$3
    local k=$4
    local additional_args=$5
    local heur=$6

    echo "Running $name: topology=$topology, load=$traffic_load, k=$k, heur=$heur"

    output=$($PYTHON_PATH $SCRIPT_PATH \
        --topology_name "$topology" \
        --load "$traffic_load" \
        --k "$k" \
        --TOTAL_TIMESTEPS 13000 \
        --NUM_ENVS 1 \
        --modulations_csv_filepath "./modulations/modulations_deeprmsa.csv" \
        --path_heuristic "$heur" \
        $additional_args 2>&1 | tee /dev/tty)

    blocking_mean=$(echo "$output" | grep "Blocking Probability mean:" | sed 's/.*: \(.*\)/\1/')
    blocking_std=$(echo "$output" | grep "Blocking Probability std:" | sed 's/.*: \(.*\)/\1/')
    blocking_iqr_lower=$(echo "$output" | grep "Blocking Probability IQR lower:" | sed 's/.*: \(.*\)/\1/')
    blocking_iqr_upper=$(echo "$output" | grep "Blocking Probability IQR upper:" | sed 's/.*: \(.*\)/\1/')
    block_count_mean=$(echo "$output" | grep "Block Count mean:" | sed 's/.*: \(.*\)/\1/')
    block_count_std=$(echo "$output" | grep "Block Count std:" | sed 's/.*: \(.*\)/\1/')
    block_count_iqr_lower=$(echo "$output" | grep "Block Count IQR lower:" | sed 's/.*: \(.*\)/\1/')
    block_count_iqr_upper=$(echo "$output" | grep "Block Count IQR upper:" | sed 's/.*: \(.*\)/\1/')
    fix_count_mean=$(echo "$output" | grep "Fix Count mean:" | sed 's/.*: \(.*\)/\1/')
    fix_count_std=$(echo "$output" | grep "Fix Count std:" | sed 's/.*: \(.*\)/\1/')
    fix_count_iqr_lower=$(echo "$output" | grep "Fix Count IQR lower:" | sed 's/.*: \(.*\)/\1/')
    fix_count_iqr_upper=$(echo "$output" | grep "Fix Count IQR upper:" | sed 's/.*: \(.*\)/\1/')
    fix_ratio_mean=$(echo "$output" | grep "Fix Ratio mean:" | sed 's/.*: \(.*\)/\1/')
    fix_ratio_std=$(echo "$output" | grep "Fix Ratio std:" | sed 's/.*: \(.*\)/\1/')
    fix_ratio_iqr_lower=$(echo "$output" | grep "Fix Ratio IQR lower:" | sed 's/.*: \(.*\)/\1/')
    fix_ratio_iqr_upper=$(echo "$output" | grep "Fix Ratio IQR upper:" | sed 's/.*: \(.*\)/\1/')

    echo "$name,$topology,$traffic_load,$k,$heur,$blocking_mean,$blocking_std,$blocking_iqr_lower,$blocking_iqr_upper,$block_count_mean,$block_count_std,$block_count_iqr_lower,$block_count_iqr_upper,$fix_count_mean,$fix_count_std,$fix_count_iqr_lower,$fix_count_iqr_upper,$fix_ratio_mean,$fix_ratio_std,$fix_ratio_iqr_lower,$fix_ratio_iqr_upper" >> $OUTPUT_FILE
}

# DeepRMSA, Reward-RMSA, GCN-RMSA  Experiments
args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --continuous_operation --truncate_holding_time"
for traffic_load in 232 240 250 260; do
  run_experiment "DeepRMSA~Reward-RMSA~GCN-RMSA" "nsfnet_deeprmsa_directed" "$traffic_load" "50" "$args" "ksp_ff"
done
for traffic_load in 500 540 550 575 600; do
  run_experiment "DeepRMSA~Reward-RMSA~GCN-RMSA" "cost239_deeprmsa_directed" "$traffic_load" "50" "$args" "ksp_ff"
done
for traffic_load in 450 460 475 478 500; do
  run_experiment "DeepRMSA~Reward-RMSA~GCN-RMSA" "usnet_gcnrnn_directed" "$traffic_load" "50" "$args" "ksp_ff"
done

# MaskRSA NSFNET
args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12 --continuous_operation"
for traffic_load in 122; do #125 130 135; do
run_experiment "MaskRSA" "nsfnet_deeprmsa_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
# MaskRSA JPN48
for traffic_load in 220 230 240; do
run_experiment "MaskRSA" "jpn48_undirected" "$traffic_load" "50" "$args" "ff_ksp"
done
# PtrNet-RSA
base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10 --continuous_operation"
var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"

 # NSFNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 225 230 240 250; do
run_experiment "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
# COST239 PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 450 460 470; do
run_experiment "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
# USNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 250 255 260 270; do
run_experiment "PtrNet-RSA-40" "usnet_ptrnet_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done

# NSFNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 300 310 320; do
run_experiment "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
# COST239 PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 600 608 615 630; do
run_experiment "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
# USNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 350 360 370; do
run_experiment "PtrNet-RSA-80" "usnet_ptrnet_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
