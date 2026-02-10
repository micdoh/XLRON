#!/bin/bash

PYTHON_PATH="./.venv/bin/python"
DEFRAG_PATH="./xlron/bounds/reconfigurable_routing_bounds_sequential.py"
OUTPUT_FILE="experiment_results_reconfigurable_bounds.csv"

echo "experiment,topology,load,k,heur,blocking_prob_mean,blocking_prob_std,blocking_prob_iqr_lower,blocking_prob_iqr_upper,block_count_mean,block_count_std,block_count_iqr_lower,block_count_iqr_upper,fix_count_mean,fix_count_std,fix_count_iqr_lower,fix_count_iqr_upper,fix_ratio_mean,fix_ratio_std,fix_ratio_iqr_lower,fix_ratio_iqr_upper" > $OUTPUT_FILE

run_reconfigurable_routing_bound() {
    local name=$1
    local topology=$2
    local traffic_load=$3
    local k=$4
    local additional_args=$5
    local heur=$6

    echo "Running $name: topology=$topology, load=$traffic_load, k=$k, heur=$heur"

    output=$($PYTHON_PATH -u $DEFRAG_PATH \
        --topology_name "$topology" \
        --load "$traffic_load" \
        --k "$k" \
        --TOTAL_TIMESTEPS 13000 \
        --NUM_ENVS 1 \
        --modulations_csv_filepath "./xlron/data/modulations/modulations_deeprmsa.csv" \
        --path_heuristic "$heur" \
        --COMPILE_RR_BOUNDS \
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
for traffic_load in 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300; do
  run_reconfigurable_routing_bound "DeepRMSA~Reward-RMSA~GCN-RMSA" "nsfnet_deeprmsa_directed" "$traffic_load" "50" "$args" "ksp_ff"
done
for traffic_load in 400 410 420 430 440 450 460 470 480 500 510 520 530 540 550 560 570 580 590 600 610 620 630; do
  run_reconfigurable_routing_bound "DeepRMSA~Reward-RMSA~GCN-RMSA" "cost239_deeprmsa_directed" "$traffic_load" "50" "$args" "ksp_ff"
done
for traffic_load in 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 510; do
  run_reconfigurable_routing_bound "DeepRMSA~Reward-RMSA~GCN-RMSA" "usnet_gcnrnn_directed" "$traffic_load" "50" "$args" "ksp_ff"
done

# MaskRSA NSFNET
args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12 --continuous_operation"
for traffic_load in 90 95 100 105 110 115 120 125 130 135 140 145; do
run_reconfigurable_routing_bound "MaskRSA" "nsfnet_deeprmsa_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
# MaskRSA JPN48
for traffic_load in 160 170 180 190 200 210 220 230 240 250 260; do
run_reconfigurable_routing_bound "MaskRSA" "jpn48_undirected" "$traffic_load" "50" "$args" "ff_ksp"
done
# PtrNet-RSA
base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10 --continuous_operation"
var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"

 # NSFNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 200 210 220 230 240 250 260 270; do
run_reconfigurable_routing_bound "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
# COST239 PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 420 430 440 450 460 470 480 490 500; do
run_reconfigurable_routing_bound "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
# USNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 210 220 230 240 250 260 270 280 290 300 310; do
run_reconfigurable_routing_bound "PtrNet-RSA-40" "usnet_ptrnet_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done

# NSFNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 210 220 230 240 250 260 270 280 290 300 310 320 330 340; do
run_reconfigurable_routing_bound "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
# COST239 PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630 640 650 660 670; do
run_reconfigurable_routing_bound "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
# USNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380; do
run_reconfigurable_routing_bound "PtrNet-RSA-80" "usnet_ptrnet_undirected" "$traffic_load" "50" "$args" "ksp_ff"
done
