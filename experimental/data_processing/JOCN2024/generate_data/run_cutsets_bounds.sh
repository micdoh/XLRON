#!/bin/bash

PYTHON_PATH="./.venv/bin/python"
SCRIPT_PATH="-m xlron.bounds.cutsets_bounds"
OUTPUT_FILE="experiment_results_cutsets_bounds.jsonl"

# Fixed cutset flags
CUTSET_FLAGS="--CUTSET_EXHAUSTIVE --CUTSET_BATCH_SIZE=512 --CUTSET_ITERATIONS=32 --CUTSET_TOP_K=256 --cutset_link_selection_mode=least_congested"

# Clear output file
> $OUTPUT_FILE

run_experiment() {
    local name=$1
    local topology=$2
    local traffic_load=$3
    local k=$4
    local additional_args=$5

    echo "Running $name: topology=$topology, load=$traffic_load, k=$k"

    local cutset_flags="${6:-$CUTSET_FLAGS}"

    $PYTHON_PATH $SCRIPT_PATH \
        --topology_name "$topology" \
        --load "$traffic_load" \
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
for traffic_load in 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300; do
  run_experiment "DeepRMSA~Reward-RMSA~GCN-RMSA" "nsfnet_deeprmsa_directed" "$traffic_load" "50" "$args"
done
for traffic_load in 400 410 420 430 440 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630 640 650 660 670; do
  run_experiment "DeepRMSA~Reward-RMSA~GCN-RMSA" "cost239_deeprmsa_directed" "$traffic_load" "50" "$args"
done
for traffic_load in 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 510 520 530 540; do
  run_experiment "DeepRMSA~Reward-RMSA~GCN-RMSA" "usnet_gcnrnn_directed" "$traffic_load" "50" "$args"
done

# MaskRSA NSFNET
args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12 --continuous_operation"
for traffic_load in 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175; do
  run_experiment "MaskRSA" "nsfnet_deeprmsa_undirected" "$traffic_load" "50" "$args"
done
# MaskRSA JPN48 (too many nodes for exhaustive search, use shortest-paths method)
JPN48_CUTSET_FLAGS="--CUTSET_TOP_K=256 --cutset_link_selection_mode=least_congested"
for traffic_load in 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300; do
  run_experiment "MaskRSA" "jpn48_undirected" "$traffic_load" "50" "$args" "$JPN48_CUTSET_FLAGS"
done

# PtrNet-RSA
base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10 --continuous_operation"
var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"

# NSFNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 200 210 220 230 240 250 260 270; do
  run_experiment "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" "$traffic_load" "50" "$args"
done
# COST239 PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 420 430 440 450 460 470 480 490 500; do
  run_experiment "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" "$traffic_load" "50" "$args"
done
# USNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 210 220 230 240 250 260 270 280 290 300 310; do
  run_experiment "PtrNet-RSA-40" "usnet_ptrnet_undirected" "$traffic_load" "50" "$args"
done

# NSFNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 210 220 230 240 250 260 270 280 290 300 310 320 330 340; do
  run_experiment "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" "$traffic_load" "50" "$args"
done
# COST239 PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630 640 650 660 670; do
  run_experiment "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" "$traffic_load" "50" "$args"
done
# USNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380; do
  run_experiment "PtrNet-RSA-80" "usnet_ptrnet_undirected" "$traffic_load" "50" "$args"
done
