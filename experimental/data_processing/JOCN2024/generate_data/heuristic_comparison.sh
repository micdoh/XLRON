#!/usr/bin/bash

PYTHON_PATH="/home/uceedoh/xlron_env/bin/python3.11"
SCRIPT_PATH="/home/uceedoh/git/XLRON/xlron/train/train.py"

# Define arrays for parameter combinations
declare -A TOPOLOGY_LOADS=(
    ["nsfnet_deeprmsa_directed"]="145"
    ["cost239_deeprmsa_directed"]="317"
    ["usnet_gcnrnn_directed"]="265"
    ["jpn48_directed"]="115"
)
HEURISTICS=("ksp_ff" "ff_ksp" "ksp_bf" "bf_ksp" "kme_ff" "kca_ff") # "kmc_ff" "kmf_ff"
K_VALUES=(2 5 8 11 14 17 20 23 26)

OUTPUT_FILE="experiment_results.jsonl"

# Clear output file
> $OUTPUT_FILE

for HEUR in "${HEURISTICS[@]}"; do
    for TOPOLOGY in "${!TOPOLOGY_LOADS[@]}"; do
        LOAD=${TOPOLOGY_LOADS[$TOPOLOGY]}
        for K in "${K_VALUES[@]}"; do
            echo "Running experiment: HEUR=$HEUR, TOPOLOGY=$TOPOLOGY, LOAD=$LOAD, K=$K"

            $PYTHON_PATH $SCRIPT_PATH \
                --env_type=rmsa \
                --load=$LOAD \
                --k=$K \
                --topology_name=$TOPOLOGY \
                --link_resources=100 \
                --max_requests=1e3 \
                --mean_service_holding_time=10 \
                --continuous_operation \
                --ENV_WARMUP_STEPS=0 \
                --TOTAL_TIMESTEPS 30000000 \
                --NUM_ENVS 3000 \
                --EVAL_HEURISTIC \
                --VISIBLE_DEVICES 3 \
                --path_heuristic $HEUR \
                --DATA_OUTPUT_FILE "$OUTPUT_FILE"

            echo "Completed experiment: HEUR=$HEUR, TOPOLOGY=$TOPOLOGY, LOAD=$LOAD, K=$K"
        done
    done
done
