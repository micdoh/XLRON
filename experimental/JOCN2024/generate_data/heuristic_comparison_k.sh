#!/usr/bin/bash

PYTHON_PATH="/home/uceedoh/xlron_env/bin/python3.11"
SCRIPT_PATH="/home/uceedoh/git/XLRON/xlron/train/train.py"

# Define arrays for parameter combinations
declare -A TOPOLOGY_LOADS
# Initialize arrays of loads for each topology
NSFNET_LOADS=(90 103 119 137 157 181)
COST239_LOADS=(190 218 251 289 332 382)
USNET_LOADS=(140 160 184 211 243 279)
JPN48_LOADS=(52 61 72 91 100 120)

# Associate loads with topologies
TOPOLOGY_LOADS["nsfnet_deeprmsa_directed"]="${NSFNET_LOADS[*]}"
TOPOLOGY_LOADS["cost239_deeprmsa_directed"]="${COST239_LOADS[*]}"
TOPOLOGY_LOADS["usnet_gcnrnn_directed"]="${USNET_LOADS[*]}"
TOPOLOGY_LOADS["jpn48_directed"]="${JPN48_LOADS[*]}"

HEURISTICS=("ksp_ff")
K_VALUES=(2 5 8 11 14 17 20 25 30 40)

OUTPUT_FILE="experiment_results_k.jsonl"

# Clear output file
> $OUTPUT_FILE

for HEUR in "${HEURISTICS[@]}"; do
    for TOPOLOGY in "${!TOPOLOGY_LOADS[@]}"; do
        # Get the array of loads for this topology and iterate through them
        loads_string=${TOPOLOGY_LOADS[$TOPOLOGY]}
        # Convert string to array
        loads_array=($loads_string)

        for LOAD in "${loads_array[@]}"; do
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
                    --VISIBLE_DEVICES 2 \
                    --path_heuristic $HEUR \
                    --DATA_OUTPUT_FILE "$OUTPUT_FILE"

                echo "Completed experiment: HEUR=$HEUR, TOPOLOGY=$TOPOLOGY, LOAD=$LOAD, K=$K"
            done
        done
    done
done
