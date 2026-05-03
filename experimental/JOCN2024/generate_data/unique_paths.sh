#!/bin/bash

PYTHON_PATH="/home/uceedoh/xlron_env/bin/python3.11"
SCRIPT_PATH="/home/uceedoh/git/XLRON/xlron/train/train.py"

HEURISTICS=("ksp_ff")
K_VALUES=(5 50)
TOPOLOGIES=(
    "nsfnet_deeprmsa_directed"
    "cost239_deeprmsa_directed"
    "usnet_gcnrnn_directed"
    "jpn48_directed"
)

OUTPUT_FILE="experiment_results_unique_paths.jsonl"

# Clear output file
> $OUTPUT_FILE

for HEUR in "${HEURISTICS[@]}"; do
    for TOPOLOGY in "${TOPOLOGIES[@]}"; do
        for K in "${K_VALUES[@]}"; do
          for weight in "--weight=weight" ""; do
            echo "Running experiment: HEUR=$HEUR, TOPOLOGY=$TOPOLOGY, LOAD=inf, K=$K weight=$weight"

            $PYTHON_PATH $SCRIPT_PATH \
                --env_type=rmsa \
                --k=$K \
                $weight \
                --topology_name=$TOPOLOGY \
                --link_resources=100 \
                --ENV_WARMUP_STEPS=0 \
                --TOTAL_TIMESTEPS 30000000 \
                --EVAL_HEURISTIC \
                --log_actions \
                --incremental_loading \
                --end_first_blocking \
                --path_heuristic $HEUR \
                --VISIBLE_DEVICES 3 \
                --NUM_ENVS 3000 \
                --log_actions \
                --DATA_OUTPUT_FILE "$OUTPUT_FILE"

            echo "Completed experiment: HEUR=$HEUR, TOPOLOGY=$TOPOLOGY, LOAD=$LOAD, K=$K weight=$weight"
          done
        done
    done
done
