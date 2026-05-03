#!/bin/bash

EVAL_PATH="./xlron/train/train.py"
HEURISTIC_OUTPUT="experiment_results_heuristic_nsfnet_deeprmsa_20k.jsonl"
TRANSFORMER_OUTPUT="experiment_results_transformer_nsfnet_deeprmsa_20k.jsonl"

k=50
NUM_ENVS=200
# 20k requests per env: 20000 * 200 = 4,000,000
TOTAL_TIMESTEPS=4000000

common_args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --truncate_holding_time"

# Clear output files
> $HEURISTIC_OUTPUT
> $TRANSFORMER_OUTPUT

echo "=== Heuristic Evaluation (KSP-FF) ==="
uv run $EVAL_PATH \
    --min_load=150 \
    --max_load=300 \
    --step_load=10 \
    --k=$k \
    --topology_name=nsfnet_deeprmsa_directed \
    --continuous_operation \
    --ENV_WARMUP_STEPS=0 \
    --TOTAL_TIMESTEPS $TOTAL_TIMESTEPS \
    --STEPS_PER_INCREMENT 200000 \
    --NUM_ENVS $NUM_ENVS \
    --EVAL_HEURISTIC \
    --path_heuristic ksp_ff \
    --modulations_csv_filepath "./xlron/data/modulations/modulations_deeprmsa.csv" \
    --PROJECT "DeepRMSA_Heuristic_20k" \
    --DATA_OUTPUT_FILE "$HEURISTIC_OUTPUT" \
    $common_args

echo "=== Transformer Model Evaluation ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=.98 uv run $EVAL_PATH \
    --min_load=150 \
    --max_load=300 \
    --step_load=10 \
    --k=$k \
    --topology_name=nsfnet_deeprmsa_directed \
    --continuous_operation \
    --ENV_WARMUP_STEPS=0 \
    --TOTAL_TIMESTEPS $TOTAL_TIMESTEPS \
    --NUM_ENVS $NUM_ENVS \
    --EVAL_MODEL \
    --MODEL_PATH "./episodic_20_8_10_2.eqx" \
    --USE_TRANSFORMER \
    --transformer_num_layers 2 \
    --transformer_num_heads 4 \
    --aggregate_slots 20 \
    --SEPARATE_VF_OPTIMIZER \
    --ROLLOUT_LENGTH 64 \
    --modulations_csv_filepath "./xlron/data/modulations/modulations_deeprmsa.csv" \
    --PROJECT "DeepRMSA_Transformer_20k" \
    --DATA_OUTPUT_FILE "$TRANSFORMER_OUTPUT" \
    $common_args

echo "=== Done ==="
echo "Heuristic results: $HEURISTIC_OUTPUT"
echo "Transformer results: $TRANSFORMER_OUTPUT"
