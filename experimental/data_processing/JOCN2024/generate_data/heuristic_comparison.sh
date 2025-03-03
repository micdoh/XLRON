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

# Create/overwrite output CSV file with headers
OUTPUT_FILE="experiment_results.csv"
echo "HEUR,TOPOLOGY,LOAD,K,returns_mean,returns_std,returns_iqr_lower,returns_iqr_upper,lengths_mean,lengths_std,lengths_iqr_lower,lengths_iqr_upper,cum_returns_mean,cum_returns_std,cum_returns_iqr_lower,cum_returns_iqr_upper,accepted_services_mean,accepted_services_std,accepted_services_iqr_lower,accepted_services_iqr_upper,accepted_bitrate_mean,accepted_bitrate_std,accepted_bitrate_iqr_lower,accepted_bitrate_iqr_upper,total_bitrate_mean,total_bitrate_std,total_bitrate_iqr_lower,total_bitrate_iqr_upper,utilisation_mean,utilisation_std,utilisation_iqr_lower,utilisation_iqr_upper,service_blocking_probability_mean,service_blocking_probability_std,service_blocking_probability_iqr_lower,service_blocking_probability_iqr_upper,bitrate_blocking_probability_mean,bitrate_blocking_probability_std,bitrate_blocking_probability_iqr_lower,bitrate_blocking_probability_iqr_upper" > $OUTPUT_FILE

for HEUR in "${HEURISTICS[@]}"; do
    for TOPOLOGY in "${!TOPOLOGY_LOADS[@]}"; do
        LOAD=${TOPOLOGY_LOADS[$TOPOLOGY]}
        for K in "${K_VALUES[@]}"; do
            echo "Running experiment: HEUR=$HEUR, TOPOLOGY=$TOPOLOGY, LOAD=$LOAD, K=$K"

            # Run the experiment
            OUTPUT=$($PYTHON_PATH $SCRIPT_PATH \
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
                --path_heuristic $HEUR)

            # Extract metrics using awk and store in CSV format
            echo "$OUTPUT" | awk -v heur="$HEUR" -v topo="$TOPOLOGY" -v l="$LOAD" -v k="$K" '
                BEGIN {
                    started=0
                    # Define the order of metrics we want in output
                    metrics[1]="returns mean"
                    metrics[2]="returns std"
                    metrics[3]="returns IQR lower"
                    metrics[4]="returns IQR upper"
                    metrics[5]="lengths mean"
                    metrics[6]="lengths std"
                    metrics[7]="lengths IQR lower"
                    metrics[8]="lengths IQR upper"
                    metrics[9]="cum_returns mean"
                    metrics[10]="cum_returns std"
                    metrics[11]="cum_returns IQR lower"
                    metrics[12]="cum_returns IQR upper"
                    metrics[13]="accepted_services mean"
                    metrics[14]="accepted_services std"
                    metrics[15]="accepted_services IQR lower"
                    metrics[16]="accepted_services IQR upper"
                    metrics[17]="accepted_bitrate mean"
                    metrics[18]="accepted_bitrate std"
                    metrics[19]="accepted_bitrate IQR lower"
                    metrics[20]="accepted_bitrate IQR upper"
                    metrics[21]="total_bitrate mean"
                    metrics[22]="total_bitrate std"
                    metrics[23]="total_bitrate IQR lower"
                    metrics[24]="total_bitrate IQR upper"
                    metrics[25]="utilisation mean"
                    metrics[26]="utilisation std"
                    metrics[27]="utilisation IQR lower"
                    metrics[28]="utilisation IQR upper"
                    metrics[29]="service_blocking_probability mean"
                    metrics[30]="service_blocking_probability std"
                    metrics[31]="service_blocking_probability IQR lower"
                    metrics[32]="service_blocking_probability IQR upper"
                    metrics[33]="bitrate_blocking_probability mean"
                    metrics[34]="bitrate_blocking_probability std"
                    metrics[35]="bitrate_blocking_probability IQR lower"
                    metrics[36]="bitrate_blocking_probability IQR upper"
                }
                /^EXECUTION:/ { started=1; next }
                started && /mean:/ {
                    split($0, parts, ":")
                    metric=parts[1]
                    value=parts[2]
                    gsub(/ /, "", value)
                    values[metric]=value
                }
                started && /std:/ {
                    split($0, parts, ":")
                    metric=parts[1]
                    value=parts[2]
                    gsub(/ /, "", value)
                    values[metric]=value
                }
                started && /IQR lower:/ {
                    split($0, parts, ":")
                    metric=parts[1]
                    value=parts[2]
                    gsub(/ /, "", value)
                    values[metric]=value
                }
                started && /IQR upper:/ {
                    split($0, parts, ":")
                    metric=parts[1]
                    value=parts[2]
                    gsub(/ /, "", value)
                    values[metric]=value
                }
                END {
                    printf "%s,%s,%s,%s", heur, topo, l, k
                    # Output values in the defined order
                    for (i=1; i<=36; i++) {
                        if (values[metrics[i]] != "") {
                            printf ",%s", values[metrics[i]]
                        } else {
                            printf ",NA"  # Handle missing values
                        }
                    }
                    printf "\n"
                }' >> $OUTPUT_FILE

            echo "Completed experiment: HEUR=$HEUR, TOPOLOGY=$TOPOLOGY, LOAD=$LOAD, K=$K"
        done
    done
done