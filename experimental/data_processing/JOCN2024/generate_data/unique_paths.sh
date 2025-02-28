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

# Create/overwrite output CSV file with headers
OUTPUT_FILE="experiment_results_unique_paths.csv"
echo "HEUR,TOPOLOGY,LOAD,K,weight,returns_mean,returns_std,returns_iqr_lower,returns_iqr_upper,lengths_mean,lengths_std,lengths_iqr_lower,lengths_iqr_upper,cum_returns_mean,cum_returns_std,cum_returns_iqr_lower,cum_returns_iqr_upper,accepted_services_mean,accepted_services_std,accepted_services_iqr_lower,accepted_services_iqr_upper,accepted_bitrate_mean,accepted_bitrate_std,accepted_bitrate_iqr_lower,accepted_bitrate_iqr_upper,total_bitrate_mean,total_bitrate_std,total_bitrate_iqr_lower,total_bitrate_iqr_upper,utilisation_mean,utilisation_std,utilisation_iqr_lower,utilisation_iqr_upper,service_blocking_probability_mean,service_blocking_probability_std,service_blocking_probability_iqr_lower,service_blocking_probability_iqr_upper,bitrate_blocking_probability_mean,bitrate_blocking_probability_std,bitrate_blocking_probability_iqr_lower,bitrate_blocking_probability_iqr_upper,avg_path_length_mean,avg_path_length_std,avg_path_length_iqr_upper,avg_path_length_iqr_lower,avg_hops_mean,avg_hops_std,avg_hops_iqr_upper,avg_hops_iqr_lower,avg_successful_path_length_mean,avg_successful_path_length_std,avg_successful_path_length_iqr_upper,avg_successful_path_length_iqr_lower,avg_successful_hops_mean,avg_successful_hops_std,avg_successful_hops_iqr_upper,avg_successful_hops_iqr_lower,unique_paths_fraction,successful_unique_paths_mean,successful_unique_paths_std,successful_unique_paths_iqr_upper,successful_unique_paths_iqr_lower" > $OUTPUT_FILE

for HEUR in "${HEURISTICS[@]}"; do
    for TOPOLOGY in "${TOPOLOGIES[@]}"; do
        for K in "${K_VALUES[@]}"; do
          for weight in "--weight=weight" ""; do
            echo "Running experiment: HEUR=$HEUR, TOPOLOGY=$TOPOLOGY, LOAD=inf, K=$K weight=$weight"

            # Run the experiment
            OUTPUT=$($PYTHON_PATH $SCRIPT_PATH \
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
                --log_actions)

            echo "$OUTPUT"

            # Extract metrics using awk and store in CSV format
            echo "$OUTPUT" | awk -v heur="$HEUR" -v topo="$TOPOLOGY" -v l="1000000" -v k="$K" -v weight="$weight" '
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
                    metrics[37]="Average path length mean"
                    metrics[38]="Average path length std"
                    metrics[39]="Average path length IQR upper"
                    metrics[40]="Average path length IQR lower"
                    metrics[41]="Average number of hops mean"
                    metrics[42]="Average number of hops std"
                    metrics[43]="Average number of hops IQR upper"
                    metrics[44]="Average number of hops IQR lower"
                    metrics[45]="Average path length for successful actions mean"
                    metrics[46]="Average path length for successful actions std"
                    metrics[47]="Average path length for successful actions IQR upper"
                    metrics[48]="Average path length for successful actions IQR lower"
                    metrics[49]="Average number of hops for successful actions mean"
                    metrics[50]="Average number of hops for successful actions std"
                    metrics[51]="Average number of hops for successful actions IQR upper"
                    metrics[52]="Average number of hops for successful actions IQR lower"
                    metrics[53]="Fraction of paths that are unique to ordering"
                    metrics[54]="Fraction of successful actions that use unique paths mean"
                    metrics[55]="Fraction of successful actions that use unique paths std"
                    metrics[56]="Fraction of successful actions that use unique paths IQR upper"
                    metrics[57]="Fraction of successful actions that use unique paths IQR lower"
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
                /Fraction of paths that are unique to ordering:/ {
                    split($0, parts, ":")
                    value=parts[2]
                    gsub(/ /, "", value)
                    values["Fraction of paths that are unique to ordering"]=value
                }
                END {
                    printf "%s,%s,%s,%s", heur, topo, l, k
                    # Output values in the defined order
                    for (i=1; i<=57; i++) {
                        if (values[metrics[i]] != "") {
                            printf ",%s", values[metrics[i]]
                        } else {
                            printf ",NA"  # Handle missing values
                        }
                    }
                    printf "\n"
                }' >> $OUTPUT_FILE

            echo "Completed experiment: HEUR=$HEUR, TOPOLOGY=$TOPOLOGY, LOAD=$LOAD, K=$K weight=$weight"
          done
        done
    done
done