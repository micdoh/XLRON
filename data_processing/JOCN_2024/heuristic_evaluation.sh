#!/bin/bash

PYTHON_PATH="/Users/michaeldoherty/Library/Caches/pypoetry/virtualenvs/xlron-QeH3eSKC-py3.11/bin/python"
SCRIPT_PATH="/home/uceedoh/git/XLRON/xlron/train/train.py"

# Create/overwrite output CSV file with headers
OUTPUT_FILE="experiment_results_eval.csv"
echo "HEUR,TOPOLOGY,LOAD,K,returns_mean,returns_std,returns_iqr_lower,returns_iqr_upper,lengths_mean,lengths_std,lengths_iqr_lower,lengths_iqr_upper,cum_returns_mean,cum_returns_std,cum_returns_iqr_lower,cum_returns_iqr_upper,accepted_services_mean,accepted_services_std,accepted_services_iqr_lower,accepted_services_iqr_upper,accepted_bitrate_mean,accepted_bitrate_std,accepted_bitrate_iqr_lower,accepted_bitrate_iqr_upper,total_bitrate_mean,total_bitrate_std,total_bitrate_iqr_lower,total_bitrate_iqr_upper,utilisation_mean,utilisation_std,utilisation_iqr_lower,utilisation_iqr_upper,service_blocking_probability_mean,service_blocking_probability_std,service_blocking_probability_iqr_lower,service_blocking_probability_iqr_upper,bitrate_blocking_probability_mean,bitrate_blocking_probability_std,bitrate_blocking_probability_iqr_lower,bitrate_blocking_probability_iqr_upper" > $OUTPUT_FILE

run_experiment() {
    local name=$1
    local topology=$2
    local traffic_load=$3
    local k=$4
    local additional_args=$5

    echo "Running $name: topology=$topology, load=$traffic_load, k=$k"

    # Run the experiment
    OUTPUT=$($PYTHON_PATH $SCRIPT_PATH \
        --load=$LOAD \
        --k=$K \
        --topology_name=$TOPOLOGY \
        --max_requests=1e3 \
        --max_timesteps=1e3 \
        --continuous_operation \
        --ENV_WARMUP_STEPS=3000 \
        --TOTAL_TIMESTEPS 100000 \
        --NUM_ENVS 10 \
        --EVAL_HEURISTIC \
        --path_heuristic ksp_ff \
        --modulations_csv_filepath "./modulations/modulations_deeprmsa.csv" \
        $additional_args)

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
}

for weight in "--weight=weight" ""; do

  for k in 5 $([ -z "$weight" ] && echo "" || echo "50"); do

    # DeepRMSA Experiments
    run_experiment "DeepRMSA" "nsfnet_deeprmsa_directed" "250" "$k" "--env_type rmsa --link_resources 100 --mean_service_holding_time 25 --truncate_holding_time $weight"
    run_experiment "DeepRMSA" "cost239_deeprmsa_directed" "600" "$k" "--env_type rmsa --link_resources 100 --mean_service_holding_time 30 --truncate_holding_time $weight"

    # Reward-RMSA
    args="--env_type rmsa --link_resources 100 --mean_service_holding_time 14 --truncate_holding_time $weight"
    for traffic_load in 168 182 196 210; do
        run_experiment "Reward-RMSA" "nsfnet_deeprmsa_directed" "$traffic_load" "$k" "$args"
    done

    # MaskRSA NSFNET
    args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12 $weight"
    for traffic_load in 80 90 100 110 120 130 140 150 160; do
        run_experiment "MaskRSA" "nsfnet_deeprmsa_undirected" "$traffic_load" "$k" "$args"
    done

    # MaskRSA JPN48
    for traffic_load in 120 130 140 150 160; do
        run_experiment "MaskRSA" "jpn48_undirected" "$traffic_load" "$k" "$args"
    done

    # GCN-RMSA NSFNET
    args="--env_type rmsa --link_resources 100 --mean_service_holding_time 14 $weight"
    for traffic_load in 154 168 182 196 210; do
        run_experiment "GCN-RMSA" "nsfnet_deeprmsa_directed" "$traffic_load" "$k" "$args"
    done

    # GCN-RMSA COST239
    args="--env_type rmsa --link_resources 100 --mean_service_holding_time 23 $weight"
    for traffic_load in 368 391 414 437 460; do
        run_experiment "GCN-RMSA" "cost239_deeprmsa_directed" "$traffic_load" "$k" "$args"
    done

    # GCN-RMSA USNET
    args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 $weight"
    for traffic_load in 320 340 360 380 400; do
        run_experiment "GCN-RMSA" "usnet_gcnrnn_directed" "$traffic_load" "$k" "$args"
    done

    # PtrNet-RSA-40 Experiments
    base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10 $weight"

    # NSFNET PtrNet-RSA-40
    args="$base_args --link_resources 40 --values_bw 1"
    for traffic_load in 180 190 200 210 220 230 240; do
        run_experiment "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" "$traffic_load" "$k" "$args"
    done

    # NSFNET PtrNet-RSA-80
    var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"
    args="$base_args --link_resources 80 --values_bw $var_bw"
    for traffic_load in 200 210 220 230 240; do
        run_experiment "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" "$traffic_load" "$k" "$args"
    done

    # COST239 PtrNet-RSA-40
    args="$base_args --link_resources 40 --values_bw 1"
    for traffic_load in 340 360 380 400 420; do
        run_experiment "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" "$traffic_load" "$k" "$args"
    done

    # COST239 PtrNet-RSA-80
    args="$base_args --link_resources 80 --values_bw $var_bw"
    for traffic_load in 420 440 460; do
        run_experiment "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" "$traffic_load" "$k" "$args"
    done

    # USNET PtrNet-RSA-40
    args="$base_args --link_resources 40 --values_bw 1"
    for traffic_load in 210 220 230 240 250 260 270 280; do
        run_experiment "PtrNet-RSA-40" "usnet_ptrnet_undirected" "$traffic_load" "$k" "$args"
    done

    # USNET PtrNet-RSA-80
    args="$base_args --link_resources 80 --values_bw $var_bw"
    for traffic_load in 260 270 280 290 300 310 320 330; do
        run_experiment "PtrNet-RSA-80" "usnet_ptrnet_undirected" "$traffic_load" "$k" "$args"
    done

  done

done