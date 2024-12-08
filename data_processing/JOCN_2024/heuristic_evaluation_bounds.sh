#!/bin/bash

PYTHON_PATH="/home/uceedoh/xlron_env/bin/python3.11"
SCRIPT_PATH="/home/uceedoh/git/XLRON/xlron/train/train.py"

# Create/overwrite output CSV file with headers
OUTPUT_FILE="experiment_results_eval_bounds.csv"
echo "NAME,HEUR,TOPOLOGY,LOAD,K,returns_mean,returns_std,returns_iqr_lower,returns_iqr_upper,lengths_mean,lengths_std,lengths_iqr_lower,lengths_iqr_upper,cum_returns_mean,cum_returns_std,cum_returns_iqr_lower,cum_returns_iqr_upper,accepted_services_mean,accepted_services_std,accepted_services_iqr_lower,accepted_services_iqr_upper,accepted_bitrate_mean,accepted_bitrate_std,accepted_bitrate_iqr_lower,accepted_bitrate_iqr_upper,total_bitrate_mean,total_bitrate_std,total_bitrate_iqr_lower,total_bitrate_iqr_upper,utilisation_mean,utilisation_std,utilisation_iqr_lower,utilisation_iqr_upper,service_blocking_probability_mean,service_blocking_probability_std,service_blocking_probability_iqr_lower,service_blocking_probability_iqr_upper,bitrate_blocking_probability_mean,bitrate_blocking_probability_std,bitrate_blocking_probability_iqr_lower,bitrate_blocking_probability_iqr_upper" > $OUTPUT_FILE

run_experiment() {
    local name=$1
    local topology=$2
    local traffic_load=$3
    local k=$4
    local heur=$5
    local additional_args=$6

    echo "Running $name: topology=$topology, load=$traffic_load, k=$k"

    # Run the experiment
    OUTPUT=$($PYTHON_PATH $SCRIPT_PATH \
        --load=$traffic_load \
        --k=$k \
        --topology_name=$topology \
        --max_requests=1e3 \
        --max_timesteps=1e3 \
        --continuous_operation \
        --ENV_WARMUP_STEPS=3000 \
        --TOTAL_TIMESTEPS 20000000 \
        --NUM_ENVS 2000 \
        --VISIBLE_DEVICES 0 \
        --EVAL_HEURISTIC \
        --path_heuristic $heur \
        --modulations_csv_filepath "./modulations/modulations_deeprmsa.csv" \
        $additional_args)

    # Extract metrics using awk and store in CSV format
    echo "$OUTPUT" | awk -v name="$name" -v heur="$heur" -v topo="$topology" -v l="$traffic_load" -v k="$k" '
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
            printf "%s,%s,%s,%s,%s,%s", name, heur, topo, l, k, weight
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

k=50

# Deep/Reward/GCN-RMSA Experiments
args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --truncate_holding_time"
for traffic_load in 160 180 200 220 240; do
    run_experiment "DeepRMSA" "nsfnet_deeprmsa_directed" "$traffic_load" "$k" "ksp_ff" "$args"
done
args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --truncate_holding_time"
for traffic_load in 425 450 475 500 525 550; do
    run_experiment "DeepRMSA" "cost239_deeprmsa_directed" "$traffic_load" "$k" "ksp_ff" "$args"
done
args="--env_type rmsa --link_resources 100 --mean_service_holding_time 20 --truncate_holding_time"
for traffic_load in 325 350 375 400 425 450; do
    run_experiment "GCN-RMSA" "usnet_gcnrnn_directed" "$traffic_load" "$k" "ksp_ff" "$args"
done

# MaskRSA Experiments
args="--env_type rmsa --link_resources 80 --max_bw 50 --guardband 0 --slot_size 12.5 --mean_service_holding_time 12"
for traffic_load in 90 100 110 120 130; do
    run_experiment "MaskRSA" "nsfnet_deeprmsa_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
for traffic_load in 100 125 150 175 185 200 210 220; do
    run_experiment "MaskRSA" "jpn48_undirected" "$traffic_load" "$k" "ff_ksp" "$args"
done

# PtrNet-RSA-40 Experiments
base_args="--env_type rsa --slot_size 1 --guardband 0 --mean_service_holding_time 10"
var_bw="1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,4"
# NSFNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 200 210 220 230 240 250; do
    run_experiment "PtrNet-RSA-40" "nsfnet_deeprmsa_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
# NSFNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 220 240 260 280 300; do
    run_experiment "PtrNet-RSA-80" "nsfnet_deeprmsa_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
# COST239 PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 420 430 440 450 460 470; do
    run_experiment "PtrNet-RSA-40" "cost239_ptrnet_real_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
# COST239 PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 460 480 500 520 540 550; do
    run_experiment "PtrNet-RSA-80" "cost239_ptrnet_real_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
# USNET PtrNet-RSA-40
args="$base_args --link_resources 40 --values_bw 1"
for traffic_load in 220 230 240 250 260 270 275; do
    run_experiment "PtrNet-RSA-40" "usnet_ptrnet_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done
# USNET PtrNet-RSA-80
args="$base_args --link_resources 80 --values_bw $var_bw"
for traffic_load in 225 240 250 260 270 280 290 300; do
    run_experiment "PtrNet-RSA-80" "usnet_ptrnet_undirected" "$traffic_load" "$k" "ksp_ff" "$args"
done