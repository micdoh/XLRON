# imports go here
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm

# Define the file paths here
output_dir = pathlib.Path("../../../data/lp_sweep/")
output_filename = "lp_sweep_log_actions"

# Check if the data directory exists, create if not
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Check if the data directory is empty
skip_runs = False
if len(os.listdir(output_dir)) > 0:
    print("Data directory is not empty. Skipping bash commands.")
    skip_runs = True

# Run a series of bash commands to generate the data
if not skip_runs:
    # Use tqdm for progress tracking
    for lp in tqdm(np.arange(-8, -4, 0.1), desc="Running launch power sweep"):
        output_filename = f"lp_sweep_log_actions_{lp}.csv"
        environment_flags = (
            "--env_type rsa_gn_model "
            "--k 5 "
            "--incremental_loading "
            "--end_first_blocking "
            "--topology_name nsfnet_deeprmsa_directed "
            "--link_resources 115 "
            "--slot_size 100 "
            "--values_bw 400 "
            "--guardband 0 "
            "--coherent "
            "--interband_gap 100 "
            "--snr_margin=3.0 "
            "--modulations_csv_filepath=../../../modulations/modulations_gn_model.csv "
        )
        run_flags = (
            "--TOTAL_TIMESTEPS 20000 "
            "--NUM_ENVS 20 "
            "--EVAL_HEURISTIC "
            "--path_heuristic ksp_ff "
            "--log_actions "
            "--VISIBLE_DEVICES 2 "
            f"--TRAJ_DATA_OUTPUT_FILE {output_dir / output_filename} "
            f"--launch_power {lp} "
        )
        base_command = "python3 ../../../xlron/train/train.py "
        command = base_command + environment_flags + run_flags

        # Execute the command and handle potential errors
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command for launch power {lp}: {e}")
            continue


# Load the files and add the rows to a single dataframe
def load_and_combine_data(directory):
    # Get all CSV files in the directory
    csv_files = list(directory.glob("lp_sweep_log_actions_*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory}")

    # Initialize empty list to store dataframes
    dfs = []

    # Process each file
    for file_path in tqdm(csv_files, desc="Loading CSV files"):
        try:
            # Extract launch power from filename
            lp = float(file_path.stem.split('_')[-1])

            # Read CSV and add launch power column
            df = pd.read_csv(file_path)
            df['launch_power'] = lp

            dfs.append(df)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Combine all dataframes
    if not dfs:
        raise ValueError("No data was successfully loaded")

    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by launch power
    combined_df = combined_df.sort_values('launch_power')

    return combined_df


# Load and combine the data
try:
    final_df = load_and_combine_data(output_dir)

    # Save combined dataset
    output_path = output_dir / "combined_lp_sweep_results.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Combined data saved to {output_path}")

    # Print summary statistics
    print("\nSummary statistics:")
    print(f"Total number of records: {len(final_df)}")
    print("\nRecords per launch power:")
    print(final_df.groupby('launch_power').size())

except Exception as e:
    print(f"Error in data processing: {e}")

