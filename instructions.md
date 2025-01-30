# Instructions for getting trajectory data CSV and what to do with it


## Overview
This guide will walk you through setting up and running simulations with either a heuristic to select the actions or a reinforcement learning agent. The output of the simulation will be a CSV file containing detailed trajectory data for each traffic request (e.g. source and destination nodes, bandwidth, path, etc.). Using this, you can then visualize the sequence of states-actions that are taken.

## Environment Setup

### Prerequisites
- Python installed on your system
- Git installed on your system
- The downloaded repository

### Setting up the Virtual Environment

1. First, install Poetry (the dependency management tool):
```bash
pip install poetry
```

2. Navigate to the repository directory and install dependencies:
```bash
cd XLRON
poetry install
```

Alternative method using requirements.txt (not recommended):
```bash
pip install -r requirements.txt
```

## Running the Simulation

Make sure the run the simulations from the XLRON directory as your present working directory. This is so that relative filepaths to e.g. the modulations file (which defines the modulation formats) can be found.

N.B. THE OUTPUT OF THE COMMAND WILL BE A CSV FILE THAT IS SAVED TO THE FILEPATH INDICATED BY THE `--TRAJ_DATA_OUTPUT_FILE` ARGUMENT. THIS FILE WILL CONTAIN THE TRAJECTORY DATA OF THE SIMULATION.

### Basic Simulation with KSP-FF Heuristic
Here's the command to run a basic simulation using the K-Shortest Path First-Fit (KSP-FF) heuristic:

```bash
poetry run python xlron/train/train.py \
    --env_type rmsa \
    --load 10 \
    --mean_service_holding_time=1 \
    --continuous_operation \
    --k=3 \
    --topology_name=5node \
    --link_resources=10 \
    --slot_size=12.5 \
    --values_bw=50,100 \
    --max_timesteps 10000 \
    --TOTAL_TIMESTEPS 10000 \
    --EVAL_HEURISTIC \
    --path_heuristic=ksp_ff \
    --log_actions \
    --PLOTTING \
    --TRAJ_DATA_OUTPUT_FILE test_data.csv
```

### Detailed Parameter Explanation

#### Environment Configuration
- `--env_type rmsa`: Sets the environment type to RMSA (Routing, Modulation and Spectrum Assignment)
- `--load 10`: Traffic load intensity (Erlang) - higher values mean more traffic
- `--mean_service_holding_time=1`: Average duration of service requests (in time units)
- `--continuous_operation`: Keeps the environment running continuously without resetting between episodes
- `--k=3`: Number of shortest paths to consider for each source-destination pair
- `--topology_name=5node`: Uses the 5-node network topology defined in the JSON file
- `--link_resources=10`: Number of frequency slots available on each link
- `--slot_size=12.5`: Width of each frequency slot in GHz
- `--values_bw=50,100`: Possible bandwidth values for requests in Gbps (requests will randomly choose between 50 and 100 Gbps)

#### Simulation Length Parameters
- `--max_timesteps 10000`: Maximum number of steps in a single episode
- `--TOTAL_TIMESTEPS 10000`: Total number of timesteps to simulate

#### Algorithm Selection
- `--EVAL_HEURISTIC`: Enables heuristic evaluation mode instead of RL
- `--path_heuristic=ksp_ff`: Uses K-Shortest Path First-Fit heuristic
  - K-Shortest Path (KSP) finds the k shortest paths between source and destination
  - First-Fit (FF) assigns the first available continuous block of spectrum

#### Output Configuration
- `--log_actions`: Enables detailed logging of all actions taken
- `--PLOTTING`: Generates plots of the results
- `--TRAJ_DATA_OUTPUT_FILE test_data.csv`: Specifies where to save the trajectory data

### Training a Reinforcement Learning Agent
To train an RL agent instead of using the heuristic, remove the `--EVAL_HEURISTIC` flag and add RL-specific parameters. Here's an example:

```bash
poetry run python train.py \
    --env_type rmsa \
    --load 10 \
    --mean_service_holding_time=1 \
    --continuous_operation \
    --k=3 \
    --topology_name=5node \
    --link_resources=10 \
    --slot_size=12.5 \
    --values_bw=50,100 \
    --max_timesteps 10000 \
    --TOTAL_TIMESTEPS 1000000 \
    --NUM_LAYERS 3 \
    --NUM_UNITS 64 \
    --LR 1e-4 \
    --GAMMA 0.99 \
    --NUM_ENVS 16 \
    --ROLLOUT_LENGTH 100 \
    --UPDATE_EPOCHS 10 \
    --log_actions \
    --PLOTTING \
    --TRAJ_DATA_OUTPUT_FILE rl_data.csv
```

#### Additional RL Parameters Explained
- `--NUM_LAYERS 3`: Number of layers in the neural network
- `--NUM_UNITS 64`: Number of units per layer
- `--LR 1e-4`: Learning rate for the optimizer
- `--GAMMA 0.99`: Discount factor for future rewards
- `--NUM_ENVS 16`: Number of parallel environments for training
- `--ROLLOUT_LENGTH 100`: Length of each rollout for collecting experience
- `--UPDATE_EPOCHS 10`: Number of training epochs per update

## Understanding the Output Data
The simulation generates a CSV file with detailed information about each traffic request and its allocation:

- `request_source`: Source node of the traffic request
- `request_dest`: Destination node of the traffic request
- `request_data_rate`: Requested bandwidth in Gbps (will be either 50 or 100 based on our configuration)
- `arrival_time`: Time step when the request arrived
- `departure_time`: Time step when the request will depart (arrival_time + holding_time)
- `path_indices`: Index of the chosen path from the k-shortest paths (0 to k-1)
- `slot_indices`: Starting frequency slot index allocated to this request (0 to link_resources-1)
- `returns`: Reward received for this allocation
- `path_links`: Binary array indicating which links are used in the path (e.g., [1,0,1,0] means links 0 and 2 are used)
- `path_spectral_efficiency`: Spectral efficiency of the chosen modulation format (bits/s/Hz)
  - Higher values mean more efficient use of spectrum
  - Depends on the path length - shorter paths can use more efficient modulation formats
- `required_slots`: Number of frequency slots needed for this request
  - Calculated as: ⌈request_data_rate / (slot_size * path_spectral_efficiency)⌉
- `bitrate_blocking_probability`: Probability of blocking based on bit rate (running average)
- `service_blocking_probability`: Probability of blocking based on service count (running average)

## Visualizing the Results
[This section will be completed with specific visualization instructions...]
