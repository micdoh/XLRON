# Instructions to generate data and recreate plots from 2024 JOCN paper

### N.B. please activate the virtual environment as described in the README.md in order to complete these steps.


## Generate heuristic data for both by running this notebook:`XLRON/data_processing/k_path_optimisation.ipynb`

## Generate heuristic data tables and charts by running this notebook: `XLRON/data_processing/k_path_plots.ipynb`


## Case study 1. - DeepRMSA

## Train DeepRMSA model from original code
### Use the modified code available at this repository: `https://github.com/micdoh/DeepRMSA`
#### Code is modified to use tensorflow 2.0 and not truncate the traffic request holding time.

##  Train DeepRMSA XLRON 16 envs
```bash
python train.py --env_type=deeprmsa --continuous_operation --load=250 --k=5 --topology_name=nsfnet_deeprmsa --link_resources=100 --max_requests=1e3 --max_timesteps=1e3 --mean_service_holding_time=25 --ROLLOUT_LENGTH=100 --continuous_operation --NUM_LAYERS 5 --NUM_UNITS 128 --NUM_ENVS
 6 --VISIBLE_DEVICES 3 --TOTAL_TIMESTEPS 5000000 --DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/deeprmsa_train_xlron.csv --PLOTTING --ENV_WARMUP_STEPS 5000 --LR 5e-4 --WARMUP_PEAK_MULTIPLIER
2 --LR_SCHEDULE warmup_cosine --UPDATE_EPOCHS 2 --GAE_LAMBDA 0.9 --GAMMA 0.95
```


## Train DeepRMSA XLRON 2000 envs
```bash
-python train.py --env_type=deeprmsa --continuous_operation --load=250 --k=5 --topology_name=nsfnet_deeprmsa --link_resources=100 --max_requests=1e3 --max_timesteps=1e3 --mean_service_holding_time=25 --ROLLOUT_LENGTH=200 --continuous_operation --NUM_LAYERS 5 --NUM_UNITS 128 --NUM_ENVS 2000 --VISIBLE_DEVICES 0 --TOTAL_TIMESTEPS 500000000 --DATA_OUTPUT_FILE /home/uceedoh/git/XLRON/data/deeprmsa_train_xlron_2000_masking_retrain_8.csv --PLOTTING --ENV_WARMUP_STEPS 20000 --LR 5e-5 --LR_SCHEDULE linear --UPDATE_EPOCHS 1 --GAE_LAMBDA 0.95 --GAMMA 0.99 --NUM_MINIBATCHES 10 --ACTION_MASKING --LOAD_MODEL --SAVE_MODEL --MODEL_PATH /home/uceedoh/git/XLRON/models/JOCN_DEEPRMSA_MASKED_8
```

## Evaluate the trained model
### Run the notebook `XLRON/data_processing/deeprmsa_model_eval.ipynb`


## Plot charts
### Run the notebook `XLRON/data_processing/deeprmsa_train.ipynb`
### Run the notebook `XLRON/data_processing/k_path_plots.ipynb`



# Case study 2. - RWA with Lightpath Reuse

## Train RWA-LR XLRON 2000 envs
```bash
python train.py --env_type rwa_lightpath_reuse --incremental_loading --k 5 --topology_name nsfnet --link_resources 100 --max_requests 10000 --max_timesteps 10000 --values_bw 100 --TOTAL_TIMESTEPS 200000000 --UPDATE_EPOCHS 10 --NUM_ENVS 100 --NUM_SEEDS 1 --ACTION_MASKING --PROJECT RWA_LR_SWEEP --WANDB --VISIBLE_DEVICES 1 --DOWNSAMPLE_FACTOR 100 --scale_factor 0.2 --LR_SCHEDULE warmup_cosine --WARMUP_END_FRACTION 0.1 --WARMUP_STEPS_FRACTION 0.1 --USE_GNN --gnn_mlp_layers 2 --message_passing_steps 3 --output_nodes_size 1 --output_globals_size 1 --GAE_LAMBDA=0.9842338134444694 --GAMMA=0.9186343961191545 --LR=1.9432016603757272e-05 --NUM_STEPS=150 --WARMUP_PEAK_MULTIPLIER=2 --gnn_latent=128
```

## Evaluate the trained model
```bash
python train.py --env_type=rwa_lightpath_reuse --k=5 --topology_name=nsfnet --link_resources=100 --max_requests=1e4 --max_timesteps=1e4 --values_bw=100 --ROLLOUT_LENGTH 150 --TOTAL_TIMESTEPS 1000000 --NUM_ENVS 1 --ACTION_MASKING --incremental_loading --USE_GNN --gnn_latent 128 --message_passing_steps 3 --output_nodes_size 1 --output_globals_size 1 --gnn_mlp_layers 2 --EVAL_MODEL --MODEL_PATH /Users/michaeldoherty/git/XLRON/models/RWA_LR_JOCN_retrain --DATA_OUTPUT_FILE /Users/michaeldoherty/git/XLRON/data/JOCN_SI/rwalr_model_firstblock_eval.csv --end_first_blocking
```

## Plot charts
### Run the notebook `XLRON/data_processing/rwa_lr_training.ipynb`
### Run the notebook `XLRON/data_processing/jocn_benchmarks_rwalr.ipynb`
### Run the notebook `XLRON/data_processing/k_path_plots.ipynb`


