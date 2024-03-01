# Quickstart

Like CleanRL and [PureJaxRL](https://github.com/luchris429/purejaxrl), XLRON is not a modular library and is not (currently) meant to be imported. Instead, the repository should be cloned, virtual environment set up, and scripts run from the command line. See the [installation guide](installation.md) for details.

By having full access to the source code, users can easily modify the environments, agents, and training loops to suit their needs. This is particularly useful for research and experimentation, where the ability to quickly prototype and test new ideas is crucial.


## Example commands

All of the commands shown here make use of flags to define the environment and training parameters. For a full list of flags and their descriptions, see the [Commandline Flags](flags_reference.md).

---
### Training runs

```bash
python train.py --env_type=rwa_lightpath_reuse --load=200 --k=5 --topology_name=nsfnet --link_resources=100 --max_requests=1e4 --max_timesteps=1e4 --mean_service_holding_time=10 --values_bw=100 --TOTAL_TIMESTEPS 10000000 --VISIBLE_DEVICES 1 --NUM_ENVS 100 --NUM_SEEDS 1 --ACTION_MASKING --incremental_loading --USE_GNN --gnn_latent 128 --message_passing_steps 3 --output_nodes_size 1 --output_globals_size 1 --gnn_mlp_layers 2 --WANDB --DOWNSAMPLE_FACTOR 100 --scale_factor 0.2 --LR 0.0003 --LR_SCHEDULE warmup_cosine --SCHEDULE_MULTIPLIER 100 --WARMUP_PEAK_MULTIPLIER 3 --WARMUP_END_FRACTION 0.1
```


---
### Set up wandb hyperparameter sweep

1. **Define the sweep configuration** in a file `config.yaml` e.g. [here](example_sweep_config.yml) (see [wandb documentation](https://docs.wandb.ai/guides/sweeps/configuration) for details)
2. Run the following command to **set up the sweep**. This will output a sweep ID, which you will need for the next step.

```bash
wandb sweep config.yaml
```

3. Run the following command to **start a sweep agent**. Multiple sweep agents can be run in parallel to speed up the search for optimal hyperparameters.

```bash
wandb agent <SWEEP_ID>
```

---
