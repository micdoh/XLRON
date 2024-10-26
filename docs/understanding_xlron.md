# Understanding XLRON

###  --- PAGE UNDER DEVELOPMENT --- 

This page provides a overview of the conceptual underpinnings of XLRON and description of its more advanced features. It explains how it is different from other network simulators that are reliant on standard graph libraries such as networkx, and instead uses an array-based approach to represent the network state. This allows JIT-compilation using JAX, parallelisation on GPU, and resulting fast generation of state transitions and reduction in training times.

For a primer on how to begin training and evaluating agents, jump to section 5 of this document or see the [quick start guide](quickstart.md).


## 1. Functional programming

## 2. Environment state and params

## 3. Data initialisation

### 3.1 Key data structures:

For `RSAEnv` (including the RMSA and RWA problems), the following data structures are used:

`path_link_array`: Contains indices of consitutnet links of paths
`link_slot_array`: Represents occupancy of slots on links
`link_slot_departure_array`: Contains departure times of services occupying slots
`request_array`: 

For `RWALightpathReuseEnv`, in addition to the data structures used in RSAEnv, the following data structures are used:

`path_index_array`:  Contains indices of lightpaths in use on slots
`path_capacity_array`:  Contains remaining capacity of each lightpath
`link_capacity_array`:  Contains remaining capacity of lightpath on each link-slot


For `VONEEnv`, in addition to the data structures used in RSAEnv, the following data structures are used:

`node_capacity_array`:  Contains remaining capacity of each node
`node_resource_array`:  Contains remaining resources of each node
`node_departure_array`:  Contains departure times of each node
``



## 4. Environment transitions

### 4.1 Implement action

### 4.2 Check

### 4.3 Reward / Undo

## 5. Training an agent

## Other topics

### Masking

### Slot aggregation

### Weights & Biases (wandb) integration

XLRON features support for wandb experiment tracking and hyperparameter sweeps. The following commandline flags, when running the 'train.py' script, will enable wandb integration:

```bash
  
  --[no]WANDB: Use wandb
    (default: 'false')
  --EXPERIMENT_NAME: Name of experiment (equivalent to run name in wandb) 
  (auto-generated based on other flags if unspecified)
    (default: '')
  --PROJECT: Name of project (same as experiment name if unspecified)
    (default: '')
  --DOWNSAMPLE_FACTOR: Downsample factor to reduce data uploaded to wandb
    (default: '1')
    (an integer)
  --[no]SAVE_MODEL: Save model (will be saved to --MODEL_PATH locally and uploaded to wandb if --WANDB is True)
    (default: 'false')
```


### GNNs with Jraph

