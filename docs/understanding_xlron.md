# Understanding XLRON

This page provides an overview of the conceptual underpinnings of XLRON and description of its more advanced features. It explains how it is different from other network simulators that are reliant on standard graph libraries such as networkx, and instead uses an array-based approach to represent the network state. This allows JIT-compilation using JAX, parallelisation on GPU, and resulting fast generation of state transitions and reduction in training times.

For a primer on how to begin training and evaluating agents, jump to section 5 of this document or see the [quick start guide](quickstart.md).


## 1. Functional programming and JAX

XLRON is built using the [JAX](https://jax.readthedocs.io/en/latest/) high-performance array computing framework. JAX is designed for use with functional programs - programs which are defined by applying and composing [pure functions](https://jax.readthedocs.io/en/latest/glossary.html#term-pure-function). This functional paradigm allows JAX programs to be JIT (Just In Time)-compiled to [XLA](https://openxla.org/xla) (Accelerated Linear Algebra) and run on GPU or TPU hardware.

**How does this affect XLRON?** While programming in JAX has many advantages, it also imposes constraints on how programs are written. This means that the code for XLRON environments is written in a different style to that of other network simulators, which are often written in an object-oriented style with standard control flow and rely on graph libraries such as networkx.

The chief constraints that XLRON obeys to be compatible with JAX are:

- **Pure functions**: Functions must be pure, meaning that they have no side effects and return the same output for the same input. This is necessary for JAX to be able to compile and run the function on a GPU or TPU.
- **Static array shapes**: Array shapes must be known at compile time and immutable. This is necessary for JAX to be able to compile and run the function on a GPU or TPU.
- **No Python control flow**: JAX does not support Python's built-in `for` loops, and instead requires the use of `jax.lax.scan` or `jax.lax.fori_loop` for iteration. Similarly, JAX does not support Python's built-in `if` statements, and instead requires the use of `jax.lax.cond` for branching.

## 2. Environment state and params

In order to satisfy the constraints of JAX, XLRON environments are implemented as a series of pure functions that take a state and a set of parameters as input, and return an updated state as output. (Other arguments may also be passed to the functions e.g. [pseudo random number generator keys](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html)). Both the state and the parameters are defined as custom dataclasses.

**The environment state** repres

**The environment parameters** 

Each of the main environments supported by XLRON (RSA, DeepRMSA, RWALightpathReuse, VONE) has a custom state and parameters. The state and parameters are defined as dataclasses, and are passed as arguments to the functions that define the environment's transitions.

## 3. Data initialisation



### 3.1 Key data structures

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

### Invalid Action Masking

[Invalid action masking](https://arxiv.org/pdf/2006.14171.pdf) is a technique used to prevent the agent from selecting invalid actions. This is particularly important in the context of optical network resource allocation problems, where the action space is large and many actions are invalid. Each XLRON environment provides a method `action_mask` to generate a mask of valid actions for a given state. This mask can be used to prevent the agent from selecting invalid actions.

Invalid action masking during training is activated by using the flag `--ACTION_MASKING` when running the 'train.py' script. 

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

