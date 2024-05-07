# Understanding XLRON

###  --- PAGE UNDER DEVELOPMENT --- 

This page provides an overview of the conceptual underpinnings of XLRON and description of its more advanced features. It explains how it is different from other network simulators that are reliant on standard graph libraries such as networkx, and instead uses an array-based approach to represent the network state. This allows JIT-compilation using JAX, parallelisation on GPU, and resulting fast generation of state transitions and reduction in training times.

For a primer on how to begin training and evaluating agents, jump to section 5 of this document or see the [quick start guide](quickstart.md).


## 1. Functional programming and JAX

XLRON is built using the [JAX](https://jax.readthedocs.io/en/latest/) high-performance array computing framework. JAX is designed for use with functional programs - programs which are defined by applying and composing [pure functions](https://jax.readthedocs.io/en/latest/glossary.html#term-pure-function). This functional paradigm allows JAX programs to be JIT (Just In Time)-compiled to [XLA](https://openxla.org/xla) (Accelerated Linear Algebra) and run on GPU or TPU hardware.

**How does this affect XLRON?** While programming in JAX has many advantages, it also imposes constraints on how to program. This means that the code for XLRON environments is in a different style to that of other network simulators, which are often object-oriented with standard control flow and rely on graph libraries such as networkx.

The chief constraints that XLRON obeys to be compatible with JAX are:

- **Pure functions**: Functions must be pure, meaning that they have no side effects and return the same output for the same input. This is necessary for JAX to be able to compile and run the function on a GPU or TPU.
- **Static array shapes**: Array shapes must be known at compile time and immutable. This is necessary for JAX to be able to compile and run the function on a GPU or TPU.
- **No Python control flow**: JAX does not support Python's built-in `for` loops, and instead requires the use of `jax.lax.scan` or `jax.lax.fori_loop` for iteration. Similarly, JAX does not support Python's built-in `if` statements, and instead requires the use of `jax.lax.cond` for branching.

## 2. Environment state and params

In order to satisfy the constraints of JAX, XLRON environments are implemented as a series of pure functions that take a state and a set of parameters as input, and return an updated state as output. (Other arguments may also be passed to the functions e.g. [pseudo random number generator keys](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html)). Both the state and the parameters are defined as custom dataclasses. Each of the main environments supported by XLRON (RSA, DeepRMSA, RWALightpathReuse, VONE) has a custom state and parameters.

**The environment state** represents the current state of our network including currently active traffic requests, occupancy of spectral slots on links, and other relevant information that can change over the course of an episode. The state is updated by the environment's `step()` transition function.

**The environment parameters** represent the parameters of the environment, such as the topology of the network, the capacity of links, the number of slots on each link, and other relevant information. The parameters are fixed and do not change during the course of an episode. Parameters are specified as static arguments to functions, to indicate their values are known at compile time.


## 3. Data initialisation

To satisfy the constraint of static array shapes, XLRON environments are initialised with a fixed number of slots on each link, and a fixed number of resources at each node. This is done by specifying the number of slots and resources as parameters to the environment. The number of slots and resources are then used to initialise arrays of zeros to represent the occupancy of slots on links and the remaining resources at nodes. These arrays are then passed as part of the environment state.

### Routing representation
In order to capture the topological information of the network in array form, the k-shortest paths between each node pair on the network is calculated and the constituent links of each path are encoded as a binary array for each row of `path_link_array`. 


### Key data structures

For `RSAEnv` (including the RMSA and RWA problems), the following data structures are used:

`path_link_array`: Contains indices of constituent links of paths
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



N.B. The return values of the `step()` and `reset()` methods differ for XLRON compared to other gym-style environments ((`observation`, `state`, `reward`, `done`, `info`) vs. (`observation`, `reward`, `terminated`, `truncated`, `info`)). To match the gym API exactly for use with other RL libraries such as stable-baselines3, use the `GymnaxToGymWrapper` from the [gymnax](https://github.com/RobertTLange/gymnax/blob/main/gymnax/wrappers/gym.py) library.

### 4.1 Implement action

### 4.2 Check

### 4.3 Reward / Undo

## 5. Training an agent

XLRON contains an implementation of [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) as the main RL algorithm. PPO is a policy gradient method that is computationally efficient and has been used to achieve state-of-the-art results in a number of domains. It is appropriate for use in environments with stochastic environment dynamics (such as optical network resource allocation problems), since it can retain stochasticity in its policy. The PPO implementation in XLRON is based on the excellent [PureJaxRL](https://github.com/luchris429/purejaxrl). Users can modify `ppo.py` if they want to experiment with different RL algorithms.

Training is done using the `train.py` script, which is a wrapper around the `train` function in `train.py`. The script takes a number of commandline arguments to specify the environment, the agent, the training hyperparameters, and other settings. The script then calls the `train` function with these arguments. See the [quick start guide](quickstart.md) for example commands.

The train script is also used for evaluation of both agents and heuristics using the flags `--EVAL_MODEL` and `--EVAL_HEURISTIC` respectively.

## Other topics

### Invalid Action Masking

[Invalid action masking](https://arxiv.org/pdf/2006.14171.pdf) is a technique used to prevent the agent from selecting invalid actions. This is particularly important in the context of optical network resource allocation problems, where the action space is large and many actions are invalid. Each XLRON environment provides a method `action_mask` to generate a mask of valid actions for a given state. This mask can be used to prevent the agent from selecting invalid actions.

Invalid action masking is activated by using the flag `--ACTION_MASKING` when running the 'train.py' script. 

### Slot aggregation

Slot aggregation is a technique used to reduce the action space in the context of optical network resource allocation problems. It is particularly useful for the RWA problem, where the action space is large and many actions are invalid. Each XLRON environment provides a method `aggregate_slots` to aggregate slots on links. This groups the available slots into blocks of size N. The agent then selects a block of slots as an action, and  first fit allocation is used to select the initial slot for the service within the selected block of slots.

Slot aggregation is activated by using the flag `--aggregate_slots=N` when running the 'train.py' script.

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

### Learning rate schedules
XLRON supports diverse learning rate schedules for the agent.
See the flags for learning rate schedules in the [commandline options](flags_reference.md) section.


### GNNs with Jraph

We use the [Jraph](https://github.com/google-deepmind/jraph/tree/master) library for graph neural networks in JAX to implement the policy and/or value networks of our agent, while retaining the advantages of JIT compilation and accelerator hardware. Watch this space for more updates on the implementation! 

