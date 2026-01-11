
<img src="docs/images/xlron_logo_upscaled.png" width="500" class="center">


[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/micdoh/ONDRLax/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/micdoh/XLRON/graph/badge.svg?token=UW9CCLRAFJ)](https://codecov.io/gh/micdoh/XLRON)



## See the documentation at https://micdoh.github.io/XLRON/

---

### *_As presented at [Optical Fibre Communication Conference (OFC)](https://www.ofcconference.org/en-us/home/about/) 2024_* - see the paper [here](ofc_paper.pdf)

---

### To recreate plots from papers follow instructions in `/examples` directory


### To cite XLRON in your work, please use the following BibTeX entry:

```
@INPROCEEDINGS{10526884,
  author={Doherty, Michael and Beghelli, Alejandra},
  booktitle={2024 Optical Fiber Communications Conference and Exhibition (OFC)}, 
  title={XLRON: Accelerated Reinforcement Learning Environments for Optical Networks}, 
  year={2024},
  volume={},
  number={},
  pages={1-3},
  keywords={Training;Graphics processing units;Reinforcement learning;Optical fiber networks;Resource management},
  doi={}}
```
---

## Overview

XLRON ("ex-el-er-on") is an open-source project that provides a suite of gym-style environments for simulating resource allocation problems in optical networks and applying reinforcement learning (RL) techniques. Unlike similar libraries, it is built on the JAX machine learning framework, enabling accelerated training on GPU/TPU/XPU hardware. This gives orders of magnitude faster training than other optical network simulation libraries (e.g. [optical-rl-gym](https://github.com/carlosnatalino/optical-rl-gym), [DeepRMSA](https://github.com/xiaoliangchenUCD/DeepRMSA), [RSA-RL](https://github.com/Optical-Networks-Group/rsa-rl)) due to:

- JIT compilation of the entire training loop
- Massive parallelism (1000s of parallel environments) on accelerator hardware
- Co-location of environment and agent on GPU avoids CPU-XPU data transfer bottleneck
- Bypasses the Python Global Interpreter Lock (GIL)

XLRON is a product of my PhD research, which focuses on a set of combinatorial optimisation problems related to resource allocation in optical networks. XLRON aims to overcome the computational bottleneck in applying RL to these problems. The project is in active development.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/micdoh/XLRON.git
cd XLRON

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

---

## Quick Start

### Training an RL Agent

Train a PPO agent on the RSA (Routing and Spectrum Assignment) problem:

```bash
python -m xlron.train.train \
  --env_type=rsa \
  --topology_name=nsfnet_chen \
  --link_resources=100 \
  --k_paths=5 \
  --load=250 \
  --TOTAL_TIMESTEPS=1000000 \
  --NUM_ENVS=100 \
  --LR=0.0005
```

Train on DeepRMSA (with modulation format selection):

```bash
python -m xlron.train.train \
  --env_type=deeprmsa \
  --topology_name=nsfnet_chen \
  --link_resources=100 \
  --k_paths=5 \
  --load=250 \
  --TOTAL_TIMESTEPS=1000000 \
  --NUM_ENVS=100
```

### Evaluating Heuristics

Evaluate classical heuristic algorithms without training:

```bash
python -m xlron.train.train \
  --env_type=rsa \
  --topology_name=nsfnet_chen \
  --link_resources=100 \
  --max_requests=1000 \
  --EVAL_HEURISTIC \
  --path_heuristic=ksp_ff
```

Available heuristics:
- `ksp_ff` - K-Shortest Paths with First-Fit spectrum assignment
- `ksp_bf` - K-Shortest Paths with Best-Fit spectrum assignment
- `ff_ksp` - First-Fit spectrum selection with KSP
- `ksp_mu` - K-Shortest Paths with Most-Used spectrum assignment
- `ksp_lu` - K-Shortest Paths with Least-Used spectrum assignment
- `ksp_mf` - K-Shortest Paths with Most-Fragmented spectrum assignment
- `kca_ff` - K Congestion-Aware with First-Fit

---

## Features

### Environment Types

| Environment | Description |
|-------------|-------------|
| `rsa` | Basic Routing and Spectrum Assignment |
| `rmsa` | Routing, Modulation and Spectrum Assignment |
| `deeprmsa` | DeepRMSA with path statistics observation space |
| `rwa_lightpath_reuse` | Routing and Wavelength Assignment with lightpath reuse |
| `vone` | Virtual Optical Network Embedding |
| `rsa_gn_model` | RSA with GN model physical layer impairments |
| `rmsa_gn_model` | RMSA with GN model physical layer impairments |
| `rsa_multiband` | RSA with multi-band transmission |

### Differentiable Mode

XLRON supports differentiable operations for gradient-based optimization through the environment. This is useful for research into differentiable discrete optimization and end-to-end learning.

```bash
# Enable differentiable mode
python -m xlron.train.train \
  --env_type=deeprmsa \
  --topology_name=4node \
  --differentiable=True \
  --temperature=1.0 \
  ...
```

When `--differentiable=True`, discrete operations (comparisons, argmax, indexing, rounding) use differentiable approximations based on:
- Straight-through gradient estimators
- Temperature-controlled soft functions (sigmoid, softmax)

When `--differentiable=False` (default), standard non-differentiable operations are used for maximum performance.

### Network Topologies

Built-in topologies include:
- `4node` - Simple 4-node test network
- `nsfnet_chen` - 14-node NSFNET topology
- `conus` - 75-node continental US topology
- `dtag` - German Telekom topology
- `euro28` - 28-node European network
- Custom topologies via `--topology_directory`

### Physical Layer Modeling

For `rsa_gn_model` and `rmsa_gn_model` environments:

```bash
python -m xlron.train.train \
  --env_type=rmsa_gn_model \
  --topology_name=nsfnet_chen \
  --consider_modulation_format=True \
  --path_snr=True \
  --snr_margin=0.5 \
  ...
```

### Graph Neural Networks

XLRON supports GNN-based agents:

```bash
python -m xlron.train.train \
  --env_type=rsa \
  --USE_GNN=True \
  --gnn_latent=64 \
  --message_passing_steps=3 \
  ...
```

---

## Key Configuration Options

### Environment Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--env_type` | Environment type | `rsa` |
| `--topology_name` | Network topology | `nsfnet_chen` |
| `--link_resources` | Frequency slots per link | `100` |
| `--k_paths` | Number of candidate paths | `5` |
| `--load` | Traffic load (Erlangs) | `250` |
| `--max_requests` | Max requests per episode | `1e4` |
| `--mean_service_holding_time` | Mean holding time | `10` |
| `--slot_size` | Slot bandwidth (GHz) | `12.5` |
| `--guardband` | Guard band slots | `1` |

### Training Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--TOTAL_TIMESTEPS` | Total training timesteps | `1e7` |
| `--NUM_ENVS` | Parallel environments | `100` |
| `--NUM_LEARNERS` | Independent learners | `1` |
| `--LR` | Learning rate | `0.0005` |
| `--GAMMA` | Discount factor | `0.999` |
| `--CLIP_EPS` | PPO clip epsilon | `0.2` |
| `--ENT_COEF` | Entropy coefficient | `0.0` |
| `--VF_COEF` | Value function coefficient | `0.5` |
| `--UPDATE_EPOCHS` | PPO update epochs | `1` |

### Differentiable Mode

| Flag | Description | Default |
|------|-------------|---------|
| `--differentiable` | Enable differentiable operations | `False` |
| `--temperature` | Temperature for soft approximations | `1.0` |

---

## Project Structure

```
xlron/
├── environments/           # JAX-based optical network environments
│   ├── dataclasses.py     # Flax struct dataclasses for state/params
│   ├── env_funcs.py       # Core environment functions
│   ├── diff_utils.py      # Differentiable operation approximations
│   ├── make_env.py        # Environment factory
│   └── wrappers.py        # Gym-style wrappers
├── train/
│   ├── train.py           # Main training entry point
│   ├── train_utils.py     # Training utilities
│   └── ppo.py             # PPO implementation
├── heuristics/            # Classical algorithms (KSP, First-Fit, etc.)
├── dtype_config.py        # Device-aware dtype configuration
└── parameter_flags.py     # Command-line flags
```

---

## Acknowledgements

This work was supported by the Engineering and Physical Sciences Research Council (EPSRC) grant EP/S022139/1 - the Centre for Doctoral Training in Connected Electronic and Photonic Systems - and EPSRC Programme Grant TRANSNET (EP/R035342/1)


## License

Copyright (c) Michael Doherty 2023. 
This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
