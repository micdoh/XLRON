
<img src="docs/images/xlron_logo_upscaled.png" width="500" class="center">


[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/micdoh/ONDRLax/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/micdoh/XLRON/graph/badge.svg?token=UW9CCLRAFJ)](https://codecov.io/gh/micdoh/XLRON)



## Documentation: https://micdoh.github.io/XLRON/

---

### *_As presented at [Optical Fibre Communication Conference (OFC)](https://www.ofcconference.org/en-us/home/about/) 2024_* - see the paper [here](ofc_paper.pdf)

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

# Install with GUI
uv sync --extra gui

# If running on GPU
uv sync --group gpu

# If running on TPU
uv sync --group tpu

# Or with pip
pip install -e .
pip install -e ".[gui]"   # include GUI
```

---

## Quick Start

### GUI (recommended for new users)

XLRON includes a browser-based GUI for configuring and launching experiments without memorising CLI flags. Install with the `gui` extra, then run:

```bash
uv sync --extra gui
xlron
```

This opens a Streamlit app where you can select environment type, topology, traffic parameters, model architecture, PPO hyperparameters, and more — then launch runs directly from the browser. Output streams live in the right-hand pane.

**Remote server?** Use SSH port forwarding (`ssh -L <port>:localhost:<port> user@remote-host`) and open the URL in your local browser.

### Training an RL Agent (CLI)

Train a PPO agent on the RMSA problem with the NSFNET topology:

```bash
python -m xlron.train.train \
  --env_type=rmsa \
  --topology_name=nsfnet_deeprmsa_directed \
  --link_resources=100 \
  --k=50 \
  --load=250 \
  --continuous_operation \
  --ENV_WARMUP_STEPS=3000 \
  --TOTAL_TIMESTEPS=5000000 \
  --NUM_ENVS=64 \
  --LR=5e-4
```

See the full **[Training with PPO](https://micdoh.github.io/XLRON/training/)** guide for details on hyperparameters, model architectures (MLP, GNN, Transformer), algorithmic features (reward centering, prioritized sampling, VTrace), schedules, and more.

### Evaluating Heuristics

Evaluate classical heuristic algorithms without training:

```bash
python -m xlron.train.train \
  --env_type=rmsa \
  --topology_name=nsfnet_deeprmsa_directed \
  --link_resources=100 \
  --k=50 \
  --load=250 \
  --continuous_operation \
  --ENV_WARMUP_STEPS=3000 \
  --TOTAL_TIMESTEPS=20000000 \
  --NUM_ENVS=2000 \
  --EVAL_HEURISTIC \
  --path_heuristic=ksp_ff
```

See the full **[Heuristic Evaluation](https://micdoh.github.io/XLRON/heuristic_evaluation/)** guide for all available heuristics, traffic configuration, and examples.

### Capacity Bound Estimation

Estimate theoretical performance bounds using cut-set or reconfigurable routing methods:

```bash
# Cut-set bounds
python -m xlron.bounds.cutsets_bounds \
  --topology_name=nsfnet_deeprmsa_directed \
  --env_type=rmsa \
  --link_resources=100 --k=50 --load=250 \
  --continuous_operation --truncate_holding_time \
  --num_sim_requests=100000 --num_trials=10 \
  --sim_min_load=150 --sim_max_load=300 --sim_step_load=10 \
  --CUTSET_EXHAUSTIVE --CUTSET_TOP_K=256

# Reconfigurable routing bounds
python xlron/bounds/reconfigurable_routing_bounds.py \
  --topology_name=nsfnet_deeprmsa_directed \
  --env_type=rmsa \
  --link_resources=100 --k=50 --load=250 \
  --continuous_operation --truncate_holding_time \
  --path_heuristic=ksp_ff \
  --TOTAL_TIMESTEPS=13000 --NUM_ENVS=1 --COMPILE_RR_BOUNDS
```

See the full **[Capacity Bound Estimation](https://micdoh.github.io/XLRON/capacity_bounds/)** guide for details.

---

## Features

### Environment Types

| Environment | Description |
|-------------|-------------|
| `rsa` | Routing and Spectrum Assignment |
| `rmsa` | Routing, Modulation and Spectrum Assignment |
| `rwa` | Routing and Wavelength Assignment (convenience wrapper: unit bandwidth, slot size 1) |
| `deeprmsa` | DeepRMSA-compatible observation/action space |
| `rwa_lightpath_reuse` | RWA with lightpath reuse |
| `vone` | Virtual Optical Network Embedding |
| `rsa_gn_model` / `rmsa_gn_model` | With GN model physical layer impairments |
| `rsa_multiband` | Multi-band transmission |

### Network Topologies

Topologies are loaded from JSON files in `xlron/data/topologies/`. Each topology is available in directed and/or undirected variants. Built-in topologies include:

- **NSFNET** (`nsfnet_deeprmsa_directed`, `nsfnet_deeprmsa_undirected`, `nsfnet_nevin_undirected`)
- **COST239** (`cost239_deeprmsa_directed`, `cost239_ptrnet_real_undirected`, etc.)
- **USNET** (`usnet_gcnrnn_directed`, `usnet_ptrnet_undirected`, etc.)
- **JPN48** (`jpn48_directed`, `jpn48_undirected`)
- **German17** (`german17_directed`, `german17_undirected`)
- **CONUS** (`conus_directed`, `conus_undirected`)
- **5-node** (`5node_directed`, `5node_undirected`)
- **119 real-world topologies** from [TopologyBench](https://github.com/TopologyBench/Real-Topologies) (e.g. `coronet`, `geant`, `abilene`, `germany50`, `japan25`, ...)
- Custom topologies via `--topology_directory`

All 119 [TopologyBench](https://github.com/TopologyBench/Real-Topologies) real-world network topologies are included natively, pre-converted in both directed and undirected variants. Use them like any built-in topology:

```bash
python -m xlron.train.train --topology_name=coronet_directed ...
```

To list all available TopologyBench topologies or regenerate from source:

```bash
python xlron/data/topologies/topology_bench_to_xlron_conversion.py --list
python xlron/data/topologies/topology_bench_to_xlron_conversion.py --download  # re-download from GitHub
```

Optional topology node attributes for rendering:
- `longitude`
- `latitude`

When present in topology JSON nodes, XLRON render uses these to seed geographic node placement.

### Model Architectures

- **MLP** (default) -- simple multi-layer perceptron
- **GNN** (`--USE_GNN`) -- Jraph-based graph neural network with optional GATv2 attention
- **Transformer** (`--USE_TRANSFORMER`) -- transformer encoder with WIRE graph-aware positional encodings

### Physical Layer Modeling

The `rsa_gn_model` and `rmsa_gn_model` environments include a closed-form ISRS GN model for estimating physical layer impairments (SNR). This enables modulation format selection based on estimated path GSNR and launch power optimization:

```bash
python -m xlron.train.train \
  --env_type=rmsa_gn_model \
  --topology_name=nsfnet_deeprmsa_directed \
  --link_resources=100 \
  --snr_margin=0.5 \
  --launch_power_type=fixed \
  ...
```

### Differentiable Mode

XLRON supports differentiable operations for gradient-based optimization through the environment. This is useful for research into differentiable discrete optimization and end-to-end learning.

```bash
python -m xlron.train.train \
  --env_type=rmsa \
  --topology_name=nsfnet_deeprmsa_directed \
  --differentiable \
  --temperature=1.0 \
  ...
```

When `--differentiable` is enabled, discrete operations (comparisons, argmax, indexing, rounding) use differentiable approximations based on straight-through gradient estimators and temperature-controlled soft functions (sigmoid, softmax). When disabled (default), standard non-differentiable operations are used for maximum performance.

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
├── heuristics/            # Classical algorithms (KSP-FF, etc.)
├── bounds/                # Capacity bound estimation (cut-sets, reconfigurable routing)
├── models/                # Neural network architectures (MLP, GNN, Transformer)
├── gui/                   # Streamlit GUI (optional, install with `xlron[gui]`)
├── dtype_config.py        # Device-aware dtype configuration
└── parameter_flags.py     # Command-line flags
```

---

## Acknowledgements

This work was supported by the Engineering and Physical Sciences Research Council (EPSRC) grant EP/S022139/1 - the Centre for Doctoral Training in Connected Electronic and Photonic Systems - and EPSRC Programme Grant TRANSNET (EP/R035342/1)


## License

Copyright (c) Michael Doherty 2023. 
This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
