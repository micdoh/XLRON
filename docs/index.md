# Welcome to XLRON

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/micdoh/XLRON/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/micdoh/XLRON/graph/badge.svg?token=UW9CCLRAFJ)](https://codecov.io/gh/micdoh/XLRON)

<p align="center">
  <video autoplay loop muted playsinline preload="metadata" width="640" style="max-width: 100%;" aria-label="Render of resource allocation decisions taken by a Graph Transformer agent trained with RL">
    <source src="images/demos/deeprmsa_transformer.webm" type="video/webm">
    <source src="images/demos/deeprmsa_transformer.mp4" type="video/mp4">
    Your browser does not support HTML5 video.
  </video>
  <br>
  <em>Live render of the per-link spectrum allocations made by a <a href="features/transformer/">Graph Transformer agent</a> trained with RL on DeepRMSA-NSFNET.</em>
</p>

___

**XLRON** ("ex-el-er-on") is a JAX-based simulation framework for resource allocation in optical networks. It combines a fast simulation engine that runs entirely on accelerator hardware, an integrated PPO trainer, classical heuristics, capacity bound estimators, an end-to-end differentiable physical layer, and a browser GUI — all in one library.

If you are deciding whether XLRON is the right tool for your work, the short version is:

- It is the **most comprehensive** of the open-source optical-network simulation libraries — see [Comparisons & Speed](features/speed.md).
- It is **fast**: 222–1,494× faster end-to-end RL training than other libraries; up to $6 \times 10^6$ steps/s on a single A100. See [Comparisons & Speed](features/speed.md).
- It includes an accurate **ISRS GN physical layer model with distributed Raman amplification** that agrees with state-of-the-art C+L-band experiments to within 0.5 dB. See [Physical Layer](features/physical_layer.md).
- It is the **first fully differentiable optical-network simulator**: gradients flow through the entire pipeline, enabling gradient-based pump optimization and direct RSA optimization. See [Differentiable Simulation](features/differentiable.md).
- It ships **all 119 real-world topologies from [TopologyBench](https://github.com/TopologyBench/Real-Topologies)** out of the box, including very large ones like USA100 and TataInd. See [Topologies](features/topologies.md).
- It is the framework behind the **first Graph Transformer trained with RL** to consistently match or beat the strongest heuristics for dynamic RMSA. See [Graph Transformer for RMSA](features/transformer.md).
- It exposes everything through both a CLI and a browser **GUI** designed to lower the barrier to entry. See [GUI](features/gui.md).
- It is **fully tested and comprehensively documented** — a unit-test suite covers the environments, training loop, heuristics, bounds, and GN model, and every flag, environment, and execution mode is documented on this site.
- The flat-flag **CLI is designed to be driven by LLM coding agents** as well as humans: every option is a single hyphenated flag, every parameter is documented, and the same interface is used by the GUI under the hood. See [GUI](features/gui.md#cli-is-still-the-foundation).

<p align="center">
  <img src="images/papers/jocn_xlron/interfaces/gui.png" width="850" alt="XLRON GUI">
</p>

___

## Get started

- New here? Start with the [Installation](installation.md) and [Quick Start](quickstart.md) guides, then [Understanding XLRON](understanding_xlron.md) for the conceptual model.
- Looking to train an agent? See [Training with PPO](training.md).
- Just want to evaluate heuristics? See [Heuristic Evaluation](heuristic_evaluation.md).
- Doing physical-layer-aware simulation? See [GN Model Physical Layer](gn_model.md) and the [Differentiable DRA Pipeline](differentiable_dra.md).
- Estimating capacity bounds? See [Capacity Bound Estimation](capacity_bounds.md).

___

## Papers and reproduction guides

If you are here because you read one of our papers and want to reproduce a figure, the per-paper reproduction guides give the exact commands and scripts:

- **XLRON: Accelerated Reinforcement Learning Environments for Optical Networks** — OFC 2024
- **Reinforcement Learning with Graph Attention for Routing and Wavelength Assignment with Lightpath Reuse** — ONDM 2025 ([arXiv:2502.14741](https://arxiv.org/abs/2502.14741)) — [Reproduce](reproduce_rwalr.md)
- **Reinforcement Learning for Dynamic Resource Allocation in Optical Networks: Hype or Hope?** — JOCN **17**(9), D1 (2025), DOI [10.1364/JOCN.559990](https://doi.org/10.1364/JOCN.559990), [arXiv:2406.01919](https://arxiv.org/abs/2406.01919) — [Reproduce](reproduce_jocn2024.md)
- **Comparison of Dynamic Elastic Optical Network Capacity Bound Estimation Methods** — *submitted to ECOC 2026* — [Reproduce](reproduce_ecoc2026.md)
- **XLRON: A Framework for Hardware-Accelerated and Differentiable Simulation of Optical Networks** — *in preparation* — [Reproduce](reproduce_jocn_xlron.md)
- **Graph Transformers and Stabilized Reinforcement Learning for Large-Scale Dynamic Routing, Modulation and Spectrum Allocation in Elastic Optical Networks** — *in preparation* ([arXiv:2605.02075](https://arxiv.org/abs/2605.02075)) — [Reproduce](reproduce_jocn_transformer.md)

A consolidated list of papers with BibTeX entries lives on the [Papers](papers.md) page.

___

## Related projects

- **[TopologyBench](https://github.com/TopologyBench/Real-Topologies)** — 119 real-world optical network topologies, all bundled with XLRON.
- **[Gymnax](https://github.com/RobertTLange/gymnax)** — gym-style API in JAX; XLRON's environment interface follows the Gymnax pattern.
- **[PureJaxRL](https://github.com/luchris429/purejaxrl)** — XLRON's PPO implementation derives from PureJaxRL.
- **[Stoix](https://github.com/EdanToledo/Stoix)** — multi-device sharding inspiration.

___

## Contact and acknowledgements

Maintained by **Michael Doherty** (michael.doherty.21@ucl.ac.uk) at UCL. Supported by EPSRC grant EP/S022139/1 (CDT in Connected Electronic and Photonic Systems) and EPSRC Programme Grant TRANSNET (EP/R035342/1).
