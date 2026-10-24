# Papers

XLRON is a research project. The features documented on this site come from the following papers — each links to a step-by-step reproduction guide (where applicable) that walks through the exact commands used to generate every figure.

---

## Published

### XLRON: Accelerated Reinforcement Learning Environments for Optical Networks
*Doherty, M., Beghelli, A. — OFC 2024 — [Optica](https://doi.org/10.1364/OFC.2024.Th2A.21) · [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10185312/) · [Repository PDF](https://github.com/micdoh/XLRON/blob/main/papers/xlron_ofc_2024.pdf)*

The original announcement of XLRON, focused on the JAX implementation, GPU parallelism, and the DeepRMSA reproduction.

```bibtex
@INPROCEEDINGS{doherty_xlron_2024,
  author    = {Doherty, Michael and Beghelli, Alejandra},
  booktitle = {2024 Optical Fiber Communications Conference and Exhibition (OFC)},
  title     = {{XLRON}: Accelerated Reinforcement Learning Environments for Optical Networks},
  year      = {2024},
  pages     = {1-3},
  doi       = {10.1364/OFC.2024.Th2A.21},
  url       = {https://doi.org/10.1364/OFC.2024.Th2A.21}
}
```

---

### Reinforcement Learning with Graph Attention for Routing and Wavelength Assignment with Lightpath Reuse
*Doherty, M., Beghelli, A. — Optical Network Design and Modelling (ONDM) 2025 — DOI: [10.23919/ONDM65745.2025.11029354](https://doi.org/10.23919/ONDM65745.2025.11029354) · [arXiv:2502.14741](https://arxiv.org/abs/2502.14741)*

A Graph-Attention RL agent for the RWA-with-Lightpath-Reuse formulation: the agent must balance routing decisions against capacity packing on already-established lightpaths. Trained on NSFNET via the `rwa_lightpath_reuse` environment using XLRON's GNN actor-critic.

→ **[Reproduce all figures](reproduce_rwalr.md)**

```bibtex
@inproceedings{doherty_rwalr_2025,
  author    = {Doherty, Michael and Beghelli, Alejandra},
  title     = {Reinforcement Learning with Graph Attention for Routing and Wavelength Assignment with Lightpath Reuse},
  booktitle = {Optical Network Design and Modelling (ONDM)},
  year      = {2025},
  doi       = {10.23919/ONDM65745.2025.11029354},
  note      = {arXiv:2502.14741},
  url       = {https://doi.org/10.23919/ONDM65745.2025.11029354}
}
```

---

### Reinforcement Learning for Dynamic Resource Allocation in Optical Networks: Hype or Hope?
*Doherty, M., Beghelli, A. — JOCN **17**(9), D1 (2025) — DOI: [10.1364/JOCN.559990](https://doi.org/10.1364/JOCN.559990) · [arXiv:2406.01919](https://arxiv.org/abs/2406.01919)*

A systematic survey of ~100 RL-for-RMSA papers. Reproduces five highly-cited published RL solutions in matched simulation settings and shows that simple KSP-FF / FF-KSP heuristics with sufficient candidate paths and an appropriate path-sort criterion match or beat them. Also introduces a defragmentation-based capacity bound estimator (reconfigurable routing) and uses cut-set bounds as an upper-bound capacity reference.

→ **[Reproduce all figures](reproduce_jocn2024.md)**

```bibtex
@article{doherty_reinforcement_2025,
  author  = {Doherty, Michael and Beghelli, Alejandra},
  title   = {Reinforcement Learning for Dynamic Resource Allocation in Optical Networks: Hype or Hope?},
  journal = {Journal of Optical Communications and Networking},
  volume  = {17},
  number  = {9},
  pages   = {D1},
  year    = {2025},
  doi     = {10.1364/JOCN.559990},
  url     = {https://opg.optica.org/abstract.cfm?URI=jocn-17-9-D1}
}
```

---

## Submitted / pending publication

### Comparison of Dynamic Elastic Optical Network Capacity Bound Estimation Methods
*Doherty, M.\*, Deng, B.\*, Beghelli, A., Toni, L., Bayvel, P. — Submitted to ECOC 2026. \*Equal contribution.*

A systematic comparison of three capacity bound estimation methods — cut-set (CS), resource-prioritized defragmentation (RPD), and KSP heuristics — across the 118 real-world topologies of TopologyBench. Median gap over KSP at 0.1 % blocking is 5.8–9.4 % for CS (top-100 / top-10 cuts) and 17.0 % for RPD, indicating that spectrum fragmentation is a larger source of lost capacity than routing inefficiency.

→ **[Reproduce all figures](reproduce_ecoc2026.md)**

```bibtex
@unpublished{doherty_deng_capacity_bounds_2026,
  author  = {Doherty, Michael and Deng, Bohua and Beghelli, Alejandra and Toni, Laura and Bayvel, Polina},
  title   = {Comparison of Dynamic Elastic Optical Network Capacity Bound Estimation Methods},
  note    = {Submitted to ECOC 2026},
  year    = {2026}
}
```

---

## Preprints

### XLRON: A Framework for Hardware-Accelerated and Differentiable Simulation of Optical Networks
*Doherty, M., Jarmolovičius, M., Deng, B., Killey, R., Bayvel, P., Beghelli, A., Toni, L. — Optica Open preprint (2026), version 1, published 25 August 2026 — [Optica Open](https://preprints.opticaopen.org/articles/preprint/XLRON_A_Framework_for_Hardware-Accelerated_and_Differentiable_Simulation_of_Optical_Networks/33317361?file=67772697) · [Repository PDF](https://github.com/micdoh/XLRON/blob/main/papers/xlron_journal_2026.pdf)*

Introduces XLRON's architecture and benchmarks it against other open-source simulation libraries. The preprint reports physical-layer agreement with the Gerard *et al.* 2025 C+L-band experiment within 0.6 dB for per-band generalized OSNR and 0.6% for throughput, and demonstrates routing and spectrum allocation optimized by gradient descent.

→ **[Reproduce all figures](reproduce_jocn_xlron.md)**

```bibtex
@misc{doherty_xlron_framework,
  author    = {Doherty, Michael and Jarmolovi\v{c}ius, Mindaugas and Deng, Bohua and Killey, Robert and Bayvel, Polina and Beghelli, Alejandra and Toni, Laura},
  title     = {{XLRON}: A Framework for Hardware-Accelerated and Differentiable Simulation of Optical Networks},
  publisher = {Optica Open},
  note      = {Preprint, version 1},
  year      = {2026},
  doi       = {10.1364/opticaopen.33317361.v1},
  url       = {https://doi.org/10.1364/opticaopen.33317361.v1}
}
```

---

### Graph Transformers and Stabilized Reinforcement Learning for Large-Scale Dynamic Routing, Modulation and Spectrum Allocation in Elastic Optical Networks
*Doherty, M., Beghelli, A., Toni, L. — In preparation, targeted at JOCN. Preprint: [arXiv:2605.02075](https://arxiv.org/abs/2605.02075).*

Presents the first transformer architecture trained from scratch with RL for dynamic RMSA. Combines off-policy invalid action masking, a valid-mass log-barrier loss, per-step damping, hard gating, Pre-LayerNorm, and Wavelet-Induced Rotary Encodings (WiRE) to inject graph structure. Beats the strongest heuristic on every standard benchmark (NSFNET, COST239, USNET, JPN48 — DeepRMSA / RewardRMSA / GCN-RMSA / MaskRSA / PtrNet-RSA settings), then scales to USA100 (100 nodes) and TataInd (143 nodes) — the largest dynamic RMSA instances ever attempted with RL.

→ **[Reproduce all figures](reproduce_jocn_transformer.md)** · **[Highlights](features/transformer.md)**

```bibtex
@unpublished{doherty_graph_transformer,
  author  = {Doherty, Michael and Beghelli, Alejandra and Toni, Laura},
  title   = {Graph Transformers and Stabilized Reinforcement Learning for Large-Scale Dynamic Routing, Modulation and Spectrum Allocation in Elastic Optical Networks},
  note    = {In preparation. arXiv:2605.02075},
  url     = {https://arxiv.org/abs/2605.02075},
  year    = {2026}
}
```

---

## Related projects we build on or use

- **[TopologyBench](https://github.com/TopologyBench/Real-Topologies)** — 119 real-world optical network topologies, all bundled with XLRON. See [Topologies](features/topologies.md).
- **[Gymnax](https://github.com/RobertTLange/gymnax)** — JAX gym-style API; XLRON's environment interface follows the same pattern.
- **[PureJaxRL](https://github.com/luchris429/purejaxrl)** — XLRON's PPO implementation derives from PureJaxRL.
- **[Stoix](https://github.com/EdanToledo/Stoix)** — multi-device sharding inspiration.
- **[GNPy](https://github.com/Telecominfraproject/oopt-gnpy)** — vendor-agnostic optical-network planning, complementary to XLRON for multi-vendor production planning.
- **[Optical-Networking-Gym](https://github.com/carlosnatalino/optical-networking-gym)** — Cython-based RMSA simulation library, benchmarked against XLRON in the *XLRON framework* paper above.

---

## Acknowledgements

This work was supported by the EPSRC grant **EP/S022139/1** (Centre for Doctoral Training in Connected Electronic and Photonic Systems) and EPSRC Programme Grant **TRANSNET (EP/R035342/1)**.
