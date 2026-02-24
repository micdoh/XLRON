<img src="docs/images/xlron_logo_upscaled.png" width="500" class="center">

---

> ## 🚀 MAJOR UPDATE COMING MARCH 2026 🚀
>
> A big new release of XLRON is on the way! Here's what's coming:
>
> - **GUI** for interactive environment visualisation and configuration
> - **Fast ISRS GN model** with **distributed Raman amplification** support
> - **Updated benchmarks** across all environments
> - **Updated documentation**
> - **Agentic coding tool support files** for seamless integration with AI coding assistants
> - And more!
>
> **Don't miss it — watch the repo for updates!**
>
> To get notified, click the **Watch** button at the top-right of the [XLRON GitHub repo](https://github.com/micdoh/XLRON) and select your notification preferences (e.g. "Releases" for release-only notifications, or "All Activity" to stay fully in the loop). You can also **Star** ⭐ the repo to bookmark it and show your support.

---

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
## 🌐 Overview 🌐
XLRON ("ex-el-er-on") is an open-source project that provides a suite of gym-style environments for simulating resource allocation problems in optical networks and applying reinforcement learning (RL) techniques. Unlike similar libraries, it is built on the JAX machine learning framework, enabling accelerated training on GPU/TPU/XPU hardware. This gives orders of magnitude faster training than other optical network simulation libraries (e.g. [optical-rl-gym](https://github.com/carlosnatalino/optical-rl-gym), [DeepRMSA](https://github.com/xiaoliangchenUCD/DeepRMSA), [RSA-RL](https://github.com/Optical-Networks-Group/rsa-rl)) due to:
<!---
[SDONSim](https://github.com/SDNNetSim/SDON_simulator)
-->
- JIT compilation of the entire training loop
- Massive parallelism (1000s of parallel environments) on accelerator hardware
- Co-location of environment and agent on GPU avoids CPU-XPU data transfer bottleneck
- Bypasses the Python Global Interpreter Lock (GIL)
XLRON is a product of my PhD research, which focuses on a set of combinatorial optimisation problems related to resource allocation in optical networks. XLRON aims to overcome the computational bottleneck in applying RL to these problems. The project is in active development.
___
### 💸 Acknowledgements 💸
This work was supported by the Engineering and Physical Sciences Research Council (EPSRC) grant EP/S022139/1 - the Centre for Doctoral Training in Connected Electronic and Photonic Systems - and EPSRC Programme Grant TRANSNET (EP/R035342/1)
### License
Copyright (c) Michael Doherty 2023. 
This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
