
[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/micdoh/ONDRLax/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/micdoh/XLRON/graph/badge.svg?token=UW9CCLRAFJ)](https://codecov.io/gh/micdoh/XLRON)


<img src="docs/images/xlron_background.png">


### _Accelerated Learning and Resource Allocation for Optical Networks_

See the documentation at https://micdoh.github.io/XLRON/
___

## 🌎 Overview 🌎 

XLRON ("ex-el-er-on") is an open-source project that provides a suite of gym-style environments for simulating resource allocation problems in optical networks and applying reinforcement learning techniques. It is built on the JAX machine learning framework, enabling accelerated training on GPU and TPU hardware.

XLRON is a product of my PhD research, which is focused on the application of Reinforcement Learning (RL) to a set of combinatorial optimisation problems related to resource allocation in optical networks. The project is currently in the early stages of development.

### Key Features

- Gym-style environments for optical network resource allocation problems.
- Powered by JAX for accelerated training on GPU and TPU.
- Facilitates the development and discovery of optimised resource allocation policies.
- Implementations of heuristics (kSP-FF, etc.) for benchmarking and comparison.
- Ideal for research, experimentation, and innovation in optical network optimization.

---

## 🏎️ Speed-up 🏎️ 
### compared to [Optical RL gym](https://github.com/carlosnatalino/optical-rl-gym)-style environments

Expect minimum 500x speed-up! 🚀

Figures are for training with XLRON with full invalid action masking vs. training using numpy-based environment with no invalid action masking.

GPU is Nvidia A100.
CPU is 10-core Apple M1 Pro.

Experiment names on y-axis follow the naming convention: topology name (NSFNET or CONUS) - number of FSU per link - JAX or numpy environment - device type - number of vectorised environments.

![fps_xlron.png](docs%2Fimages%2Ffps_xlron.png)
![spmf_xlron.png](docs%2Fimages%2Fspmf_xlron.png)
![compilation_xlron.png](docs%2Fimages%2Fcompilation_xlron.png)


___
### Acknowledgements
Financial support from EPSRC Centre for Doctoral Training in Connected Electronic and Photonic Systems (CEPS CDT) and EPSRC Programme Grant TRANSNET (EP/R035342/1) is gratefully acknowledged.


### License
Copyright (c) Michael Doherty 2023. 
This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
