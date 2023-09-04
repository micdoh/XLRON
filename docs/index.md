# **Welcome to XLRON's documentation!**

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/micdoh/XLRON/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/micdoh/XLRON/graph/badge.svg?token=UW9CCLRAFJ)](https://codecov.io/gh/micdoh/XLRON)

___

## _Accelerated Learning and Resource Allocation for Optical Networks_


XLRON ("ex-el-er-on") is an open-source project that provides a suite of gym-style environments for simulating resource allocation problems in optical networks using reinforcement learning techniques. It is built on the JAX machine learning framework, enabling accelerated training on GPU and TPU hardware.

XLRON is a product of my PhD research, which is focused on the application of Reinforcement Learning (RL) to the same set of resource allocation and combinatorial optimisation problems in Optical Networks. The project is currently in the early stages of development.

### Key Features

- Gym-style environments for optical network resource allocation problems.
- Powered by JAX for accelerated training on GPU and TPU.
- Facilitates the development and discovery of optimal resource allocation policies.
- Implementations of heuristics (kSP-FF, etc.) for benchmarking and comparison.
- Ideal for research, experimentation, and innovation in optical network optimization.


---
## Related work

The gym-style environments follow the example set in [Gymnax](https://github.com/RobertTLange/gymnax)

The PPO implementation in this project derives from the excellent [PureJaxRL](https://github.com/luchris429/purejaxrl)

___

## Acknowledgements

Financial support from EPSRC Centre for Doctoral Training in Connected Electronic and Photonic Systems (CEPS CDT) and EPSRC Programme Grant TRANSNET (EP/R035342/1) is gratefully acknowledged.

___

## Contact

If you have any questions or comments, please feel free to contact me at michael.doherty.21@ucl.ac.uk