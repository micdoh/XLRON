# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['xlron',
 'xlron.environments',
 'xlron.heuristics',
 'xlron.models',
 'xlron.train']

package_data = \
{'': ['*']}

install_requires = \
['Cython>=0.29.32,<0.30.0',
 'absl-py>=1.4.0,<2.0.0',
 'black>=22.12.0,<23.0.0',
 'cffi>=1.15.1,<2.0.0',
 'chex>=0.1.82,<0.2.0',
 'distrax>=0.1.4,<0.2.0',
 'dm-haiku>=0.0.10,<0.0.11',
 'gymnax>=0.0.6,<0.0.7',
 'jax[cpu]>=0.4.11,<0.5.0',
 'joblib>=1.2.0,<2.0.0',
 'jraph==0.0.6.*',
 'jupyter>=1.0.0,<2.0.0',
 'mkdocs-material>=9.2.6,<10.0.0',
 'mkdocs>=1.5.2,<2.0.0',
 'mkdocstrings[python]>=0.22.0,<0.23.0',
 'ml-dtypes==0.2.0',
 'networkx>=2.8.7,<3.0.0',
 'numpy>=1.23.4,<2.0.0',
 'orbax>=0.1.9,<0.2.0',
 'pillow>=9.3.0,<10.0.0',
 'pip>=22.2,<23.0',
 'progress>=1.6,<2.0',
 'pymongo>=4.3.2,<5.0.0',
 'pytest-cov>=4.1.0,<5.0.0',
 'pytest>=7.2.0,<8.0.0',
 'pyyaml>=6.0,<7.0',
 'rlax>=0.1.6,<0.2.0',
 'ruff>=0.0.286,<0.0.287',
 'scikit-learn>=1.1.3,<2.0.0',
 'scipy>=1.9.3,<2.0.0',
 'seaborn>=0.12.2,<0.13.0',
 'tqdm>=4.64.1,<5.0.0',
 'wandb>=0.13.4,<0.14.0']

setup_kwargs = {
    'name': 'XLRON',
    'version': '0.1.0',
    'description': 'Accelerated Learning and Resource Allocation for Optical Networks',
    'long_description': '\n[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/micdoh/ONDRLax/LICENSE)\n[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)\n[![codecov](https://codecov.io/gh/micdoh/XLRON/graph/badge.svg?token=UW9CCLRAFJ)](https://codecov.io/gh/micdoh/XLRON)\n\n\n<img src="docs/images/xlron_background.png">\n\n\n### _Accelerated Learning and Resource Allocation for Optical Networks_\n\nSee the documentation at https://micdoh.github.io/XLRON/\n___\n\n### *_Accepted to [Optical Fibre Communication Conference (OFC)](https://www.ofcconference.org/en-us/home/about/) - San Diego, CA, 24-28 March 2024_*\n\n___\n\n## 🌎 Overview 🌎 \n\nXLRON ("ex-el-er-on") is an open-source project that provides a suite of gym-style environments for simulating resource allocation problems in optical networks and applying reinforcement learning techniques. It is built on the JAX machine learning framework, enabling accelerated training on GPU and TPU hardware.\n\nXLRON is a product of my PhD research, which is focused on the application of Reinforcement Learning (RL) to a set of combinatorial optimisation problems related to resource allocation in optical networks. The project is currently in the early stages of development.\n\n### Key Features\n\n- Gym-style environments for optical network resource allocation problems.\n- Powered by JAX for accelerated training on GPU and TPU.\n- Facilitates the development and discovery of optimised resource allocation policies.\n- Implementations of heuristics (kSP-FF, etc.) for benchmarking and comparison.\n- Ideal for research, experimentation, and innovation in optical network optimization.\n\n---\n\n## 🏎️ Speed-up 🏎️ \n### compared to [Optical RL gym](https://github.com/carlosnatalino/optical-rl-gym)-style environments\n\n#### tldr: Expect approximately 500x speed-up! 🚀\n\nXLRON is faster than CPU-based training because of the following factors:\n\n- End-to-end JAX implementation (both environment and RL algorithm) allows entire training loop to be compiled and optimised as a single program\n- GPU-compatiblity allows parallelisation to make maximum use of accelerator hardware (GPU or TPU)\n- Running entirely on GPU avoids CPU-GPU data transfer bottleneck and eliminates any overhead from Python interpreter\n\nFor the comparisons shown, the CPU is 10-core Apple M1 Pro and the GPU is Nvidia A100.\n\n### Case study 1\n\nTo fairly assess the speed-up offered by XLRON, we implement a "DeepRMSA" environment and agent (exactly like in the canonical [DeepRMSA paper](https://ieeexplore.ieee.org/document/8738827)) and compare with the equivalent example from [optical-rl-gym](https://github.com/carlosnatalino/optical-rl-gym/blob/main/examples/stable_baselines3/DeepRMSA.ipynb), which uses stables_baselines3 (SB3) for training.\n\nThe below figure shows the training curves for both implementations, with 250 or 2000 parallel envs shown for XLRON. Shaded areas indicate the standard deviation of values across environments (each with a unique random seed) for XLRON and across 3 random seeds for SB3. The left figure shows the training progression with episode count, the right figure shows training progression with time on a log scale.\n\n![ofc2023_comp_all.png](docs%2Fimages%2Fofc2023_comp_all.png)\n\nIncreasing the number of parallel environments decreases the time required to train on a given number of environment steps, but changes the training dynamics so hyperparameters should be tuned accordingly for different numbers of parallel environments.\n\n\n\n### Case study 2\n\nFor the virtual optical network embedding problem, XLRON is compared with the environments from an ECOC 2023 paper (publication pending). The below figure compares the time it takes to train on 1M environment steps for two different topologies (NSFNET or CONUS) and either 100 or 320 frequency slot units (FSU) per link.\n\nThere are 4 horizontal bars per experiment:\n\n- sb3 training with 1 vectorised environment on CPU\n- sb3 training with 10 vectorised environments on CPU\n- XLRON training with 1 vectorised environment on CPU\n- XLRON training with 2000 vectorised environments on GPU\n\nExperiment names on y-axis follow the naming convention: topology name (NSFNET or CONUS) - number of FSU per link - JAX or numpy environment - device type - number of vectorised environments.\n\n![ofc2023_vone_comparison.png](docs%2Fimages%2Fofc2023_vone_comparison.png)\n\n\n#### Compilation times\n\nSee below figure for compilatiion times of different environments. Compilation typically takes a few seconds, therefore adds very little overhead to the training process.\n![compilation_xlron.png](docs%2Fimages%2Fcompilation_xlron.png)\n\n\n___\n### Acknowledgements\nThis work was supported by the Engineering and Physical Sciences Research Council (EPSRC) grant EP/S022139/1 - the Centre for Doctoral Training in Connected Electronic and Photonic Systems - and EPSRC Programme Grant TRANSNET (EP/R035342/1)\n\n\n### License\nCopyright (c) Michael Doherty 2023. \nThis project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.\n',
    'author': 'Michael Doherty',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<3.12',
}


setup(**setup_kwargs)

