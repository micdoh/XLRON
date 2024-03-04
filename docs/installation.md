# Installation

XLRON is designed to be downloaded from Github rather than imported as a modular library. The repository should be cloned, virtual environment set up, and scripts run from the command line. This allows easy access to the source code, so users can modify the environments, agents, and training loops to suit their needs. This is particularly useful for research and experimentation, where the ability to quickly prototype and test new ideas is crucial.

If you want XLRON environments to conform to the standard gym-style API (e.g. step returns `observation`, `reward`, `terminated`, `truncated`, `info`) for use with other RL libraries such as stable-baselines3, use the `GymnaxToGymWrapper` from the [gymnax](https://github.com/RobertTLange/gymnax/blob/main/gymnax/wrappers/gym.py) library.

## Clone the repo

```bash
git clone https://github.com/micdoh/XLRON.git
```

## Set up the virtual environment

### with pyproject.toml (recommended)

Poetry is a tool for dependency management and packaging in Python.
Install poetry if you don't have it already. Follow the instructions here: https://python-poetry.org/docs/#installation

Before setting up the environment, you need to decide if you wish to use the CPU-only version of JAX/jaxlib or the GPU version. `pyproject.toml` contains the CPU version by default. If you want to use the GPU (or TPU) version, edit `pyproject.toml` by (un)commenting the appropriate lines. See the JAX documentation for more information: https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu

To create the virtual environment and install the dependencies from `pyproject.toml`, run the following commands:

```bash
cd XLRON
poetry install
```

### with requirements.txt

If you don't want to use poetry, you can create a virtual environment and install the dependencies from `requirements.txt`:

```bash
cd XLRON
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Please note that `requirements.txt` is generated from `pyproject.toml` using the command `poetry export -f requirements.txt --output requirements.txt`. The requirements.txt contains the CPU version of JAX by default. If you want to use the GPU version, edit `pyproject.toml` and re-export, or install the GPU version of JAX manually.
