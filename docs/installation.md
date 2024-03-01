# Installation

XLRON is designed to be downloaded from Github and run from the command line. This is because the environments, agents, and training loops are not fully modular and are not (currently) meant to be imported. Instead, the repository should be cloned, virtual environment set up, and scripts run from the command line. This allows full access to the source code, so users can easily modify the environments, agents, and training loops to suit their needs. This is particularly useful for research and experimentation, where the ability to quickly prototype and test new ideas is crucial.

## Clone the repo

```bash
git clone https://github.com/micdoh/XLRON.git
```

## Set up the virtual environment

### with pyproject.toml (recommended)

Poetry is a tool for dependency management and packaging in Python.
Install poetry if you don't have it already. Follow the instructions here: https://python-poetry.org/docs/#installation

TO create the virtual environment and install the dependencies from `pyproject.toml`, run the following commands:

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
