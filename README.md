# Biscuits

# Biscuits
A set of **experiments** inspired by the **paper** ["Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs"](https://arxiv.org/abs/2003.00152) by Jonathan Frankle, David J. Schwab, Ari S. Morcos

Presented as **final project** for the "**Deep Learning and Applied AI**" exam, taught in A.Y. 2021/22 of Computer Science MSc by **Prof Emanuele Rodolà**

<p align="center">
    <a href="https://github.com/dansolombrino/biscuits/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/dansolombrino/biscuits/Test%20Suite/main?label=main%20checks></a>
    <a href="https://dansolombrino.github.io/biscuits"><img alt="Docs" src=https://img.shields.io/github/deployments/dansolombrino/biscuits/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.2.2-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

 


## Installation

```bash
pip install git+ssh://git@github.com/dansolombrino/biscuits.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:dansolombrino/biscuits.git
cd biscuits
conda env create -f env.yaml
conda activate biscuits
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
