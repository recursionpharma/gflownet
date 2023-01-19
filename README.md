

[![Build-and-Test](https://github.com/recursionpharma/gflownet/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/recursionpharma/gflownet/actions/workflows/build-and-test.yaml)
[![Code Quality](https://github.com/recursionpharma/gflownet/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/recursionpharma/gflownet/actions/workflows/code-quality.yaml)
[![Python versions](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# gflownet

GFlowNet-related training and environment code on graphs.

**Primer**

[GFlowNet](https://yoshuabengio.org/2022/03/05/generative-flow-networks/), short for Generative Flow Network, is a novel generative modeling framework, particularly suited for discrete, combinatorial objects. Here in particular it is implemented for graph generation.

The idea behind GFN is to estimate flows in a (graph-theoretic) directed acyclic network*. The network represents all possible ways of constructing an object, and so knowing the flow gives us a policy which we can follow to sequentially construct objects. Such a sequence of partially constructed objects is a _trajectory_. *Perhaps confusingly, the _network_ in GFN refers to the state space, not a neural network architecture.

Here the objects we construct are themselves graphs (e.g. graphs of atoms), which are constructed node by node. To make policy predictions, we use a graph neural network. This GNN outputs per-node logits (e.g. add an atom to this atom, or add a bond between these two atoms), as well as per-graph logits (e.g. stop/"done constructing this object").

The GNN model can be trained on a mix of existing data (offline) and self-generated data (online), the latter being obtained by querying the model sequentially to obtain trajectories. For offline data, we can easily generate trajectories since we know the end state.

## Repo overview

- [algo](src/gflownet/algo), contains GFlowNet algorithms implementations (only [Trajectory Balance](https://arxiv.org/abs/2201.13259) for now), as well as some baselines. These implement how to sample trajectories from a model and compute the loss from trajectories.
- [data](src/gflownet/data), contains dataset definitions, data loading and data sampling utilities.
- [envs](src/gflownet/envs), contains environment classes; a graph-building environment base, and a molecular graph context class. The base environment is agnostic to what kind of graph is being made, and the context class specifies mappings from graphs to objects (e.g. molecules) and torch geometric Data.
- [examples](docs/examples), contains simple example implementations of GFlowNet.
- [models](src/gflownet/models), contains model definitions.
- [tasks](src/gflownet/tasks), contains training code.
    -  [qm9](src/gflownet/tasks/qm9/qm9.py), temperature-conditional molecule sampler based on QM9's HOMO-LUMO gap data as a reward.
    -  [seh_frag](src/gflownet/tasks/seh_frag.py), reproducing Bengio et al. 2021, fragment-based molecule design targeting the sEH protein
    -  [seh_frag_moo](src/gflownet/tasks/seh_frag_moo.py), same as the above, but with multi-objective optimization (incl. QED, SA, and molecule weight objectives).
- [utils](src/gflownet/utils), contains utilities (multiprocessing).
- [`train.py`](src/gflownet/train.py), defines a general harness for training GFlowNet models.

## Getting started

A good place to get started is with the [sEH fragment-based MOO task](src/gflownet/tasks/seh_frag_moo.py). The file `seh_frag_moo.py` is runnable as-is (although you may want to change the default configuration in `main()`).

## Installation

### PIP

This package is installable as a PIP package, but since it depends on some torch-geometric package wheels, the `--find-links` arguments must be specified as well:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-1.10.0+cu113.html
```
Or for CPU use:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-1.10.0+cpu.html
```

To install or [depend on](https://matiascodesal.com/blog/how-use-git-repository-pip-dependency/) a specific tag, for example here `v0.0.10`, use the following scheme:
```bash
pip install git+https://github.com/recursionpharma/gflownet.git@v0.0.10 --find-links ...
```

## Developing & Contributing

TODO: Write Contributing.md.
