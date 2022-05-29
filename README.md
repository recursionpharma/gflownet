

[![Build-and-Test](https://github.com/recursionpharma/gflownet/actions/workflows/build-and-test.yaml/badge.svg)](https://github.com/recursionpharma/gflownet/actions/workflows/build-and-test.yaml)
[![Code Quality](https://github.com/recursionpharma/gflownet/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/recursionpharma/gflownet/actions/workflows/code-quality.yaml)
[![Python versions](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# gflownet

GFlowNet-related training and environment code on graphs.

**Primer**

[GFlowNet](https://yoshuabengio.org/2022/03/05/generative-flow-networks/), short for Generative Flow Network, is a novel generative modeling framework, particularly suited for discrete, combinatorial objects. Here in particular it is implemented for graph generation.

The idea behind GFN is to estimate flows in a (graph-theoretic) directed acyclic network*. The network represents all possible ways of constructing an object, and so knowing the flow gives us a policy which we can follow to sequentially construct objects. Such a sequence of partially constructed objects is a _trajectory_. *Perhaps confusingly, network here refers to the state space, not a neural network architecture.

Here the objects we construct are themselves graphs (e.g. graphs of atoms), which are constructed node by node. To make policy predictions, we use a graph neural network. This GNN outputs per-node logits (e.g. add an atom to this atom, or add a bond between these two atoms), as well as per-graph logits (e.g. stop/"done constructing this object").

The GNN model can be trained on a mix of existing data (offline) and self-generated data (online), the latter being obtained by querying the model sequentially to obtain trajectories. For offline data, we can easily generate trajectories since we know the end state.

## Repo overview

- [algo](https://github.com/recursionpharma/gflownet/tree/trunk/gflownet/algo), contains GFlowNet algorithms implementations (only [Trajectory Balance](https://arxiv.org/abs/2201.13259) for now). These implement how to sample trajectories from a model and compute the loss from trajectories.
- [data](https://github.com/recursionpharma/gflownet/tree/trunk/gflownet/data), contains dataset definitions, data loading and data sampling utilities.
- [envs](https://github.com/recursionpharma/gflownet/tree/trunk/gflownet/envs), contains environment classes; a graph-building environment base, and a molecular graph context class. The base environment is agnostic to what kind of graph is being made, and the context class specifies mappings from graphs to objects (e.g. molecules) and torch geometric Data.
- [examples](https://github.com/recursionpharma/gflownet/tree/trunk/gflownet/examples), contains simple example implementations of GFlowNet.
- [models](https://github.com/recursionpharma/gflownet/tree/trunk/gflownet/models), contains model definitions.
- [tasks](https://github.com/recursionpharma/gflownet/tree/trunk/gflownet/tasks), contains training code.
    -  [qm9](https://github.com/recursionpharma/gflownet/blob/trunk/gflownet/tasks/qm9/qm9.py), temperature-conditional molecule sampler based on QM9's HOMO-LUMO gap data as a reward.
- [utils](gflownet/utils), contains utilities (multiprocessing).
- [`train.py`](gflownet/train.py), general GFlowNet training setup.
## Installation

### PIP

This package is installable via [Nexus](nexus.rxrx.io). You should configure your PIP using the instructions listed at
[roadie: PIP Configuration](https://github.com/recursionpharma/roadie#pip-configuration). Then, execute the following:

```bash
pip install gflownet
```

Alternatively, `pip install -e .`.


## Developing

Please refer to [Developing Python at Recursion](https://github.com/recursionpharma/roadie/blob/trunk/Developing.md)
for all tasks related to development of this codebase.
