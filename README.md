# gflownet

GFlowNet related training and environment code

## Installation

### PIP

This package is installable via [Nexus](nexus.rxrx.io). You should configure your PIP using the instructions listed at
[roadie: PIP Configuration](https://github.com/recursionpharma/roadie#pip-configuration). Then, execute the following:

```bash
pip install gflownet
```

### Conda

You will need [anaconda cloud setup](https://github.com/recursionpharma/drug-discovery/wiki/Anaconda-Setup#setup-anaconda-cloud-locally)
in order to access the Recursion conda packages. Once that is setup you can install this library like so:

```
conda config --add channels conda-forge
conda config --add channels defaults
conda config --add channels recursion
conda install gflownet
```

## Developing

Please refer to [Developing Python at Recursion](https://github.com/recursionpharma/roadie/blob/trunk/Developing.md)
for all tasks related to development of this codebase.
