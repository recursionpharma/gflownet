# Getting Started 

For an introduction to the library, see [this colab notebook](https://colab.research.google.com/drive/1wANyo6Y-ceYEto9-p50riCsGRb_6U6eH).

For an introduction to using `wandb` to log experiments, see [this demo](../src/gflownet/hyperopt/wandb_demo).

For more general introductions to GFlowNets, check out the following:
- The 2023 [GFlowNet workshop](https://gflownet.org/) has several introductory talks and colab tutorials.
- This high-level [GFlowNet colab tutorial](https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR) (updated versions of which were written for the 2023 workshop, in particular for continuous GFNs).

A good place to get started immediately is with the [sEH fragment-based MOO task](src/gflownet/tasks/seh_frag_moo.py). The file `seh_frag_moo.py` is runnable as-is (although you may want to change the default configuration in `main()`).