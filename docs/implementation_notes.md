# Implementation notes

This repo is centered around training GFlowNets that produce graphs, although sequences are also supported. While we intend to specialize towards building molecules, we've tried to keep the implementation moderately agnostic to that fact, which makes it able to support other graph-generation environments.

## Environment, Context, Task, Trainers

We separate experiment concerns in four categories:
- The Environment is the graph abstraction that is common to all; think of it as the base definition of the MDP.
- The Context provides an interface between the agent and the environment, it 
    - maps graphs to torch_geometric `Data` 
  instances
    - maps GraphActions to action indices
    - produces action masks
    - communicates to the model what inputs it should expect
- The Task class is responsible for computing the reward of a state, and for sampling conditioning information 
- The Trainer class is responsible for instanciating everything, and running the training & testing loop

Typically one would setup a new experiment by creating a class that inherits from `GFNTask` and a class that inherits from `GFNTrainer`. To implement a new MDP, one would create a class that inherits from `GraphBuildingEnvContext`. 


## Graphs

This library is built around the idea of generating graphs. We use the `networkx` library to represent graphs, and we use the `torch_geometric` library to represent graphs as tensors for the models. There is a fair amount of code that is dedicated to converting between the two representations.

Some notes:
- graphs are (for now) assumed to be _undirected_. This is encoded for `torch_geometric` by duplicating the edges (contiguously) in both directions. Models still only produce one logit(-row) per edge, so the policy is still assumed to operate on undirected graphs.
- When converting from `GraphAction`s (nx) to so-called `aidx`s, the `aidx`s are encoding-bound, i.e. they point to specific rows and columns in the torch encoding.


### Graph policies & graph action categoricals

The code contains a specific categorical distribution type for graph actions, `GraphActionCategorical`. This class contains logic to sample from concatenated sets of logits accross a minibatch. 

Consider for example the `AddNode` and `SetEdgeAttr` actions, one applies to nodes and one to edges. An efficient way to produce logits for these actions would be to take the node/edge embeddings and project them (e.g. via an MLP) to a `(n_nodes, n_node_actions)` and `(n_edges, n_edge_actions)` tensor respectively. We thus obtain a list of tensors representing the logits of different actions, but logits are mixed between graphs in the minibatch, so one cannot simply apply a `softmax` operator on the tensor. 

The `GraphActionCategorical` class handles this and can be used to compute various other things, such as entropy, log probabilities, and so on; it can also be used to sample from the distribution.
