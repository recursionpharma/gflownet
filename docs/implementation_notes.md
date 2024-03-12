# Implementation notes

This repo is centered around training GFlowNets that produce graphs, although sequences are also supported. While we intend to specialize towards building molecules, we've tried to keep the implementation moderately agnostic to that fact, which makes it able to support other graph-generation environments.

## Environment, Context, Task, Trainers

We separate experiment concerns in four categories:
- The Environment is the graph abstraction that is common to all; think of it as the base definition of the MDP.
- The Context provides an interface between the agent and the environment, it 
    - maps graphs to torch_geometric `Data` 
  instances
    - maps GraphActions to action indices
    - communicates to the model what inputs it should expect
- The Task class is responsible for computing the reward of a state, and for sampling conditioning information 
- The Trainer class is responsible for instanciating everything, and running the training & testing loop

Typically one would setup a new experiment by creating a class that inherits from `GFNTask` and a class that inherits from `GFNTrainer`. To implement a new MDP, one would create a class that inherits from `GraphBuildingEnvContext`. 


## Graphs

This library is built around the idea of generating graphs. We use the `networkx` library to represent graphs, and we use the `torch_geometric` library to represent graphs as tensors for the models. There is a fair amount of code that is dedicated to converting between the two representations.

Some notes:
- graphs are (for now) assumed to be _undirected_. This is encoded for `torch_geometric` by duplicating the edges (contiguously) in both directions. Models still only produce one logit(-row) per edge, so the policy is still assumed to operate on undirected graphs.
- When converting from `GraphAction`s (nx) to `ActionIndex`s (tuple of ints), the action indexes are encoding-bound, i.e. they point to specific rows and columns in the torch encoding.


### Graph policies & graph action categoricals

The code contains a specific categorical distribution type for graph actions, `GraphActionCategorical`. This class contains logic to sample from concatenated sets of logits accross a minibatch. 

Consider for example the `AddNode` and `SetEdgeAttr` actions, one applies to nodes and one to edges. An efficient way to produce logits for these actions would be to take the node/edge embeddings and project them (e.g. via an MLP) to a `(n_nodes, n_node_actions)` and `(n_edges, n_edge_actions)` tensor respectively. We thus obtain a list of tensors representing the logits of different actions, but logits are mixed between graphs in the minibatch, so one cannot simply apply a `softmax` operator on the tensor. 

The `GraphActionCategorical` class handles this and can be used to compute various other things, such as entropy, log probabilities, action masks and so on; it can also be used to sample from the distribution.

To expand, the logits are always 2d tensors, and there’s going to be one such tensor per “action type” that the agent is allowed to take.
Since graphs have variable number of nodes, and since each node has `n_node_actions` associated possible action/logits, then the `(n_nodes, n_node_actions)` tensor will vary from minibatch to minibatch.  
In addition, the nodes in said logit tensor belong to different graphs in the minibatch; this is indicated by a `batch` tensor of shape `(n_nodes,)` for nodes (for e.g. edges it would be of shape `(n_edges,)`).

Here’s an example: say we have 2 graphs in a minibatch, the first has 3 nodes, the second 2 nodes. The logits associated with AddNode  will be of shape `(5, n)` (assuming there are `n` types of nodes in the problem). Say `n=2`, and `logits[AddNode] = [[1,2],[3,4],[5,6],[7,8],[9,0]]`, and `batch=[0,0,0,1,1]`.  
Then to compute the policy, we have to compute a softmax appropriately, i.e. the softmax for the first graph would be `softmax([1,2,3,4,5,6])` and for the second `softmax([7,8,9,0])` . This is possible thanks to `batch` and is what `GraphActionCategorical` does behind the scenes.  
Now that would be for when we only have the `AddNode` action. With more than one action we also have to compute the log-softmax log-normalization factor over the logits of these other tensors, log-add them together and then substract it from all corresponding logits.

## Data sources

The data used for training GFlowNets can come from a variety of sources. `DataSource` implements these different use-cases as individual iterators that collectively assemble the training batches before passing it to the trainer. Some of these use-cases include:
- Generating new trajectories on-policy
- Sampling trajectories from passed policies from a replay buffer
- Sampling trajectories from a fixed, offline dataset 

`DataSource` also covers validation sets, including cases such as:
- Generating new trajectories (w.r.t a fixed dataset of conditioning goals)
- Evaluating the model's likelihood on trajectories from a fixed, offline dataset