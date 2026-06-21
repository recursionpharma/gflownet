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

## Multiprocessing

We use the multiprocessing features of torch's `DataLoader` to parallelize data generation and featurization. This is done by setting the `num_workers` (via `cfg.num_workers`) parameter of the `DataLoader` to a value greater than 0. Because workers cannot (easily) use a CUDA handle, we have to resort to a number of tricks.

Because training models involves sampling them, the worker processes need to be able to call the models. This is done by passing a wrapped model (and possibly wrapped replay buffer) to the workers, using `gflownet.utils.multiprocessing_proxy`. These wrappers ensure that model calls are routed to the main worker process, where the model lives (e.g. in CUDA), and that the returned values are properly serialized and sent back to the worker process. These wrappers are also designed to be API-compatible with models, e.g. `model(input)` or `model.method(input)` will work as expected, regardless of whether `model` is a torch module or a wrapper. Note that it is only possible to call methods on these wrappers, direct attribute access is not supported.

Note that the workers do not use CUDA, therefore have to work entirely on CPU, but the code is designed to be somewhat agnostic to this fact. By using `get_worker_device`, code can be written without assuming too much; again, calls such as `model(input)` will work as expected.

On message serialization, naively sending batches of data and results (`Batch` and `GraphActionCategorical`) through multiprocessing queues is fairly inefficient. Torch tries to be smart and will use shared memory for tensors that are sent through queues, which unfortunately is very slow because creating these shared memory files is slow, and because `Data` `Batch`es tend to contain lots of small tensors, which is not a good fit for shared memory.

We implement two solutions to this problem (in order of preference):
- using `SharedPinnedBuffer`s, which are shared tensors of fixed size (`cfg.mp_buffer_size`), but initialized once and pinned. This is the fastest solution, but requires that the size of the largest possible batch/return value is known in advance. This should work for any message, but has only been tested with `Batch` and `GraphActionCategorical` messages.
- using `cfg.pickle_mp_messages`, which simply serializes messages with `pickle`. This prevents the creation of lots of shared memory files, but is slower than the `SharedPinnedBuffer` solution. This should work for any message that `pickle` can handle.
