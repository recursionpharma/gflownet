# Implementation notes

This repo is centered around training GFlowNets that produce graphs. While we intend to specialize towards building molecules, we've tried to keep the implementation moderately agnostic to that fact, which makes it able to support other graph-generation environments.

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
