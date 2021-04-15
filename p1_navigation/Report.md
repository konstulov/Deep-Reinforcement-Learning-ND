[//]: # (Image References)

[image1]: dueling_ddqn_training.png "Training Progress"

# Project 1: Report

### Model

### Introduction

For this project, we trained an agent to collect bananas in a square world.
The state space has 37 dimensions and can choose one of 4 actions in each state.
We implement a Neural Network model and train it using different variants of Deep Q-learning algorithm.

### Network Architecture
In order to solve this environment, we used a Neural Network Architecture with 2 hidden fully connected layers (each with 64 units), with the input containing 37 units (state space size) and the output containing 4 units (number of actions). We apply ReLU non-linearities after each linear layer.
Additionally, we also implement a Dueling Network architecture with two separate streams - one to model the value of each state and one to model the advantage of each (state, action) pair. This architecture also has 2 hidden layers, one of which is shared between the two streams and one is separately applied within each stream. The value and advantage streams are combined in the output using the following formula:
output = value + advantage - mean_advantage.

### Learning algorithms
We experimented with 4 variations of the Deep Q-learning algorithm to find the best approach to training the agent.
1. Deep Q-learning (DQN section in the Jupyter notebook) solves the environment by training the neural network using a combination of the online network (called `qnetwork_local` in dqn_agent.py) and the target network (`qnetwork_target` in the code). At each step in the training algorithm, the target network is applied to the next state to compute the target Q values in the loss function. At the same time, the online network is applied to the current state to get the expected Q values. MSE loss is used as the training criterion

### Training
In all cases, agents are trained using experience replay, implemented using the `ReplayBuffer` in `dqn_agent.py`. During each training episode, the agent follows the epsilon-greedy policy (lines 81-84 in `dqn_agent.py`) defined by the online network (`qnetwork_local`), adds the new experiences to the replay memory, and periodically (every 4 steps) calls the agent's `learn()` method to update the online network weights using a batch of sampled experiences. The target network weights are updated gradually using a `soft_update` method, which applies a linear combination of the target and local networks, putting most of the weight on the previous target weights to avoid noisy oscillations at every update.

### Results

All of the training results can be seen in `Navigation.ipynb` notebook.

![Training Progress][image1]

### Idea for Future

There are a number of ideas we could explore to further improve the training/score performance of our DRL agent.
1. Parameter tuning
2. Prioritized experience replay
3. Different network architecture
4. Different target network update strategy (update the target network less often but with more weight on the local network weights)
