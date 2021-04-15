[//]: # (Image References)

[image1]: dueling_ddqn_training.png "Training Progress"

# Project 1: Report

### Introduction

For this project, we trained a Deep Reinforcement Learning (DRL) agent to collect bananas in a square world.
The state space has 37 dimensions and the agent can choose one of 4 actions in each state.
We implement a Neural Network model and train it using different variants of Deep Q-learning algorithm.

### Network Architecture
In order to solve this environment, we designed a Neural Network Architecture with 2 hidden fully connected linear layers (each with 64 units), with the input containing 37 units (state space dimension) and the output containing 4 units (number of actions). We apply ReLU non-linearities after each hidden linear layer.
Additionally, we also implemented a Dueling Network architecture with two separate streams - one to model the value of each state and one to model the advantage of each (state, action) pair (where advantage is defined as the difference between the state value and the Q-value `A(s, a) = V(s) - Q(s, a)`). The Dueling Network architecture also has 2 hidden layers, one of which is shared between the two streams and the other is applied separately within each stream. The value and advantage streams are combined in the output using the following formula (proposed in [3]):
`output = value + advantage - mean_advantage`.

### DRL Algorithms
We experimented with 4 variants of the Deep Q-learning algorithm to find the best approach for training the agent in this environment.
1. Deep Q-learning (described in [1]) solves the environment by training the neural network using a combination of the online network (called `qnetwork_local` in dqn_agent.py) and the target network (`qnetwork_target` in the code). At each step of the training algorithm, the target network is applied to the next state to compute the target Q values in the loss function. At the same time, the online network is applied to the current state to get the expected Q values. MSE loss is used as the training criterion to optimize the network weights using back-propagation algorithm.
See `DQN` section in `Navigation.ipynb`.
2. Double DQN (described in [2]) is an improvement on Deep Q-learning algorithm, which aims to decouple the action selection and evaluation by applying the online network to select the action and the target network to compute the Q value at the next state.
See `Double DQN` section in `Navigation.ipynb`.
3. Dueling DQN (described in [3]) splits the sequential design of the DQN architecture by modeling state value and state-action advantage separately in the parallel streams. These two streams are joint at the output to produce Q-values for each (state, action) pair.
See `Dueling DQN` section in `Navigation.ipynb`.
4. Lastly we implement Dueling Double DQN by using the Dueling Network architecture in the Double Q-learning setting.
See `Dueling Double DQN` section in `Navigation.ipynb`.

### Training
In all experiments, agents were trained using samples from experience replay memory (as described in [1]), implemented using `ReplayBuffer` in `dqn_agent.py`. During each training episode, the agent follows the epsilon-greedy policy (lines 81-84 in `dqn_agent.py`) defined by the online network (`qnetwork_local`), adds the new experiences to the replay memory, and periodically (every 4 steps) calls the DQN agent's `learn()` method to update the online network weights using a batch of randomly sampled experiences. The target network weights are updated gradually using the `soft_update()` method, which takes a linear combination of the target and local networks' weights, putting most of the weight on the previous target weights to avoid noisy oscillations at every update.
The agent is trained for a maximum of 2000 episodes or until it reaches the desired score.
We did not perform any search over the hyper-parameter space (e.g. batch size, epsilon decay strategy, update frequency, update weight `tau`, etc) and instead focused on trying out different DRL algorithms with the fixed hyper-parameters.

### Results

All training results can be seen in `Navigation.ipynb` notebook.
All of the learning algorithms achieved about similar maximum score (of ~16.5) when trained for the full 2000 episodes, although Dueling (Double) DQN seemed to perform slightly worse than (Double) DQN.
Additionally, there was a slight improvement in the training convergence speed (to achieve the score of 15.0) going from DQN to Double/Dueling DQN. However, the best training performance (to achieve the score of 15.0) was achieved by applying the Dueling Network in the Double DQN setting. The plot below shows the training progress of the Dueling Double DQN algorithm.

![Training Progress][image1]

### Ideas for the Future

There are a number of ideas we could explore to further improve the training speed/max score performance of our DRL agent. Here are some of the more obvious next steps:
1. Hyper-parameter tuning
2. Prioritized experience replay as described in [4] (instead of uniform random sampling)
3. Different network architecture (try different number of layers/neurons)
4. Different target network update strategy (update the target network less often but with more weight on the local network weights)

### References

* [1] [Human-level control through deep reinforcement learning, Mnih](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
* [2] [Deep Reinforcement Learning with Double Q-learning (arxiv), van Hasselt)](https://arxiv.org/abs/1509.06461)
* [3] [Dueling Network Architectures for Deep Reinforcement Learning (arxiv), Wang](https://arxiv.org/abs/1511.06581)
* [4] [Prioritized Experience Replay (arxiv), Schaul](https://arxiv.org/abs/1511.05952)
