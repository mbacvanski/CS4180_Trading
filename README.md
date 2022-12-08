# What the hell

are these agents

* DQN1 is foolish and ain't work
* DQN2 uses a CNN on the entire observation space, with a target network and probably with a replay buffer. It sucks
  because it doesn't use one-hot vectors to encode past positions.
* DQN3 is like DQN2 but with a teeny tiny NN that uses precomputed features: a 200 day and 50 day moving average.
  Like DQN2 it also sucks because it doesn't use one-hot vectors to encode positions.
* DQN4 finally figures out that you need to one-hot the position history. Duh. This uses a large feedforward neural
  net and finally produces some resemblance of profit.
* DQN5 is like DQN4 but uses a small neural network and precomputed features: 200-day moving average, 50-day moving
  average, and current position, one-hotted.
* DQN6 is like DQN4 but uses a CNN for function approximation over the entire observation space.
