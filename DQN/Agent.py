import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class DQNAgent:
    """
    The DQNAgent class implements a Deep Q-Network (DQN) agent for reinforcement learning. This class is responsible for managing the agent's interactions with the environment, storing experiences, and learning from them to improve its decision-making policy.

    Attributes:
    - state_size: The size of the state space, representing the number of features in the input state.
    - action_size: The number of possible actions the agent can take in the environment.
    - memory: A deque used to store experiences (state, action, reward, next_state, done) for training.
    - gamma: The discount factor used to weigh future rewards.
    - epsilon: The exploration rate, determining the probability of taking a random action instead of an optimal one.
    - epsilon_min: The minimum exploration rate.
    - epsilon_decay: The rate at which the exploration rate decays after each episode.
    - learning_rate: The learning rate for the optimizer.
    - model: A neural network model used to approximate the Q-values for each action given a state.
    - optimizer: An optimizer used to update the model's weights based on the loss.

    Methods:
    - __init__: Initializes the DQNAgent with the specified state and action sizes, and sets up the memory, model, and optimizer.
    - build_model: Constructs the neural network model for approximating Q-values.
    - remember: Stores an experience in the agent's memory.
    - act: Chooses an action based on the current state, either by exploring randomly or exploiting the learned policy.
    - replay: Samples a batch of experiences from memory to train the model, updating the Q-values using the Bellman equation.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.98  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.998
        self.learning_rate = 1e-6
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 15),
            nn.ReLU(),
            nn.Linear(15, 5),
            nn.ReLU(),
            nn.Linear(5, self.action_size)

        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size)  # Random action
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state).detach().clone()
            target_f[action] = target
            loss = nn.MSELoss()(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
