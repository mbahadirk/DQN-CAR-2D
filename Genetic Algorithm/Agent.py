import torch
import random
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=0.05, epsilon_min=0.01, epsilon_decay=0.9995,
                 learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate

        self.model = self._build_model()

    def _build_model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, self.action_size),
        )

    def act(self, state):
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()
