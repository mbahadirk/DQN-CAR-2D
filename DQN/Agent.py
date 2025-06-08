import torch
import random
from collections import deque
import numpy as np
import torch.nn.functional as F



class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=500, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = 0.99  # default gamma
        self.lr = learning_rate
        self.done = False

        self.memory = deque(maxlen=buffer_size)
        self.model = self._build_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.replay_buffer = self.memory  # alias for compatibility

    def _build_model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.action_size),
        )

    def act(self, state):
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print(reward)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q = self.model(torch.FloatTensor(next_state)).detach().max().item()
                target = reward + self.gamma * next_q

            current_q = self.model(torch.FloatTensor(state)).squeeze(0)[action]
            loss = F.mse_loss(current_q, torch.tensor(target, dtype=torch.float32))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
