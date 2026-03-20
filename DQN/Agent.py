import torch
import random
from collections import deque
import numpy as np
import torch.nn.functional as F



class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=100000, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
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
        self.target_model = self._build_model()
        self.update_target_network()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.replay_buffer = self.memory  # alias for compatibility

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _build_model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_size),
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

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor(np.array([m[0] for m in minibatch]))
        actions = torch.LongTensor(np.array([m[1] for m in minibatch])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch]))
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).unsqueeze(1)

        current_q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
