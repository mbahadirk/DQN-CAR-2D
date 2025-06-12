import torch
import random
from collections import deque
import numpy as np
import torch.nn.functional as F



class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=500, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.9995, learning_rate=0.01, target_update_freq=1000):
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
        self.update_target_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_step = 0
        self.target_update_freq = target_update_freq

        self.memory = deque(maxlen=buffer_size)
        self.model = self._build_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.replay_buffer = self.memory  # alias for compatibility

    def _build_model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.action_size),
        )

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

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
        self.train_step += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor([exp[0] for exp in minibatch])
        actions = torch.LongTensor([exp[1] for exp in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([exp[2] for exp in minibatch])
        next_states = torch.FloatTensor([exp[3] for exp in minibatch])
        dones = torch.FloatTensor([float(exp[4]) for exp in minibatch])

        # Q(s, a)
        q_values = self.model(states).gather(1, actions).squeeze()

        # max_a' Q_target(s', a')
        next_q_values = self.target_model(next_states).max(1)[0].detach()

        # target Q değerleri
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f"Loss: {loss.item()}")
