from torch import nn, optim
import torch
from collections import deque, namedtuple
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DeepNetwork(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DeepNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.model(x)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return np.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent():
    def __init__(self, id, obs_len, actionSpace, batchSize=128, gamma=0.99, epsStart=0.9, epsEnd=0.05, epsDecay=1000, tau=0.005, lr=1e-4) -> None:
        self.id = id
        self.obs_len = obs_len
        self.actionSpace_n = len(actionSpace)
        self.batchSize = batchSize
        self.gamma = gamma
        self.epsStart = epsStart
        self.epsEnd = epsEnd
        self.epsDecay = epsDecay
        self.tau = tau
        self.lr = lr
        self.policy_net = DeepNetwork(
            obs_len, self.actionSpace_n).to(device)
        self.target_net = DeepNetwork(
            obs_len, self.actionSpace_n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

    # Convert to tensors

    def choose_greedy_action(self, s) -> int:
        return np.argmax(self.q_table[s])

    def choose_eps_action(self, s) -> int:
        p = np.random.random()
        if p < self.epsilon:
            return np.random.randint(self.actionSpace_n)
        return self.choose_greedy_action(s)

    def choose_random_action(self):
        return np.random.randint(0, self.actionSpace_n)
