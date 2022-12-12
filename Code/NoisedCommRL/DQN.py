from torch import nn
import torch
from collections import deque,namedtuple
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
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return np.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class GridAgent():
    def __init__(self, id, obs_dim,actionSpace, learning_rate=0.1, discount=0.9, epsilon_decay=0.9) -> None:
        self.id = id
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.actionSpace = actionSpace
        self.actionSpace_n = len(actionSpace)
        self.q_table = np.random.uniform(
            low=-2, high=0, size=obs_dim+(self.actionSpace_n,))

    def policy(s):
        pass

    def maxQVal(self, s):
        return max(self.q_table[s])

    def choose_greedy_action(self, s) -> int:
        return np.argmax(self.q_table[s])

    def choose_eps_action(self, s) -> int:
        p = np.random.random()
        if p < self.epsilon:
            return np.random.randint(self.actionSpace_n)
        return self.choose_greedy_action(s)

    def choose_action(self,s,eps=False):
        if eps:
            return self.choose_eps_action(s)
        return self.choose_greedy_action(s)

    

class DQN():
    def __init__(self,batchSize=128,gamma=0.99,epsStart=0.9,epsEnd=0.05,epsDecay=1000,tau=0.005,alpha=1e-4) -> None:
        self.batchSize = batchSize
        self.gamma = gamma
        self.epsStart = epsStart
        self.epsEnd = epsEnd
        self.epsDecay = epsDecay
        self.tau = tau
        self.alpha = alpha


    #Convert to tensors
    def choose_greedy_action(self, s) -> int:
        return np.argmax(self.q_table[s])

    def choose_eps_action(self, s) -> int:
        p = np.random.random()
        if p < self.epsilon:
            return np.random.randint(self.actionSpace_n)
        return self.choose_greedy_action(s)

    def choose_random_action(self):
        return np.random.randint(0,self.actionSpace_n)
    