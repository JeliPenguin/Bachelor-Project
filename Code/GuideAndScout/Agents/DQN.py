from torch import nn, optim
import torch
from collections import deque, namedtuple
import numpy as np
import random
from typing import List
from const import device

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DeepNetwork(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DeepNetwork, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self._model(x)


class SpreadDeepNetwork(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(SpreadDeepNetwork, self).__init__()
        self._model = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self._model(x)


class ReplayMemory():

    def __init__(self, capacity):
        self._memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self._memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def clear(self):
        self._memory.clear()

    def __len__(self):
        return len(self._memory)


class DQNAgent():
    def __init__(self, id: int, n_observations: int, actionSpace: List[str], hyperParam) -> None:
        self._id = id
        self._symbol = str(id)
        self._n_observations = n_observations
        self._n_actions = len(actionSpace)
        self._batchSize = hyperParam["batchSize"]
        self._gamma = hyperParam["gamma"]
        self._epsStart = hyperParam["epsStart"]
        self._epsEnd = hyperParam["epsEnd"]
        self._epsDecay = hyperParam["epsDecay"]
        self._tau = hyperParam["tau"]
        self._lr = hyperParam["lr"]
        self._memory = ReplayMemory(50000)
        self._eps_done = 0
        self.setNetwork("Default")

    def setNetwork(self, type):
        if type == "Spread":
            print("Set network type to Spread for agent: ", self._id)
            self._policy_net = SpreadDeepNetwork(
                self._n_observations, self._n_actions).to(device)
            self._target_net = SpreadDeepNetwork(
                self._n_observations, self._n_actions).to(device)
        else:
            self._policy_net = DeepNetwork(
                self._n_observations, self._n_actions).to(device)
            self._target_net = DeepNetwork(
                self._n_observations, self._n_actions).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._optimizer = optim.Adam(
            self._policy_net.parameters(), lr=self._lr)

    def getSetting(self):
        return (self._policy_net, self._eps_done)

    def loadSetting(self, setting: tuple):
        self._policy_net, self._eps_done = setting

    def getID(self):
        return self._id

    def getSymbol(self):
        return self._symbol

    def memorize(self, stateTensor: torch.Tensor, actionTensor: torch.Tensor, sPrimeTensor: torch.Tensor, rewardTensor: torch.Tensor):
        self._memory.push(stateTensor, actionTensor,
                          sPrimeTensor, rewardTensor)

    def choose_greedy_action(self, s: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def choose_random_action(self) -> torch.Tensor:
        randAction = np.random.randint(0, self._n_actions)
        return torch.tensor([[randAction]], device=device)

    def choose_action(self, s: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def optimize(self):
        if len(self._memory) < self._batchSize:
            return
        transitions = self._memory.sample(self._batchSize)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self._policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self._batchSize, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self._target_net(
                non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self._gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 100)
        self._optimizer.step()
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self._target_net.state_dict()
        policy_net_state_dict = self._policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self._tau + target_net_state_dict[key]*(1-self._tau)
        self._target_net.load_state_dict(target_net_state_dict)
