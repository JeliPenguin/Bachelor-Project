from const import *
from DQN import DQNAgent
import numpy as np


class ScoutAgent(DQNAgent):
    def __init__(self, id, obs_dim, actionSpace, batchSize=128, gamma=0.99, epsStart=0.9, epsEnd=0.05, epsDecay=1000, tau=0.005, alpha=1e-4) -> None:
        super().__init__(id, obs_dim, actionSpace, batchSize,
                         gamma, epsStart, epsEnd, epsDecay, tau, alpha)
        self.symbol = "S"

    def choose_action(self, s, eps=False):
        return self.choose_random_action()


class GuideAgent(DQNAgent):
    def __init__(self, id, obs_dim, actionSpace, batchSize=128, gamma=0.99, epsStart=0.9, epsEnd=0.05, epsDecay=1000, tau=0.005, alpha=1e-4) -> None:
        super().__init__(id, obs_dim, actionSpace, batchSize,
                         gamma, epsStart, epsEnd, epsDecay, tau, alpha)
        self.symbol = "G"

    def choose_action(self, s):
        return STAY
