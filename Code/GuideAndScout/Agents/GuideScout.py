from const import *
from Agents.CommAgent import CommAgent
import torch
import numpy as np
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GUIDEID = 0


class ScoutAgent(CommAgent):
    def __init__(self, id, obs_dim, actionSpace) -> None:
        super().__init__(id, obs_dim, actionSpace, epsDecay=10000)
        self.symbol = str(id)

    def choose_greedy_action(self) -> torch.Tensor:
        guideMsg = self.messageReceived[GUIDEID]
        stateTensor, _, _, _ = self.tensorize(guideMsg)
        with torch.no_grad():
            return self.policy_net(stateTensor).max(1)[1].view(1, 1)

    def choose_action(self) -> torch.Tensor:
        p = np.random.random()
        epsThresh = self.epsEnd + \
            (self.epsStart - self.epsEnd) * \
            np.exp(-1. * self.eps_done / self.epsDecay)
        # print(f"EpsThresh: {epsThresh} Eps done: {self.eps_done}")
        if p > epsThresh:
            return self.choose_greedy_action()
        return self.choose_random_action()

    def memorize(self):
        """
            Unpacks message recieved from Guide and memorize the states
        """
        guideMsg = self.messageReceived[GUIDEID]
        stateTensor, actionTensor, sPrimeTensor, rewardTensor = self.tensorize(
            guideMsg)
        super().memorize(stateTensor, actionTensor, sPrimeTensor, rewardTensor)

    def updateEps(self):
        self.eps_done += 1


class GuideAgent(CommAgent):
    def __init__(self, id, obs_dim, actionSpace) -> None:
        super().__init__(id, obs_dim, actionSpace)
        self.symbol = "G"

    def choose_action(self) -> torch.Tensor:
        """ Returns STAY as Guide can only stay at allocated position"""
        return torch.tensor([[STAY]], device=device)

    def choose_random_action(self) -> torch.Tensor:
        randAction = STAY
        return torch.tensor([[randAction]], device=device)
