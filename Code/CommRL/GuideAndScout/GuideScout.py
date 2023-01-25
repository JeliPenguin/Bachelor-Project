from const import *
from CommAgent import CommAgent
import torch
import numpy as np
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GUIDEID = 0

class ScoutAgent(CommAgent):
    def __init__(self, id, obs_dim, actionSpace) -> None:
        super().__init__(id, obs_dim, actionSpace)
        self.symbol = "S"

    def choose_greedy_action(self) -> torch.Tensor:
        s = self.messageReceived[GUIDEID]["state"]
        with torch.no_grad():
            return self.policy_net(s).max(1)[1].view(1, 1)

    def choose_action(self) -> torch.Tensor:
        p = np.random.random()
        epsThresh = self.epsEnd + \
            (self.epsStart - self.epsEnd) * \
            np.exp(-1. * self.steps_done / self.epsDecay)
        self.steps_done += 1
        if p > epsThresh:
            return self.choose_greedy_action()
        return self.choose_random_action()

    def memorize(self):
        """
            Unpacks message recieved from Guide and memorize the states
        """
        msgDict = self.messageReceived[GUIDEID]
        stateTensor = msgDict["state"]
        actionTensor = msgDict["action"]
        sPrimeTensor = msgDict["sPrime"]
        rewardTensor = msgDict["reward"]
        super().memorize(stateTensor, actionTensor, sPrimeTensor, rewardTensor)


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



