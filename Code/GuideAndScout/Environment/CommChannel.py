from typing import Tuple
from const import *
import torch
import numpy as np


class CommChannel():
    def __init__(self, agents, noiseP, noised=False) -> None:
        self.noised = noised
        self.noiseP = noiseP
        self.agents = agents

    def setupChannel(self):
        for agent in self.agents:
            agent.setChannel(self)

    def addNoise(self, msg):
        noise = np.random.random(msg.shape) < self.noiseP
        noiseAdded = []
        for m, n in zip(msg, noise):
            if n == 0:
                noiseAdded.append(m)
            else:
                noiseAdded.append(1-m)
        noiseAdded = np.array(noiseAdded)
        return (noiseAdded)

    def sendMessage(self, senderID, receiverID, msg):
        receiver = self.agents[receiverID]
        if self.noised:
            msg = self.addNoise(msg)

        receiver.recieveMessage(senderID, msg)
