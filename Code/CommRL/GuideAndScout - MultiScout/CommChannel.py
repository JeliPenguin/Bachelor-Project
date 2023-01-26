from typing import Tuple
from const import device
import torch


class CommChannel():
    def __init__(self, agents, noised=False) -> None:
        self.noised = noised
        self.agents = agents
        
    
    def setupChannel(self):
        for agent in self.agents:
            agent.setChannel(self)

    def addNoise(self, message, p=0.005):
        noisedMsg = message

        return noisedMsg

    def sendMessage(self, senderID, receiverID, msg, tag):
        # print(tag, msg)
        receiver = self.agents[receiverID]
        if self.noised:
            msg = self.addNoise(msg)
        receiver.recieveMessage(senderID, msg, tag)
