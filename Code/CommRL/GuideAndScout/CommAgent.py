from const import *
from DQN import DQNAgent
from CommChannel import CommChannel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CommAgent(DQNAgent):
    def __init__(self, id, n_observations, actionSpace, batchSize=32, gamma=0.99, epsStart=0.9, epsEnd=0.05, epsDecay=1000, tau=0.005, lr=0.0001) -> None:
        super().__init__(id, n_observations, actionSpace, batchSize, gamma, epsStart, epsEnd, epsDecay, tau, lr)
        self.messageReceived = {}
        self.messageSent = {}
    
    def sendMessage(self,msg,recieverID:int,channel:CommChannel):
        channel.sendMessage(self.id,recieverID,msg)
        if recieverID not in self.messageSent:
            self.messageSent[recieverID] = [msg]
        else:
            self.messageSent[recieverID].append(msg)
    
    def recieveMessage(self,senderID:int,msg):
        if senderID not in self.messageReceived:
            self.messageReceived[senderID] = [msg]
        else:
            self.messageReceived[senderID].append(msg)


class ScoutAgent(DQNAgent):
    def __init__(self, id, obs_dim, actionSpace) -> None:
        super().__init__(id, obs_dim, actionSpace)
        self.symbol = "S"



class GuideAgent(DQNAgent):
    def __init__(self, id, obs_dim, actionSpace) -> None:
        super().__init__(id, obs_dim, actionSpace)
        self.symbol = "G"

    def choose_action(self, s):
        # Guide can only stay at allocated position
        return torch.tensor([[STAY]],device=device)
