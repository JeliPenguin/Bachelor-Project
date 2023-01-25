from DQN import DQNAgent
from CommChannel import CommChannel

class CommAgent(DQNAgent):
    def __init__(self, id, n_observations, actionSpace, batchSize=32, gamma=0.99, epsStart=0.9, epsEnd=0.05, epsDecay=1000, tau=0.005, lr=0.0001) -> None:
        super().__init__(id, n_observations, actionSpace, batchSize, gamma, epsStart, epsEnd, epsDecay, tau, lr)
        self.messageReceived = {}
        self.messageSent = {}
    
    def setChannel(self,channel:CommChannel):
        self.channel = channel
    
    def sendMessage(self,recieverID:int,msg,tag:str):
        self.channel.sendMessage(self.id,recieverID,msg,tag)
        if recieverID not in self.messageSent:
            self.messageSent[recieverID] = {tag:msg}
        else:
            self.messageSent[recieverID][tag]=(msg)
    
    def recieveMessage(self,senderID:int,msg,tag:str):
        if senderID not in self.messageReceived:
            self.messageReceived[senderID] = {tag:msg}
        else:
            self.messageReceived[senderID][tag]= (msg)