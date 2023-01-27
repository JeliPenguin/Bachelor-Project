from DQN import DQNAgent
from CommChannel import CommChannel
import numpy as np
import torch
from const import device


class CommAgent(DQNAgent):
    def __init__(self, id, n_observations, actionSpace, batchSize=32, gamma=0.99, epsStart=0.9, epsEnd=0.05, epsDecay=1000, tau=0.005, lr=0.0001) -> None:
        super().__init__(id, n_observations, actionSpace,
                         batchSize, gamma, epsStart, epsEnd, epsDecay, tau, lr)
        self.messageReceived = {}
        self.messageSent = {}
        self.messageMemory = {}

    def setChannel(self, channel: CommChannel):
        self.channel = channel

    def encodeMessage(self, msg):
        encodedMsg = msg
        return encodedMsg

    def decodeMessage(self, encodedMsg):
        decodedMsg = encodedMsg
        parse = {}
        parse["state"] = decodedMsg[:8]
        parse["action"] = decodedMsg[8]
        parse["reward"] = decodedMsg[9]
        parse["sPrime"] = decodedMsg[10:]
        if parse["sPrime"][0] == [-1]:
            parse["sPrime"] = None
        return parse

    def prepareMessage(self, msg, tag: str):
        self.messageMemory[tag] = msg

    def sendMessage(self, recieverID: int):
        """
        Sending Order:
        State - Action - Reward - sPrime
        """
        msgString = np.concatenate(
            (self.messageMemory["state"], self.messageMemory["action"], self.messageMemory["reward"], self.messageMemory["sPrime"]))
        msgString = self.encodeMessage(msgString)
        self.channel.sendMessage(self.id, recieverID, msgString)

    def recieveMessage(self, senderID: int, msg):
        # Assumes message recieved in inorder
        parse = self.decodeMessage(msg)
        for tag, content in parse.items():
            if tag == "action":
                content = torch.tensor(
                    [[content]], dtype=torch.int64, device=device)
            elif tag == "state" or tag == "sPrime":
                if content is not None:
                    content = torch.tensor(content, dtype=torch.float32,
                                           device=device).unsqueeze(0)
            elif tag == "reward":
                content = torch.tensor(
                    [content], dtype=torch.float32, device=device)
            if senderID not in self.messageReceived:
                self.messageReceived[senderID] = {tag: content}
            else:
                self.messageReceived[senderID][tag] = (content)
