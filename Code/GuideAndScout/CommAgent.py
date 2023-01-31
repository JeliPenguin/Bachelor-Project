from DQN import DQNAgent
from CommChannel import CommChannel
import numpy as np
import torch
from const import device


class CommAgent(DQNAgent):
    def __init__(self, id, n_observations, actionSpace, batchSize=32, gamma=0.99, epsStart=0.9, epsEnd=0.05, epsDecay=1000, tau=0.005, lr=0.0001) -> None:
        super().__init__(id, n_observations, actionSpace,
                         batchSize, gamma, epsStart, epsEnd, epsDecay, tau, lr)
        self.reset()

    def reset(self):
        self.messageReceived = {}
        self.messageSent = {}
        self.messageMemory = {
            "state": None,
            "action": None,
            "reward": None,
            "sPrime": None
        }

    def setChannel(self, channel: CommChannel):
        self.channel = channel
        self.reset()

    def encodeMessage(self):
        """
        Message Order: State - Action - Reward - sPrime each as unsigned 8 bits
        Unsigned 255 used to represents -1
        """
        if self.messageMemory["action"] is None and self.messageMemory["reward"] is None and self.messageMemory["sPrime"] is None:
            # Case state only
            msgString = self.messageMemory["state"]
        elif self.messageMemory["sPrime"] is None:
            # Case termination
            msgString = np.concatenate(
                (self.messageMemory["state"], self.messageMemory["action"], self.messageMemory["reward"]))
        else:
            msgString = np.concatenate(
                (self.messageMemory["state"], self.messageMemory["action"], self.messageMemory["reward"], self.messageMemory["sPrime"]))
        formatted = np.array(msgString, dtype=np.uint8)
        encoded = np.unpackbits(formatted)
        return encoded

    def prepareMessage(self, msg, tag: str):
        self.messageMemory[tag] = msg

    def sendMessage(self, recieverID: int):
        print("Agent: ",recieverID)
        print("Original: ",self.messageMemory)
        msgString = self.encodeMessage()
        self.channel.sendMessage(self.id, recieverID, msgString)

    def decodeMessage(self, encodedMsg):
        decodedMsg = np.packbits(encodedMsg)
        msgLen = len(decodedMsg)
        obsLen = self.n_observations
        parse = {
            "state": None,
            "action": None,
            "reward": None,
            "sPrime": None
        }
        parse["state"] = decodedMsg[:obsLen]
        if msgLen > self.n_observations:
            parse["action"] = [decodedMsg[obsLen]]
            parse["reward"] = [decodedMsg[obsLen+1]]
            if msgLen > self.n_observations + 2:
                parse["sPrime"] = decodedMsg[obsLen+2:]
        if parse["reward"] == [255]:
            parse["reward"] = [-1]
        return parse

    def recieveMessage(self, senderID: int, msg):
        # Assumes message recieved in inorder
        parse = self.decodeMessage(msg)
        for tag, content in parse.items():
            if content is not None:
                if tag == "action":
                    content = torch.tensor(
                        [content], dtype=torch.int64, device=device)
                elif tag == "state" or tag == "sPrime":
                    if content is not None:
                        content = torch.tensor(content, dtype=torch.float32,
                                               device=device).unsqueeze(0)
                elif tag == "reward":
                    content = torch.tensor(
                        content, dtype=torch.float32, device=device)
            if senderID not in self.messageReceived:
                self.messageReceived[senderID] = {tag: content}
            else:
                self.messageReceived[senderID][tag] = (content)
        # print(self.messageReceived)
