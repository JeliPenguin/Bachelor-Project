import torch
import numpy as np
from typing import *
from const import device

sampleMessage = {
    'state': np.array([8., 6., 4., 2., 4., 3., 8., 2., 3., 8.]),
    'action': [1],
    'reward': [255],
    'sPrime': np.array([8., 6., 4., 1., 4., 2., 8., 2., 3., 8.])}

sampleMessage2 = {
    'state': np.array([8., 6., 4., 2., 4., 3., 8., 2., 3., 8.]),
    'action': [1],
    'reward': [255],
    'sPrime': None}

sampleMessage3 = {
    'state': np.array([8., 6., 4., 2., 8., 2., 3., 8.]),
    'action': [1],
    'reward': [255],
    'sPrime': None}


class simulation():
    def __init__(self) -> None:
        self.messageMemory = sampleMessage2
        self.messageReceived = {}
        self.n_obs = len(self.messageMemory["state"])

    def addNoise(self, msg, p=0.01):
        noise = np.random.random(msg.shape) < p
        noiseAdded = []
        for m, n in zip(msg, noise):
            if n == 0:
                noiseAdded.append(m)
            else:
                noiseAdded.append(1-m)
        noiseAdded = np.array(noiseAdded)
        print(noiseAdded)
        return (noiseAdded)

    def encoder(self):
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

    def decodeMessage(self, msg):
        decodedMsg = np.packbits(msg)
        msgLen = len(decodedMsg)
        parse = {
            "state": None,
            "action": None,
            "reward": None,
            "sPrime": None
        }
        parse["state"] = decodedMsg[:self.n_obs]
        if msgLen > self.n_obs:
            parse["action"] = [decodedMsg[self.n_obs]]
            parse["reward"] = [decodedMsg[self.n_obs+1]]
            if msgLen > self.n_obs + 2:
                parse["sPrime"] = decodedMsg[self.n_obs+2:]
        if parse["reward"] == [255]:
            parse["reward"] = [-1]
        return parse

    def sendMessage(self):
        encoded = self.encoder()
        # encoded = addNoise(encoded)
        self.recieveMessage(0, encoded)

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
                    if content == [255]:
                        content = [-1]
                    content = torch.tensor(
                        content, dtype=torch.float32, device=device)
            if senderID not in self.messageReceived:
                self.messageReceived[senderID] = {tag: content}
            else:
                self.messageReceived[senderID][tag] = (content)
        print(self.messageReceived)

    """
    Ending format
    {0: 
        {
            'state': tensor([[3., 9., 2., 4., 9., 8., 5., 7., 0., 0.]]), 
            'action': tensor([[3]]), 
            'reward': tensor([-1.]), 
            'sPrime': tensor([[3., 9., 3., 4., 9., 8., 5., 7., 0., 0.]])
        }
    }
    """


t = simulation()
t.sendMessage()
