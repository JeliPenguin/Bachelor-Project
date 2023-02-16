import numpy as np
from typing import *
import hashlib

sampleMessage = {
    'state': np.array([8., 6., 4., 2., 4., 3., 8., 2., 3., 8.]),
    'reward': [255],
    'sPrime': np.array([8., 6., 4., 1., 4., 2., 8., 2., 3., 8.])}

sampleMessage2 = {
    'state': np.array([8., 6., 4., 2., 4., 3., 8., 2., 3., 8.]),
    'reward': [255],
    'sPrime': None}

sampleMessage3 = {
    'state': np.array([8., 6., 4., 2., 8., 2., 3., 8.]),
    'reward': [255],
    'sPrime': None}


class simulation():
    def __init__(self) -> None:
        self.messageMemory = sampleMessage2
        self.messageReceived = {}
        self.n_observations = len(self.messageMemory["state"])
        self.action = [1]
        self.k = 8

    def addNoise(self, msg, p=0.05):
        noise = np.random.random(msg.shape) < p
        noiseAdded = []
        for m, n in zip(msg, noise):
            if n == 0:
                noiseAdded.append(m)
            else:
                noiseAdded.append(1-m)
        noiseAdded = np.array(noiseAdded)
        return (noiseAdded)

    def genChecksum(self, encoded: str):
        # Normal encoded length 168
        # Terminal state encoded length 88
        res = []

        digitSum = 0
        for i in range(int(len(encoded)/self.k)):
            digitSum += int(encoded[self.k*i:self.k*(i+1)], 2)

        digitSum = bin(digitSum)[2:]

        if(len(digitSum) > self.k):
            x = len(digitSum)-self.k
            digitSum = bin(int(digitSum[0:x], 2)+int(digitSum[x:], 2))[2:]
        if(len(digitSum) < self.k):
            digitSum = '0'*(self.k-len(digitSum))+digitSum

        for i in digitSum:
            if(i == '1'):
                res.append(0)
            else:
                res.append(1)

        return np.array(res, dtype=np.uint8)

    def checkChecksum(self, receivedMsg: str):
        # receivedMsg includes checksum as the final 8 bits

        digitSum = 0
        for i in range(int(len(receivedMsg)/self.k)):
            # print(receivedMsg[self.k*i:self.k*(i+1)])
            digitSum += int(receivedMsg[self.k*i:self.k*(i+1)], 2)

        digitSum = bin(digitSum)[2:]

        # Adding the overflow bits
        if(len(digitSum) > self.k):
            x = len(digitSum)-self.k
            digitSum = bin(
                int(digitSum[0:x], 2)+int(digitSum[x:], 2))[2:]

        checksum = 0
        for i in digitSum:
            if(i == '1'):
                checksum += 0
            else:
                checksum += 1
        return checksum == 0

    def encodeMessage(self):
        """
        Message Order: State - Reward - sPrime each as unsigned 8 bits
        Unsigned 255 used to represents -1
        """
        if self.messageMemory["reward"] is None and self.messageMemory["sPrime"] is None:
            # Case state only
            msgString = self.messageMemory["state"]
        elif self.messageMemory["sPrime"] is None:
            # Case termination
            msgString = np.concatenate(
                (self.messageMemory["state"], self.messageMemory["reward"]))
        else:
            msgString = np.concatenate(
                (self.messageMemory["state"], self.messageMemory["reward"], self.messageMemory["sPrime"]))
        formatted = np.array(msgString, dtype=np.uint8)
        encoded = np.unpackbits(formatted)
        return encoded

    def decodeMessage(self, encodedMsg):
        decodedMsg = np.packbits(encodedMsg[self.k:])
        msgLen = len(decodedMsg)
        obsLen = self.n_observations
        parse = {
            "state": None,
            "reward": None,
            "sPrime": None
        }
        parse["state"] = decodedMsg[:obsLen]
        if msgLen > self.n_observations:
            parse["reward"] = [decodedMsg[obsLen]]
            if msgLen > self.n_observations + 1:
                parse["sPrime"] = decodedMsg[obsLen+1:]
        if parse["reward"] == [255]:
            parse["reward"] = [-1]
        return parse

    def stringify(self, encoded):
        encodedString = ""
        for b in encoded:
            encodedString += str(b)
        return encodedString

    def sendMessage(self):
        print("Message sent: ", self.messageMemory)
        encoded = self.encodeMessage()
        # print("Message Sent Encoded: ", encoded)
        stringified = self.stringify(encoded)
        checksum = self.genChecksum(stringified)
        encoded = np.concatenate([checksum, encoded])
        # print("Concatenated sent msg: ", encoded)
        # encoded = self.addNoise(encoded)
        self.recieveMessage(0, encoded)

    def recieveMessage(self, senderID: int, msg):
        # Assumes message recieved in inorder
        stringified = self.stringify(msg)
        print("Checksum check: ", self.checkChecksum(stringified))
        parse = self.decodeMessage(msg)
        parse["action"] = self.action
        print("Msg Recieved: ", parse)


t = simulation()
t.sendMessage()
