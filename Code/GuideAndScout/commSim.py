import numpy as np
from typing import *
from collections import deque
from copy import deepcopy
import scipy.stats

sampleMessage = {
    'state': np.array([8., 6., 4., 2., 4., 3., 8., 2., 3., 8.]),
    'reward': [-1],
    'sPrime': np.array([8., 6., 4., 1., 4., 2., 8., 2., 3., 8.])}

sampleMessage2 = {
    'state': np.array([8., 6., 4., 2., 4., 3., 8., 2., 3., 8.]),
    'reward': [-1],
    'sPrime': None}

sampleMessage3 = {
    'state': np.array([8., 6., 4., 2., 4., 3., 8., 2., 3., 8.]),
    'reward': None,
    'sPrime': None}

sampleHistory = deque([
    {0: {'state': np.array([2, 2, 1, 0, 1, 1, 0, 3, 2, 1], dtype=np.uint8), 'reward': [-5],
         'sPrime': np.array([2, 2, 0, 0, 1, 2, 0, 3, 2, 1], dtype=np.uint8), 'action': [2]}},
    {0: {'state': np.array([2, 2, 0, 0, 1, 2, 0, 3, 2, 1], dtype=np.uint8), 'reward': [-5],
         'sPrime': np.array([2, 2, 0, 0, 1, 2, 0, 3, 2, 1], dtype=np.uint8), 'action': [3]}},
    {0: {'state': np.array([2, 2, 0, 0, 1, 2, 0, 3, 2, 1], dtype=np.uint8), 'reward': [-5],
         'sPrime': np.array([2, 2, 0, 1, 1, 2, 0, 3, 2, 1], dtype=np.uint8), 'action': [3]}},
    {0: {'state': np.array([2, 2, 0, 1, 1, 2, 0, 3, 2, 1], dtype=np.uint8), 'reward': [-5],
         'sPrime': np.array([2, 2, 1, 1, 1, 2, 0, 3, 2, 1], dtype=np.uint8), 'action': [4]}},
    {0: {'state': np.array([2, 2, 1, 1, 1, 2, 0, 3, 2, 1], dtype=np.uint8), 'reward': [-5],
         'sPrime': np.array([2, 2, 1, 1, 0, 2, 0, 3, 2, 1], dtype=np.uint8), 'action': [0]}},
])


class simulation():
    def __init__(self) -> None:
        self._messageMemory = sampleMessage2
        self._messageReceived = {}
        self._n_observations = len(self._messageMemory["state"])
        self._action = [1]
        self._recievedHistory = deque()
        self._k = 8
        self._majorityNum = 5
        self._majorityMem = []

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
        for i in range(int(len(encoded)/self._k)):
            digitSum += int(encoded[self._k*i:self._k*(i+1)], 2)

        digitSum = bin(digitSum)[2:]

        if(len(digitSum) > self._k):
            x = len(digitSum)-self._k
            digitSum = bin(int(digitSum[0:x], 2)+int(digitSum[x:], 2))[2:]
        if(len(digitSum) < self._k):
            digitSum = '0'*(self._k-len(digitSum))+digitSum

        for i in digitSum:
            if(i == '1'):
                res.append(0)
            else:
                res.append(1)

        return np.array(res, dtype=np.uint8)

    def checkChecksum(self, receivedMsg: str):
        # receivedMsg includes checksum as the final 8 bits

        digitSum = 0
        for i in range(int(len(receivedMsg)/self._k)):
            # print(receivedMsg[self._k*i:self._k*(i+1)])
            digitSum += int(receivedMsg[self._k*i:self._k*(i+1)], 2)

        digitSum = bin(digitSum)[2:]

        # Adding the overflow bits
        if(len(digitSum) > self._k):
            x = len(digitSum)-self._k
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
        For rewards, unsigned 129-255 used to represents -127 - -1
        """
        if self._messageMemory["reward"] is None and self._messageMemory["sPrime"] is None:
            # Case state only
            msgString = self._messageMemory["state"]
        elif self._messageMemory["sPrime"] is None:
            # Case termination
            msgString = np.concatenate(
                (self._messageMemory["state"], self._messageMemory["reward"]))
        else:
            msgString = np.concatenate(
                (self._messageMemory["state"], self._messageMemory["reward"], self._messageMemory["sPrime"]))
        formatted = np.array(msgString, dtype=np.uint8)
        encoded = np.unpackbits(formatted)
        return encoded

    def stringify(self, encoded):
        encodedString = ""
        for b in encoded:
            encodedString += str(b)
        return encodedString

    def sendMessage(self):
        print("Message sent: ", self._messageMemory)
        msgString = self.encodeMessage()
        stringified = self.stringify(msgString)
        checksum = self.genChecksum(stringified)
        msgString = np.concatenate([checksum, msgString])
        # print("Encoded sent message: ", msgString)
        for _ in range(self._majorityNum):
            self.recieveMessage(0, msgString)

    def decodeMessage(self, encodedMsg):
        decodedMsg = np.packbits(encodedMsg[self._k:])
        msgLen = len(decodedMsg)
        obsLen = self._n_observations
        parse = {
            "state": None,
            "reward": None,
            "sPrime": None
        }
        parse["state"] = decodedMsg[:obsLen]
        if msgLen > self._n_observations:
            parse["reward"] = [decodedMsg[obsLen]]
            if msgLen > self._n_observations + 1:
                parse["sPrime"] = decodedMsg[obsLen+1:]
        if parse["reward"] is not None and parse["reward"][0] > 129:
            parse["reward"] = [parse["reward"][0]-256]
        return parse

    def attemptRecovery(self, parse):
        # Attempt in recovering original message by looking at history of correctly received messages

        #
        return parse

    def rememberRecieved(self):
        # Make a copy of all recieved messages
        # Stroing 5 past messages max
        if len(self._recievedHistory) >= 5:
            self._recievedHistory.popleft()
        self._recievedHistory.append(deepcopy(self._messageReceived))
        print("Recieved history: ")
        for hist in self._recievedHistory:
            print(hist)
        print("\n")

    def majorityVote(self):
        res = scipy.stats.mode(np.stack(self._majorityMem),
                               axis=0, keepdims=True).mode[0]
        return res

    def recieveMessage(self, senderID: int, msg):
        # Assumes message recieved in inorder
        msg = self.addNoise(msg)
        self._majorityMem.append(msg)
        if len(self._majorityMem) == self._majorityNum:
            msg = self.majorityVote()
            self._majorityMem.clear()
            stringified = self.stringify(msg)
            msgChecksumPass = self.checkChecksum(stringified)
            print("Checksum check: ", msgChecksumPass)
            parse = self.decodeMessage(msg)
            if not msgChecksumPass:
                parse = self.attemptRecovery(parse)
            # Action independent of the message as agent itself knows what action has been executed (deterministic policy)
            if parse:
                parse["action"] = self._action
                print("Message Received: ", parse)
                print("\n")
                for tag, content in parse.items():
                    if senderID not in self._messageReceived:
                        self._messageReceived[senderID] = {tag: content}
                    else:
                        self._messageReceived[senderID][tag] = (content)
            else:
                print("Message not recovered")


t = simulation()
t.sendMessage()
