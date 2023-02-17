from Agents.DQN import DQNAgent
from Environment.CommChannel import CommChannel
import numpy as np
import torch
from const import *
from collections import deque
from copy import deepcopy


class CommAgent(DQNAgent):
    def __init__(self, id, n_observations, actionSpace, batchSize=128, gamma=1, epsStart=0.9, epsEnd=0.05, epsDecay=1000, tau=0.005, lr=0.0001) -> None:
        super().__init__(id, n_observations, actionSpace,
                         batchSize, gamma, epsStart, epsEnd, epsDecay, tau, lr)
        self.reset()
        self.k = 8

    def reset(self):
        self.messageReceived = {}
        self.recievedHistory = deque()
        self.messageSent = {}
        self.action = None
        self.messageMemory = {
            "state": None,
            "reward": None,
            "sPrime": None
        }

    def setChannel(self, channel: CommChannel):
        self.channel = channel
        self.reset()

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
        Unsigned 129-255 used to represents -127 - -1
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

    def clearPreparedMessage(self):
        self.messageMemory = {
            "state": None,
            "reward": None,
            "sPrime": None
        }

    def prepareMessage(self, msg, tag: str):
        self.messageMemory[tag] = msg

    def rememberAction(self, action):
        self.action = action

    def stringify(self, encoded):
        encodedString = ""
        for b in encoded:
            encodedString += str(b)
        return encodedString

    def sendMessage(self, recieverID: int):
        if getVerbose() >= 2:
            print("Sending to Agent: ", recieverID)
            print("Message sent: ", self.messageMemory)
        msgString = self.encodeMessage()
        stringified = self.stringify(msgString)
        checksum = self.genChecksum(stringified)
        msgString = np.concatenate([checksum, msgString])
        # if getVerbose() >= 3:
        #     print("Checksum: ", checksum)
        if getVerbose() >= 4:
            print("Encoded sent message: ", msgString)
        self.channel.sendMessage(self.id, recieverID, msgString)

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
        if parse["reward"] is not None and parse["reward"][0] > 129:
            parse["reward"] = [parse["reward"][0]-256]
        return parse

    def attemptRecovery(self):
        # Attempt in recovering original message by looking at history of correctly received messages
        pass

    def rememberRecieved(self):
        # Make a copy of all recieved messages
        # Stroing 5 past messages max
        if len(self.recievedHistory) >= 5:
            self.recievedHistory.popleft()
        self.recievedHistory.append(deepcopy(self.messageReceived))
        if getVerbose() >= 3:
            print("Recieved history: ")
            for hist in self.recievedHistory:
                print(hist)
            print("\n")

    def recieveMessage(self, senderID: int, msg):
        # Assumes message recieved in inorder
        stringified = self.stringify(msg)
        msgChecksum = self.checkChecksum(stringified)
        if getVerbose() >= 3:
            print("Checksum check: ", msgChecksum)
        parse = self.decodeMessage(msg)
        # Action independent of the message as agent itself knows what action has been executed (deterministic policy)
        parse["action"] = self.action
        if getVerbose() >= 2:
            print("Message Received: ", parse)
            print("\n")
        for tag, content in parse.items():
            if senderID not in self.messageReceived:
                self.messageReceived[senderID] = {tag: content}
            else:
                self.messageReceived[senderID][tag] = (content)
        # print(self.messageReceived)
