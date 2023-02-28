import numpy as np
from typing import *
from collections import deque
from copy import deepcopy
import scipy.stats
from const import decodeAction, transition

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

sampleMessage4 = {
    'state': np.array([0., 2., 0., 3., 1., 2., 2., 0., 0., 0.]),
    'reward': [-3],
    'sPrime': np.array([0., 2., 1., 3., 1., 2., 2., 0., 0., 0.])}

sampleHistory = deque([
    {0: {'state': np.array([0, 2, 1, 3, 2, 1, 2, 0, 0, 0], dtype=np.uint8), 'reward': [-3],
         'sPrime': np.array([0, 2, 1, 3, 2, 2, 2, 0, 0, 0], dtype=np.uint8), 'action': [4]}},
    {0: {'state': np.array([0, 2, 1, 3, 2, 2, 2, 0, 0, 0], dtype=np.uint8), 'reward': [-3],
         'sPrime': np.array([0, 2, 0, 3, 1, 2, 2, 0, 0, 0], dtype=np.uint8), 'action': [0]}},
    {0: {'state': np.array([0, 2, 0, 3, 1, 2, 2, 0, 0, 0], dtype=np.uint8), 'reward': [-3],
         'sPrime': np.array([0, 2, 0, 3, 1, 2, 2, 0, 0, 0], dtype=np.uint8), 'action': [1]}}
])


class simulation():
    def __init__(self) -> None:
        self._messageMemory = sampleMessage3
        self._messageReceived = {}
        self._n_observations = len(self._messageMemory["state"])
        self._action = [3]
        # self._recievedHistory = deque()
        self._recievedHistory = sampleHistory
        self._k = 8
        self._majorityNum = 3
        self._majorityMem = []
        self._noiseHandling = True
        self._id = 1

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
        encoded = self.stringify(encoded)
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

    def checkChecksum(self, receivedMsg):
        # receivedMsg includes checksum as the final 8 bits
        receivedMsg = self.stringify(receivedMsg)
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
        checksum = self.genChecksum(msgString)
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

    def attemptRecovery(self, senderID, parse):
        # Attempt in recovering original message by looking at history of correctly received messages

        # TODO Not robust atm, assumes environment has only 2 treats
        print("Recovering Message: ")
        fixedState = parse["state"]
        fixedReward = parse["reward"]
        fixedsPrime = parse["sPrime"]

        def recoverMyState(recentsPrime):
            fixedState[self._id*2:self._id*2 +
                       2] = recentsPrime[self._id*2:self._id*2+2]
            myState = recentsPrime[self._id*2:self._id*2+2]
            actionTaken = decodeAction(self._action[0])
            newState = np.array(
                transition(tuple(myState), actionTaken), dtype=np.uint8)
            if fixedsPrime:
                fixedsPrime[self._id*2:self._id*2 + 2] = newState

        def recoverSPrime():
            pass

        def recoverStandard(recentsPrime):
            # Assumes 2 treats in environment
            fixedState[0:2] = recentsPrime[0:2]
            fixedState[self._n_observations -
                       4:] = recentsPrime[self._n_observations-4:]
            if fixedsPrime:
                fixedsPrime[0:2] = recentsPrime[0:2]
                fixedsPrime[self._n_observations -
                            4:] = recentsPrime[self._n_observations-4:]

        if self._recievedHistory:
            recentRecord = self._recievedHistory[-1][senderID]
            recentState = recentRecord["state"]
            # Guide, treat positions are fixed hence can be recovered directly
            recoverStandard(recentRecord["sPrime"])
            # Current scout's s and sPrime can be estimated using previous s and action
            recoverMyState(recentRecord["sPrime"])
        else:
            # No history of previous states
            return parse

        print("\n")
        return {
            "state": fixedState,
            "reward": fixedReward,
            "sPrime": fixedsPrime
        }

    def majorityVote(self):
        res = scipy.stats.mode(np.stack(self._majorityMem),
                               axis=0, keepdims=True).mode[0]
        self._majorityMem.clear()
        return res

    def storeRecievedMessage(self, senderID, parse):
        # Action independent of the message as agent itself knows what action has been executed
        # Policy assumed to be a deterministic policy
        parse["action"] = self._action
        print("Message Received: ", parse)
        print("\n")
        for tag, content in parse.items():
            if senderID not in self._messageReceived:
                self._messageReceived[senderID] = {tag: content}
            else:
                self._messageReceived[senderID][tag] = (content)

    def recieveMessage(self, senderID: int, msg):
        # Assumes message recieved in inorder
        msg = self.addNoise(msg)
        self._majorityMem.append(msg)
        if len(self._majorityMem) == self._majorityNum:
            # Majority vote the messages
            msg = self.majorityVote()
            msgChecksumPass = self.checkChecksum(msg)
            print("Checksum check: ", msgChecksumPass)
            parse = self.decodeMessage(msg)
            if not msgChecksumPass:
                # If majority voting unable to fix noise, attempt recovery of message using previous history
                parse = self.attemptRecovery(senderID, parse)

            if parse:
                self.storeRecievedMessage(senderID, parse)
            else:
                print("Message not recovered")


t = simulation()
t.sendMessage()
