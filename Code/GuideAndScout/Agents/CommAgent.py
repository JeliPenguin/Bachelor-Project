from Agents.DQN import DQNAgent
from Environment.CommChannel import CommChannel
import numpy as np
import torch
from const import *
from collections import deque
from copy import deepcopy
import scipy.stats

class CommAgent(DQNAgent):
    def __init__(self, id, n_observations, actionSpace, noiseHandling=False, batchSize=128, gamma=1, epsStart=0.9, epsEnd=0.05, epsDecay=1000, tau=0.005, lr=0.0001) -> None:
        super().__init__(id, n_observations, actionSpace,
                         batchSize, gamma, epsStart, epsEnd, epsDecay, tau, lr)

        self._k = 8
        self._historySize = 20
        self._majorityNum = 3
        self._noiseHandling = noiseHandling

        self._treatNum = 2
        self.reset()

    def reset(self):
        self._messageReceived = {}
        self._recievedHistory = deque(maxlen=self._historySize)
        self._messageSent = {}
        self._action = None
        self._messageMemory = {
            "state": None,
            "reward": None,
            "sPrime": None
        }
        self._majorityMem = []
        self._anchoredGuidePos = None
        self._anchoredTreatPos = None

    def setNoiseHandling(self, noiseHandling):
        self._noiseHandling = noiseHandling

    def setChannel(self, channel: CommChannel):
        self._channel = channel
        self.reset()

    def calcDigitsum(self, binString: str):
        digitSum = 0
        for i in range(int(len(binString)/self._k)):
            digitSum += int(binString[self._k*i:self._k*(i+1)], 2)
        digitSum = bin(digitSum)[2:]
        return digitSum

    def genChecksum(self, encoded):
        # Normal encoded length 168
        # Terminal state encoded length 88
        res = []
        encoded = self.stringify(encoded)
        digitSum = self.calcDigitsum(encoded)
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
        digitSum = self.calcDigitsum(receivedMsg)
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

    def tensorize(self, msg):
        stateTensor = None
        actionTensor = None
        sPrimeTensor = None
        rewardTensor = None
        for tag, content in msg.items():
            if content is not None:
                if tag == "action":
                    actionTensor = torch.tensor(
                        [content], dtype=torch.int64, device=device)
                elif tag == "state":
                    stateTensor = torch.tensor(content, dtype=torch.float32,
                                               device=device).unsqueeze(0)
                elif tag == "sPrime":
                    sPrimeTensor = torch.tensor(content, dtype=torch.float32,
                                                device=device).unsqueeze(0)
                elif tag == "reward":
                    rewardTensor = torch.tensor(
                        content, dtype=torch.float32, device=device)
        return stateTensor, actionTensor, sPrimeTensor, rewardTensor

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

    def clearPreparedMessage(self):
        self._messageMemory = {
            "state": None,
            "reward": None,
            "sPrime": None
        }

    def prepareMessage(self, msg, tag: str):
        self._messageMemory[tag] = msg

    def rememberAction(self, action):
        self._action = action

    def stringify(self, encoded):
        encodedString = ""
        for b in encoded:
            encodedString += str(b)
        return encodedString

    def sendMessage(self, recieverID: int):
        if getVerbose() >= 2:
            print("Sending to Agent: ", recieverID)
            print("Message sent: ", self._messageMemory)
        msgString = self.encodeMessage()
        if getVerbose() >= 5:
            print("Encoded sent message: ", msgString)
        if self._noiseHandling:
            checksum = self.genChecksum(msgString)
            # Checksum sent along with msg, hence can be noised as well
            msgString = np.concatenate([checksum, msgString])
            if getVerbose() >= 5:
                print("Checksum: ", checksum)
            for _ in range(self._majorityNum):
                self._channel.sendMessage(self._id, recieverID, msgString)
        else:
            self._channel.sendMessage(self._id, recieverID, msgString)

    def decodeMessage(self, encodedMsg):
        msgLen = len(encodedMsg)
        obsLen = self._n_observations
        parse = {
            "state": None,
            "reward": None,
            "sPrime": None
        }
        parse["state"] = encodedMsg[:obsLen]
        if msgLen > self._n_observations:
            parse["reward"] = [encodedMsg[obsLen]]
            if msgLen > self._n_observations + 1:
                parse["sPrime"] = encodedMsg[obsLen+1:]
        if parse["reward"] is not None and parse["reward"][0] > 129:
            parse["reward"] = [parse["reward"][0]-256]
        return parse

    def majorityVote(self):
        res = scipy.stats.mode(np.stack(self._majorityMem),
                               axis=0, keepdims=True).mode[0]
        self._majorityMem.clear()
        return res

    def attemptRecovery(self, senderID, parse):
        # Attempt in recovering original message by looking at history of correctly received messages
        # Could be checksum got corrupted, msg got corrupted or both

        # TODO Now assumes the last history is 100% accurate
        # TODO Not robust atm, assumes environment has only 2 treats
        # print("Recovering Message: ")
        fixedState = parse["state"]
        fixedReward = parse["reward"]
        fixedsPrime = parse["sPrime"]
        hasSPrime = fixedsPrime is not None

        def recoverSPrime():
            # Detect skip in states
            pass

        if self._recievedHistory:
            recentRecord = self._recievedHistory[-1][senderID]
            recentState = recentRecord["state"]
            recentsPrime = recentRecord["sPrime"]
            history = recentsPrime
            if history is None:
                history = recentState

            # Guide, treat positions are fixed hence can be recovered directly
            # Assumes 2 treats only in environment
            if self._anchoredGuidePos is not None:
                guidePos = self._anchoredGuidePos
            else:
                guidePos = history[0:2]

            if self._anchoredTreatPos is not None:
                treatPos = self._anchoredTreatPos
            else:
                treatPos = history[self._n_observations-2*self._treatNum:]

            fixedState[0:2] = guidePos
            fixedState[self._n_observations -
                       2*self._treatNum:] = treatPos
            if hasSPrime:
                fixedsPrime[0:2] = guidePos
                fixedsPrime[self._n_observations -
                        2*self._treatNum:] = treatPos

            # Current scout's s and sPrime can be estimated using previous s and action
            fixedState[self._id*2:self._id*2 +
                       2] = history[self._id*2:self._id*2+2]
            myState = history[self._id*2:self._id*2+2]
            actionTaken = decodeAction(self._action[0])
            newState = np.array(
                transition(tuple(myState), actionTaken), dtype=np.uint8)
            if hasSPrime:
                fixedsPrime[self._id*2:self._id*2 + 2] = newState
        else:
            # No history of previous states
            return parse
        return {
            "state": fixedState,
            "reward": fixedReward,
            "sPrime": fixedsPrime
        }

    def rememberRecieved(self, correctChecksum):
        # Make a copy of all recieved messages
        self._recievedHistory.append(deepcopy(self._messageReceived))
        if correctChecksum:
            for id in self._messageReceived.keys():
                state = self._messageReceived[id]["state"]
                guidePos = state[:2]
                treatPos = state[self._n_observations - 4:]
                self._anchoredGuidePos = guidePos
                self._anchoredTreatPos = treatPos

        if getVerbose() >= 3:
            print("Recieved history: ")
            for hist in self._recievedHistory:
                print(hist)
            print("\n")

    def storeRecievedMessage(self, senderID, parse, correctChecksum=True):
        # Action independent of the message as agent itself knows what action has been executed
        # Policy assumed to be a deterministic policy
        parse["action"] = self._action
        if getVerbose() >= 2:
            print("Message Received: ", parse)
            print("\n")
        for tag, content in parse.items():
            if senderID not in self._messageReceived:
                self._messageReceived[senderID] = {tag: content}
            else:
                self._messageReceived[senderID][tag] = (content)
        if self._noiseHandling:
            # Remember msg recieved
            self.rememberRecieved(correctChecksum)

    def recieveNoisyMessage(self, senderID: int, msg):
        # Assumes message recieved in inorder
        self._majorityMem.append(msg)
        if len(self._majorityMem) == self._majorityNum:
            # Majority vote the messages
            msg = self.majorityVote()
            msgChecksumPass = self.checkChecksum(msg)
            if getVerbose() >= 3:
                print("Checksum check: ", msgChecksumPass)
            msg = np.packbits(msg[self._k:])
            decoded = self.decodeMessage(msg)
            if getVerbose() >= 3:
                print("Before recovery: ",decoded)
            if not msgChecksumPass:
                # If majority voting unable to fix noise, attempt recovery of message using previous history
                decoded = self.attemptRecovery(senderID, decoded)
            if getVerbose() >= 3:
                print("Anchors:")
                print(f"Guide Pos: {self._anchoredGuidePos}")
                print(f"Treat Pos: {self._anchoredTreatPos}")
            self.storeRecievedMessage(senderID, decoded, msgChecksumPass)

    def recieveNormalMessage(self, senderID: int, msg):
        msg = np.packbits(msg)
        decoded = self.decodeMessage(msg)
        self.storeRecievedMessage(senderID, decoded)

    def recieveMessage(self, senderID: int, msg):
        if self._noiseHandling:
            self.recieveNoisyMessage(senderID, msg)
        else:
            self.recieveNormalMessage(senderID, msg)
