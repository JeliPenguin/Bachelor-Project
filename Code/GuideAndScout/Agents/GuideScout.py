from const import verbPrint, device, getVerbose
from Environment.EnvUtilities import *
from Agents.CommAgent import CommAgent
from Agents.MessageRecoverer import MessageRecoverer
import torch
import numpy as np
import scipy.stats
from collections import deque


class ScoutAgent(CommAgent):
    def __init__(self, id, obs_dim, actionSpace, noiseHandling, hyperParam) -> None:
        super().__init__(id, obs_dim, actionSpace, noiseHandling, hyperParam)
        self._symbol = str(id)
        self._falseCount = 0
        self._recieveCount = 0
        self._MASampleSize = 3
        self._falseLimit = 0.3
        self.recoverer = MessageRecoverer(self._id, self._totalTreatNum)
        self._historySize = 20
        self._recievedHistory = deque(maxlen=self._historySize)

    def choose_greedy_action(self) -> torch.Tensor:
        guideMsg = self._messageReceived[GUIDEID]
        stateTensor, _, _, _ = self.tensorize(guideMsg)
        with torch.no_grad():
            return self._policy_net(stateTensor).max(1)[1].view(1, 1)

    def choose_action(self) -> torch.Tensor:
        """ 
            Ordinary Epsilon greedy 
        """
        p = np.random.random()
        epsThresh = self._epsEnd + \
            (self._epsStart - self._epsEnd) * \
            np.exp(-1. * self._eps_done / self._epsDecay)
        # print(f"EpsThresh: {epsThresh} Eps done: {self._eps_done}")
        if p > epsThresh:
            return self.choose_greedy_action()
        return self.choose_random_action()

    def memorize(self):
        """
            Unpacks message recieved from Guide and memorize the states
        """
        guideMsg = self._messageReceived[GUIDEID]
        stateTensor, actionTensor, sPrimeTensor, rewardTensor = self.tensorize(
            guideMsg)
        super().memorize(stateTensor, actionTensor, sPrimeTensor, rewardTensor)

    def updateEps(self):
        self._eps_done += 1

    def majorityVote(self):

        res = scipy.stats.mode(np.stack(self._majorityMem),
                               axis=0, keepdims=True).mode[0]
        self._majorityMem.clear()
        return res

    def majorityAdjust(self, checksumCheck):
        self._recieveCount += 1
        if not checksumCheck:
            self._falseCount += 1
        if self._recieveCount >= self._MASampleSize:
            falseRatio = self._falseCount / self._recieveCount
            if falseRatio >= self._falseLimit:
                self._majorityNum = min(self._majorityNum + 2, self._bandwidth)
                # verbPrint(
                #     f"Increased majority num, now is : {self._majorityNum}", 4)
                self._falseCount = 0
                self._recieveCount = 0
                self.broadcastMajority()

    def recieveNoisyMessage(self, senderID: int, msg):
        # Assumes message recieved in inorder
        self._majorityMem.append(msg)
        if len(self._majorityMem) == self._majorityNum:
            # Majority vote the messages
            msg = self.majorityVote()
            noError, msg = self.errorDetector.decode(msg)
            msg = np.packbits(msg)
            decoded = self.decodeMessage(msg)
            state = decoded["state"]
            guidePos = state[:2]
            treatPos = state[self._n_observations - 2*self._totalTreatNum:]
            self.recoverer.computeGuideAnchor(guidePos, noError)
            self.recoverer.computeTreatAnchor(treatPos, noError)
            if not noError:
                # If majority voting unable to fix noise, attempt recovery of message using previous history
                decoded = self.recoverer.attemptRecovery(
                    decoded, self._recievedHistory, self._action)
            self.majorityAdjust(noError)
            self.storeRecievedMessage(senderID, decoded, noError)

    def broadcastMajority(self):
        self.broadcastSignal(np.array([0]))

    def recieveBroadcast(self, signal):
        # Currently assumes the only signal broadcasted is for majority updates
        self._majorityNum = min(self._majorityNum + 2, self._bandwidth)

    def parseState(self, state):
        guidePos = state[:2]
        treatsPos = state[self._n_observations - 2*self._totalTreatNum:]
        scout1Pos = state[2:4]
        scout2Pos = state[4:6]
        return [guidePos, scout1Pos, scout2Pos, treatsPos]

    def formatHistory(self):
        history = self._messageReceived[0]
        formatted = {}
        formatted["state"] = self.parseState(history["state"])
        formatted["action"] = history["action"]
        if history["sPrime"] is None:
            formatted["sPrime"] = None
        else:
            formatted["sPrime"] = self.parseState(history["sPrime"])
        return formatted

    def rememberRecieved(self, correctChecksum):
        # Make a copy of all recieved messages
        history = self.formatHistory()
        history["checksum"] = correctChecksum
        self._recievedHistory.append(history)

        # if getVerbose() >= 3:
        #     print("Recieved history: ")
        #     for hist in self._recievedHistory:
        #         print(hist)

    def storeRecievedMessage(self, senderID, parse, correctChecksum=True):
        super().storeRecievedMessage(senderID, parse)
        if self._noiseHandling:
            # Remember msg recieved
            self.rememberRecieved(correctChecksum)


class GuideAgent(CommAgent):

    def __init__(self, id, obs_dim, actionSpace, noiseHandling, hyperParam) -> None:
        super().__init__(id, obs_dim, actionSpace, noiseHandling, hyperParam)
        self._symbol = "G"

    def choose_action(self) -> torch.Tensor:
        """ Returns STAY as Guide can only stay at allocated position"""
        return torch.tensor([[STAY]], device=device)

    def choose_random_action(self) -> torch.Tensor:
        """ Returns STAY as Guide can only stay at allocated position"""
        return self.choose_action()

    def recieveBroadcast(self, signal):
        self._majorityNum = min(self._majorityNum + 2, self._bandwidth)

    def sendMessage(self, recieverID: int):
        msgString = self.encodeMessage()
        if self._noiseHandling:
            errorDetectionCode = self.errorDetector.encode(msgString)
            # error detection code sent along with msg
            msgString = np.concatenate([errorDetectionCode, msgString])
            for _ in range(self._majorityNum):
                self._channel.sendMessage(self._id, recieverID, msgString)
        else:
            self._channel.sendMessage(self._id, recieverID, msgString)
