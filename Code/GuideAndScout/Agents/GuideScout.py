from const import verbPrint, device, getVerbose
from Environment.EnvUtilities import *
from Agents.CommAgent import CommAgent
from Agents.MessageRecoverer import MessageRecoverer
import torch
import numpy as np
from typing import Tuple
from copy import deepcopy
import scipy.stats
from collections import deque


class ScoutAgent(CommAgent):
    def __init__(self, id, obs_dim, actionSpace, noiseHandling, epsDecay) -> None:
        super().__init__(id, obs_dim, actionSpace,
                         noiseHandling=noiseHandling, epsDecay=epsDecay)
        self._symbol = str(id)
        self._falseCount = 0
        self._recieveCount = 0
        self._MASampleSize = 3
        self._falseLimit = 0.4
        self.recoverer = MessageRecoverer(self._id, self._totalTreatNum)
        self._historySize = 5
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
                self._majorityNum = min(self._majorityNum + 2,self._bandwidth)
                verbPrint(
                    f"Increased majority num, now is : {self._majorityNum}", 4)
                self._falseCount = 0
                self._recieveCount = 0
                self.broadcastMajority()

    def recieveNoisyMessage(self, senderID: int, msg):
        # Assumes message recieved in inorder
        self._majorityMem.append(msg)
        if len(self._majorityMem) == self._majorityNum:
            # Majority vote the messages
            msg = self.majorityVote()
            msgChecksumPass, msg = self.errorDetector.decode(msg)
            verbPrint(f"Checksum check: {msgChecksumPass}", 2)
            msg = np.packbits(msg)
            decoded = self.decodeMessage(msg)
            verbPrint(f"Before recovery: {decoded}", 3)
            if not msgChecksumPass:
                # If majority voting unable to fix noise, attempt recovery of message using previous history
                decoded = self.recoverer.attemptRecovery(
                    senderID, decoded, self._recievedHistory, self._action)
                # reEncode = self.encodeMessage()
                # checksumRecheck = False
                # verbPrint(f"Checksum check after recovery: {checksumRecheck}",3)
            self.majorityAdjust(msgChecksumPass)
            verbPrint("Anchors:", 3)
            verbPrint(f"Guide Pos: {self.recoverer._anchoredGuidePos}", 3)
            verbPrint(f"Treat Pos: {self.recoverer._anchoredTreatPos}", 3)
            self.storeRecievedMessage(senderID, decoded, msgChecksumPass)

    def broadcastMajority(self):
        self.broadcastSignal(np.array([0]))

    def recieveBroadcast(self, signal):
        self._majorityNum = min(self._majorityNum + 2,self._bandwidth)

    def parseState(self,state):
        guidePos = state[:2]
        treatsPos = state[self._n_observations - 2*self._totalTreatNum:]
        scout1Pos = state[2:4]
        scout2Pos = state[4:6]
        return [guidePos,scout1Pos,scout2Pos,treatsPos]


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
        if correctChecksum:
            # for id in self._messageReceived.keys():
            state = self._messageReceived[GUIDEID]["state"]
            guidePos = state[:2]
            treatPos = state[self._n_observations - 2*self._totalTreatNum:]
            self.recoverer.computeGuideAnchor(guidePos)
            self.recoverer.computeTreatAnchor(treatPos)

        if getVerbose() >= 3:
            print("Recieved history: ")
            for hist in self._recievedHistory:
                print(hist)

    def storeRecievedMessage(self, senderID, parse, correctChecksum=True):
        super().storeRecievedMessage(senderID, parse)
        if self._noiseHandling:
            # Remember msg recieved
            self.rememberRecieved(correctChecksum)


class GuideAgent(CommAgent):
    def __init__(self, id, obs_dim, actionSpace, noiseHandling) -> None:
        super().__init__(id, obs_dim, actionSpace, noiseHandling=noiseHandling)
        self._symbol = "G"

    def choose_action(self) -> torch.Tensor:
        """ Returns STAY as Guide can only stay at allocated position"""
        return torch.tensor([[STAY]], device=device)

    def recieveNoisyMessage(self):
        return

    def choose_random_action(self) -> torch.Tensor:
        """ Returns STAY as Guide can only stay at allocated position"""
        return self.choose_action()

    def recieveBroadcast(self, signal):
        self._majorityNum = min(self._majorityNum + 2,self._bandwidth)

    def sendMessage(self, recieverID: int):
        if getVerbose() >= 2:
            print("Sending to Agent: ", recieverID)
            print("Message sent: ", self._messageMemory)
        msgString = self.encodeMessage()
        verbPrint(f"Encoded sent message: {msgString}", 5)
        if self._noiseHandling:
            checksum = self.errorDetector.encode(msgString)
            # Checksum sent along with msg, hence can be noised as well
            msgString = np.concatenate([checksum, msgString])
            verbPrint(f"Checksum: {checksum}", 5)
            for _ in range(self._majorityNum):
                self._channel.sendMessage(self._id, recieverID, msgString)
        else:
            self._channel.sendMessage(self._id, recieverID, msgString)
