from const import *
from Agents.CommAgent import CommAgent
import torch
import numpy as np
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GUIDEID = 0


class ScoutAgent(CommAgent):
    def __init__(self, id, obs_dim, actionSpace, noiseHandling, epsDecay) -> None:
        super().__init__(id, obs_dim, actionSpace,
                         noiseHandling=noiseHandling, epsDecay=epsDecay)
        self._symbol = str(id)
        self._falseCount = 0
        self._recieveCount = 0
        self._MASampleSize = 3
        self._falseLimit = 0.4

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

    def majorityAdjust(self,checksumCheck):
        self._recieveCount += 1
        if not checksumCheck:
            self._falseCount += 1
        if self._recieveCount >= self._MASampleSize:
            falseRatio = self._falseCount / self._recieveCount
            if falseRatio >= self._falseLimit:
                self._majorityNum += 2
                if getVerbose() >= 4:
                    print("Increased majority num, now is : ",self._majorityNum)
                self._falseCount = 0
                self._recieveCount = 0
                self.broadcastMajority()

    def recieveNoisyMessage(self, senderID: int, msg):
        # Assumes message recieved in inorder
        self._majorityMem.append(msg)
        if len(self._majorityMem) == self._majorityNum:
            # Majority vote the messages
            msg = self.majorityVote()
            msgChecksumPass = self.checkChecksum(msg)
            if getVerbose() >= 2:
                print("Checksum check: ", msgChecksumPass)
            msg = np.packbits(msg[self._k:])
            decoded = self.decodeMessage(msg)
            if getVerbose() >= 3:
                print("Before recovery: ",decoded)
            if not msgChecksumPass:
                # If majority voting unable to fix noise, attempt recovery of message using previous history
                decoded = self.attemptRecovery(senderID, decoded)
            self.majorityAdjust(msgChecksumPass)
            if getVerbose() >= 3:
                print("Anchors:")
                print(f"Guide Pos: {self._anchoredGuidePos}")
                print(f"Treat Pos: {self._anchoredTreatPos}")
            self.storeRecievedMessage(senderID, decoded, msgChecksumPass)

    def broadcastMajority(self):
        self.broadcastSignal(np.array([0]))
    
    def recieveBroadcast(self, signal):
        self._majorityNum+=2


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
        self._majorityNum+=2
