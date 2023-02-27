import numpy as np
from const import *

class MessageRecoverer():
    def __init__(self,id,totalTreatNum) -> None:
        self._anchoredGuidePos = None
        self._anchoredTreatPos = None
        self._id = id
        self._totalTreatNum = totalTreatNum

    def computeGuideAnchor(self,anchorVal):
        if self._anchoredGuidePos is None:
            self._anchoredGuidePos = anchorVal
        
    def computeTreatAnchor(self,anchorVal):
        self._anchoredTreatPos = anchorVal

    def attemptRecovery(self, senderID, parse,recievedHistory,action):
        # Attempt in recovering original message by looking at history of correctly received messages
        # Could be checksum got corrupted, msg got corrupted or both
        # print("Recovering Message: ")
        fixedState = parse["state"]
        fixedReward = parse["reward"]
        fixedsPrime = parse["sPrime"]
        hasSPrime = fixedsPrime is not None

        def resolveOtherAgents():
            # Detect skip in states
            pass

        # Guide, treat positions are fixed hence can be recovered directly
        # Assumes 2 treats only in environment
        if self._anchoredGuidePos is not None:
            guidePos = self._anchoredGuidePos
            fixedState[0:2] = guidePos
            if hasSPrime:
                fixedsPrime[0:2] = guidePos

        if self._anchoredTreatPos is not None:
            treatStart = len(fixedState) - 2*self._totalTreatNum
            treatPos = self._anchoredTreatPos
            fixedState[treatStart:] = treatPos
            if hasSPrime:
                fixedsPrime[treatStart:] = treatPos

        if recievedHistory:
            recentRecord = recievedHistory[-1][senderID]
            recentState = recentRecord["state"]
            recentsPrime = recentRecord["sPrime"]
            history = recentsPrime
            if history is None:
                history = recentState
            # Current scout's state and s' can be estimated using previous s and action
            fixedState[self._id*2:self._id*2 +
                       2] = history[self._id*2:self._id*2+2]
            myState = history[self._id*2:self._id*2+2]
            actionTaken = decodeAction(action[0])
            mySPrime = np.array(
                transition(tuple(myState), actionTaken), dtype=np.uint8)
            if hasSPrime:
                fixedsPrime[self._id*2:self._id*2 + 2] = mySPrime
            #resolving other agents' (excluding guide) s and s'
            resolveOtherAgents()

        return {
            "state": fixedState,
            "reward": fixedReward,
            "sPrime": fixedsPrime
        }
