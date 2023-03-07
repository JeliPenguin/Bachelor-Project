import numpy as np
from Environment.EnvUtilities import decodeAction, transition
from const import verbPrint

class MessageRecoverer():
    def __init__(self, id, totalTreatNum) -> None:
        self._anchoredGuidePos = None
        self._anchoredTreatPos = None
        self._id = id
        self._totalTreatNum = totalTreatNum

    def computeGuideAnchor(self, anchorVal):
        if self._anchoredGuidePos is None:
            self._anchoredGuidePos = anchorVal

    def computeTreatAnchor(self, anchorVal):
        self._anchoredTreatPos = anchorVal

    def attemptRecovery(self, senderID, parse, recievedHistory, action):
        # Attempt in recovering original message by looking at history of correctly received messages
        # Could be checksum got corrupted, msg got corrupted or both
        fixedState = parse["state"]
        fixedReward = parse["reward"]
        fixedsPrime = parse["sPrime"]
        hasSPrime = fixedsPrime is not None
        treatStart = len(fixedState) - 2*self._totalTreatNum

        # hardcoded for 2 scouts environment
        if self._id == 1:
            otherAgentID = 2
        else:
            otherAgentID = 1
            
        # Guide, treat positions are fixed hence can be recovered directly
        if self._anchoredGuidePos is not None:
            guidePos = self._anchoredGuidePos
            fixedState[0:2] = guidePos
            if hasSPrime:
                fixedsPrime[0:2] = guidePos

        if self._anchoredTreatPos is not None:
            treatPos = self._anchoredTreatPos
            fixedState[treatStart:] = treatPos
            if hasSPrime:
                fixedsPrime[treatStart:] = treatPos

        if recievedHistory:
            recentRecord = recievedHistory[-1]
            recentState = recentRecord["state"]
            recentsPrime = recentRecord["sPrime"]
            history = recentsPrime
            if history is None:
                history = recentState
            # Current scout's state and s' can be estimated using previous s and action
            fixedState[self._id*2:self._id*2 +
                       2] = history[self._id]
            myState = history[self._id]
            actionTaken = decodeAction(action[0])
            mySPrime = np.array(
                transition(tuple(myState), actionTaken), dtype=np.uint8)
            if hasSPrime:
                fixedsPrime[self._id*2:self._id*2 + 2] = mySPrime

            # resolving other agents' s and s'
            otherAgentStates = []
            startRecall = False
            for record in recievedHistory:
                startRecall = startRecall or record["checksum"]
                if startRecall:
                    agentS = record["state"][otherAgentID]
                    agentSPrime = record["sPrime"]
                    if agentSPrime is not None:
                        agentSPrime = agentSPrime[otherAgentID]
                    otherAgentStates.append([agentS,agentSPrime])
            verbPrint(f"Agent: {self._id}",-1)
            verbPrint("Other Agent States:",-1)
            verbPrint(otherAgentStates,-1)

        return {
            "state": fixedState,
            "reward": fixedReward,
            "sPrime": fixedsPrime
        }
