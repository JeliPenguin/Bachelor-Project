from Environment.CommGridEnv import CommGridEnv
from Agents.CommAgent import CommAgent
from typing import Tuple
from Environment.EnvUtilities import *
import numpy as np


class FindingTreat(CommGridEnv):
    def __init__(self, row: int, column: int, agents: Tuple[CommAgent], treatNum, render=True, numpify=True) -> None:
        super().__init__(row, column, agents, treatNum,
                         render, numpify, envName="Finding Treat")

    def distanceToTreats(self) -> float:
        """ Returns maximum of all scout's euclidean distance to closest treat """
        distances = []
        for i in range(GUIDEID+1, self._agentNum):
            sumDist = 0
            for treatLoc in self._treatLocations:
                crtAgentState = self._agentInfo[i]["state"]
                euclidDistance = np.sqrt(
                    (crtAgentState[0] - treatLoc[0])**2 + (crtAgentState[1] - treatLoc[1])**2)
                sumDist += euclidDistance
            distances.append(sumDist)
            self._agentInfo[i]["sumDist"] = sumDist
        # return 0
        return int(max(distances))

    def rewardFunction(self, sPrimes, ateTreatRecord, doneRecord):
        """ 
        Calculate reward in simulatneous manner and returns a unified team reward
        ateTreat: Boolean indicating whether a treat has been eaten
        done: Boolean indicating state of the game
        Cannot set reward > 128 due to message encodings
        """
        time_penalty = -1
        treat_penalty = -2
        treatReward = 10
        # doneReward = 50

        # if doneRecord[-1]:
        #     return doneReward

        reward = 0
        for ateTreat in ateTreatRecord:
            if ateTreat:
                reward += treatReward

        # Penalised for taking extra timesteps
        reward += time_penalty
        # Penalise for remaining treats
        reward += treat_penalty * self._treatCount
        # reward -= self.distanceToTreats()

        return reward
    
    def agentStep(self, agentID: int, action: int):
        """
        Taking one step for an agent specified by its ID
        """
        s = self._agentInfo[agentID]["state"]
        sPrime = self.takeAction(s, action)
        ateTreat = False
        agentSymbol = self._agentInfo[agentID]["symbol"]
        done = False
        if s != sPrime:
            self._agentInfo[agentID]["state"] = sPrime
            self._grid[s[0]][s[1]] = EMPTY
            if self._grid[sPrime[0]][sPrime[1]] == TREAT:
                self._treatLocations.remove(sPrime)
                ateTreat = True
                self._treatCount -= 1
            done = self._treatCount <= 0
            self._grid[sPrime[0]][sPrime[1]] = agentSymbol

        return sPrime, ateTreat, done
