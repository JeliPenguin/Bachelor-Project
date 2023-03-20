from Environment.CommGridEnv import CommGridEnv
from Agents.CommAgent import CommAgent
from typing import Tuple
from Environment.EnvUtilities import *
import numpy as np


class FindingTreat(CommGridEnv):
    """
    A gridworld with N Agents (A Guide + N-1 scouts) and M treats, scouts must learn to cooperate and 
    find all treats with minimal time.
    """
    def __init__(self, row: int, column: int, agents: Tuple[CommAgent], treatNum, render=True, numpify=True) -> None:
        super().__init__(row, column, agents, treatNum,
                         render, numpify, envName="Finding Treat")

    def rewardFunction(self, ateTreatRecord, doneRecord):
        """ 
        Calculate reward in simulatneous manner and returns a unified team reward
        Cannot set reward > 128 due to message encodings
        """
        time_penalty = -1
        treat_penalty = -2
        treatReward = 10

        reward = 0
        for ateTreat in ateTreatRecord:
            if ateTreat:
                reward += treatReward

        # Penalised for taking extra timesteps
        reward += time_penalty
        # Penalise for remaining treats
        reward += treat_penalty * self._treatCount

        return reward
    
    def agentStep(self, agentID: int, action: int):
        """
        Taking one step for an agent specified by its ID
        """
        s = self._agentInfo[agentID]["state"]
        sPrime,_ = self.takeAction(s, action)
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
