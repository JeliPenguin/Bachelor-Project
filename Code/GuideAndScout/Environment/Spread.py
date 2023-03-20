from Environment.CommGridEnv import CommGridEnv
from Agents.CommAgent import CommAgent
from typing import Tuple
from Environment.EnvUtilities import *
import numpy as np


class Spread(CommGridEnv):
    """
    An envrionment simulating the MPE Simple Spread problem in gridworld.
    There would be N Agents (Guide + N-1 scouts) and N-1 landmarks, scouts must learn to cover all landmarks
    with minimal time. Agents would be penalised for colliding into each other or with the wall.
    https://pettingzoo.farama.org/environments/mpe/simple_spread/
    """
    def __init__(self, row: int, column: int, agents: Tuple[CommAgent], treatNum, render=True, numpify=True) -> None:
        super().__init__(row, column, agents, treatNum, render, numpify, envName="Spread")
        self._covered = 0

    def distanceToTreats(self) -> float:
        """ Returns sum of minial distances from treats """
        distances = []
        for i in range(GUIDEID+1, self._agentNum):
            minDist = np.inf
            for treatLoc in self._treatLocations:
                crtAgentState = self._agentInfo[i]["state"]
                euclidDistance = np.sqrt(
                    (crtAgentState[0] - treatLoc[0])**2 + (crtAgentState[1] - treatLoc[1])**2)
                minDist = min(euclidDistance,minDist)
            distances.append(minDist)
            self._agentInfo[i]["minDist"] = minDist
        return sum(distances)

    def rewardFunction(self, sPrimes, ateTreatRecord, doneRecord):
        # """
        # Calculate reward in simulatneous manner and returns a unified team reward
        # Cannot set reward > 128 and reward < -129 due to message encodings
        # """

        reward = 0

        # # Penalised for taking extra timesteps
        # reward += time_penalty
        # # Penalise for remaining treats
        # reward += treat_penalty * self._treatCount

        # In worst case, given 5x5 grid for 2 scouts, distances = 120
        distances = int(self.distanceToTreats() * 10)
        reward -= distances

        return reward

    def agentStep(self, agentID: int, action: int):
        """
        Taking one step for an agent specified by its ID
        """
        s = self._agentInfo[agentID]["state"]
        sPrime = self.takeAction(s, action)
        agentSymbol = self._agentInfo[agentID]["symbol"]
        done = False
        if s != sPrime:
            self._agentInfo[agentID]["state"] = sPrime
            if s in self._treatLocations:
                self._grid[s[0]][s[1]] = TREAT
                self._covered -= 1
            else:
                self._grid[s[0]][s[1]] = EMPTY
            if self._grid[sPrime[0]][sPrime[1]] == TREAT:
                self._covered += 1
            #     self._treatLocations.remove(sPrime)
            #     self._treatCount -= 1
            # done = self._treatCount <= 0
            done = self._covered == self._treatCount
            self._grid[sPrime[0]][sPrime[1]] = agentSymbol
        else:
            # Case collision, hence remained at same state
            pass

        return sPrime, False, done
