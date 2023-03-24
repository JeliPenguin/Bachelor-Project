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

    def initGrid(self):
        self._covered = 0
        return super().initGrid()

    def distanceToTreats(self) -> float:
        """ Returns sum of minial distances from treats """
        distances = 0
        for i in range(GUIDEID+1, self._agentNum):
            minDist = np.inf
            for treatLoc in self._treatLocations:
                crtAgentState = self._agentInfo[i]["state"]
                euclidDistance = np.sqrt(
                    (crtAgentState[0] - treatLoc[0])**2 + (crtAgentState[1] - treatLoc[1])**2)
                minDist = min(euclidDistance, minDist)
            distances += (minDist)
            self._agentInfo[i]["minDist"] = minDist
        return distances

    def rewardFunction(self, collisionRecord, doneRecord):
        # """
        # Calculate reward in simulatneous manner and returns a unified team reward
        # Cannot set reward > 128 and reward < -129 due to message encodings
        # """

        reward = 0
        collisionPenalty = -2
        # Penalised for collision

        for collision in collisionRecord[1:]:
            if collision:
                reward += collisionPenalty

        distances = -int(self.distanceToTreats() * 10)
        # In worst case, given 5x5 grid for 2 scouts, distances = -120 and collisions = -4
        reward += distances

        return reward

    def agentStep(self, agentID: int, action: int):
        """
        Taking one step for an agent specified by its ID
        """
        s = self._agentInfo[agentID]["state"]
        sPrime, collision = self.takeAction(s, action)
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

            self._grid[sPrime[0]][sPrime[1]] = agentSymbol
        done = self._covered == self._treatCount
        return sPrime, collision, done

    def additionalAgentInfo(self, agentID):
        toWrite = ""
        minDist = None
        if "minDist" in self._agentInfo[agentID]:
            minDist = self._agentInfo[agentID]["minDist"]
        if agentID != GUIDEID:
            toWrite += f", min dist: {minDist}"

        return toWrite
