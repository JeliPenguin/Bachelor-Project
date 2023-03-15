from Environment.CommGridEnv import CommGridEnv
from Agents.CommAgent import CommAgent
from typing import Tuple

class FindingTreat(CommGridEnv):
    def __init__(self, row: int, column: int, agents: Tuple[CommAgent], treatNum, render=True, numpify=True) -> None:
        super().__init__(row, column, agents, treatNum, render, numpify,envName="Finding Treat")

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