import numpy as np
import sys
from Environment.EnvUtilities import *
from typing import List, Tuple
from Agents.CommAgent import CommAgent


class CommGridEnv():
    def __init__(self, row: int, column: int, agents: Tuple[CommAgent], treatNum, render=True, numpify=True, envName="Base") -> None:
        self._row = row
        self._column = column
        self._treatNum = treatNum
        self._agents = agents
        self._agentNum = len(agents)
        self._agentSymbol = set([str(agent.getSymbol())
                                for agent in self._agents])
        self._action_space = ACTIONSPACE
        self._state_space = self._row * self._column
        self._toRender = render
        self._toNumpify = numpify
        self._seed = None
        self._envName = envName

    def envName(self):
        return self._envName

    def setSeed(self, seed):
        self._seed = seed
        np.random.seed(self._seed)

    def initGrid(self) -> List[tuple]:
        self._teamReward = None
        self._steps = 0
        self._treatCount = self._treatNum
        self._initLoc = set()
        self._agentInfo = {}
        self._treatLocations = set()
        self._grid = [([EMPTY]*self._column) for _ in range(self._row)]
        self._teamReward = 0
        initState = []
        for _ in range(self._treatNum):
            loc = self.addComponent(TREAT)
            self._treatLocations.add(loc)
        for agent in self._agents:
            loc = self.addComponent(agent.getSymbol())
            initState.append(loc)
            self._agentInfo[agent.getID()] = {
                "state": loc, "last-action": -1, "reward": 0, "symbol": agent.getSymbol()}
        if self._toRender:
            self.render()
            self._toRender = False

        if self._toNumpify:
            return self.numpifiedState()
        return initState

    def addComponent(self, compSymbol: str):
        """ Add specified component to random location on the grid"""
        loc = tuple(np.random.randint(0, self._row, 2))
        while loc in self._initLoc:
            loc = tuple(np.random.randint(0, self._row, 2))
        self._grid[loc[0]][loc[1]] = compSymbol
        self._initLoc.add(loc)
        return loc

    # def numpifiedState(self):
    #     # Returns numpy array of shape (numComponents,row,column) for DQN
    #     # numComponents being number of agents + 1 (treat)
    #     state = np.zeros((self.agentNum+1, self.row, self.column))
    #     layer = 0

    #     for info in self.agentInfo.values():
    #         agentLoc = info["state"]
    #         x = agentLoc[1]
    #         y = agentLoc[0]
    #         state[layer][y][x] = 1
    #         layer += 1

    #     for treatLoc in self.treatLocations:
    #         x = treatLoc[1]
    #         y = treatLoc[0]
    #         state[layer][y][x] = 1

    #     return state
    def numpifiedState(self) -> np.ndarray:
        state = np.zeros((self._agentNum*2+self._treatNum*2,))
        index = 0
        for info in self._agentInfo.values():
            agentLoc = info["state"]
            x = agentLoc[1]
            y = agentLoc[0]
            state[index] = y
            state[index+1] = x
            index += 2

        for treatLoc in self._treatLocations:
            x = treatLoc[1]
            y = treatLoc[0]
            state[index] = y
            state[index+1] = x
            index += 2

        return state

    def takeAction(self, state: tuple, action: int):
        """ 
        Given current state as x,y coordinates and an action, return coordinate of resulting new state
        """
        movement = decodeAction(action)
        newState = transition(state, movement)
        ''' If the new state is outside the grid then remain at same state
            Allows agents to be on same state'''
        if min(newState) < 0 or max(newState) > min(self._row-1, self._column-1) or self._grid[newState[0]][newState[1]] in self._agentSymbol:
            return state
        return newState

    def agentStep(self, agentID: int, action: int):
        raise NotImplementedError

    # def rewardFunc(self, ateTreat: bool, done: bool) -> float:
    #     """
    #         ateTreat: Boolean indicating whether a treat has been eaten
    #         done: Boolean indicating state of the game
    #         Cannot set reward > 128 due to message encodings
    #     """
    #     # if done:
    #     #     return self.doneReward
    #     if ateTreat or done:
    #         return self.treatReward
    #     reward = self.time_penalty
    #     # reward -= self.distanceToTreats()
    #     # Penalise for remaining treats
    #     reward += self.treat_penalty * self.treatCount
    #     return reward

    def rewardFunction(self, sPrimes, ateTreatRecord, doneRecord):
        raise NotImplementedError

    def step(self, actions: List[int]):
        """
        Taking one step for all agents in the environment
        """
        sPrimes: List[tuple] = []
        ateTreatRecord = []
        doneRecord = []
        # Agents take turn to do their action
        # Guide -> Scout1 -> Scout2
        for agentID, agentAction in enumerate(actions):
            self._agentInfo[agentID]["last-action"] = agentAction
            sPrime, ateTreat, done = self.agentStep(agentID, agentAction)
            sPrimes.append(sPrime)
            ateTreatRecord.append(ateTreat)
            doneRecord.append(done)
        self._steps += 1
        self._teamReward = self.rewardFunction(
            sPrimes, ateTreatRecord, doneRecord)
        done = doneRecord[-1]

        if self._toRender:
            self.render()

        if self._toNumpify:
            sPrimes = self.numpifiedState()
        return sPrimes, self._teamReward, done, self._agentInfo

    def write(self, content):
        sys.stdout.write("\r%s" % content)

    def formatGridInfo(self):
        """
        Generateing the environment grid with text symbols
        """
        toWrite = ""
        # toWrite += "="*20+"\n"
        toWrite += "-"*(self._column * 2 + 3) + "\n"
        for row in range(self._row):
            rowContent = "| "
            for column in range(self._column):
                rowContent += self._grid[row][column] + " "
            rowContent += "|\n"
            toWrite += rowContent
        toWrite += "-"*(self._column * 2 + 3)+"\n"
        toWrite += f"Treats: {self._treatCount}"
        toWrite += f"\nTreat Pos: {self._treatLocations}"
        return toWrite

    def formatAgentInfo(self):
        """
        Generating detailed information for each agent
        """
        toWrite = f"Team Reward: {self._teamReward}\n"
        for agentID in range(self._agentNum):
            agentState = self._agentInfo[agentID]["state"]
            lastAction = ACTIONSPACE[self._agentInfo[agentID]["last-action"]]
            symbol = self._agentInfo[agentID]["symbol"]
            # sumDist = None
            # indiReward = None
            # if "sumDist" in self._agentInfo[agentID]:
            #     sumDist = self._agentInfo[agentID]["sumDist"]
            # if "indiReward" in self._agentInfo[agentID]:
            #     indiReward = self._agentInfo[agentID]["indiReward"]
            aType = "Guide"
            if symbol != "G":
                aType = "Scout"
            toWrite += f"{aType}: {symbol}, Current State: {agentState}, Last chose action: {lastAction}"
            # if agentID != GUIDEID:
            #     toWrite += f", Sum dist: {sumDist}, Indi Reward: {indiReward}"
            toWrite += "\n"
        return toWrite

    def render(self, inplace=False):
        toWrite = f"Step: {self._steps}\n{self.formatGridInfo()}\n{self.formatAgentInfo()}"
        if inplace:
            sys.stdout.write("\r%s" % toWrite)
            sys.stdout.flush()
        else:
            print(toWrite)
        # time.sleep(1)

    def reset(self):
        return self.initGrid()
