import numpy as np
import sys
from const import *
from typing import List, Tuple
from Agents.CommAgent import CommAgent
from Agents.GuideScout import GUIDEID


def decodeAction(num: int):
    mapping = {
        0: (-1, 0),
        1: (0, -1),
        2: (0, 1),
        3: (1, 0),
        4: (0, 0)
    }
    return mapping[num]


class CommGridEnv():
    def __init__(self, row: int, column: int, agents: Tuple[CommAgent], treatNum, render=True, numpify=True) -> None:
        self.row = row
        self.column = column
        self.treatNum = treatNum
        self.agents = agents
        self.agentNum = len(agents)
        self.agentSymbol = set([str(agent.symbol) for agent in self.agents])
        self.action_space = ACTIONSPACE
        self.state_space = self.row * self.column
        self.time_penalty = -1
        self.treat_penalty = -5
        self.treatReward = 10
        self.doneReward = 50
        self.teamReward = None
        self.toRender = render
        self.toNumpify = numpify

    def addComponent(self, compSymbol: str):
        """ Add specified component to random location on the grid"""
        loc = tuple(np.random.randint(0, self.row, 2))
        while loc in self.initLoc:
            loc = tuple(np.random.randint(0, self.row, 2))
        self.grid[loc[0]][loc[1]] = compSymbol
        self.initLoc.add(loc)
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
        state = np.zeros((self.agentNum*2+self.treatNum*2,))
        index = 0
        for info in self.agentInfo.values():
            agentLoc = info["state"]
            x = agentLoc[1]
            y = agentLoc[0]
            state[index] = y
            state[index+1] = x
            index += 2

        for treatLoc in self.treatLocations:
            x = treatLoc[1]
            y = treatLoc[0]
            state[index] = y
            state[index+1] = x
            index += 2

        return state

    def initGrid(self) -> List[tuple]:
        self.done = False
        self.steps = 0
        self.treatCount = self.treatNum
        self.initLoc = set()
        self.agentInfo = {}
        self.treatLocations = set()
        self.grid = [([EMPTY]*self.column) for _ in range(self.row)]
        initState = []
        for _ in range(self.treatNum):
            loc = self.addComponent(TREAT)
            self.treatLocations.add(loc)
        for agent in self.agents:
            loc = self.addComponent(agent.symbol)
            initState.append(loc)
            self.agentInfo[agent.id] = {
                "state": loc, "last-action": -1, "reward": 0, "symbol": agent.symbol}
        if self.toRender:
            self.render()

        if self.toNumpify:
            return self.numpifiedState()
        return initState

    def distanceToTreats(self) -> float:
        """ Returns maximum of all scout's euclidean distance to closest treat """
        distances = []
        for i in range(GUIDEID+1, self.agentNum):
            sumDist = 0
            for treatLoc in self.treatLocations:
                crtAgentState = self.agentInfo[i]["state"]
                euclidDistance = np.sqrt(
                    (crtAgentState[0] - treatLoc[0])**2 + (crtAgentState[1] - treatLoc[1])**2)
                sumDist += euclidDistance
            distances.append(sumDist)
            self.agentInfo[i]["sumDist"] = sumDist
        # return 0
        return int(max(distances))

    def rewardFunction(self, ateTreat: bool, done: bool) -> float:
        """
            ateTreat: Boolean indicating whether a treat has been eaten
            done: Boolean indicating state of the game
            Cannot set reward > 128 due to message encodings
        """
        # if done:
        #     return self.doneReward
        if ateTreat or done:
            return self.treatReward
        reward = self.time_penalty
        # reward -= self.distanceToTreats()
        # Penalise for remaining treats
        reward += self.treat_penalty * self.treatCount
        return reward

    def takeAction(self, state: tuple, action: int):
        """ 
        Given current state as x,y coordinates and an action, return coordinate of resulting new state
        """
        def tupleAdd(xs, ys): return tuple(x + y for x, y in zip(xs, ys))
        movement = decodeAction(action)
        newState = tupleAdd(state, movement)
        ''' If the new state is outside the grid then remain at same state
            Allows agents to be on same state'''
        if min(newState) < 0 or max(newState) > min(self.row-1, self.column-1) or self.grid[newState[0]][newState[1]] in self.agentSymbol:
            return state
        return newState

    def agentStep(self, agentID: int, action: int):
        """
        Taking one step for an agent specified by its ID
        """
        s = self.agentInfo[agentID]["state"]
        sPrime = self.takeAction(s, action)
        ateTreat = False
        agentSymbol = self.agentInfo[agentID]["symbol"]
        if s != sPrime:
            self.agentInfo[agentID]["state"] = sPrime
            self.grid[s[0]][s[1]] = EMPTY
            if self.grid[sPrime[0]][sPrime[1]] == TREAT:
                self.treatLocations.remove(sPrime)
                ateTreat = True
                self.treatCount -= 1

            self.done = self.treatCount <= 0
            self.grid[sPrime[0]][sPrime[1]] = agentSymbol

        reward = self.rewardFunction(ateTreat, self.done)
        self.agentInfo[agentID]["indiReward"] = reward

        return sPrime, reward

    def step(self, actions: List[int]):
        """
        Taking one step for all agents in the environment
        """
        sPrimes: List[tuple] = []
        rewards: List[float] = []
        for agentID, agentAction in enumerate(actions):
            self.agentInfo[agentID]["last-action"] = agentAction
            sPrime, reward = self.agentStep(agentID, agentAction)
            sPrimes.append(sPrime)
            rewards.append(reward)
        self.steps += 1
        self.teamReward = sum(rewards[1:])

        if self.toRender:
            self.render()

        if self.toNumpify:
            sPrimes = self.numpifiedState()
        return sPrimes, self.teamReward, self.done, self.agentInfo

    def write(self, content):
        sys.stdout.write("\r%s" % content)

    def formatGridInfo(self):
        """
        Generateing the environment grid with text symbols
        """
        toWrite = ""
        # toWrite += "="*20+"\n"
        toWrite += "-"*(self.column * 2 + 3) + "\n"
        for row in range(self.row):
            rowContent = "| "
            for column in range(self.column):
                rowContent += self.grid[row][column] + " "
            rowContent += "|\n"
            toWrite += rowContent
        toWrite += "-"*(self.column * 2 + 3)+"\n"
        toWrite += f"Treats: {self.treatCount}"
        return toWrite

    def formatAgentInfo(self):
        """
        Generating detailed information for each agent
        """
        toWrite = f"Team Reward: {self.teamReward}\n"
        for agentID in range(self.agentNum):
            agentState = self.agentInfo[agentID]["state"]
            lastAction = ACTIONSPACE[self.agentInfo[agentID]["last-action"]]
            symbol = self.agentInfo[agentID]["symbol"]
            sumDist = None
            indiReward = None
            if "sumDist" in self.agentInfo[agentID]:
                sumDist = self.agentInfo[agentID]["sumDist"]
            if "indiReward" in self.agentInfo[agentID]:
                indiReward = self.agentInfo[agentID]["indiReward"]
            aType = "Guide"
            if symbol != "G":
                aType = "Scout"
            toWrite += f"{aType}: {symbol}, Current State: {agentState}, Last chose action: {lastAction}, Sum dist: {sumDist}"
            if agentID != GUIDEID:
                toWrite += f", Indi Reward: {indiReward}"
            toWrite += "\n"
        return toWrite

    def render(self, inplace=False):
        toWrite = f"Step: {self.steps}\n{self.formatGridInfo()}\n{self.formatAgentInfo()}"
        if inplace:
            sys.stdout.write("\r%s" % toWrite)
            sys.stdout.flush()
        else:
            print(toWrite)
        # time.sleep(1)

    def reset(self):
        return self.initGrid()
