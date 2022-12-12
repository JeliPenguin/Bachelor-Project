import numpy as np
from enum import Enum
import sys
import time

EMPTY = "-"
TREAT = "$"
AGENT = "*"

class GridAction(Enum):
    UP = (-1,0)
    LEFT = (0,-1)
    RIGHT = (0,1)
    DOWN = (1,0)
    STAY = (0,0)

def encodeAction(num):
    mapping = {
        0:GridAction.UP,
        1:GridAction.LEFT,
        2:GridAction.RIGHT,
        3:GridAction.DOWN,
        4:GridAction.STAY
    }
    return mapping[num]


class GridEnv():
    def __init__(self, row, column, agents, treatNum=2) -> None:
        self.row = row
        self.column = column
        self.treatNum = treatNum
        self.agents = agents
        self.agentNum = len(agents)
        self.agentSymbol = [str(i) for i in range(self.agentNum)]
        self.action_space = list(GridAction)
        self.state_space = self.row * self.column
        self.time_penalty = -1
        self.treat_penalty = -1
        self.treatReward = 5
        self.teamReward = None

    def addComponent(self, compSymbol):
        # Add specified component to random location on the grid
        loc = tuple(np.random.randint(0, self.row, 2))
        while loc in self.initLoc:
            loc = tuple(np.random.randint(0, self.row, 2))
        self.grid[loc[0]][loc[1]] = compSymbol
        self.initLoc.add(loc)
        return loc

    def initGrid(self):
        self.done = False
        self.treatCount = self.treatNum
        self.initLoc = set()
        self.agentInfo = {}
        self.treatLocations = []
        self.grid = [([EMPTY]*self.column) for _ in range(self.row)]
        initState = []
        for _ in range(self.treatNum):
            loc = self.addComponent(TREAT)
            self.treatLocations.append(loc)
        for agent in self.agents:
            loc = self.addComponent(self.agentSymbol[agent.id])
            initState.append(loc)
            self.agentInfo[agent.id] = {"state": loc,"last-action":-1,"reward":0}
        return initState

    def distanceToTreats(self):
        # Returns minimum of all agent's euclidean distance to closest treat
        distances = []
        for treatLoc in self.treatLocations:
            sumDist = 0
            for i in range(self.agentNum):
                crtAgentState = self.agentInfo[i]["state"]
                euclidDistance = np.sqrt(
                    (crtAgentState[0] - treatLoc[0])**2 + (crtAgentState[1] - treatLoc[1])**2)
                sumDist += euclidDistance
            distances.append(sumDist)
        return min(distances)

    def rewardFunction(self,ateTreat,done):
        reward = self.time_penalty
        if ateTreat:
            reward += self.treatReward
        if done:
            reward = self.treatReward
        else:
            # Penalise for remaining treats
            reward -= self.distanceToTreats()
            # Penalised for number of remaining treats and wasted time
            reward += self.treat_penalty * self.treatCount
        return reward

    def takeAction(self, state: tuple, action: int):
        def tupleAdd(xs, ys): return tuple(x + y for x, y in zip(xs, ys))
        encodedAction = encodeAction(action)
        newState = tupleAdd(state, encodedAction.value)
        #print(newState, (self.row, self.column))
        # print(action)
        ''' If the new state is outside then remain at same state
            Allows agents to be on same state'''

        if min(newState) < 0 or max(newState) > min(self.row-1, self.column-1) or self.grid[newState[0]][newState[1]] in self.agentSymbol:
            return state
        return newState

    def agentStep(self, agentID: int, action: int):
        s = self.agentInfo[agentID]["state"]
        sPrime = self.takeAction(s, action)
        ateTreat = False
        reward = self.time_penalty
        agentSymbol = self.agentSymbol[agentID]
        if s != sPrime:
            self.agentInfo[agentID]["state"] = sPrime
            self.grid[s[0]][s[1]] = EMPTY
            if self.grid[sPrime[0]][sPrime[1]] == TREAT:
                self.treatLocations.remove(sPrime)
                ateTreat = True
                self.treatCount -= 1

            self.done = self.treatCount <= 0
            self.grid[sPrime[0]][sPrime[1]] = agentSymbol

        reward = self.rewardFunction(ateTreat,self.done)
        self.agentInfo[agentID]["reward"] = reward
        return sPrime, reward

    def step(self, actions: list):
        sPrimes = []
        rewards = []
        for agentID,agentAction in enumerate(actions):
            self.agentInfo[agentID]["last-action"] = agentAction
            sPrime, reward = self.agentStep(agentID, agentAction)
            sPrimes.append(sPrime)
            rewards.append(reward)
        return sPrimes, rewards, self.done

    def write(self,content):
        sys.stdout.write("\r%s"%content)

    def formatGridInfo(self):
        toWrite = "-"*(self.column * 2 + 3) + "\n"
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
        toWrite = ""
        for agentID in range(self.agentNum):
            agentState = self.agentInfo[agentID]["state"]
            lastAction = encodeAction(self.agentInfo[agentID]["last-action"])
            reward = self.agentInfo[agentID]["reward"]
            toWrite += f"Agent ID: {agentID}, Current State: {agentState}, Last chose action: {lastAction}, Reward: {reward}\n"
        return toWrite

    def render(self,inplace=False):
        toWrite = f"{self.formatGridInfo()}\n{self.formatAgentInfo()}"
        if inplace:
            sys.stdout.write("\r%s"%toWrite)
            sys.stdout.flush()
        else:
            print(toWrite)
        #time.sleep(1)

    def reset(self):
        return self.initGrid()
