import numpy as np
from Enums import GridSymbol, Action, encodeAction
from Agent import Agent


class GridEnv():
    def __init__(self, row, column, agentNum, treatNum=2) -> None:
        self.row = row
        self.column = column
        self.treatNum = treatNum
        self.agentNum = agentNum
        self.action_space = len(list(Action))
        self.state_space = self.row * self.column
        self.time_penalty = -1
        self.treat_penalty = -1
        self.treatReward = 5

    def addComponent(self, compType):
        # Add specified component to random location on the grid
        loc = tuple(np.random.randint(0, self.row, 2))
        while loc in self.initLoc:
            loc = tuple(np.random.randint(0, self.row, 2))
        self.grid[loc[0]][loc[1]] = compType
        self.initLoc.add(loc)
        return loc

    def initGrid(self):
        self.done = False
        self.treatCount = self.treatNum
        self.initLoc = set()
        self.agentInfo = {}
        self.treatLocations = []
        self.grid = [([GridSymbol.EMPTY]*self.column) for _ in range(self.row)]
        initState = []
        for _ in range(self.treatNum):
            loc = self.addComponent(GridSymbol.TREAT)
            self.treatLocations.append(loc)
        for i in range(self.agentNum):
            loc = self.addComponent(GridSymbol.AGENT)
            agent = Agent(i, (self.row, self.column))
            initState.append(loc)
            self.agentInfo[agent.id] = {"agent": agent, "state": loc}
        return initState

    def takeAction(self, state: tuple, action: int):
        def tupleAdd(xs, ys): return tuple(x + y for x, y in zip(xs, ys))
        encodedAction = encodeAction(action)
        newState = tupleAdd(state, encodedAction.value)
        #print(newState, (self.row, self.column))
        # print(action)
        ''' If the new state is outside then remain at same state
            Allows agents to be on same state'''
        if min(newState) < 0 or max(newState) > min(self.row-1, self.column-1):
            return state
        return newState

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

    def agentStep(self, agentID: int, action: int):
        s = self.agentInfo[agentID]["state"]
        sPrime = self.takeAction(s, action)
        reward = self.time_penalty
        if s != sPrime:
            self.agentInfo[agentID]["state"] = sPrime
            self.grid[s[0]][s[1]] = GridSymbol.EMPTY
            if self.grid[sPrime[0]][sPrime[1]] == GridSymbol.TREAT:
                self.treatLocations.remove(sPrime)
                reward += self.treatReward
                self.treatCount -= 1

            if self.treatCount <= 0:
                # All treats eaten
                reward = self.treatReward
                self.done = True
            else:
                # Penalise for remaining treats
                reward -= self.distanceToTreats()
                # Penalised for number of remaining treats and wasted time
                reward += self.treat_penalty * self.treatCount
            self.grid[sPrime[0]][sPrime[1]] = GridSymbol.AGENT
        return sPrime, reward

    def step(self, actions: list):
        sPrimes = []
        rewards = []
        done = False
        for i in range(self.agentNum):
            sPrime, reward, agentDone = self.agentStep(i, actions[i])
            sPrimes.append(sPrime)
            rewards.append(reward)
            done = done or agentDone
        return sPrimes, rewards, done

    def render(self):
        print("-"*(self.column * 2 + 3))
        for row in range(self.row):
            rowContent = "| "
            for column in range(self.column):
                rowContent += str(self.grid[row][column].value) + " "
            rowContent += "|"
            print(rowContent)
        print("-"*(self.column * 2 + 3))
        print("Treats: ", self.treatCount)

    def reset(self):
        return self.initGrid()
