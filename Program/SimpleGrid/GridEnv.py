import numpy as np
from Enums import GridSymbol, Action


class GridEnv():
    def __init__(self, row=5, column=5, treatNum=2) -> None:
        self.row = row
        self.column = column
        self.treatNum = treatNum
        self.action_space = 4
        self.state_space = self.row * self.column
        self.done = False
        self.initLoc = set()
        self.agentInfo = {}
        self.grid = None
        self.initGrid()

    def addComponent(self, compType):
        # Add specified component to random location on the grid
        loc = tuple(np.random.randint(0, self.row, 2))
        while loc in self.initLoc:
            loc = tuple(np.random.randint(0, self.row, 2))
        self.grid[loc[0]][loc[1]] = compType
        self.initLoc.add(loc)
        return loc

    def initGrid(self):
        self.grid = [([GridSymbol.EMPTY]*self.column) for _ in range(self.row)]
        for _ in range(self.treatNum):
            self.addComponent(GridSymbol.TREAT)

    def addAgent(self, agentID):
        loc = self.addComponent(GridSymbol.AGENT)
        self.agentInfo[agentID] = {"state": loc}

    def takeAction(self, state: tuple, action: Action):
        # If the new state has an agent then remain at same state
        def tupleAdd(xs, ys): return tuple(x + y for x, y in zip(xs, ys))
        newState = tupleAdd(state, action.value)
        print(newState, (self.row,self.column))
        if newState < (0, 0) or newState > (self.row-1, self.column-1) or self.grid[newState[0]][newState[1]] == GridSymbol.AGENT:
            return state
        return newState

    def step(self, agentID, action):
        s = self.agentInfo[agentID]["state"]
        sPrime = self.takeAction(s, action)
        reward = -1
        if s != sPrime:
            self.agentInfo[agentID]["state"] = sPrime
            self.grid[s[0]][s[1]] = GridSymbol.EMPTY
            if self.grid[sPrime[0]][sPrime[1]] == GridSymbol.TREAT:
                reward = 1
                self.treatNum -= 1
                if not self.treatNum:
                    self.done = True
            self.grid[sPrime[0]][sPrime[1]] = GridSymbol.AGENT
        return sPrime, reward, self.done

    def render(self):
        print("-"*(self.column * 2 + 3))
        for row in range(self.row):
            rowContent = "| "
            for column in range(self.column):
                rowContent += str(self.grid[row][column].value) + " "
            rowContent += "|"
            print(rowContent)
        print("-"*(self.column * 2 + 3))

    def reset(self):
        pass


env = GridEnv()
env.addAgent(1)
env.render()
env.step(1,Action.DOWN)
env.render()
