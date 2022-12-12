from GridEnv import GridEnv,GridAction
from Agent import GridAgent

# In discrete gridworld environment
row,column = (4,4)
num_episode = 50000
agentNum = 3
agents = []
for i in range(agentNum):
    agent = GridAgent(i,(row,column),list(GridAction))
    agents.append(agent)

env = GridEnv(row, column, agents,treatNum=1)

def testRun():
    done = False
    env.reset()
    while not done:
        actions = []
        for agent in agents:
            actions.append(agent.choose_random_action())
        _, _, done = env.step(actions)
        env.render()

testRun()
