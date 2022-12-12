
from CommGridEnv import CommGridEnv
from CommAgent import ScoutAgent, GuideAgent
from const import *

row, column = (3, 3)
num_episode = 1000
scout = ScoutAgent(0, (row, column), ACTIONSPACE)
guide = GuideAgent(1, (row, column), ACTIONSPACE)
agents = [scout, guide]
env = CommGridEnv(row, column, agents, treatNum=2)


def train():
    num_episode = 5000
    for eps in range(num_episode):
        done = False
        s = env.reset()
        env.render()
        while not done:
            # actions = []
            # for agent in agents:
            #     actions.append(agent.choose_action(s))
            # s, r, done = env.step(actions)
            # env.render()
            done = True


def testCommRun():
    done = False
    s = env.reset()
    env.render()
    while not done:
        actions = []
        for agent in agents:
            actions.append(agent.choose_action(s))
        s, r, done, info = env.step(actions)
        env.render()


def testNumpify():
    s = env.reset()
    a = env.numpifiedGrid()
    env.render()
    print(a)

    actions = []
    for agent in agents:
        actions.append(agent.choose_action(s))
    s, r, done, info = env.step(actions)
    a = env.numpifiedGrid()
    env.render()
    print(a)


testCommRun()
# testNumpify()
