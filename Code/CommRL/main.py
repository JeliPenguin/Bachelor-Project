
from CommGridEnv import CommGridEnv
from CommAgent import ScoutAgent, GuideAgent
from const import *
import torch

row, column = (10, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def testCommRun():
    agent = ScoutAgent(0, row*column*2, ACTIONSPACE)
    agents = [agent]
    env = CommGridEnv(row, column, agents, treatNum=2,render=False)
    done = False
    s = env.reset()
    while not done:
        actions = []
        for agent in agents:
            actions.append(agent.choose_random_action().item())
        s, r, done, info = env.step(actions)


def testNumpify():
    agent = ScoutAgent(0, row*column*2, ACTIONSPACE)
    agents = [agent]
    env = CommGridEnv(row, column, agents, treatNum=2,render=True)
    s = env.reset()
    a = env.numpifiedState()
    print(a)

    actions = []
    for agent in agents:
        actions.append(agent.choose_random_action().item())
    s, r, done, info = env.step(actions)
    a = env.numpifiedState()
    print(a)


#testCommRun()
#testNumpify()
