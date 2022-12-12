
from CommGridEnv import CommGridEnv
from CommAgent import ScoutAgent,GuideAgent
from const import *

row,column = (4,4)
num_episode = 1000
agentNum = 2
scout = ScoutAgent(0,(row,column),ACTIONSPACE)
guide = GuideAgent(1,(row,column),ACTIONSPACE)
agents = [guide,scout]
env = CommGridEnv(row, column, agents,treatNum=1)

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
        s, r, done = env.step(actions)
        env.render()

testCommRun()