from GridEnv import GridEnv
from Agent import Agent
import numpy as np
from Enums import Action

# In discrete gridworld environment
row = 8
column = 8
num_episode = 50000
agentNum = 3
env = GridEnv(row, column, agentNum)


def testRun():
    done = False
    env.reset()
    while not done:
        actions = np.random.choice([0, 1, 2, 3, 4], 2)
        _, _, done = env.step(actions)
        env.render()


def IQLTrain():
    # Implementing independant Q-Learning
    displayCount = 500
    for eps in range(num_episode):
        steps = 0
        env.reset()
        if eps % displayCount == 0:
            print("-----------------------------------")
            print("Episode ", eps)
        if eps == num_episode - 1:
            env.render()
        while not env.done:
            for i in range(agentNum):
                crtAgent = env.agentInfo[i]["agent"]
                crtState = env.agentInfo[i]["state"]
                action = crtAgent.choose_eps_action(crtState)
                sPrime, reward = env.agentStep(i, action)
                # print(sPrimes,rewards,done)
                if not env.done:
                    q_error = reward + crtAgent.discount * \
                        crtAgent.maxQVal(sPrime) - \
                        crtAgent.q_table[crtState][action]
                    crtAgent.q_table[crtState][action] = crtAgent.q_table[crtState][action] + \
                        crtAgent.learning_rate * q_error
            if eps == num_episode - 1:
                env.render()
            steps += 1
        for i in range(agentNum):
            crtAgent = env.agentInfo[i]["agent"]
            if crtAgent.epsilon > 0.05 and (eps % 100 == 0):
                crtAgent.epsilon *= crtAgent.epsilon_decay
        if eps % displayCount == 0:
            print(f"Episode {eps} finished with {steps} steps")
            print("--------------------------------------")


IQLTrain()
# testRun()
