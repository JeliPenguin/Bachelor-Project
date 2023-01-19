import torch
from CommGridEnv import CommGridEnv
from DQN import DQNAgent
from const import *
from joblib import dump, load
import matplotlib.pyplot as plt
from tqdm import tqdm

row, column = (10, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agentNum = 1
treatNum = 2
n_obs = 2 * (agentNum + treatNum)


def train():
    agent = DQNAgent(0, n_obs, ACTIONSPACE)
    agents = [agent]
    env = CommGridEnv(row, column, agents, treatNum=treatNum, render=False)
    num_episode = 5000
    episodicRewards = []
    for eps in tqdm(range(num_episode)):
        # Initialize the environment and get it's state
        env.reset()
        numpifiedState = env.numpifiedState()
        stateTensor = torch.tensor(
            numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        episodicReward = 0
        while not done:
            action = agent.choose_action(stateTensor)
            _, reward, done, _ = env.step([action.item()])
            numpifiedSPrime = env.numpifiedState()
            rewardTensor = torch.tensor(
                [reward], dtype=torch.float32, device=device)
            if done:
                sPrimeTensor = None
            else:
                sPrimeTensor = torch.tensor(
                    numpifiedSPrime, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.memory.push(stateTensor, action, sPrimeTensor, rewardTensor)

            # Move to the next state
            stateTensor = sPrimeTensor

            # Perform one step of the optimization (on the policy network)
            agent.optimize()
            episodicReward += (reward)
            if done:
                break
        episodicRewards.append(episodicReward)
        #print(f"Episode {eps} done, Eps Reward: {episodicReward}")
    dump(agent, "singleAgent")
    dump(episodicRewards, "episodicRewards")


def test():
    agent = load("singleAgent")
    agent.symbol = "A"
    agents = [agent]
    env = CommGridEnv(row, column, agents, treatNum=treatNum, render=True)
    env.reset()
    numpifiedState = env.numpifiedState()
    stateTensor = torch.tensor(
        numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    while not done:
        action = agent.choose_action(stateTensor)
        _, reward, done, _ = env.step([action.item()])
        numpifiedState = env.numpifiedState()
        stateTensor = torch.tensor(
            numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)


def plot():
    history = load("episodicRewards")
    plt.plot(history)
    plt.show()


# train()
test()
plot()
