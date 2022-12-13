import torch
from CommGridEnv import CommGridEnv
from CommAgent import ScoutAgent, GuideAgent
from const import *
from joblib import dump,load

row, column = (10, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agentNum = 1
treatNum = 2
n_obs = 2 * (agentNum + treatNum)

def train():
    scout = ScoutAgent(0, n_obs, ACTIONSPACE)
    guide = GuideAgent(1,n_obs,ACTIONSPACE)
    agents = [scout,guide]
    env = CommGridEnv(row, column, agents, treatNum=treatNum,render=False)
    episodicRewards = []
    num_episode = 5000

    for eps in range(num_episode):
        # Initialize the environment and get it's state
        env.reset()
        numpifiedState = env.numpifiedState()
        stateTensor = torch.tensor(numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        episodicReward = 0
        while not done:
            actions = []
            for agent in agents:
                action = agent.choose_action(stateTensor)
                actions.append(action.item())

            _, reward, done, _ = env.step(actions)
            numpifiedSPrime = env.numpifiedState()
            rewardTensor = torch.tensor([reward],dtype=torch.float32, device=device)
            if done:
                sPrimeTensor = None
            else:
                sPrimeTensor = torch.tensor(numpifiedSPrime, dtype=torch.float32, device=device).unsqueeze(0)

            for agent in agents:
                # Store the transition in memory
                agent.memory.push(stateTensor, action, sPrimeTensor, rewardTensor)
                # Perform one step of the optimization (on the policy network)
                agent.optimize()

            # Move to the next state
            stateTensor = sPrimeTensor
            episodicReward += sum(reward)

        episodicRewards.append(episodicReward)
        print(f"Episode {eps} done, Eps Reward: {episodicReward}")
    dump(agent,"singleAgent")
    dump(episodicRewards,"episodicRewards")

def test():
    pass
