import torch
from CommGridEnv import CommGridEnv
from GuideScout import *
from const import *
from joblib import dump,load
from CommChannel import CommChannel
from typing import List

"""
Gridworld with treats, a Guide agent and a Scout agent

Guide agent cannot move but can observe the environment and send messages

Scout agent can move but cannot observe the environment and send messages

Guide and scout need to cooporate through communication to obtain all treats

Communication currently with no noise added
"""

row, column = (8, 8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

treatNum = 2

def train():
    agents = instantiateAgents(treatNum)
    guide = agents[GUIDEID]
    scout = agents[SCOUTID]
    env = CommGridEnv(row, column, agents, treatNum=treatNum,render=False,numpify=True)
    channel = CommChannel(agents)
    guide.setChannel(channel)
    scout.setChannel(channel)
    episodicRewards = []
    num_episode = 5000
    print(f"Running {num_episode} epochs:")
    for eps in range(num_episode):
        # Initialize the environment and get it's state
        
        # State observerd by guide
        numpifiedState = env.reset()
        stateTensor = torch.tensor(numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        episodicReward = 0
        while not done:
            channel.sendMessage(GUIDEID,SCOUTID,stateTensor,"state")
            #Scout chooses epsilon greedy action solely on recieved message
            scoutAction = scout.choose_action()
            guideAction = guide.choose_action()
            actionTensor:torch.Tensor = scoutAction
            actions:List[int] = [scoutAction.item(),guideAction.item()]
            numpifiedSPrime, reward, done, _ = env.step(actions)
            rewardTensor = torch.tensor([reward],dtype=torch.float32, device=device)
            if done:
                sPrimeTensor = None
            else:
                sPrimeTensor = torch.tensor(numpifiedSPrime, dtype=torch.float32, device=device).unsqueeze(0)

            # for agent,actionTensor in zip(agents,actionTensors):
            #     # Store the transition in memory
            #     agent.memorize(stateTensor, actionTensor, sPrimeTensor, rewardTensor)
            #     # Perform one step of the optimization (on the policy network)
            #     agent.optimize()
            channel.sendMessage(GUIDEID,SCOUTID,actionTensor,"action")
            channel.sendMessage(GUIDEID,SCOUTID,rewardTensor,"reward")
            channel.sendMessage(GUIDEID,SCOUTID,sPrimeTensor,"sPrime")
            scout.memorize()
            scout.optimize()

            # Move to the next state
            stateTensor = sPrimeTensor
            episodicReward += reward

        episodicRewards.append(episodicReward)
        print(f"Episode {eps} done, Eps Reward: {episodicReward}")
    dump(scout,"scout")
    dump(guide,"guide")
    dump(guide,"episodicRewards")

def test():
    scout = load("scout")
    guide = load("guide")
    agents = tuple([scout,guide])
    env = CommGridEnv(row, column, agents, treatNum=treatNum,render=True)
    env.reset()
    numpifiedState = env.numpifiedState()
    stateTensor = torch.tensor(numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    while not done:
        actions = []
        for agent in agents:
            action = agent.choose_action(stateTensor)
            actions.append(action.item())

        _, reward, done, _ = env.step(actions)
        numpifiedState = env.numpifiedState()
        stateTensor = torch.tensor(numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)

train()