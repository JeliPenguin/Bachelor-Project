import torch
from CommGridEnv import CommGridEnv
from GuideScout import *
from const import *
from joblib import dump, load
from CommChannel import CommChannel
from typing import List
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Run():
    def __init__(self, envSetting) -> None:
        self.row = envSetting["row"]
        self.column = envSetting["column"]
        self.treatNum = envSetting["treatNum"]
        self.noised = envSetting["noised"]

    def setup(self, setupType):
        render = setupType != "train"
        if setupType == "train" or setupType == "rand":
            agents = instantiateAgents(self.treatNum)
            guide = agents[GUIDEID]
            scout = agents[SCOUTID]
        else:
            scout = load("scout")
            guide = load("guide")
            agents = tuple([scout, guide])
        env = CommGridEnv(self.row, self.column, agents, self.treatNum,
                          render)
        channel = CommChannel(agents, self.noised)
        guide.setChannel(channel)
        scout.setChannel(channel)
        return guide, scout, env, channel

    def doStep(self, guide, scout, channel, env, stateTensor):
        channel.sendMessage(GUIDEID, SCOUTID, stateTensor, "state")
        # Scout chooses epsilon greedy action solely on recieved message
        scoutAction = scout.choose_action()
        guideAction = guide.choose_action()
        actionTensor: torch.Tensor = scoutAction
        actions: List[int] = [scoutAction.item(), guideAction.item()]
        numpifiedSPrime, reward, done, info = env.step(actions)
        rewardTensor = torch.tensor(
            [reward], dtype=torch.float32, device=device)
        if done:
            sPrimeTensor = None
        else:
            sPrimeTensor = torch.tensor(
                numpifiedSPrime, dtype=torch.float32, device=device).unsqueeze(0)

        # for agent,actionTensor in zip(agents,actionTensors):
        #     # Store the transition in memory
        #     agent.memorize(stateTensor, actionTensor, sPrimeTensor, rewardTensor)
        #     # Perform one step of the optimization (on the policy network)
        #     agent.optimize()
        channel.sendMessage(GUIDEID, SCOUTID, actionTensor, "action")
        channel.sendMessage(GUIDEID, SCOUTID, rewardTensor, "reward")
        channel.sendMessage(GUIDEID, SCOUTID, sPrimeTensor, "sPrime")

        return sPrimeTensor, reward, done, info

    def train(self):
        guide, scout, env, channel = self.setup("train")
        episodicRewards = []
        num_episode = 5000
        print(f"Running {num_episode} epochs:")
        for eps in tqdm(range(num_episode)):
            # Initialize the environment and get it's state
            # State observerd by guide
            numpifiedState = env.reset()
            stateTensor = torch.tensor(
                numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
            done = False
            episodicReward = 0
            while not done:
                sPrimeTensor, reward, done, _ = self.doStep(
                    guide, scout, channel, env, stateTensor)
                scout.memorize()
                scout.optimize()
                # Move to the next state
                stateTensor = sPrimeTensor
                episodicReward += reward

            episodicRewards.append(episodicReward)
            #print(f"Episode {eps} done, Eps Reward: {episodicReward}")
        dump(scout, "scout")
        dump(guide, "guide")
        dump(episodicRewards, "episodicRewards")

    def test(self):
        guide, scout, env, channel = self.setup("test")
        env.reset()
        numpifiedState = env.numpifiedState()
        stateTensor = torch.tensor(
            numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            sPrimeTensor, reward, done, _ = self.doStep(
                guide, scout, channel, env, stateTensor)
            stateTensor = sPrimeTensor

    def randomRun(self):
        guide, scout, env, channel = self.setup("rand")
        steps = 10
        stp = 0
        env.reset()
        numpifiedState = env.numpifiedState()
        stateTensor = torch.tensor(
            numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done and stp < steps:
            sPrimeTensor, reward, done, _ = self.doStep(
                guide, scout, channel, env, stateTensor)
            stateTensor = sPrimeTensor
            stp += 1
