import torch
from CommGridEnv import CommGridEnv
from GuideScout import *
from const import *
from joblib import dump, load
from CommChannel import CommChannel
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCOUTID = GUIDEID + 1
SCOUT2ID = GUIDEID + 2


class Run():
    def __init__(self, envSetting) -> None:
        self.row = envSetting["row"]
        self.column = envSetting["column"]
        self.treatNum = envSetting["treatNum"]
        self.noised = envSetting["noised"]
        self.TRAIN_EPS = envSetting["TRAIN_EPS"]
        self.TEST_MAX_EPS = envSetting["TEST_MAX_EPS"]
        self.RAND_EPS = envSetting["RAND_EPS"]
        self.scoutSaveDir = "./Saves/scout"
        self.guideSaveDir = "./Saves/guide"
        self.rewardsSaveDir = "./Saves/episodicRewards"

    def instantiateAgents(self, treatNum: int) -> Tuple[GuideAgent, ScoutAgent, ScoutAgent]:
        agentNum = 2
        n_obs = 2 * (agentNum + treatNum)
        guide = GuideAgent(GUIDEID, n_obs, ACTIONSPACE)
        scout1 = ScoutAgent(SCOUTID, n_obs, ACTIONSPACE)
        # scout2 = ScoutAgent(SCOUT2ID, n_obs, ACTIONSPACE)
        agents: Tuple[GuideAgent, ScoutAgent] = tuple([guide, scout1])
        return agents

    def setup(self, setupType):
        render = setupType != "train"
        if setupType == "train" or setupType == "rand":
            agents = self.instantiateAgents(self.treatNum)
            guide = agents[GUIDEID]
            scout = agents[SCOUTID]
            # scout2 = agents[SCOUT2ID]
        else:
            scout = load(self.scoutSaveDir)
            guide = load(self.guideSaveDir)
            agents = tuple([scout, guide])
        channel = CommChannel(agents, self.noised)
        env = CommGridEnv(self.row, self.column, agents, self.treatNum,
                          render)

        return guide, scout, env, channel

    def doStep(self, guide: GuideAgent, scout: ScoutAgent, channel: CommChannel, env: CommGridEnv, stateTensor):
        # print("------------------------------------------------------\n\n\n")
        channel.sendMessage(GUIDEID, SCOUTID, stateTensor, "state")
        # Scout chooses epsilon greedy action solely on recieved message
        scoutAction = scout.choose_action()
        guideAction = guide.choose_action()
        actionTensor: torch.Tensor = scoutAction
        actions: List[int] = [guideAction.item(), scoutAction.item()]
        numpifiedSPrime, reward, done, info = env.step(actions)
        rewardTensor = torch.tensor(
            [reward], dtype=torch.float32, device=device)
        if done:
            sPrimeTensor = None
        else:
            sPrimeTensor = torch.tensor(
                numpifiedSPrime, dtype=torch.float32, device=device).unsqueeze(0)

        channel.sendMessage(GUIDEID, SCOUTID, actionTensor, "action")
        channel.sendMessage(GUIDEID, SCOUTID, rewardTensor, "reward")
        channel.sendMessage(GUIDEID, SCOUTID, sPrimeTensor, "sPrime")

        return sPrimeTensor, reward, done, info

    def train(self):
        guide, scout, env, channel = self.setup("train")
        episodicRewards = []
        print(f"Running {self.TRAIN_EPS} epochs:")
        for eps in tqdm(range(self.TRAIN_EPS)):
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
        dump(scout, self.scoutSaveDir)
        dump(guide, self.guideSaveDir)
        dump(episodicRewards, self.rewardsSaveDir)

    def test(self, plot=False):
        guide, scout, env, channel = self.setup("test")
        env.reset()
        numpifiedState = env.numpifiedState()
        stateTensor = torch.tensor(
            numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        step = 0
        while not done and step < self.TEST_MAX_EPS:
            sPrimeTensor, reward, done, _ = self.doStep(
                guide, scout, channel, env, stateTensor)
            stateTensor = sPrimeTensor
            step += 1

        if plot:
            rewardPlot = load(self.rewardsSaveDir)
            plt.plot(rewardPlot)
            plt.show()

    def randomRun(self):
        guide, scout, env, channel = self.setup("rand")
        stp = 0
        env.reset()
        numpifiedState = env.numpifiedState()
        stateTensor = torch.tensor(
            numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done and stp < self.RAND_EPS:
            sPrimeTensor, reward, done, _ = self.doStep(
                guide, scout, channel, env, stateTensor)
            stateTensor = sPrimeTensor
            stp += 1
