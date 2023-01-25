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

    def instantiateAgents(self, treatNum: int):
        agentNum = 3
        n_obs = 2 * (agentNum + treatNum)
        guide = GuideAgent(GUIDEID, n_obs, ACTIONSPACE)
        scout1 = ScoutAgent(SCOUTID, n_obs, ACTIONSPACE)
        scout2 = ScoutAgent(SCOUT2ID, n_obs, ACTIONSPACE)
        agents = [guide, scout1, scout2]
        return agents

    def setup(self, setupType):
        render = setupType != "train"
        if setupType == "train" or setupType == "rand":
            agents = self.instantiateAgents(self.treatNum)
        else:
            scouts = load(self.scoutSaveDir)
            guide = load(self.guideSaveDir)
            agents = [guide]+scouts
        channel = CommChannel(agents, self.noised)
        env = CommGridEnv(self.row, self.column, agents, self.treatNum,
                          render)

        return agents, env, channel

    def doStep(self, agents, channel: CommChannel, env: CommGridEnv, stateTensor):
        # print("------------------------------------------------------\n\n\n")
        for scoutID in range(1, len(agents)):
            channel.sendMessage(GUIDEID, scoutID, stateTensor, "state")
        # Scout chooses epsilon greedy action solely on recieved message
        guide = agents[0]
        scout = agents[1]
        scout2 = agents[2]
        guideAction = guide.choose_action()
        scoutAction = scout.choose_action()
        scout2Action = scout2.choose_action()
        scoutActionTensor: torch.Tensor = scoutAction
        scout2ActionTensor: torch.Tensor = scout2Action
        # 0 Added as placeholder for scoutID structure
        scoutActions = [scoutActionTensor, scout2ActionTensor]
        actions: List[int] = [guideAction.item(), scoutAction.item(),
                              scout2Action.item()]
        numpifiedSPrime, reward, done, info = env.step(actions)
        rewardTensor = torch.tensor(
            [reward], dtype=torch.float32, device=device)
        if done:
            sPrimeTensor = None
        else:
            sPrimeTensor = torch.tensor(
                numpifiedSPrime, dtype=torch.float32, device=device).unsqueeze(0)

        for scoutID in range(1, len(agents)):
            channel.sendMessage(
                GUIDEID, scoutID, scoutActions[scoutID-1], "action")
            channel.sendMessage(GUIDEID, scoutID, rewardTensor, "reward")
            channel.sendMessage(GUIDEID, scoutID, sPrimeTensor, "sPrime")

        return sPrimeTensor, reward, done, info

    def train(self):
        agents, env, channel = self.setup("train")
        guide = agents[0]
        scouts = agents[1:]
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
                    agents, channel, env, stateTensor)
                for scout in scouts:
                    scout.memorize()
                    scout.optimize()
                # Move to the next state
                stateTensor = sPrimeTensor
                episodicReward += reward

            episodicRewards.append(episodicReward)
            #print(f"Episode {eps} done, Eps Reward: {episodicReward}")
        dump(scouts, self.scoutSaveDir)
        dump(guide, self.guideSaveDir)
        dump(episodicRewards, self.rewardsSaveDir)

    def test(self, plot=False):
        agents, env, channel = self.setup("test")
        env.reset()
        numpifiedState = env.numpifiedState()
        stateTensor = torch.tensor(
            numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        step = 0
        while not done and step < self.TEST_MAX_EPS:
            sPrimeTensor, reward, done, _ = self.doStep(
                agents, channel, env, stateTensor)
            stateTensor = sPrimeTensor
            step += 1

        if plot:
            rewardPlot = load(self.rewardsSaveDir)
            plt.plot(rewardPlot)
            plt.show()

    def randomRun(self):
        agents, env, channel = self.setup("rand")
        guide = agents[0]
        scouts = agents[1:]
        stp = 0
        env.reset()
        numpifiedState = env.numpifiedState()
        stateTensor = torch.tensor(
            numpifiedState, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done and stp < self.RAND_EPS:
            sPrimeTensor, _, done, _ = self.doStep(
                agents, channel, env, stateTensor)
            stateTensor = sPrimeTensor
            stp += 1
