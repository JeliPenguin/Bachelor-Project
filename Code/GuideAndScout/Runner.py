import torch
from CommGridEnv import CommGridEnv
from GuideScout import *
from const import *
from joblib import dump, load
from CommChannel import CommChannel
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime

startingScoutID = GUIDEID + 1


class Runner():
    def __init__(self, envSetting, saveName="Default") -> None:
        self.setupEnvSetting(envSetting)
        self.constructSaves(saveName, envSetting)

    def constructSaves(self, saveName, envSetting):
        # now = datetime.now()
        # dt_string = now.strftime("-%m-%d_%H-%M")
        # saveFolderDir = "./Saves/" + saveName + dt_string + "/"
        saveFolderDir = "./Saves/" + saveName + "/"
        if not os.path.exists(saveFolderDir):
            os.mkdir(saveFolderDir)
        self.agentsSaveDir = saveFolderDir + "agents"
        self.rewardsSaveDir = saveFolderDir + "episodicRewards"
        self.stepsSaveDir = saveFolderDir + "episodicSteps"
        self.envSaveDir = saveFolderDir + "envSetting"
        dump(envSetting, self.envSaveDir)

    def setupEnvSetting(self, envSetting):
        configuredEnvSetting = {
            "row": 5,
            "column": 5,
            "treatNum": 2,
            "scoutsNum": 1,
            "noised": False,
            "TRAIN_EPS": 5,
            "TEST_MAX_EPS": 30,
            "RAND_EPS": 1,
        }
        for key in envSetting.keys():
            configuredEnvSetting[key] = envSetting[key]

        self.row = configuredEnvSetting["row"]
        self.column = configuredEnvSetting["column"]
        self.treatNum = configuredEnvSetting["treatNum"]
        self.scoutsNum = configuredEnvSetting["scoutsNum"]
        self.noised = configuredEnvSetting["noised"]
        self.TRAIN_EPS = configuredEnvSetting["TRAIN_EPS"]
        self.TEST_MAX_EPS = configuredEnvSetting["TEST_MAX_EPS"]
        self.RAND_EPS = configuredEnvSetting["RAND_EPS"]

    def instantiateAgents(self, treatNum: int):
        agentNum = 1 + self.scoutsNum
        n_obs = 2 * (agentNum + treatNum)
        guide = GuideAgent(GUIDEID, n_obs, ACTIONSPACE)

        agents = [guide]
        for i in range(self.scoutsNum):
            scout = ScoutAgent(startingScoutID + i, n_obs, ACTIONSPACE)
            agents.append(scout)
        return agents

    def setupRun(self, setupType):
        render = setupType != "train"
        if setupType == "train" or setupType == "rand":
            agents = self.instantiateAgents(self.treatNum)
        else:
            agents = load(self.agentsSaveDir)
        channel = CommChannel(agents, self.noised)
        channel.setupChannel()
        env = CommGridEnv(self.row, self.column, agents, self.treatNum,
                          render)

        return agents, env

    def doStep(self, agents, env: CommGridEnv, state):
        # Guide only chooses action STAY
        # Scouts choose epsilon greedy action solely on recieved message
        guide = agents[GUIDEID]
        for scoutID in range(startingScoutID, len(agents)):
            # Other part of the message kept as None
            guide.prepareMessage(state, "state")
            guide.sendMessage(scoutID)
        actions: List[int] = [a.choose_action().item() for a in agents]
        # One timestep forward in the environment based on agents' actions
        sPrime, reward, done, info = env.step(actions)
        if done:
            # indicates end of episode
            sPrime = None

        guide: GuideAgent = agents[GUIDEID]
        for scoutID in range(startingScoutID, len(agents)):
            guide.prepareMessage([actions[scoutID]], "action")
            guide.prepareMessage([reward], "reward")
            guide.prepareMessage(sPrime, "sPrime")
            guide.sendMessage(scoutID)

        return sPrime, reward, done, info

    def train(self):
        agents, env = self.setupRun("train")
        scouts = agents[startingScoutID:]
        episodicRewards = []
        episodicSteps = []
        print(f"Running {self.TRAIN_EPS} epochs:")
        for _ in tqdm(range(self.TRAIN_EPS)):
            # Initialize the environment and get it's state
            # State only observerd by the guide
            state = env.reset()
            done = False
            episodicReward = 0
            step = 0
            while not done:
                sPrime, reward, done, _ = self.doStep(
                    agents, env, state)
                for scout in scouts:
                    scout.memorize()
                    scout.optimize()
                # Move to the next state
                state = sPrime
                episodicReward += reward
                step += 1

            episodicSteps.append(step)
            episodicRewards.append(episodicReward)
        dump(agents, self.agentsSaveDir)
        dump(episodicRewards, self.rewardsSaveDir)
        dump(episodicSteps, self.stepsSaveDir)

    def test(self, plot=False):
        envSetting = load(self.envSaveDir)
        self.setupEnvSetting(envSetting)
        agents, env = self.setupRun("test")
        env.reset()
        state = env.numpifiedState()
        done = False
        step = 0
        while not done and step < self.TEST_MAX_EPS:
            sPrime, _, done, _ = self.doStep(
                agents, env, state)
            state = sPrime
            step += 1

        if plot:
            rewardPlot = load(self.rewardsSaveDir)
            plt.plot(rewardPlot)
            plt.show()

    def randomRun(self):
        agents, env = self.setupRun("rand")
        stp = 0
        env.reset()
        state = env.numpifiedState()
        done = False
        while not done and stp < self.RAND_EPS:
            sPrime, _, done, _ = self.doStep(
                agents, env, state)
            state = sPrime
            stp += 1
