import torch
from Environment.CommGridEnv import CommGridEnv
from Agents.GuideScout import *
from const import *
from joblib import dump, load
from Environment.CommChannel import CommChannel
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import List
import wandb

startingScoutID = GUIDEID + 1


class Runner():
    def __init__(self, envSetting, saveName="Default") -> None:
        self.crtEnvSetting = envSetting
        self.constructSaves(saveName, envSetting)
        self.setupEnvSetting()

    def setupEnvSetting(self, loadSave=False):
        self.configuredEnvSetting = {
            "row": 5,
            "column": 5,
            "treatNum": 2,
            "scoutsNum": 2,
            "noised": False,
            "noiseP": 0.005,
            "TRAIN_EPS": 5,
            "TEST_MAX_EPS": 30,
            "RAND_EPS": 1,
        }
        if loadSave:
            envSetting = load(self.envSaveDir)
        else:
            envSetting = self.crtEnvSetting

        for key in envSetting.keys():
            self.configuredEnvSetting[key] = envSetting[key]

        self.row = self.configuredEnvSetting["row"]
        self.column = self.configuredEnvSetting["column"]
        self.treatNum = self.configuredEnvSetting["treatNum"]
        self.scoutsNum = self.configuredEnvSetting["scoutsNum"]
        self.noised = self.configuredEnvSetting["noised"]
        self.noiseP = self.configuredEnvSetting["noiseP"]
        self.TRAIN_EPS = self.configuredEnvSetting["TRAIN_EPS"]
        self.TEST_MAX_EPS = self.configuredEnvSetting["TEST_MAX_EPS"]
        self.RAND_EPS = self.configuredEnvSetting["RAND_EPS"]

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
        self.crtEnvSetting = envSetting
        dump(envSetting, self.envSaveDir)

    def instantiateAgents(self):
        agentNum = 1 + self.scoutsNum
        n_obs = 2 * (agentNum + self.treatNum)
        guide = GuideAgent(GUIDEID, n_obs, ACTIONSPACE)

        agents = [guide]
        for i in range(self.scoutsNum):
            scout = ScoutAgent(startingScoutID + i, n_obs, ACTIONSPACE)
            agents.append(scout)
        return agents

    def setupRun(self, setupType):
        render = setupType != "train"
        if setupType == "train" or setupType == "rand":
            agents = self.instantiateAgents()
        else:
            agents = load(self.agentsSaveDir)
        channel = CommChannel(agents, self.noiseP,self.noised)
        channel.setupChannel()
        env = CommGridEnv(self.row, self.column, agents, self.treatNum,
                          render)

        return agents, env

    def doStep(self, agents, env: CommGridEnv, state):
        # Guide only chooses action STAY
        # Scouts choose epsilon greedy action solely on recieved message
        if getVerbose() >= 1:
            print("---------------------------------------\n")
        guide = agents[GUIDEID]
        if getVerbose() >= 2:
            print("SENDING CURRENT STATE")
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
        if getVerbose() >= 2:
            print("SENDING REWARD AND SPRIME")
        for scoutID in range(startingScoutID, len(agents)):
            # Action not included in the message as the agent themselves already
            # know what action they performed and shouldn't be noised
            agents[scoutID].rememberAction([actions[scoutID]])
            guide.prepareMessage([reward], "reward")
            guide.prepareMessage(sPrime, "sPrime")
            guide.sendMessage(scoutID)

        return sPrime, reward, done, info

    def train(self, verbose=0,wandbLog=True):
        setVerbose(verbose)
        if wandbLog:
            wandb.init(project="Comm-Noised MARL", entity="jelipenguin")
            wandb.config = self.configuredEnvSetting
        agents, env = self.setupRun("train")
        scouts = agents[startingScoutID:]
        episodicRewards = []
        episodicSteps = []
        print(f"Running {self.TRAIN_EPS} epochs:")
        for eps in tqdm(range(self.TRAIN_EPS)):
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

            if wandbLog:
                wandb.log({"episodicStep": step})
                wandb.log({"episodicReward": episodicReward})
            episodicSteps.append(step)
            episodicRewards.append(episodicReward)
        dump(agents, self.agentsSaveDir)
        dump(episodicRewards, self.rewardsSaveDir)
        dump(episodicSteps, self.stepsSaveDir)

    def test(self, loadSavedEnvSetting=True, plot=False):
        self.setupEnvSetting(loadSave=loadSavedEnvSetting)
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
