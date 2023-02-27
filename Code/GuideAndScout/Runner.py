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
        """
        Standardized Runner for handling interaction between agents and the environment
        """
        self._crtEnvSetting = envSetting
        self.constructSaves(saveName, envSetting)
        self.setupEnvSetting()

    def constructSaves(self, saveName, envSetting):
        # now = datetime.now()
        # dt_string = now.strftime("-%m-%d_%H-%M")
        # saveFolderDir = "./Saves/" + saveName + dt_string + "/"
        saveFolderDir = "./Saves/" + saveName + "/"
        if not os.path.exists(saveFolderDir):
            os.mkdir(saveFolderDir)
        self._agentsSaveDir = saveFolderDir + "agents"
        self._rewardsSaveDir = saveFolderDir + "episodicRewards"
        self._stepsSaveDir = saveFolderDir + "episodicSteps"
        self._envSaveDir = saveFolderDir + "envSetting"
        self._crtEnvSetting = envSetting
        dump(envSetting, self._envSaveDir)

    def setupEnvSetting(self, loadSave=False):
        self._configuredEnvSetting = {
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
            envSetting = load(self._envSaveDir)
        else:
            envSetting = self._crtEnvSetting

        for key in envSetting.keys():
            self._configuredEnvSetting[key] = envSetting[key]

        self._row = self._configuredEnvSetting["row"]
        self._column = self._configuredEnvSetting["column"]
        self._treatNum = self._configuredEnvSetting["treatNum"]
        self._scoutsNum = self._configuredEnvSetting["scoutsNum"]
        self._noised = self._configuredEnvSetting["noised"]
        self._noiseP = self._configuredEnvSetting["noiseP"]
        self._TRAIN_EPS = self._configuredEnvSetting["TRAIN_EPS"]
        self._TEST_MAX_EPS = self._configuredEnvSetting["TEST_MAX_EPS"]
        self._RAND_EPS = self._configuredEnvSetting["RAND_EPS"]

    def instantiateAgents(self):
        agentNum = 1 + self._scoutsNum
        n_obs = 2 * (agentNum + self._treatNum)
        guide = GuideAgent(GUIDEID, n_obs, ACTIONSPACE,
                           noiseHandling=self._noised)

        agents = [guide]
        for i in range(self._scoutsNum):
            scout = ScoutAgent(startingScoutID + i, n_obs,
                               ACTIONSPACE, noiseHandling=self._noised, epsDecay=12000)
            agents.append(scout)
        return agents

    def setupRun(self, setupType):
        render = getVerbose() != 0
        if setupType == "train" or setupType == "random":
            agents = self.instantiateAgents()
        else:
            agents = load(self._agentsSaveDir)
        noised = self._noised
        if setupType == "train":
            noised = False
        for agent in agents:
            agent.setNoiseHandling(noised)
        print(f"Noised: {noised}")
        if noised:
            print(f"Noise level: {self._noiseP}")
        channel = CommChannel(agents, self._noiseP, noised)
        channel.setupChannel()
        env = CommGridEnv(self._row, self._column, agents, self._treatNum,
                          render)

        return agents, env

    def doStep(self, agents, env: CommGridEnv, state):
        # Guide only chooses action STAY
        # Scouts choose epsilon greedy action solely on recieved message
        if getVerbose() >= 1:
            print("=================================================================\n")
        guide = agents[GUIDEID]
        if getVerbose() >= 2:
            print("SENDING CURRENT STATE ONLY")
        for scoutID in range(startingScoutID, len(agents)):
            # Other part of the message kept as None
            guide.clearPreparedMessage()
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
            # print("SENDING REWARD AND SPRIME")
            print("MSG SENT:")
        for scoutID in range(startingScoutID, len(agents)):
            agents[scoutID].rememberAction([actions[scoutID]])
            guide.prepareMessage([reward], "reward")
            guide.prepareMessage(sPrime, "sPrime")
            guide.sendMessage(scoutID)

        return sPrime, reward, done, info

    def train(self, verbose=0, wandbLog=True):
        """
        Run training with given environment settings
        """
        setVerbose(verbose)
        if wandbLog:
            wandb.init(project="Comm-Noised MARL", entity="jelipenguin")
            wandb.config = self._configuredEnvSetting
        agents, env = self.setupRun("train")
        scouts = agents[startingScoutID:]
        episodicRewards = []
        episodicSteps = []
        print(f"Running {self._TRAIN_EPS} epochs:")
        for eps in tqdm(range(self._TRAIN_EPS)):
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

            for scout in scouts:
                scout.updateEps()

            if wandbLog:
                wandb.log({"episodicStep": step})
                wandb.log({"episodicReward": episodicReward})
            episodicSteps.append(step)
            episodicRewards.append(episodicReward)
        dump(agents, self._agentsSaveDir)
        dump(episodicRewards, self._rewardsSaveDir)
        dump(episodicSteps, self._stepsSaveDir)

    def test(self, verbose=2, loadSavedEnvSetting=True, plot=False):
        setVerbose(verbose)
        self.setupEnvSetting(loadSave=loadSavedEnvSetting)
        agents, env = self.setupRun("test")
        env.reset()
        state = env.numpifiedState()
        done = False
        step = 0
        while not done and step < self._TEST_MAX_EPS:
            sPrime, _, done, _ = self.doStep(
                agents, env, state)
            state = sPrime
            step += 1

        if plot:
            rewardPlot = load(self._rewardsSaveDir)
            plt.plot(rewardPlot)
            plt.show()

    def randomRun(self, verbose=1):
        setVerbose(verbose)
        agents, env = self.setupRun("random")
        stp = 0
        env.reset()
        state = env.numpifiedState()
        done = False
        while not done and stp < self._RAND_EPS:
            sPrime, _, done, _ = self.doStep(
                agents, env, state)
            state = sPrime
            stp += 1
