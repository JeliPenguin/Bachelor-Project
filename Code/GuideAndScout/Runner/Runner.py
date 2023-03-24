from Environment.FindingTreat import FindingTreat
from Environment.CommGridEnv import CommGridEnv
from Environment.Spread import Spread
from Agents.GuideScout import *
from const import verbPrint, setVerbose
from joblib import dump, load
from Environment.CommChannel import CommChannel
from typing import List
import os
from datetime import datetime
import wandb
import random
from tqdm import tqdm
from const import evalNoiseLevels


class Runner():
    def __init__(self, envType, saveName) -> None:
        """

        """
        self.envType = envType
        self.saveName = saveName
        self.constructSaves()

    def constructSaves(self):
        # now = datetime.now()
        # dt_string = now.strftime("-%m-%d_%H-%M")
        # saveFolderDir = "./Saves/" + saveName + dt_string + "/"
        saveFolderDir = "./Saves/" + self.saveName + "/"
        if not os.path.exists(saveFolderDir):
            os.mkdir(saveFolderDir)
        self._agentSettingSaveDir = saveFolderDir + "agentSettings"
        self._agentTrainSettingSaveDir = saveFolderDir + "agentTrainSetting"
        self._agentsSaveDir = saveFolderDir + "agents"
        self._rewardsSaveDir = saveFolderDir + "episodicRewards"
        self._stepsSaveDir = saveFolderDir + "episodicSteps"
        self._envSaveDir = saveFolderDir + "envSetting"

    def setupEnvSetting(self, loadSave, envSetting):
        defaultEnvSetting = {
            "row": 5,
            "column": 5,
            "treatNum": 2,
            "scoutsNum": 2,
        }
        if loadSave:
            configuredEnvSetting = load(self._envSaveDir)
        else:
            configuredEnvSetting = defaultEnvSetting
            if envSetting:
                for key in envSetting.keys():
                    configuredEnvSetting[key] = envSetting[key]
            dump(configuredEnvSetting, self._envSaveDir)

        # print(self._configuredEnvSetting)
        print(f"Environment Setting: ")
        print(configuredEnvSetting)
        return configuredEnvSetting

    def instantiateAgents(self, obsDim, scoutsNum, noised, noiseHandlingMode, loadSaved):
        trainedAgent = None
        if loadSaved:
            trainedAgent = load(self._agentSettingSaveDir)
        trainSetting = load(self._agentTrainSettingSaveDir)

        guide = GuideAgent(GUIDEID, obsDim, ACTIONSPACE,
                           noiseHandling=noised, hyperParam=trainSetting)

        agents = [guide]
        for i in range(scoutsNum):
            scout = ScoutAgent(startingScoutID + i, obsDim,
                               ACTIONSPACE, noiseHandling=noised, hyperParam=trainSetting)
            agents.append(scout)

        for i, agent in enumerate(agents):
            agent.setNoiseHandling(noiseHandlingMode)
            if trainedAgent:
                agent.loadSetting(trainedAgent[i])

        return agents

    def setupRun(self, setupType, envSetting=None, noiseP=0, noiseHandlingMode=None):
        loadSave = setupType == "test"
        configuredEnvSetting = self.setupEnvSetting(loadSave, envSetting)
        row = configuredEnvSetting["row"]
        column = configuredEnvSetting["column"]
        treatNum = configuredEnvSetting["treatNum"]
        scoutsNum = configuredEnvSetting["scoutsNum"]
        agentNum = 1+scoutsNum
        obsDim = (agentNum, treatNum)
        noised = noiseP != 0
        render = getVerbose() != 0
        agents = self.instantiateAgents(
            obsDim, scoutsNum, noised, noiseHandlingMode, loadSave)

        # print(f"Channel Noised: {self._noised}")
        # verbPrint(f"Noise level: {self._noiseP}", 1)
        print(f"Noised: {noised}")
        print(f"Noise level: {noiseP}")
        verbPrint(f"Noise Handling Mode: {noiseHandlingMode}", 1)
        self._channel = CommChannel(agents, noiseP, noised)
        self._channel.setupChannel()
        if self.envType == "FindingTreat":
            env = FindingTreat(row, column, agents, treatNum,
                               render)
        elif self.envType == "Spread":
            env = Spread(row, column, agents, treatNum,
                         render)
        else:
            print("Invalid Environment Type")
            exit()
        print("Running on environment: ", env.envName())

        return agents, env

    def doStep(self, agents, env: CommGridEnv, state):
        # verbPrint(
        #     "=================================================================\n", 1)
        guide = agents[GUIDEID]
        verbPrint("SENDING CURRENT STATE ONLY", 2)
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
        verbPrint("MSG SENT:", 2)
        for scoutID in range(startingScoutID, len(agents)):
            agents[scoutID].rememberAction([actions[scoutID]])
            guide.prepareMessage([reward], "reward")
            guide.prepareMessage(sPrime, "sPrime")
            guide.sendMessage(scoutID)

        return sPrime, reward, done, info

    def train(self, envSetting, trainSetting, verbose=0, wandbLog=True):
        """
        Run training with given environment settings
        """
        print("Training Setting: ", trainSetting)
        setVerbose(verbose)
        TRAIN_EPS = trainSetting["TRAIN_EPS"]
        trainSeed = trainSetting["seed"]
        trainMethod = trainSetting["method"]
        dump(trainSetting, self._agentTrainSettingSaveDir)
        random.seed(trainSeed)
        # if wandbLog:
        #     wandb.init(project="Comm-Noised MARL", entity="jelipenguin")
        agents, env = self.setupRun(
            "train", envSetting)
        scouts = agents[startingScoutID:]
        episodicRewards = []
        episodicSteps = []
        print(f"{trainMethod} Trainer")
        print(f"Running {TRAIN_EPS} epochs:")
        for eps in tqdm(range(TRAIN_EPS)):
            # Initialize the environment and get it's state
            # State only observerd by the guide
            if trainMethod == "Sched":
                noiseP = random.choice(evalNoiseLevels)
                self._channel.setNoiseP(noiseP)
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

            # if wandbLog:
            #     wandb.log({"episodicStep": step})
            #     wandb.log({"episodicReward": episodicReward})
            episodicSteps.append(step)
            episodicRewards.append(episodicReward)
        for a in agents:
            a._memory.clear()
        agentSettings = [a.getSetting() for a in agents]
        dump(agents, self._agentsSaveDir)
        dump(agentSettings, self._agentSettingSaveDir)
        dump(episodicRewards, self._rewardsSaveDir)
        dump(episodicSteps, self._stepsSaveDir)

    def test(self, verbose=2, noiseP=0, noiseHandlingMode=None, maxEps=30):
        setVerbose(verbose)
        agents, env = self.setupRun(
            "test", noiseP=noiseP, noiseHandlingMode=noiseHandlingMode)
        env.reset()
        state = env.numpifiedState()
        done = False
        step = 0
        rewards = 0
        while not done and step < maxEps:
            sPrime, reward, done, _ = self.doStep(
                agents, env, state)
            state = sPrime
            step += 1
            rewards += reward
        verbPrint(
            f"===================================================\nCompleted in {step} steps\n===================================================", 1)
        return step, rewards

    def randomRun(self, verbose=1, maxEps=3):
        setVerbose(verbose)
        agents, env = self.setupRun("random")
        stp = 0
        env.reset()
        state = env.numpifiedState()
        done = False
        while not done and stp < maxEps:
            sPrime, _, done, _ = self.doStep(
                agents, env, state)
            state = sPrime
            stp += 1
