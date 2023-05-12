from Environment.FindingTreat import FindingTreat
from Environment.CommGridEnv import CommGridEnv
from Environment.Spread import Spread
from Agents.GuideScout import *
from const import verbPrint, setVerbose
from joblib import dump, load
from Environment.CommChannel import CommChannel
from typing import List
import os
import random
from tqdm import tqdm
from const import evalNoiseLevels
from Seeder import Seeder


class Runner():
    def __init__(self, envType, saveName) -> None:
        """

        """
        self._envType = envType
        self._saveName = saveName
        self._seeder = Seeder()
        self.constructSaves()

    def constructSaves(self):
        """Construct save file for current run"""
        saveFolderDir = "./Saves/" + self._saveName + "/"
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
        verbPrint(f"Environment Setting: {configuredEnvSetting}", 0)
        return configuredEnvSetting

    def instantiateAgents(self, obsDim, scoutsNum, noised, noiseHandlingMode, loadSaved):
        """
        Initialising agents, whether it's loading pretained configuations or instantiating new agents
        """
        trainedAgent = None
        # Load pretrained config
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
        """
        Configure environment and communication channel settings for current run, initialise agents
        """
        loadSave = setupType == "test"
        configuredEnvSetting = self.setupEnvSetting(loadSave, envSetting)
        row = configuredEnvSetting["row"]
        column = configuredEnvSetting["column"]
        treatNum = configuredEnvSetting["treatNum"]
        scoutsNum = configuredEnvSetting["scoutsNum"]
        agentNum = 1+scoutsNum
        obsDim = (agentNum, treatNum)
        noised = noiseP != 0
        render = getVerbose() > 0
        agents = self.instantiateAgents(
            obsDim, scoutsNum, noised, noiseHandlingMode, loadSave)
        verbPrint(f"Noised: {noised}", 0)
        verbPrint(f"Noise level: {noiseP}", 0)
        verbPrint(f"Noise Handling Mode: {noiseHandlingMode}", 1)
        self._channel = CommChannel(agents, noiseP, noised)
        self._channel.setupChannel()
        if self._envType == "FindingTreat":
            env = FindingTreat(row, column, agents, treatNum,
                               render)
        elif self._envType == "Spread":
            env = Spread(row, column, agents, treatNum,
                         render)
            if not loadSave:
                for agent in agents:
                    agent.setNetwork("Spread")
        else:
            print("Invalid Environment Type")
            exit()
        verbPrint(f"Running on environment: {env.envName()}", 0)

        return agents, env

    def doStep(self, agents, env: CommGridEnv, state):
        """ Do a single timestep of the environment"""
        guide = agents[GUIDEID]
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
        for scoutID in range(startingScoutID, len(agents)):
            agents[scoutID].rememberAction([actions[scoutID]])
            guide.prepareMessage([reward], "reward")
            guide.prepareMessage(sPrime, "sPrime")
            guide.sendMessage(scoutID)

        return sPrime, reward, done, info

    def intermediateShow(self, agents, env):
        setVerbose(1)
        env._toRender = True
        env.setSeed(self._seeder.getEvalSeed())
        state = env.reset()
        step = 0
        done = False
        while not done:
            sPrime, reward, done, _ = self.doStep(
                agents, env, state)
            state = sPrime
            step += 1
        print(
            f"===================================================\nCompleted in {step} steps\n===================================================")
        setVerbose(0)
        env._toRender = False

    def train(self, envSetting, trainSetting, verbose=0, showInterTrain=False):
        """
        Run training with given environment settings
        """
        print("Training Setting: ", trainSetting)
        setVerbose(verbose)
        TRAIN_EPS = trainSetting["TRAIN_EPS"]
        trainMethod = trainSetting["method"]
        dump(trainSetting, self._agentTrainSettingSaveDir)
        agents, env = self.setupRun(
            "train", envSetting)
        random.seed(self._seeder.getTrainSeed())
        scouts = agents[startingScoutID:]
        episodicRewards = []
        episodicSteps = []
        print(f"{trainMethod} Trainer")
        print(f"Running {TRAIN_EPS} epochs:")
        for eps in tqdm(range(TRAIN_EPS)):
            # Initialize the environment and get it's state
            # State only observerd by the guide
            env.setSeed(self._seeder.getTrainSeed())
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
            episodicSteps.append(step)
            episodicRewards.append(episodicReward)

            if showInterTrain and eps % 100 == 0:
                self.intermediateShow(agents, env)

        # Dumping trained agents configurations and training stats
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
        return step, rewards

    def randomRun(self, verbose=1, maxEps=3):
        """
        All agents do random actions, for debugging
        """
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
