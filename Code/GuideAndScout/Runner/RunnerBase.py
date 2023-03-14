from Environment.CommGridEnv import CommGridEnv
from Agents.GuideScout import *
from const import verbPrint, setVerbose
from joblib import dump, load
from Environment.CommChannel import CommChannel
from typing import List
import os
from datetime import datetime


class RunnerBase():
    def __init__(self, saveName) -> None:
        """

        """
        self.saveName = saveName
        self.defaultEnvSetting = {
            "row": 5,
            "column": 5,
            "treatNum": 2,
            "scoutsNum": 2,
            "noised": False,
            "noiseP": None,
            "TRAIN_EPS": 5,
            "TEST_MAX_EPS": 30,
            "RAND_EPS": 1,
        }

    def constructSaves(self):
        # now = datetime.now()
        # dt_string = now.strftime("-%m-%d_%H-%M")
        # saveFolderDir = "./Saves/" + saveName + dt_string + "/"
        saveFolderDir = "./Saves/" + self.saveName + "/"
        if not os.path.exists(saveFolderDir):
            os.mkdir(saveFolderDir)
        self._agentSettingSaveDir = saveFolderDir + "agentSettings"
        self._agentsSaveDir = saveFolderDir + "agents"
        self._rewardsSaveDir = saveFolderDir + "episodicRewards"
        self._stepsSaveDir = saveFolderDir + "episodicSteps"
        self._envSaveDir = saveFolderDir + "envSetting"

    def setupEnvSetting(self, loadSave, envSetting):
        if loadSave:
            self._configuredEnvSetting = load(self._envSaveDir)
        else:
            self._configuredEnvSetting = self.defaultEnvSetting
            if envSetting:
                for key in envSetting.keys():
                    self._configuredEnvSetting[key] = envSetting[key]
            dump(self._configuredEnvSetting, self._envSaveDir)

        # print(self._configuredEnvSetting)

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
        obsDim = (agentNum, self._treatNum)
        guide = GuideAgent(GUIDEID, obsDim, ACTIONSPACE,
                           noiseHandling=self._noised)

        agents = [guide]
        for i in range(self._scoutsNum):
            scout = ScoutAgent(startingScoutID + i, obsDim,
                               ACTIONSPACE, noiseHandling=self._noised, epsDecay=12000)
            agents.append(scout)
        return agents

    def setupRun(self, setupType, envSetting=None, noiseLevel=None, noiseHandlingMode=None):
        self.constructSaves()
        loadSave = setupType == "test"
        self.setupEnvSetting(loadSave, envSetting)
        render = getVerbose() != 0
        agents = self.instantiateAgents()
        agentSetting = None
        if setupType == "test":
            agentSetting = load(self._agentSettingSaveDir)
        for i, agent in enumerate(agents):
            agent.setNoiseHandling(noiseHandlingMode)
            if agentSetting:
                agent.loadSetting(agentSetting[i])
        
        noised = self._noised
        if setupType == "test":
            noised = noiseLevel is not None
            if noised:
                self._noiseP = noiseLevel
            else:
                self._noiseP = 0
        # print(f"Channel Noised: {self._noised}")
            # verbPrint(f"Noise level: {self._noiseP}", 1)
        # print(f"Noised: {noised}")
        # print(f"Noise level: {self._noiseP}")
        verbPrint(f"Noise Handling Mode: {noiseHandlingMode}", 1)
        self._channel = CommChannel(agents, self._noiseP, noised)
        self._channel.setupChannel()
        env = CommGridEnv(self._row, self._column, agents, self._treatNum,
                          render)

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

    def train(self, envSetting=None, verbose=0, wandbLog=True):
        """
        Run training with given environment settings
        """
        raise NotImplementedError

    def test(self, verbose=2, noiseLevel=None, noiseHandlingMode=None):
        setVerbose(verbose)
        agents, env = self.setupRun(
            "test", noiseLevel=noiseLevel, noiseHandlingMode=noiseHandlingMode)
        env.reset()
        state = env.numpifiedState()
        done = False
        step = 0
        rewards = 0
        while not done and step < self._TEST_MAX_EPS:
            sPrime, reward, done, _ = self.doStep(
                agents, env, state)
            state = sPrime
            step += 1
            rewards += reward
        verbPrint(
            f"===================================================\nCompleted in {step} steps\n===================================================", 1)
        return step, rewards

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
