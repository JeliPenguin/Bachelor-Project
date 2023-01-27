import torch
from CommGridEnv import CommGridEnv
from GuideScout import *
from const import *
from joblib import dump, load
from CommChannel import CommChannel
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt


class Runner():
    def __init__(self, envSetting) -> None:
        envSetting = self.setupEnvSetting(envSetting)
        self.row = envSetting["row"]
        self.column = envSetting["column"]
        self.treatNum = envSetting["treatNum"]
        self.scoutsNum = envSetting["scoutsNum"]
        self.noised = envSetting["noised"]
        self.TRAIN_EPS = envSetting["TRAIN_EPS"]
        self.TEST_MAX_EPS = envSetting["TEST_MAX_EPS"]
        self.RAND_EPS = envSetting["RAND_EPS"]
        self.agentsSaveDir = "./Saves/agents"
        self.rewardsSaveDir = "./Saves/episodicRewards"

    def setupEnvSetting(self, envSetting):
        defaultEnvSetting = {
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
            defaultEnvSetting[key] = envSetting[key]
        return defaultEnvSetting

    def instantiateAgents(self, treatNum: int):
        agentNum = 1 + self.scoutsNum
        n_obs = 2 * (agentNum + treatNum)
        guide = GuideAgent(GUIDEID, n_obs, ACTIONSPACE)
        startingScoutID = GUIDEID + 1
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
        actions: List[int] = [a.choose_action().item() for a in agents]
        # One timestep forward in the environment based on agents' actions
        sPrime, reward, done, info = env.step(actions)
        if done:
            # -1  Reserved for indication of termination
            sPrime = [-1]

        guide: GuideAgent = agents[GUIDEID]
        for scoutID in range(GUIDEID+1, len(agents)):
            guide.prepareMessage(state, "state")
            guide.prepareMessage([actions[scoutID]], "action")
            guide.prepareMessage([reward], "reward")
            guide.prepareMessage(sPrime, "sPrime")
            guide.sendMessage(scoutID)

        return sPrime, reward, done, info

    def train(self):
        agents, env = self.setupRun("train")
        guide = agents[0]
        scouts = agents[1:]
        episodicRewards = []
        print(f"Running {self.TRAIN_EPS} epochs:")
        for _ in tqdm(range(self.TRAIN_EPS)):
            # Initialize the environment and get it's state
            # State only observerd by the guide
            state = env.reset()
            done = False
            episodicReward = 0
            while not done:
                sPrime, reward, done, _ = self.doStep(
                    agents, env, state)
                for scout in scouts:
                    scout.memorize()
                    scout.optimize()
                # Move to the next state
                state = sPrime
                episodicReward += reward

            episodicRewards.append(episodicReward)
            #print(f"Episode {eps} done, Eps Reward: {episodicReward}")
        dump(agents, self.agentsSaveDir)
        dump(episodicRewards, self.rewardsSaveDir)

    def test(self, plot=False):
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
