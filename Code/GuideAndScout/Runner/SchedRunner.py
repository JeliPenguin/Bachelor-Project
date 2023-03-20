from Runner.RunnerBase import RunnerBase
from const import setVerbose, trainSeed, evalNoiseLevels
from Environment.EnvUtilities import *
from tqdm import tqdm
from joblib import dump
import random


class SchedRunner(RunnerBase):
    """
        Runner with varied p for the binary symmetric channel when training
    """

    def __init__(self, envType,saveName) -> None:
        super().__init__(envType,saveName)

    def train(self, envSetting=None, verbose=0, wandbLog=False):
        """
        Run training with given environment settings, but with scheduled changes in channel noise
        """
        random.seed(trainSeed)
        setVerbose(verbose)
        agents, env = self.setupRun("train", envSetting)
        scouts = agents[startingScoutID:]
        episodicRewards = []
        episodicSteps = []
        print("Sched Runner")
        print(f"Training On: ")
        print(self._configuredEnvSetting)
        print(f"Running {self._TRAIN_EPS} epochs:")
        for eps in tqdm(range(self._TRAIN_EPS)):
            # Initialize the environment and get it's state
            # State only observerd by the guide
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
        for a in agents:
            a._memory.clear()
        agentSettings = [a.getSetting() for a in agents]
        dump(agents, self._agentsSaveDir)
        dump(agentSettings, self._agentSettingSaveDir)
        dump(episodicRewards, self._rewardsSaveDir)
        dump(episodicSteps, self._stepsSaveDir)
