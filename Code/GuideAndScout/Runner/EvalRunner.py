from Runner.RunnerBase import RunnerBase
from const import evalSeed


class EvalRunner(RunnerBase):
    def __init__(self, envType,saveName) -> None:
        super().__init__(envType,saveName)
        self.resetEvalSeed()
        self._currentSeed = self._initSeed

    def resetEvalSeed(self):
        self._initSeed = evalSeed

    def setupRun(self, setupType, envSetting=None, noiseLevel=None, noiseHandlingMode=None):
        agents, env = super().setupRun(setupType, envSetting, noiseLevel, noiseHandlingMode)
        env.setSeed(self._currentSeed)
        return agents, env

    def setupEnvSetting(self, loadSave, envSetting):
        super().setupEnvSetting(loadSave, envSetting)
        # Maximal step per episode for evaluation
        self._TEST_MAX_EPS = 1000

    def test(self, verbose=2, noiseLevel=None, noiseHandlingMode=None):
        step, rewards = super().test(verbose, noiseLevel, noiseHandlingMode)
        self._currentSeed += 1
        return step, rewards
