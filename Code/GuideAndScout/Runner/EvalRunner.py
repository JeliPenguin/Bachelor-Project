from Runner.Runner import Runner


class EvalRunner(Runner):
    """Runner for evaluation that uses evaluation seeds for environment initialization and has 5000 timestep limit"""
    def __init__(self, envType, saveName) -> None:
        super().__init__(envType, saveName)

    def setupRun(self, setupType, envSetting=None, noiseP=None, noiseHandlingMode=None):
        agents, env = super().setupRun(setupType, envSetting, noiseP, noiseHandlingMode)
        env.setSeed(self._seeder.getEvalSeed())
        return agents, env

    def test(self, verbose=-1, noiseP=0, noiseHandlingMode=None, maxEps=5000):
        step, rewards = super().test(verbose, noiseP, noiseHandlingMode, maxEps)
        return step, rewards
