class Seeder():
    def __init__(self, initSeed=0) -> None:
        self._trainSeed = initSeed
        self._evalSeed = initSeed + 1

    def getTrainSeed(self):
        self._trainSeed += 2
        return self._trainSeed

    def getEvalSeed(self):
        self._evalSeed += 2
        return self._evalSeed
