class Seeder():
    """Seed generator for training and evaluation"""

    def __init__(self, initSeed=0) -> None:
        """Avoids duplication of train and eval seeds"""
        self._trainSeed = initSeed
        self._evalSeed = initSeed + 1

    def getTrainSeed(self):
        self._trainSeed += 2
        # print("Train Seed: ",self._trainSeed)
        return self._trainSeed

    def getEvalSeed(self):
        self._evalSeed += 2
        return self._evalSeed
