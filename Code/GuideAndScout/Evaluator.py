
from Runner import Runner
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Evaluator():
    def __init__(self, normSaveName, noisedSaveName) -> None:
        normModel = (normSaveName, 0, "Norm")
        baseModel = (noisedSaveName, None, "Baseline")
        self.models = [baseModel, normModel]
        self.verbose = 0
        self.noiseLevels = [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        self.repetitions = 1000

    def testRun(self, modelName, noiseLevel, noiseHandlingMode):
        run = Runner(modelName)
        steps, reward = run.test(
            verbose=0, noiseLevel=noiseLevel, noiseHandlingMode=noiseHandlingMode)
        return steps, reward

    def reEvaluate(self, model):
        modelName = model[0]
        noiseHandling = model[1]
        saveName = model[2]
        print(f"Evaluating Model {modelName}:")
        epsRecord = []
        rwdRecord = []
        for noise in tqdm(self.noiseLevels):
            eps = []
            rwd = []
            for _ in range(self.repetitions):
                epsE, epsR = self.testRun(modelName, noise, noiseHandling)
                eps.append(epsE)
                rwd.append(epsR)
            epsRecord.append(np.mean(eps))
            rwdRecord.append(np.mean(rwd))
        dump(epsRecord, f"./Saves/Evaluation/{saveName}Eps")
        dump(rwdRecord, f"./Saves/Evaluation/{saveName}Rwd")

    def checkSaved(self):
        for model in self.models:
            modelSaveName = model[2]
            epsRecord = load(f"./Saves/Evaluation/{modelSaveName}Eps")
            plt.plot(self.noiseLevels, epsRecord, label=modelSaveName)
        plt.xlabel("Noise Level (p)")
        plt.ylabel("Average steps per episode")
        plt.legend(loc="upper left")
        plt.show()

        for model in self.models:
            modelSaveName = model[2]
            rwdRecord = load(f"./Saves/Evaluation/{modelSaveName}Rwd")
            plt.plot(self.noiseLevels, rwdRecord, label=modelSaveName)
        plt.xlabel("Noise Level (p)")
        plt.ylabel("Average return per episode")
        plt.legend(loc="upper left")
        plt.show()

    def evaluate(self, reEval=False):
        if reEval:
            for model in self.models:
                self.reEvaluate(model)
        self.checkSaved()
