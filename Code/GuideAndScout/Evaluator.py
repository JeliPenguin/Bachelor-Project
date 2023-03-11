
from Runner.NormRunner import Runner
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Evaluator():
    def __init__(self, normSaveName, noisedSaveName) -> None:
        nhModel = (normSaveName, 0, "Noise_Handling")
        normNoisedModel = (normSaveName, None, "Norm_Noised")
        normModel = (normSaveName, None, "Norm")
        baseModel = (noisedSaveName, None, "Baseline")
        self.modelToEvaluate = [nhModel, normNoisedModel, baseModel]
        self.models = self.modelToEvaluate + [normModel]
        self.verbose = 0
        self.noiseLevels = [0, 0.001, 0.005, 0.01,
                            0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
        self.repetitions = 500

    def testRun(self, modelName, noiseLevel, noiseHandlingMode):
        run = Runner(modelName, eval=True)
        steps, reward = run.test(
            verbose=0, noiseLevel=noiseLevel, noiseHandlingMode=noiseHandlingMode)
        return steps, reward

    def reEvaluate(self, model):
        modelName = model[0]
        noiseHandling = model[1]
        saveName = model[2]
        print(f"Evaluating Model {saveName}:")
        epsRecord = []
        rwdRecord = []
        for noise in tqdm(self.noiseLevels):
            eps = []
            rwd = []
            if saveName == "Norm" or noise == 0:
                noise = None
            for _ in range(self.repetitions):
                epsE, epsR = self.testRun(modelName, noise, noiseHandling)
                eps.append(epsE)
                rwd.append(epsR)
            epsRecord.append(np.mean(eps))
            rwdRecord.append(np.mean(rwd))
        dump(epsRecord, f"./Saves/Evaluation/{saveName}Eps")
        dump(rwdRecord, f"./Saves/Evaluation/{saveName}Rwd")

    def doPlot(self, plotType):
        for model in self.models:
            modelSaveName = model[2]
            epsRecord = load(f"./Saves/Evaluation/{modelSaveName}{plotType}")
            style = None
            if modelSaveName == "Norm":
                style = "dashed"
            plt.plot(self.noiseLevels, epsRecord,
                     label=modelSaveName, linestyle=style)
        plt.xlabel("Noise Level (p)")
        plt.ylabel("Average steps per episode")
        plt.legend(loc="upper left")
        plt.show()

    def checkSaved(self):
        self.doPlot("Eps")
        self.doPlot("Rwd")

    def evaluate(self, reEval=False):
        if reEval:
            for model in self.modelToEvaluate:
                self.reEvaluate(model)
        self.checkSaved()
