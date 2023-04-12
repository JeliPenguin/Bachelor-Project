
from Runner.EvalRunner import EvalRunner
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from const import evalNoiseLevels
import os


class Evaluator():
    def __init__(self) -> None:
        # randModel = ("Test",None,"Random")
        # randModel2 = ("Test",None,"Random2")
        # randModel3 = ("Test2",None,"Random3")
        # self.modelToEvaluate = [randModel,randModel2,randModel3]
        # self.models =self.modelToEvaluate
        # normSaveName = "Two5X5"
        # nhModel = (normSaveName, 0, "Noise_Handling")
        # normNoisedModel = (normSaveName, None, "Norm_Noised")
        # normModel = (normSaveName, None, "Norm")
        # schedModel = ("Sched5x5", None, "Sched")
        # self.modelToEvaluate = [normModel,
        #                         schedModel, normNoisedModel, nhModel]
        self.modelToEvaluate = []
        for name in os.listdir("./Saves/HyperParam"):
            # print(name)
            saveName = "HyperParam/"+name
            model = (saveName, 0, name)
            self.modelToEvaluate.append(model)
        self.models = self.modelToEvaluate
        self.repetitions = 50

    def testRun(self, run, noiseP, noiseHandlingMode):
        steps, reward = run.test(
            noiseP=noiseP, noiseHandlingMode=noiseHandlingMode)
        return steps, reward

    def reEvaluate(self, model):
        modelName = model[0]
        noiseHandling = model[1]
        saveName = model[2]
        run = EvalRunner("FindingTreat", modelName)
        print(f"Evaluating Model {saveName}:")
        epsDf = []
        rwdDf = []
        for noise in tqdm(evalNoiseLevels):
            testNoise = noise
            if saveName == "Norm" or noise == 0:
                testNoise = 0
            for _ in range(self.repetitions):
                epsE, epsR = self.testRun(run, testNoise, noiseHandling)
                epsDf.append([noise, epsE])
                rwdDf.append([noise, epsR])

        epsDf = pd.DataFrame(epsDf, columns=['Noise', 'Eps'])
        rwdDf = pd.DataFrame(rwdDf, columns=['Noise', 'Rwd'])

        dump(epsDf, f"./Saves/Evaluation/{saveName}Eps")
        dump(rwdDf, f"./Saves/Evaluation/{saveName}Rwd")

    def doPlot(self, plotType, range):
        for model in self.models:
            modelSaveName = model[2]
            epsRecord = load(f"./Saves/Evaluation/{modelSaveName}{plotType}")
            if range:
                epsRecord = epsRecord[epsRecord["Noise"] <= range]
            style = None
            if modelSaveName == "Norm":
                style = "dashed"
            sns.lineplot(data=epsRecord, x="Noise",
                         y=plotType, label=modelSaveName, linestyle=style)
            # plt.plot(self.noiseLevels, epsRecord,
            #          label=modelSaveName, linestyle=style)
        plt.xlabel("Noise Level (p)")
        plt.ylabel("Average steps per episode")
        plt.legend(loc="upper left")
        plt.show()

    def checkSaved(self, range):
        self.doPlot("Eps", range)
        # self.doPlot("Rwd")

    def evaluate(self, reEval=False, range=None):
        if reEval:
            for model in self.modelToEvaluate:
                self.reEvaluate(model)
        self.checkSaved(range)
