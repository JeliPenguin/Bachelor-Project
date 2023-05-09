
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
    def __init__(self, hyperParam=False, envType="FindingTreat") -> None:
        self.envType = envType
        normSaveName = "Two5X5"
        nhModel = (normSaveName, 0, "Noise_Handling")
        normNoisedModel = (normSaveName, None, "Norm_Noised")
        normModel = (normSaveName, None, "Norm")
        schedModel = ("Sched5x5", None, "Sched")
        self.modelToEvaluate = [normModel]
        if hyperParam:
            for name in os.listdir(f"./Saves/HyperParam/{envType}"):
                # print(name)
                saveName = f"HyperParam/{envType}/{name}"
                model = (saveName, 0, name)
                self.modelToEvaluate.append(model)
        else:
            self.modelToEvaluate = [normModel,
                                    schedModel, normNoisedModel, nhModel]
        # self.modelToEvaluate = self.modelToEvaluate[:4]
        self.models = self.modelToEvaluate
        self.repetitions = 100
        self.savePath = f"./Saves/Evaluation/{self.envType}/"

    def testRun(self, run: EvalRunner, noiseP, noiseHandlingMode):
        steps, reward = run.test(
            noiseP=noiseP, noiseHandlingMode=noiseHandlingMode)
        return steps, reward

    def modelEval(self, model, reEval):
        modelName = model[0]
        noiseHandling = model[1]
        saveName = model[2]
        if reEval or saveName not in os.listdir(self.savePath):
            fullSavePath = self.savePath + saveName
            os.mkdir(fullSavePath)
            run = EvalRunner(self.envType, modelName)
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

            dump(
                epsDf, f"{fullSavePath}/Eps")
            dump(
                rwdDf, f"{fullSavePath}/Rwd")

    def doPlot(self, plotType, range):
        for model in self.models:
            modelSaveName = model[2]
            epsRecord = load(
                f"{self.savePath}{modelSaveName}/{plotType}")
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
        for model in self.modelToEvaluate:
            self.modelEval(model, reEval)
        self.checkSaved(range)
